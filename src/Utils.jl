using Base.Iterators
using Base.Threads
using LinearAlgebra
using Statistics: mean, var
using CUDA
using Parameters
using Distances


CUDA.allowscalar(true)
atype = isequal(CUDA.functional(), true) ? cu : Array;



@with_kw struct SplitInfo
    change::Bool=false
    freqs::Int64 = (15e3)
    ordinary::Int64 = (25e3)
    rares::Int64 = (160e3)
end

@with_kw struct Postprocessing{T<:AbstractFloat, T1<:Integer, F<:Function}
    X::Matrix{T}
    Y::Matrix{T}
    src_idx::Array{T1}
    trg_idx::Array{T1}
    func::F
    info::SplitInfo
    validationSet::Dict
    src_voc::Array
    trg_voc::Array
end


# TODO this is a new function and this will effect the above and below code pieces for later usage....
# taken from FastText/alignment/utils.py


function topk_mean(sim, k; inplace=false)
    n = size(sim, 1)
    ans_ = (zeros(eltype(sim), n, 1)) |> typeof(sim)
    if k <= 0
        return ans
    end
    if !inplace
        sim = deepcopy(sim)
    end
    min_ = findmin(sim)[1]
    for i in 1:k
        vals_idx = findmax(sim, dims=2);
        ans_ += vals_idx[1]
        sim[vals_idx[2]] .= min_
    end
    return ans_ / k
end


function whitening(M::Matrix{Float32})
    F = svd(M)
    F.V * diagm(1 ./ F.S) * (F.Vt)
end

function whitening(M::CuArray)
    F = CUDA.CUBLAS.svd(M)
    F.V * cu(diagm(1 ./ F.S)) * (F.Vt)
end

simSize(X::T, Y::T; unsupervised_vocab::Int64=4000) where {T}  =
                unsupervised_vocab <= 0 ? min(size(X, 2) , size(Y, 2)) : min(size(X, 2) , size(Y, 2), unsupervised_vocab)

function cudaCorrelationMatrix(E::CuArray; sim_size::Int64=4000)
    F = CUDA.CUBLAS.svd(E[:, 1:sim_size]) # F is object of SVD
    (F.V .* F.S') * F.Vt;
end

function csls(sim; csls_neighborhood::Int64=10)
    knn_sim_fwd = topk_mean(sim, csls_neighborhood);
    knn_sim_bwd = topk_mean(permutedims(sim), csls_neighborhood);
    sim -= CUDA.ones(eltype(sim), size(sim)) .* (knn_sim_fwd / 2) + CUDA.ones(eltype(sim), size(sim)) .* ((knn_sim_bwd / 2));
end

cutVocabulary(X::T; vocabulary_cutoff::Int64=20000) where {T} = vocabulary_cutoff <= 0  ? size(X, 2) : min(size(X, 2), vocabulary_cutoff)


function findLowestConditions(E::L; n::Int=Int(20e3), rev::Bool=false) where {L}
    c, r = size(E)
    @views T = reshape(E, (c, 400, 500));
    C = @views map(i -> log2(cond(T[:, :, i] * T[:, :, i]')), collect(1:500))
    C_sorted = sortperm(C, rev=rev);
    s = div(n, 400)
    takens = collect(take(C_sorted, s)) |> sort;
    return reshape(T[:, :, takens], (c, n)) # makes a 20e3 Dictionary
end

function buildSubSpace(E::L; parts::Int=size(E, 2), rev::Bool=false) where {L}
    c, r = size(E)
    samples = div(r, parts)
    @views T = reshape(E, (c, samples, parts));
    C = Array{Float32}(undef, parts)
    @threads for i in 1:parts
        @views C[i] =  log2(cond(T[:, :, i] * permutedims(T[:, :, i])))
    end
    idx = sortperm(C, rev=rev) |> Array
    return reshape(T[:, :, idx], (c, samples * parts)), idx
end


function buildSeedDictionary(X::T, Y::T; sim_size::Int64=4000) where {T}
   # sims = map(cudaCorrelationMatrix, [X, Y])
    xsim = cudaCorrelationMatrix(X, sim_size=sim_size)
    ysim = cudaCorrelationMatrix(Y, sim_size=sim_size)
    sort!(ysim, dims=1)
    sort!(xsim, dims=1)
    # map(sim -> sort!(sim, dims=1), sims);
    xsim, ysim = map(normalizeEmbedding, [xsim, ysim])
    sim = xsim' * ysim; # actually this is still the cosine similarity from X -> Z.
    # csls_neighborhood = 10
    sim = csls(sim)

    src_idx = vec(vcat(collect(1:sim_size), permutedims(getindex.(argmax(sim, dims=1), 1))|> Array))
    trg_idx = vec(vcat((getindex.(argmax(sim, dims=2),2))|> Array, collect(1:sim_size)))

    return src_idx, trg_idx
end

relu(E::CuMatrix) = CUDA.CUBLAS.max.(0, E);

function updateDictionary(X, Y, keep_prob::Float64; direction::Symbol=:forward)
    sim = isequal(direction, :forward) ? permutedims(Y) * X  : permutedims(X) * Y
    knn_sim  = topk_mean(sim, 10, inplace=true)
    sim_rev  = isequal(direction, :forward) ? permutedims(X) * Y : permutedims(Y) * X
    best_sim = CUDA.CUBLAS.maximum(sim_rev, dims=2)
    sim_rev  = sim_rev - (CUDA.ones(Float32, size(sim)) .* (transpose(knn_sim / 2)))
    idx = getindex.(argmax(CUDA.CUDNN.cudnnDropoutForward(sim_rev, dropout=1 - keep_prob), dims=2), 2)
    return vec(idx), vec(best_sim)
end


function buildDictionary(metric::Symbol=:CosineDist)

    # 1. find the cosine distances between each sub-space
    cosx, cosy = map(space -> pairwise(eval(metric)(), space), [subx, suby]);
    map(space -> sort!(space, dims=1), [cosx, cosy]);
    cosx, cosy = map(normalizeEmbedding, [cosx, cosy]);
    sim = csls(cosx' * cosy);

    src_idx = vec(vcat(collect(1:sim_size), permutedims(getindex.(argmax(sim, dims=1), 1))|> Array))
    trg_idx = vec(vcat((getindex.(argmax(sim, dims=2),2))|> Array, collect(1:sim_size)))

    return src_idx, trg_idx
end


function update(A, B, keep_prob::Float64)
    sim = permutedims(A) * B
    revsim  = permutedims(sim)

    knnsim  = topk_mean(sim, 10, inplace=true)
    bestsim = CUDA.CUBLAS.maximum(revsim, dims=2)

    revsim  = revsim - (CUDA.ones(Float32, size(sim)) .* (transpose(knnsim / 2)))
    idx = getindex.(argmax(CUDA.CUDNN.cudnnDropoutForward(revsim, dropout=1 - keep_prob), dims=2), 2)

    return vec(idx), vec(bestsim)
end


function calcobjective(sim)
    bestfward = mean(CUDA.CUBLAS.maximum(sim, dims=2))
    bestbward = mean(CUDA.CUBLAS.maximum(permutedims(sim), dims=2))
    objective = (bestfward + bestbward) / 2
end


function mahalanobis(subx, suby; ssize::Int64=Int(2.2e2))
    d, w = size(subx)
    D = reshape(subx, (d, w, 1)) .- reshape(suby, (d, 1, w))
    D = reshape(D, (d, w * w))
    C = pinv(subx * suby') # 300x300
    sim = reshape(diag(D' * C' * D), (w, w))
    # src_idx = vec(vcat(collect(1:sim_size), permutedims(getindex.(argmax(sim, dims=1), 1))|> Array))
    # trg_idx = vec(vcat((getindex.(argmax(sim, dims=2),2))|> Array, collect(1:sim_size)))
    return sim #, src_idx, trg_idx
end

function mahalanobis1(subx, suby;sim_size::Int64=Int(4e3))
    d, w = size(subx)
    C = pinv(subx * suby')'
    distMah = zeros(w, w)
    for i in 1:w
        for j in 1:w
            dist = subx[:, i] - suby[:, j];
            distMah[i, j]  = dist' * C * dist
        end
    end
    return distMah
    src_idx = vec(vcat(collect(1:sim_size), permutedims(getindex.(argmax(distMah, dims=1), 1))|> Array))
    trg_idx = vec(vcat((getindex.(argmax(distMah, dims=2),2))|> Array, collect(1:sim_size)))
    return src_idx, trg_idx
end

function mahalanobisGPU(subx, suby)
    d, wx = size(subx)
    wy = size(suby, 2)
    data = hcat(subx, suby)
    C = cov(data |> permutedims)
    distMah = CUDA.zeros(wx, wy)
    for i in 1:wx
        dist = subx[:, i] .- suby;
        distMah[i, :] = sum((C \ dist) .* dist, dims=1)
    end
    @. distMah = sqrt(distMah)
    # sort!(distMah, dims=1)
    #src_idx = vec(vcat(collect(1:wx), permutedims(CUDA.CUBLAS.getindex.(CUDA.CUBLAS.argmax(distMah, dims=1), 1))|> Array));
    #trg_idx = vec(vcat((CUDA.CUBLAS.getindex.(CUDA.CUBLAS.argmax(distMah, dims=2),2))|> Array, collect(1:wy)));
    return distMah # , src_idx, trg_idx
end
    



function parallelMahalanobis1(subx, suby; sim_size::Int64=Int(4e3))
    d, w = size(subx)
    C = inv(subx * suby')
    distMah = similar(subx, w, w)
    
    @threads for j in 1:w
        for i in 1:w
            distMah[i, j] = @views ((subx[:, i] .- suby[:, j])' * C * (subx[:, i] .- suby[:, j]))
        end
    end
    #src_idx = vec(vcat(collect(1:sim_size), permutedims(getindex.(argmax(distMah, dims=1), 1))|> Array))
    #trg_idx = vec(vcat((getindex.(argmax(distMah, dims=2),2))|> Array, collect(1:sim_size)))
    return distMah #, src_idx, trg_idx
end

function buildDictionary(subx, suby; sim_size::Int64=Int(4e3))
    distXY = Mahalanobis(subx * suby', skipchecks=true)
    sim = pairwise(distXY, subx, suby) |> cu

    src_idx = vec(vcat(collect(1:sim_size), permutedims(getindex.(argmax(sim, dims=1), 1))|> Array))
    trg_idx = vec(vcat((getindex.(argmax(sim, dims=2),2))|> Array, collect(1:sim_size)))
    
    src_idx, trg_idx
end




function update(xw, y)
    

    

end    


function buildMahalanobisDictionary(subx::Matrix, suby::Matrix; sim_size::Int64=Int(4e3))
    distx, disty = map(space -> Mahalanobis(space * space'), [subx, suby])

    mahx = pairwise(distx, subx) |> cu
    mahy = pairwise(disty, suby) |> cu

    sort!(mahx, dims=1)
    sort!(mahy, dims=1)

    mahx, mahy = map(normalizeEmbedding, [mahx, mahy])

    sim = mahx' * mahy

    src_idx = vec(vcat(collect(1:sim_size), permutedims(getindex.(argmax(sim, dims=1), 1))|> Array))
    trg_idx = vec(vcat((getindex.(argmax(sim, dims=2),2))|> Array, collect(1:sim_size)))

    return src_idx, trg_idx, objective
end


function mapOrthogonal(X::T, Y::T; λ::Float32=Float32(1)) where {T}
    F = CUDA.CUBLAS.svd(X * Y')
    W = permutedims(F.U * F.Vt * cuinv((X * X') + λ .* CuMatrix{Float32}(I, 300, 300)))
    return W, F.S
end


function train2(X::T, Y::T, Wt_1::T, src_idx::T1, trg_idx::T1, src_size::Int64, trg_size::Int64, keep_prob::Float64,
                objective::T2; stop::Bool=false, lambda::Float32=Float32(.1)) where {T, T1, T2}

    src_idx_forward  = cu(collect(1:src_size));
    trg_idx_backward = collect(1:trg_size);

    W, _ = mapOrthogonal(X[:, src_idx], Y[:, trg_idx], λ=lambda)

    if stop
        return src_idx, trg_idx, objective, W  # returning the results
    end

    XW = W * X;

    src_idx, trg_idx, objective = buildMahalanobisDictionary(XW |> Array, Y |> Array, sim_size=src_size)
    #trg_idx_forward,  best_sim_forward  = update(Y, XW, keep_prob)
    #src_idx_backward, best_sim_backward = update(XW, Y, keep_prob)

    #src_idx = vcat(src_idx_forward, src_idx_backward)
    #trg_idx = vcat(trg_idx_forward, trg_idx_backward)

    #objective = (mean(best_sim_forward) + mean(best_sim_backward)) / 2
    return src_idx, trg_idx, objective, W
end







function train(X::T, Y::T, Wt_1::T, src_idx::T1, trg_idx::T1, src_size::Int64, trg_size::Int64, keep_prob::Float64,
               objective::T2; stop::Bool=false, time::Bool=false, lambda::Float32=Float32(.1), updateSV::SplitInfo=SplitInfo(false, 10, 20, 40)) where {T, T1, T2}

    src_idx_forward  = cu(collect(1:src_size));
    trg_idx_backward = collect(1:trg_size);

    # W = similar(Wt_1)
    W, _ = mapOrthogonal(X[:, src_idx], Y[:, trg_idx], λ=lambda)
    # W = mapViaPseudoInverse(X[:, src_idx], Y[:, trg_idx], λ=lambda)

    if time
        W += Wt_1
    end
    if stop
        return src_idx, trg_idx, objective, W  # returning the results
    end

    XW = W * X


    if updateSV.change
        # XW = permutedims(replaceSingulars(permutedims(XW[:,1:src_size]), freqs=Int(4e3), ordinary=Int(4e3), rares=Int(12e3)))
        XW = replaceSingulars(XW[:,1:src_size], info=updateSV)
    end

    trg_idx_forward,  best_sim_forward  = update(Y, XW, keep_prob)
    src_idx_backward, best_sim_backward = update(XW, Y, keep_prob)

    src_idx = vcat(src_idx_forward, src_idx_backward)
    trg_idx = vcat(trg_idx_forward, trg_idx_backward)

    objective = (mean(best_sim_forward) + mean(best_sim_backward)) / 2
    return src_idx, trg_idx, objective, W
end



word2idx(voc::Array{String,1}) = Dict(term => i for (i, term) in enumerate(voc))

function validate(XW::T, YW::T, validation::Dict) where {T}
    trg_indices = collect(values(validation));
    src_indices = collect(keys(validation));
    simval= XW[:, src_indices]' * YW;
    nn = getindex.(argmax(simval, dims=2), 2) |> Array
    accuracy = mean(in.(nn, trg_indices))
    j = collect(validation[src_indices[i]] for i in 1:length(src_indices))
    similarity = mean(maximum.(collect((simval[:, collect(flatten(j[i]))] for i in 1:length(j)))))
    return accuracy, similarity
end

function validateCSLS(XW::T, YW::T, validation::Dict) where {T}
    trg_indices = collect(values(validation));
    src_indices = collect(keys(validation));
    simbwd = zeros(eltype(YW), size(YW, 2))
    for i in 1:500:size(YW, 2)
        j = min(i + 500 - 1, size(YW, 2))
        simbwd[i:j] = topk_mean(permutedims(YW[:, i:j]) * XW, 10, inplace=true)
    end
    simvals = (2 .* permutedims(XW[:, src_indices]) * YW) .- CUDA.ones(Float32, (length(src_indices), Int(200e3))) .* cu(permutedims(simbwd));
    nn = getindex.(argmax(simvals, dims=2), 2) |> Array
    accuracy = mean(in.(nn, trg_indices))

    j = collect(validation[src_indices[i]] for i in 1:length(src_indices))
    similarity = mean(maximum.(collect((simvals[:, collect(flatten(j[i]))] for i in 1:length(j)))))

    return accuracy, similarity
end

function getIDX_NN(XW::T, YW::T, validation::Dict) where {T}
    trg_indices = collect(values(validation));
    src_indices = collect(keys(validation));
    simval= XW[:, src_indices]' * YW;
    nn = getindex.(argmax(simval, dims=2), 2) |> Array
    return nn, trg_indices
end

function getIDX_CSLS(XW::T, YW::T, validation::Dict) where {T}
    trg_indices = collect(values(validation));
    src_indices = collect(keys(validation));
    simbwd = zeros(eltype(YW), size(YW, 2))
    for i in 1:500:size(YW, 2)
        j = min(i + 500 - 1, size(YW, 2))
        simbwd[i:j] = topk_mean(permutedims(YW[:, i:j]) * XW, 10, inplace=true)
    end
    simvals = (2 .* permutedims(XW[:, src_indices]) * YW) .- CUDA.ones(Float32, (length(src_indices), Int(200e3))) .* cu(permutedims(simbwd));
    nn = getindex.(argmax(simvals, dims=2), 2) |> Array
    return nn, trg_indices
end


function cuinv(A)
    if size(A, 1) != size(A, 2) throw(ArgumentError("Matrix not square.")) end
    B = atype(Matrix{Float32}(I(size(A,1))))
    A, ipiv = CUDA.CUSOLVER.getrf!(A)
    return (CUDA.CUSOLVER.getrs!('N', A, ipiv, B))
end


function advancedMapping(X, Y, src_idx, trg_idx)

    @info "Inside Advanced Mapping"

    #step 1: Whitening
    Wx1 = whitening(X[src_idx, :])
    Wy1 = whitening(Y[trg_idx, :])
    XW  = X * Wx1 # kind of normalized embeddings
    YW  = Y * Wy1


    #step 2: Orthogonal mapping
    F = svd(XW[src_idx, :]' * YW[trg_idx, :], full=true)
    """
    # F represents U, S, V, and Vt
    in the original code U is wx2 , S is s and Vt is wz2_T
    """
    XW = XW * F.U # multiply by left  rotation F.U = wx2 in original code
    YW = YW * F.V # multiply by right rotation F.V = wz2 in original code

    #step 3: Reweighting
    src_reweight = 0.5
    trg_reweight = 0.5

    XW = XW * atype(diagm(F.S) .^ src_reweight);
    YW = YW * atype(diagm(F.S) .^ trg_reweight);

    #step 4: De-whitening;
    src_dewhiten = "src"  # || global src_dewhiten = "trg"
    XW = isequal(src_dewhiten, "src") ? XW * F.U' * cuinv(Wx1) * F.U : XW * F.Vt * cuinv(Wy1) * F.V

    trg_dewhiten = "trg" # || global trg_dewhiten = "trg"
    YW = isequal(trg_dewhiten, "src") ? YW * F.U' * cuinv(Wx1) * F.U : YW * F.Vt * cuinv(Wy1) * F.V

    return map(permutedims, [XW, YW])
end

splitEmbedding(X, rng) = X[:, rng]

function replaceSingulars(E; info::SplitInfo=SplitInfo())
    vsize = size(E, 2)
    freqs = info.freqs;
    ordinary = info.ordinary;
    rares = info.rares;
    Ef = splitEmbedding(E, 1:freqs)
    Eo = splitEmbedding(E, freqs+1:freqs+ordinary)
    Er = splitEmbedding(E, freqs+ordinary+1:vsize)

    Fs = map(svd, [Ef, Eo, Er])

    Er_new = Fs[3].U * cu(diagm(Fs[1].S)) * Fs[3].Vt
    E_new = hcat(Ef, Eo, Er_new)

end

transform(X::T) where {T} = X |> permutedims |> cu;

function validateModel(info::Postprocessing)
    XW, YW = advancedMapping(info.X |> transform, info.Y|> transform, info.src_idx, info.trg_idx);

    XW = XW |> info.func |> normalizeEmbedding
    YW = YW |> info.func |> normalizeEmbedding

    @info "Evaluation with NN :"
    validate(XW, YW, info.validationSet) |> printResults;
    @info "Evaluation with CSLS :"
    validateCSLS(XW, YW, info.validationSet) |> printResults

    printstyled("Do you want to save outputs ? (yes / no): ", color=:red)
    save = readline();
    if isequal(save, "yes")
        printstyled("Give the full path: (ex: /home/PhD/...etc/) ", color=:red)
        save = readline()
        writeEmbeds(save * "src_mapped", info.src_voc, XW |> Array)
        writeEmbeds(save * "trg_mapped", info.trg_voc, YW |> Array)
    end
end

function printResults((accuracy, similarity)::Tuple)
    printstyled("Accuracy : ", accuracy, color=:blue)
    printstyled(", Similarity : ", similarity, "\n", color=:blue)
end