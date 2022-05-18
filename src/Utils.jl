using Base.Iterators
using Base.Threads
using LinearAlgebra
using Statistics: mean, var
using CUDA
using Parameters
using Distances
using TransferEntropy
using Printf
using RandomizedLinAlg
using VectorizedReduction
using LoopVectorization
using MKL


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
        return ans_
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

function topk_mean_cpu(sim, k; inplace=false)
    n = size(sim, 1)
    ans_ = zeros(eltype(sim), n, 1)
    if k <= 0
        return ans_
    end
    if !inplace
        sim = deepcopy(sim)
    end
    min_ = vtfindmin(sim)[1]
    for i in 1:k
        vals_idx = vtfindmax(sim, dims=2)
        ans_ += vals_idx[1]
        sim[vals_idx[2]] .= min_
    end
    return ans_ / k
end

function topk_mean_cpu2(distM::Matrix, k; inplace=false)
    n = size(distM, 1)
    ans_ = zeros(eltype(distM), n, 1)
    if k <= 0
        return ans_
    end
    if !inplace
        distM = deepcopy(distM)
    end
    min_ = vtfindmax(distM)[1]
    for i in 1:k
        vals_idx = vtfindmin(distM, dims=2)
        ans_ += vals_idx[1]
        distM[vals_idx[2]] .= min_
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

function sqrt_eigen(subE)
    F = svd(subE)
    F.U * diagm(sqrt.(F.S)) * F.Vt
end

function correlation(E::Matrix; ssize::Int64=Int(4e3))
    F = svd(E[:, 1:ssize])
    F.V * diagm(F.S) * F.Vt
end

function cudaCorrelationMatrix(E::CuArray; sim_size::Int64=4000)
    F = CUDA.CUBLAS.svd(E[:, 1:sim_size]) # F is object of SVD
    (F.V .* F.S') * F.Vt;
end




function csls(sim; k::Int64=10)
    knn_sim_fwd = topk_mean(sim, k);
    knn_sim_bwd = topk_mean(permutedims(sim), k);
    sim -= CUDA.ones(eltype(sim), size(sim)) .* (knn_sim_fwd / 2) + CUDA.ones(eltype(sim), size(sim)) .* ((knn_sim_bwd / 2));
end

cutVocabulary(X::T; vocabulary_cutoff::Int64=20000) where {T} = vocabulary_cutoff <= 0  ? size(X, 2) : min(size(X, 2), vocabulary_cutoff)

function rotateCosine(FromCosine::Matrix, ToCosine::Matrix)
    F = rsvd(FromCosine * ToCosine, 10, 4)
    W = F.V * F.U'
    W * FromCosine
end

function buildMahalanobisDictionary(X::Matrix, Y::Matrix; sim_size::Int64=Int(4e3))
    @time mahX = pairwise(Mahalanobis(X * X'), X) .- 1# |> dist2sim
    @time mahY = pairwise(Mahalanobis(Y * Y'), Y) .- 1# |> dist2sim

    sort!(mahX, dims=1)
    sort!(mahY, dims=1)

    mahX, mahY = map(normalizeEmbedding, [mahX, mahY])
    distance = mahX' * mahY
    revDistance = permutedims(distance)

    src_idx = vec(vcat(collect(1:sim_size), permutedims(getindex.(argmax(distance, dims=1), 1))|> Array))
    trg_idx = vec(vcat((getindex.(argmax(revDistance, dims=2),2))|> Array, collect(1:sim_size)))
    return src_idx, trg_idx
end


function buildCosineDictionary(X::Matrix, Y::Matrix; dsize::Int64=Int(4e3))
    cosx = pairwise(CosineDist(), X) .- 1
    cosy = pairwise(CosineDist(), Y) .- 1

    sort!(cosx, dims=1)
    sort!(cosy, dims=1)

    cosx, cosy = map(normalizeEmbedding, [cosx, cosy])
    distance = cosx' * cosy
    revDistance = permutedims(distance)

    src_idx = getindex.(argmax(distance, dims=1), 1) |> permutedims |> vec
    trg_idx = getindex.(argmax(revDistance, dims=2), 2) |> vec

    vcat(collect(1:dsize), src_idx) , vcat(trg_idx, collect(1:dsize))
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

    sim = csls(sim, k=10)

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

dist2sim(M::Matrix) = 1 ./ exp.(M)



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


function calculateMahalanobis(M::Matrix)
    simA = pairwise(Mahalanobis(M * M'), M);
end



function calculateMahalanobis(From::Matrix, To::Matrix)
    Q = A * B';
    Q = (Q * Q') ./ 2
    distAB = pairwise(Mahalanobis(Q, skipchecks=true), A, B)
    return distAB |> dist2sim
end

function updateMahalanobis(From::Matrix, To::Matrix, keep_prob::Float64)
    mahF = calculateMahalanobis(From) |> dist2sim |> cu
    mahT = calculateMahalanobis(To) |> dist2sim |> cu

    sim = CuMatrix{Float32}(undef, size(mahF))
    CUDA.CUBLAS.gemm!('T', 'N', Float32(1.), mahF, mahT, Float32(0.), sim)

    # sim  = permutedims(mahF) * mahT
    revsim = sim |> permutedims;

    knnsim = topk_mean(sim, 10, inplace=true)
    bestsim = CUDA.CUBLAS.maximum(revsim, dims=2) |> vec |> Array
    idx = getindex.(argmax(CUDA.CUDNN.cudnnDropoutForward(revsim, dropout=1 - keep_prob), dims=2), 2) |> vec |> Array
    return idx, bestsim
end




#=
function updateMahalanobis(A::Matrix, B::Matrix, keep_prob::Float64)
    mahA = calculateMahalanobis(A);
    mahB = calculateMahalanobis(B);

    sim = permutedims(mahA) * mahB
    revsim = sim |> Array |> permutedims |> cu

    knnsim  = topk_mean(sim, 10, inplace=true)
    bestsim = CUDA.CUBLAS.maximum(revsim, dims=2)

    revsim  = revsim - (CUDA.ones(Float32, size(sim)) .* (transpose(knnsim / 2)))
    idx = getindex.(argmax(CUDA.CUDNN.cudnnDropoutForward(revsim, dropout=1 - keep_prob), dims=2), 2)

    return vec(idx), vec(bestsim)
end
=#

function buildMIDictionary(subx::Matrix, suby::Matrix)
    @printf "Calculating Mutual Information : \n"
    d, w = size(subx)
    mix = Matrix{Float32}(undef, w, w);
    miy = Matrix{Float32}(undef, w, w);
    @threads for i in axes(subx, 2)
        for j in axes(subx, 2)
            @views mix[i, j] = mutualinfo(subx[:, i], subx[:, j], Kraskov2(5))
            @views miy[i, j] = mutualinfo(suby[:, i], suby[:, j], Kraskov2(5))
        end
    end
    simX = mix |> dist2sim |> cu
    simY = miy |> dist2sim |> cu
    @printf "Calculating Similarities : \n"
    sort!(simX, dims=1)
    sort!(simY, dims=1)

    simX, simY = map(normalizeEmbedding, [simX, simY])
    sim = simX' * simY

    src_idx = vec(vcat(collect(1:w), permutedims(getindex.(argmax(sim, dims=1), 1))|> Array))
    trg_idx = vec(vcat((getindex.(argmax(sim, dims=2),2))|> Array, collect(1:w)))
    return src_idx, trg_idx
end


function buildMIDictionary2(subx::Matrix, suby::Matrix)
    @printf "Calculating Mutual Information : \n"
    d, w = size(subx);
    dist = Matrix{Float32}(undef, w, w);
    @threads for i in axes(subx, 2)
        for j in axes(suby, 2)
            dist[i, j] = @views mutualinfo(subx[:, i], suby[:, j], Kraskov2(5))
        end
    end
    sim = dist |> dist2sim |> normalizeEmbedding |> cu;
    src_idx = vec(vcat(collect(1:w), permutedims(getindex.(argmax(sim, dims=1), 1))|> Array))
    trg_idx = vec(vcat((getindex.(argmax(sim, dims=2),2))|> Array, collect(1:w)))
    return src_idx, trg_idx
end


function updateMahalanobisDictionary(From::Matrix, To::Matrix; sim_size::Int64=Int(4e3))
    mahF = pairwise(Mahalanobis(From * From'), From) .- 1 |> cu
    mahT = pairwise(Mahalanobis(To * To'), To) .-1  |> cu


    #sort!(mahF, dims=1)
    #sort!(mahT, dims=1)

    mahF, mahT = map(normalizeEmbedding, [mahF, mahT])
    sim = mahF' * mahT # similarity between the two
    revSim = permutedims(sim)

    bestsim = CUDA.CUBLAS.maximum(revSim, dims=1) |> vec |> Array
    idx = getindex.(argmax(revSim, dims=1), 1) |> vec|> Array

    return idx, bestsim
end


function update(from::CuMatrix, to::CuMatrix, keep_prob::Float64)
    sim = permutedims(from) * to
    revsim  = permutedims(sim)

    knnsim  = topk_mean(sim, 10, inplace=true)
    bestsim = CUDA.CUBLAS.maximum(revsim, dims=2) |> vec |> Array

    revsim  = revsim - (CUDA.ones(Float32, size(sim)) .* (transpose(knnsim / 2)))
    idx = getindex.(argmax(CUDA.CUDNN.cudnnDropoutForward(revsim, dropout=1 - keep_prob), dims=2), 2) |> vec |> Array

    return idx, bestsim
end

function rerankMahalanobis(from::Matrix, to::Matrix)
    mahF = pairwise(Mahalanobis(from * from'), from) |> dist2sim |> cu
    mahT = pairwise(Mahalanobis(to * to'), to) |> dist2sim |> cu
    # cosFrom = from' * from # |> Array # symmetric
    # cosTo   = to' * to # |> Array # symmetric

    # cosRFrom = rotateSimilarity(cosFrom, cosTo) # this is no more a symmetric matrix !
    # revCosRFrom = permutedims(cosRFrom)

    revFrom = permutedims(mahF)

    knnsim = topk_mean(mahF, 10, inplace=true)
    bestsim = CUDA.CUBLAS.maximum(revFrom, dims=2) |> vec |> Array

    revFrom = revFrom - (CUDA.ones(Float32, size(revFrom)) .* (transpose(knnsim / 2)))
    idx = getindex.(argmax(revFrom, dims=2), 2) |> vec |> Array

    return idx, bestsim
end


function rerank(from::CuMatrix, to::CuMatrix)
    c, r = size(from)
    F = CuMatrix{Float32}(undef, r, r)
    T = CuMatrix{Float32}(undef, r, r)
    CUDA.CUBLAS.gemm!('T', 'N', cu(Float32(1.0)), from, from, cu(Float32(0.0)), F)
    CUDA.CUBLAS.gemm!('T', 'N', cu(Float32(1.0)), to,   to,   cu(Float32(0.0)), T)

    F = F |> Array
    T = T |> Array

    rotF = rotateSimilarity(F, T) # this is no more a symmetric matrix !
    revRF = permutedims(rotF)

    knnsim = topk_mean(rotF, 10, inplace=true)
    bestsim = CUDA.CUBLAS.maximum(revRF, dims=2) |> vec |> Array

    revRF = revRF - (CUDA.ones(Float32, size(revRF)) .* (transpose(knnsim / 2)))
    idx = getindex.(argmax(revRF, dims=2), 2) |> vec |> Array

    return idx, bestsim
end


function mapOrthogonal(X::T, Y::T; λ::Float32=Float32(1)) where {T}
    F = CUDA.CUBLAS.svd(X * Y')
    W = permutedims(F.U * F.Vt * cuinv((X * X') + λ .* CuMatrix{Float32}(I, 300, 300)))
    return W, F.S
end

# taken from LoopVectorization



function rotateSimilarity(fromSim::Matrix, toSim::Matrix)::CuMatrix{Float32}
    f = 40
    M = Matrix{Float32}(undef, size(fromSim))
    T = CuMatrix{Float32}(undef, size(fromSim))
    LinearAlgebra.BLAS.gemm!('N', 'N', Float32(1.), fromSim, toSim, Float32(0.), M)

    F = rsvd(M, f, 4);
    LinearAlgebra.BLAS.gemm!('T', 'T', Float32(1.), Float32.(F.Vt), Float32.(F.U), Float32(0.), M)

    CUDA.CUBLAS.gemm!('N', 'N', cu(Float32(1.)), cu(M), cu(fromSim), cu(Float32(0.)), T)
    CUDA.CUBLAS.gemm!('N', 'T', cu(Float32(1.)), T, cu(M), cu(Float32(0.)), T)
    return T
end


function main(X, Y, src_idx, trg_idx, validation; src_size=Int(20e3), trg_size=Int(20e3), Wt::CuMatrix=cu(zeros(size(X))))
    @info "Starting Training"
    stochastic_interval   = 50
    stochastic_multiplier = 2.0
    stochastic_initial    = .1
    threshold = Float64(1e-6) # original threshold = Float64(1e-6)
    best_objective = objective = -100. # floating
    it = 1
    last_improvement = 0
    keep_prob = stochastic_initial
    stop = !true
    W = CUDA.zeros(size(X))
    λ = Float32(1)
    time = true
    while true
        printstyled("Iteration : # ", it, "\n", color=:green)
        # increase the keep probability if we have not improved in stochastic_interval iterations
        if it - last_improvement > stochastic_interval
            if keep_prob >= 1.0
                stop = true
            end
            keep_prob = min(1., stochastic_multiplier * keep_prob)
            println("Drop probability : ", 100 - 100 * keep_prob)
            last_improvement = it
        end

        if stop
            break
        end

        # src_idx, trg_idx, objective, W = XLEs.train2(X[:, 1:src_size], Y[:, 1:trg_size], Wt_1, src_idx, trg_idx, src_size, trg_size, keep_prob, objective; stop=stop, time=true, lambda=λ)


        # updating training dictionary
        src_idx, trg_idx, objective, W = train2(X[:, 1:src_size], Y[:, 1:trg_size], Wt, src_idx, trg_idx, src_size, trg_size, keep_prob, objective; stop=stop, time=time, lambda=λ)

        # time = false

        if objective - best_objective >= threshold
            last_improvement = it
            best_objective = objective
        end

        # validating
        if mod(it, 10) == 0
            accuracy, similarity = validate((W * X), Y, validation)
            @info "Accuracy on validation set :", accuracy
            @info "Validation Similarity = " , similarity
        end

        it += 1

    end
    return W, src_idx, trg_idx
end



function train2(X::T, Y::T, Wt_1::T, src_idx::T1, trg_idx::T1, src_size::Int64, trg_size::Int64, keep_prob::Float64,
               objective::T2; stop::Bool=false, time::Bool=false, lambda::Float32=Float32(.1)) where {T, T1, T2}

       W, _ = mapOrthogonal(X[:, src_idx], Y[:, trg_idx], λ=lambda)
       # W = mapViaPseudoInverse(X[:, src_idx], Y[:, trg_idx], λ=lambda)


       if stop
           return src_idx, trg_idx, objective, W  # return the results
       end

       WX = W  * X
       WY = W' * Y

       src_aligned_idx, _ = update(WX, Y, keep_prob) # from rotated x to y (Ŷ -> Y)
       trg_aligned_idx, _ = update(WY, X, keep_prob) # from rotated y to x (X̂ -> X)

       # need to keep indices !!
       srcPairs = pairs(IndexStyle(1:src_size), src_aligned_idx)
       trgPairs = pairs(IndexStyle(1:trg_size), trg_aligned_idx)

       # here we re-rank the similarity matrices
       src_aligned_idx, best_sim_src = rerank(WX[:, src_aligned_idx], Y)
       trg_aligned_idx, best_sim_trg = rerank(WY[:, trg_aligned_idx], X)

       src_idx = vcat(1:src_size, srcPairs[src_aligned_idx])
       trg_idx = vcat(trgPairs[trg_aligned_idx], 1:trg_size)

       objective = (mean(best_sim_src) + mean(best_sim_trg)) / 2
       return src_idx, trg_idx, objective, W

end



function train(X::T, Y::T, Wt_1::T, src_idx::T1, trg_idx::T1, src_size::Int64, trg_size::Int64, keep_prob::Float64,
               objective::T2; stop::Bool=false, time::Bool=false, lambda::Float32=Float32(.1),
               updateSV::SplitInfo=SplitInfo(false, 10, 20, 40)) where {T, T1, T2}

    #src_idx_forward  = cu(collect(1:src_size));
    #trg_idx_backward = collect(1:trg_size);

    # W = similar(Wt_1)
    W, _ = mapOrthogonal(X[:, src_idx], Y[:, trg_idx], λ=lambda)
    # W = mapViaPseudoInverse(X[:, src_idx], Y[:, trg_idx], λ=lambda)


    if stop
        return src_idx, trg_idx, objective, W  # returning the results
    end


    if updateSV.change
        # XW = permutedims(replaceSingulars(permutedims(XW[:,1:src_size]), freqs=Int(4e3), ordinary=Int(4e3), rares=Int(12e3)))
        XW = replaceSingulars(XW[:,1:src_size], info=updateSV)
    end

    WX = W  * X #|> Array
    WY = W' * Y #|> Array

#    src_idx_backward, best_sim_backward = update(WX, Y, keep_prob) # from wx to y
#    trg_idx_forward,  best_sim_forward  = update(Y, WX, keep_prob) # from y to wx


   src_aligned_idx, best_sim_src = update(WX, Y, keep_prob) # from rotated x to y (Ŷ -> Y)
   trg_aligned_idx, best_sim_trg = update(WY, X, keep_prob) # from rotated y to x (X̂ -> X)

    # src_aligned_idx, best_sim_src = updateMahalanobisDictionary(WX, Array(Y)) # from rotated x to y (Ŷ -> Y)
    # trg_aligned_idx, best_sim_trg = updateMahalanobisDictionary(WY, Array(X)) # from rotated y to x (X̂ -> X)


    # here we re-rank the similarity matrices

    src_idx = vcat(1:src_size, src_aligned_idx)
    trg_idx = vcat(trg_aligned_idx, 1:trg_size)

    objective = (mean(best_sim_src) + mean(best_sim_trg)) / 2
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
