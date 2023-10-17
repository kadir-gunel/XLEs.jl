"""
Asked for how to optimize the search on Discourse julia and taken from
the best working solution; you can find
@https://discourse.julialang.org/t/y-a-t-q-yet-another-threads-question/71541/9
"""
function sequenceCosine(X_sub::Matrix, X::Matrix)
    [@views cosine_dist(X_sub[i, :], X[j, :])
        for i in axes(X_sub, 1),
            j in axes(X, 1)]
end

#=
function parallelMahalanobis(X_sub::Matrix, X::Matrix)

    results = similar(X, size(X_sub, 2), size(X, 2))
    @threads for j in axes(X, 2)
        for i in axes(X_sub, 2)
            results[i, j] = @views mahalanobis
end

=#

function parallelCosine(X_sub::Matrix, X::Matrix)
    results = similar(X, size(X_sub, 2), size(X, 2))
    @threads for j in axes(X, 2)
        for i in axes(X_sub, 2)
            results[i, j] = @views cosine_dist(X_sub[:, i], X[:, j])
        end
    end
    results
end


function parallelCosine(X_sub::Matrix, X::Matrix)
    results = similar(X, size(X_sub, 2), size(X, 2))
    @threads for j in axes(X, 2)
        for i in axes(X_sub, 2)
            results[i, j] = @views cosine_dist(X_sub[:, i], X[:, j])
        end
    end
    results
end

function parallelIdx(R::Matrix; k::Int64=5)
    top_idx = Matrix{Int64}(undef, size(R, 1), k)
    @threads for i in axes(R, 1)
        top_idx[i, :] = sortperm(R[i, :])[1:k]
    end
    return top_idx
end

getTopK(voc::Array{String}, DistM::Matrix, idx::Int64; k::Int64=10) = voc[DistM[:, idx][end-k:end]]




function getDistanceIDX(S::Matrix; k::Int64=size(S, 1))
    d, w = size(S)
    if d != w
        @error "Input matrix is not square! Fatal Error !"
    end
    D = Matrix{Int64}(undef, k, w)
    @threads for i in axes(S, 1)
        D[:, i] = sortperm(S[:, i])[end-k+1:end]
    end
    return D
end


"""
    randomized Singular Value Decomposition
    X: Matrix to decompose
    r: target rank
    p: oversampling parameter
    q: power iterations
    returns Fr object.
    Target rank (r) should be << than size(X, 1) - considering that the matrix is column major.

"""
function rSVD(X::Matrix, r::Int64; p::Int64=5, q::Int64=1)

    d, n = size(X)
    P = rand(Float32, r + p, d)
    Z = P * X
    # apply power iterations
    for k in 1:q

    end
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


function calculatePseudoInverse(C)
    C = C |> Array;
    PI = pinv(C) |> cu;
    PI = PI * permutedims(PI) / 2 # hence making it symmetric again
    return PI |> Array
end

function updateMahalanobisDictionary(A, B, keep_prob::Float64)
    PI = calculatePseudoInverse(permutedims(A) * B)
    sim = pairwise(Mahalanobis(PI), A |> Array, B |> Array) |> cu
    revsim = sim |> permutedims |> cu

    knnsim  = topk_mean(sim, 10, inplace=true)
    bestsim = CUDA.CUBLAS.maximum(revsim, dims=2)

    revsim  = revsim - (CUDA.ones(Float32, size(sim)) .* (transpose(knnsim / 2)))
    idx = getindex.(argmax(CUDA.CUDNN.cudnnDropoutForward(revsim, dropout=1 - keep_prob), dims=2), 2)

    return vec(idx), vec(bestsim)
end

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

function getConditionScores(file::String)
    lines = readlines(file)
    lines = split.(lines[12:2:end])
    lines = reduce(hcat, lines)
    scores= parse.(Float64, lines[3, :])
    return scores
end


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

function csls(sim; k::Int64=10)
    knn_sim_fwd = topk_mean(sim, k);
    knn_sim_bwd = topk_mean(permutedims(sim), k);
    sim -= CUDA.ones(eltype(sim), size(sim)) .* (knn_sim_fwd / 2) + CUDA.ones(eltype(sim), size(sim)) .* ((knn_sim_bwd / 2));
end
