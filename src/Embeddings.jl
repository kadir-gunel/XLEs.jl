using Base.Threads
using Distances
using Random


function unit(matrix::T) where {T}
    norms = p_norm(matrix)
    norms[findall(isequal(0), norms)] .= 1
    return matrix ./ norms
end

function unitByVar(matrix::T) where {T}
    @warn "Use this function after center normalization !"
    d = sqrt(size(matrix, 1) - 1)
    norms = p_norm(matrix)
    norms[findall(isequal(0), norms)] .= 1
    return d * (matrix ./ norms)
end

center(matrix::T) where {T} = matrix .- (vec(mean(matrix, dims=2)))

# fast cosine similarity for Matrices
p_norm(M::T) where {T} = sqrt.(sum(real(M .* conj(M)), dims=1))
# cosine(X::T, Y::T) where {T} = diag((X ./ p_norm(X)) * (Y ./ p_norm(Y))
vecLength(X::T) where {T} = sqrt.(diag(X' * X))

normalizeEmbedding(E::T) where {T} = E |> unit |> center |> unit


"""
Asked for how to optimize the search on Discourse julia and taken from the best working solution;
    you can find @ https://discourse.julialang.org/t/y-a-t-q-yet-another-threads-question/71541/9
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


function corrAndCov(E::T, ssize::Int64=Int(4e3)) where {T}
    F = svd(E[:, 1:ssize]) # F is object of SVD
    C = (F.V .* F.S') * F.Vt;
    s = sqrt.(1.0 ./ LinearAlgebra.diag(C) )
    Corr = permutedims(s .* permutedims(C) ) .* s
    Corr[LinearAlgebra.diagind(Corr) ] .= 1.0
    C, Corr
end

function doubleCenter!(S::Matrix)
    d, w = size(S)
    if d != w
        @error "Input matrix is not square! Fatal Error!"
    end
    μ = fill(mean(S), w)
    S .= S .- mean(S, dims=1) .= mean(S, dims=2) .+ μ
    return S
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
