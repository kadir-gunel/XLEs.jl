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
    top_idx = Matrix{Int64}(undef, k, size(R, 1))
    @threads for i in axes(R, 1)
            top_idx[:, i] = sortperm(R[:, i])[1:k]
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
