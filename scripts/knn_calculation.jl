cd(@__DIR__)
include("../src/FluxVECMap.jl")

using .FluxVECMap

using OhMyREPL
using LinearAlgebra
using Statistics
using Test
using Random
using Gnuplot
using Distances
using Base.Threads

file = "../data/exp_raw/embeddings/es"
voc, X = readBinaryEmbeddings(file);
X20k = X[:, 1:Int(20e3)];


# first lets find out the most related words by using knn and cosine similarity.
# let k be 10
# we will consider the first 4k words like in the original implementation as classes.
# then we will ensemble 10 times

# lets create a search space for each words
# so for each word we will search all 200k words!

function sequenceCosine(X_sub::Matrix, X::Matrix)
    [@views cosine_dist(X_sub[i, :], X[j, :])
        for i in axes(X_sub, 1),
            j in axes(X, 1)]
end

function parallelCosine2(X_sub::Matrix, X::Matrix)
    results = similar(X, size(X_sub, 2), size(X, 2))
    @threads for j in axes(X, 2)
        for i in axes(X_sub, 2)
            results[i, j] = @views cosine_dist(X_sub[:, i], X[:, j])
        end
    end
    results
end

function parallelIdx(R::Matrix; k::Int64=5)
    top_idx = Matrix{Int64}(undef, k, size(result, 1))
    @threads for i in axes(R, 1)
            top_idx[:, i] = sortperm(R[:, i])[1:k]
        end
    return top_idx
end



# check the cosine distance of first 20k words.
@time result = parallelCosine(X20k, X);
@time top_idx = parallelIdx(result, k = 5);



file = file * "_topk"
writeBinaryEmbeddings(file, top_idx, String[])


k = 10; # top k elements
topk = top_idx[1:k, :];


1.19209f-7  1.0488    1.03069     1.10552   0.981959  1.00398   1.05661   1.03356   …  0.98891   1.11964   0.96298   1.13395   1.07702   0.995102  1.01376
 1.0488      0.0       0.618532    0.418508  0.701253  0.500407  0.994802  0.442084     1.05649   0.982523  0.919132  0.903287  1.01196   1.06549   0.898851
 1.03069     0.618532  1.19209f-7  0.772343  0.645236  0.544086  0.803002  0.671888     1.06507   1.04487   1.04408   0.888778  1.01285   0.906643  0.804992
 1.10552     0.418508  0.772343    0.0       0.784617  0.697685  0.786459  0.581317     0.947137  1.14005   0.989808  0.912624  0.950418  1.12626   1.02342
 0.981959    0.701253  0.645236    0.784617  0.0       0.728689  0.758244  0.556571     1.19752   1.14283   0.969246  0.938419  0.988834  1.1036    0.866074
