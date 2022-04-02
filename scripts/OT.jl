cd(@__DIR__)
using OhMyREPL
using XLEs
using KernelDensityEstimate
using KernelDensityEstimatePlotting
using OptimalTransport
using LinearAlgebra

function convex_initialization(subX, subY; λ=.05, niters=100, aplly_sqrt=false)
    n = size(subX, 2)
    K_X = permutedims(subX) * subX
    K_Y = permutedims(subY) * subY
    K_Y *= norm(K_X) / norm(K_Y)
    K2_X = K_X * K_X
    K2_Y = K_Y * K_Y
    P = ones(n, n) / n
    μ, ν = ones(n), ones(n)
    for it in 1:niters
        println(it)
        # G = pairwise(CosineDist(), K_X, K_Y)
        G = (P * K2_X) + (K2_Y * P) - 2(K_Y * P * K_X)
        q = sinkhorn(μ, ν, G, λ)
        α = 2. / (2. + it)
        P = (α .* q) + ((1. - α) .* P)
    end
    obj = norm((P * K_X) - (K_Y * P))
    print(obj)
    return obj
end



# read embedding files
files = "../data/exp_raw/embeddings/";
root, folder, files = first(walkdir(files))

src = "en"
trg = "es"

vocX, X = readBinaryEmbeddings(root * src)
vocY, Y = readBinaryEmbeddings(root * trg)

X, Y = map(unit, [X, Y])
X, Y = map(center, [X, Y])

subX = X[:, 1:Int(2.5e3)]
subY = Y[:, 1:Int(2.5e3)]
