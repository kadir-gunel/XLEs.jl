cd(@__DIR__)

include("../src/FluxVECMap.jl")
using .FluxVECMap
using OhMyREPL
using CUDA
using ITensors
using ITensorsGPU
using Gnuplot
using LinearAlgebra
using Test

srcfile = "../data/exp_raw/embeddings/en"
voc, X  = readBinaryEmbeddings(srcfile)

# need to convert X to 300 x 20k x 10 3D Tensor
Xd3 = reshape(X, (300, Int(20e3), 10))

i = Index(300, "i")
j = Index(Int(20e3), "j")
k = Index(div(length(voc), j.space), "k")

X3T = ITensor(Xd3, i, j, k)
# Decomposing into its components by using SVD
Fi = svd(X3T, (j, k))
Fj = svd(X3T, (i, k))
Fk = svd(X3T, (i, j))
S  = svdvals(X)


@test isapprox(S, Fi.S.tensor.storage)
