cd(@__DIR__)

include("../src/FluxVECMap.jl")

using .FluxVECMap

using Statistics
using Logging
using OhMyREPL
using CUDA
using BSON
using Distributed
using Logging
addprocs(nprocs())
using Test
using Printf
@everywhere using LinearAlgebra

# Zipf's Law
#@everywhere begin zipfs(k::Int, N::Int, s::Int=1) = (1 / (k ^ s)) / reduce(+, 1 ./ (collect(1:N).^s)) end
#""" returns the probabilities of word by using Zipfs' Law"""
#getProbabilities(vsize::Int) = pmap(zipfs, 1:vsize, repeat([vsize], inner=vsize), on_error=ex->-10)

function reorderEV(E::T1; splitSize::Int64=1000) where {T1} #, dictSize::Int64=Int(20e3)) where {T1}
    #first transform E into a Tensor
    c, r = size(E)

    n = div(r, splitSize)
    T = reshape(E, (c, splitSize, n))
    idx = map(i -> cond(T[:, :, i]), 1:n) |> sortperm

    return reshape(T[:, :, idx], (c, r))
    # rng = div(dictSize, splitSize)
    # idx = i[1:rng]
    # c, r, s =  size(T[:, :, i[idx]])
    #return reshape(T[:, :, i[idx]], (c, r * s ))
end

root, folders, Embedfiles = first(walkdir("../data/exp_raw/embeddings/"))

srcfile = root * "en"
trgfile = root * trg
valfile = "../data/exp_raw/dictionaries/en-$(trg).test.txt"

x_voc, src_embeds = readBinaryEmbeddings(srcfile)
y_voc, trg_embeds = readBinaryEmbeddings(trgfile)

# 1. normalize embeddings
X = src_embeds |> normalizeEmbedding
Y = trg_embeds |> normalizeEmbedding

splitSize = 1000
c, r = size(X)
n = div(r, splitSize)
Tx = reshape(X, (c, splitSize, n))
Ty = reshape(Y, (c, splitSize, n))
Sx = pmap(svdvals, eachslice(Tx, dims=3))
Sy = pmap(svdvals, eachslice(Ty, dims=3))

Σxy = pmap.(var, [Sx, Sy])
Sσ_min2max = Σxy .|> sortperm


conX = pmap(S -> maximum(S) / minimum(S), Sx)
conY = pmap(S -> maximum(S) / minimum(S), Sy)

i = conX |> sortperm
j = conY |> sortperm

# x = i[i .== Sσ_min2max[1]]
# y = j[j .== Sσ_min2max[2]]
x = Sσ_min2max[1]
y = Sσ_min2max[2]


newS_Subx = reshape(Tx[:, :, Sσ_min2max[1][1:15]], (c, 15000)) |> svdvals
newS_Suby = reshape(Ty[:, :, Sσ_min2max[2][1:15]], (c, 15000)) |> svdvals

SX = X |> svdvals
SY = Y |> svdvals

cond(X[:, 1:Int(1e3)])
cond(reshape(Tx[:, :, x[1:15]], (c, 15000)))

cond(Y[:, 1:Int(4e3)])
cond(reshape(Ty[:, :, y[1:15]], (c, 15000)))

# preprocessing ve post processing'de freq -> rare'e gecis yapildiginda condition number nasil degisiyor ?


SubX = X[:, 1:Int(1e3)]
F = SubX |> svd
newSubX = F.U * diagm(newS_Subx) * F.Vt
newX = hcat(newSubX, X[:, Int(1e3) + 1:end])
cond(newX)
cond(X)


SubY = Y[:, 1:Int(15e3)]
Fy = SubY |> svd
newSuby = Fy.U * diagm(newS_Suby) * Fy.Vt
newY = hcat(newSuby, Y[:, Int(15e3)+1:end])
cond(newY)
cond(Y)
