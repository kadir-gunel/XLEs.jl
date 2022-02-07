cd(@__DIR__)

include("../src/FluxVECMap.jl")

using OhMyREPL
using .FluxVECMap
using BSON: @load
using CUDA
using DelimitedFiles
using Base.Iterators
using Test
using Printf

getIdxOfIncluded(list; dict=src_w2i) = findall(list .|> i -> haskey(dict, i))
getIdxOfNotIncluded(list; dict::Dict{String, Int64}=src_w2i) = findall(list .|> i -> !haskey(dict, i))


@info "Loading embedding"
srcfile = "../data/exp_raw/embeddings/en"
voc, X  = readBinaryEmbeddings(srcfile)
src_w2i = word2idx(voc);

@info "Loading Bilingual Embedding"
@load "../data/sims/FT/NU/en-it/model_1.bson" model
src_idx = model[:src_idx]
trg_idx = model[:trg_idx]
trgfile = "../data/exp_raw/embeddings/it"
_, Y = readBinaryEmbeddings(trgfile)
X, Y = map(normalizeEmbedding, [X, Y]);
XW, YW = advancedMapping(X |> permutedims |> cu, Y |> permutedims |> cu, src_idx, trg_idx)


qw = "../../vecmap/data/analogies/questions-words.txt"
lines = readdlm(qw)
idx   = getindex.(findall(lines .== ":"), 1)
categories = lines[idx, :][:, 2] .|> String

#exclude categories from the data matrix
D = lines[setdiff(1:end, idx), :] .|> String .|> lowercase


notInLists = eachcol(D) .|> getIdxOfNotIncluded
inLists = eachcol(D) .|> getIdxOfIncluded

src1, trg1 = inLists[1][setdiff(1:end, notInLists[2])], inLists[2]
src2, trg2 = inLists[3][setdiff(1:end, notInLists[4])], inLists[4]

@test length(src1) == length(trg1)
@test length(src2) == length(trg2)
@test src1 == trg1
@test src2 == trg2


@printf  " # of Out of Vocabulary words for source 1 : %.0f" length(notInLists[2])
@printf  " # of Out of Vocabulary words for source 2 : %.0f" length(notInLists[4])

nn = Array{Int64}[]
# compute nearest neighbors
for i in 1:1000:length(src1)
    j = min(i + 1000 - 1, length(src1))
    sims = permutedims(XW) * (XW[:, src2[i:j]] - XW[:, src1[i:j]] + XW[:, trg1[i:j]])
    sims[src1[i:j], j - i] .= -1
    sims[trg1[i:j], j - i] .= -1
    sims[src2[i:j], j - i] .= -1
    push!(nn, getindex.(argmax(sims, dims=1), 2) |> Array |> vec)
end



nn = nn |> flatten |> collect

P1 = permutedims(XW[:, 1:Int(100e3)]) * (XW[:, src2] - XW[:, src1] + XW[:, trg1])
P2 = permutedims(XW[:, Int(100e3)+1:end]) * XW[:, src2]
