cd(@__DIR__)

include("../src/FluxVECMap.jl")
using .FluxVECMap
using BSON: @load
using OhMyREPL
using CUDA
using LinearAlgebra
using Statistics
using Gnuplot


root, folders, Embedfiles = first(walkdir("../data/exp_raw/embeddings/"))

srcfile = root * "en"


voc, X = readBinaryEmbeddings(srcfile)

splits = SplitInfo()
X = X |> normalizeEmbedding
S = svdvals(X)
F = svdvals(X[:,1:splits.freqs])
O = svdvals(X[:, splits.freqs+1:splits.freqs+splits.ordinary])
R = svdvals(X[:, splits.freqs+splits.ordinary+1:length(voc)])

N2S = Dict{Symbol, Array{Float32}}()
push!(N2S, :Original => (S))
push!(N2S, :Frequent => (F))
push!(N2S, :Ordinary => (O))
push!(N2S, :Rare => (R))


@gp "set grid" "set key opaque" "set key bottom" "set logscale y"
@gp :- "set title 'Singular Values for Finnish'" "set label 'Dims'"
for (k, v) in N2S
    @gp :- 1:length(v) log2.(v) "with lines tit '$(string(k)) (var=$(var(v)))'"
end
Gnuplot.save("../plots/Finnish.gp")
