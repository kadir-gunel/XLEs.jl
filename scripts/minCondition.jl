cd(@__DIR__)

using Revise

include("../src/FluxVECMap.jl")

using .FluxVECMap
using OhMyREPL
using Distributed
addprocs(nprocs())
@everywhere using LinearAlgebra
@everywhere using Statistics
# @everywhere using ITensors
using Test
using Base.Threads
using BSON: @load , @save
using CUDA

using Gnuplot


function minCondition(E; splitInit::Int64=Int(200e3))
#    Σs = []
#    while splitInit > 300
        c, r = size(E)
        n = div(r, splitInit)
        T = reshape(E, (c, n, splitInit))
        Σ = CuArray{Float32}(undef, c, size(T, 3));
        @time for i in axes(T, 3)
            @views Σ[:, i] .= (T[:, :, i]) * permutedims(T[:, :, i]) |> svdvals
        end
        σ = var(Σ, dims=1) |> vec
        idx = σ |> sortperm
    return Σ, σ, idx
end

function parallelMinCondition(E::Matrix; splitInit::Int64=Int(200e3))
#    Σs = []
#    while splitInit > 300
        c, r = size(E)
        n = div(r, splitInit)
        T = reshape(E, (c, n, splitInit))
        Σ = Array{Float32}(undef, c, size(T, 3));
        @threads for i in axes(T, 3)
            #printstyled(i, color=:red)
            #println()
            @views Σ[:, i] .= (T[:, :, i]) * permutedims(T[:, :, i]) |> svdvals
        end
        σ = var(Σ, dims=1) |> vec
        idx = σ |> sortperm
#        push!(Σs, Σ[:, idx])
#        splitInit = div(splitInit, 10)
# end
    return Σ, σ, idx
end


splitList = [Int(200e3), Int(100e3), Int(50e3), Int(20e3), Int(10e3), Int(5e3), Int(4e3), Int(2e3), Int(1e3), Int(5e2), 400, 200, 100, 10, 1]
for lang in ["en", "es", "it", "de", "fi"]
    myDict = Dict{Int, Tuple}()
    _, X = readBinaryEmbeddings("../data/exp_raw/embeddings/$(lang)")
    X = X |> normalizeEmbedding
    for each in splitList
        @time condTuple = parallelMinCondition(X, splitInit=each)
        push!(myDict, each => condTuple)
        GC.gc()
    end
    @save "../data/exp_pro/$(lang).bson" myDict
end



# for lang in ["es", "it", "de", "fi"]
for lang in ["fi"]

    _, X = readBinaryEmbeddings("../data/exp_raw/embeddings/en")
    _, Y = readBinaryEmbeddings("../data/exp_raw/embeddings/$(lang)")
    X = X |> normalizeEmbedding
    Y = Y |> normalizeEmbedding

    @info "Reading FT models"
    @load "../data/sims/FT/NU/en-$(lang)/model_3.bson" model
    src_idx = model[:src_idx]
    trg_idx = model[:trg_idx]

    X, Y = advancedMapping(X |> permutedims |> cu, Y |> permutedims |> cu, src_idx, trg_idx)
    X, Y = map(normalizeEmbedding, [X, Y])
    X, Y = map(Array, [X, Y])
#=
    for (l, E) in zip(["en", lang], [X, Y])
        evaluate(E, lang, l, "original", splitList)
    end
=#
    X_replaced = X |> cu |> replaceSingulars |> Array
    Y_replaced = Y |> cu |> replaceSingulars |> Array

    X, Y = map(normalizeEmbedding, [X_replaced, Y_replaced])
    X, Y = map(normalizeEmbedding, [X, Y])
    X, Y = map(Array, [X, Y])

    for (l, E) in zip(["en", lang], [X, Y])
        evaluate(E, lang, l, "replaced", splitList)
    end

end


function evaluate(E, lang::String, l::String, T::String, splitList)
    # splitList = [Int(200e3), Int(100e3), Int(50e3), Int(20e3), Int(10e3), Int(5e3), Int(4e3), Int(2e3), Int(1e3)]
    myDict = Dict{Int, Tuple}()
    for each in splitList
        @time condTuple = parallelMinCondition(E, splitInit=each)
        push!(myDict, each => condTuple)
        GC.gc()
    end
    @save "../data/exp_pro/en-$(lang)/$(l)_$(T).bson" myDict
end



# splitList = [Int(200e3), Int(100e3), Int(50e3), Int(20e3), Int(10e3), Int(5e3), Int(4e3), Int(2e3), Int(1e3),Int(5e2), 400, 200, 100, 10, 1]

function getKs(CΣs::Matrix)
    K = Array{Float32}(undef, size(CΣs, 2))
    @threads for i in axes(CΣs, 2)
        @views K[i] = (CΣs[:, i] |> first) / (CΣs[:, i] |> last)
    end
    return K
end


lang = ["en", "es", "it", "de", "fi"]


for (lang, lang_title) in zip(lang, lang_title)
    @load "../data/exp_pro/$(lang).bson" myDict
    splitList = collect(keys(myDict)) |> sort |> reverse
    # plotCondition(myDict, splitList)
    for i in splitList
        CΣs, σs, idx = myDict[i]
        K =  getKs(CΣs)
        color = log2.(K)
        @gp "set grid" "set key opaque" "set key top right"
        @gp :- i "set title '𝐊 for $(lang_title) '" "set label 'Dims'"
        @gp :- collect(1:length(K)) log2.(K) color "w p pt 2.5 ps 1 lc palette tit 'For $(div(splitList[1], i)) Sample (μ=$(mean(color)), σ=$(var(color)))'"  palette(:cool)
        save("../plots/$(lang)/$(i).gp")
        save(term="pngcairo size 1500,1100", output="../plots/$(lang)/$(i).png")
    end
end



targetList = ["Finnish"]
trg = ["fi"]
src = ["en"]
maptype = "replaced"


for (src, trg, target) in zip(src, trg, targetList)
    @load "../data/exp_pro/$(src)-$(trg)/$(src)_$(maptype).bson" myDict
    L1 = deepcopy(myDict)

    splitList = collect(keys(myDict)) |> sort |> reverse

    for i in splitList
        CΣs, σs, idx = L1[i]
        K =  getKs(CΣs)
        color = log2.(K)
        @gp "set grid" "set key opaque" "set key top right"
        @gp :- i "set title '𝐊 for English from Mapping of $(target) '" ylabel="𝝟" xlabel="Bag of Samples"
        @gp :- collect(1:length(K)) log2.(K) color "w p pt 2.5 ps 1 lc palette tit 'For $(div(splitList[1], i)) Sample (μ=$(mean(color)), σ=$(var(color)))'"  palette(:cool)
        save("../plots/$(src)-$(trg)/$(maptype)/$(src)/$(i).gp")
        save(term="pngcairo size 1500,1100", output="../plots/$(src)-$(trg)/$(maptype)/$(src)/$(i).png")
    end

    @load "../data/exp_pro/$(src)-$(trg)/$(trg)_$(maptype).bson" myDict
    L2 = deepcopy(myDict)



    for i in splitList
        CΣs, σs, idx = L2[i]
        K =  getKs(CΣs)
        color = log2.(K)
        @gp "set grid" "set key opaque" "set key top right"
        @gp :- i "set title '𝐊 for Mapping of $(target) '" ylabel="𝝟" xlabel="Bag of Samples"
        @gp :- collect(1:length(K)) log2.(K) color "w p pt 2.5 ps 1 lc palette tit 'For $(div(splitList[1], i)) Sample (μ=$(mean(color)), σ=$(var(color)))'"  palette(:cool)
        save("../plots/$(src)-$(trg)/$(maptype)/$(trg)/$(i).gp")
        save(term="pngcairo size 1500,1100", output="../plots/$(src)-$(trg)/$(maptype)/$(trg)/$(i).png")
    end
end



function  plotCondition(dict, splitList)
    for i in splitList
        CΣs, σs, idx = dict[i]
        K =  getKs(CΣs)
        color = log2.(K)
        @gp "set grid" "set key opaque" "set key top right"
        @gp :- i "set title '𝐊 for $(lang_title) '" "set label 'Dims'"
        @gp :- collect(1:length(K)) log2.(K) color "w p pt 2.5 ps 1 lc palette tit 'For $(div(splitList[1], i)) Sample (μ=$(mean(color)), σ=$(var(color)))'"  palette(:cool)
        save("../plots/$(lang)/$(i).gp")
        save(term="pngcairo size 1500,1100", output="../plots/$(lang)/$(i).png")
    end
end

targetList = ["Italian", "Spanish", "Finnish", "German"]
trg = ["it", "es", "fi", "de"]

for (trg, target) in zip(trg, targetList)
    training = getConditionScores("../data/sims/log_$(trg)-normal-training-condition.txt")
    updated = getConditionScores("../data/sims/log_$(trg)-updated-training-condition.txt")

    lang_title = "English-$(target)"

    @gp "set grid" "set key opaque" "set key top right"
    @gp :- "set title '𝐊 During Training of $(lang_title) '" ylabel="𝝟" xlabel="Iterations"
    @gp :- collect(1:length(training)) log2.(training) log2.(training) "w p lw 3 dt 1 tit 'Condition during Training'"
    @gp :- collect(1:length(updated)) log2.(updated) log2.(updated)    "w p lw 3 dt 1 tit 'Updated Condition during training'"
    save("../plots/$(lang_title).gp")
    save(term="pngcairo size 1500,1100", output="../plots/$(lang_title).png")
end

function getConditionScores(file::String)
    lines = readlines(file)
    lines = split.(lines[12:2:end])
    lines = reduce(hcat, lines)
    scores= parse.(Float64, lines[3, :])
    return scores
end





@load "../data/exp_pro/en.bson" myDict
S = deepcopy(myDict)

@load "../data/exp_pro/en-es/es_original.bson" myDict
T = deepcopy(myDict)

id = S |> keys |> collect |> sort |> reverse

Σs, σ, idx = S[id[7]]
K = getKs(Σs)
sorted = K |> sortperm
K_sorted = K[sorted]

@gp "set grid" "set key opaque" "set key top right"
@gp :- "set title '𝐊 for $(id[13]) '" ylabel="𝝟" xlabel="Samples"
@gp :- collect(1:length(K_sorted)) log2.(K_sorted) log2.(K_sorted) "w p lw 3 dt 1  lc palette tit 'Condition After Advanced Mapping'" palette(:cool)

@gp "set multiplot layout 5,3; set key off" :-
for i in id
    Σs, σ, idx = S[i]
    K = getKs(Σs)
    K_sorted = K |> sortperm
    @gp :- "set grid" "set key opaque" "set key top right"  :-
    @gp :- "set title '𝐊 for $(i) '" ylabel="𝝟" xlabel="Samples" :-
    @gp :- collect(1:length(K_sorted)) log2.(K_sorted) log2.(K_sorted) "w p lw 3 dt 1  lc palette tit 'Condition After Advanced Mapping'" palette(:cool) :-
end
@gp
