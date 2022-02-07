cd(@__DIR__)

include("../src/FluxVECMap.jl")

using .FluxVECMap
using LinearAlgebra
using Statistics
using OhMyREPL
using CUDA
using BSON: @save

#=
using Distributed
addprocs(nprocs())
using Logging
using Test

# Zipf's Law
@everywhere begin zipfs(k::Int, N::Int, s::Int=1) = (1 / (k ^ s)) / reduce(+, 1 ./ (collect(1:N).^s)) end
""" returns the probabilities of word by using Zipfs' Law"""
getProbabilities(vsize::Int) = pmap(zipfs, 1:vsize, repeat([vsize], inner=vsize), on_error=ex->-10)

=#


root, folders, Embedfiles = first(walkdir("../data/exp_raw/embeddings/"))

experiment = "NU"
preSplitInfo = SplitInfo(false, 15e3, 25e3, 160e3) # total is 200k
trnSplitInfo = SplitInfo(true, 17e3, 2e3, 1e3) # total is 20k
# postSplitInfo= preSplitInfo
numberOfExperiments = 1

for trg in ["es"] # , "de", "fi", "it"]
    @info "Language Pair is :" trg
    srcfile = root * "en"
    trgfile = root * trg
    valfile = "../data/exp_raw/dictionaries/en-$(trg).test.txt"

    x_voc, src_embeds = readBinaryEmbeddings(srcfile)
    y_voc, trg_embeds = readBinaryEmbeddings(trgfile)

    src_w2i = word2idx(x_voc);
    trg_w2i = word2idx(y_voc);
    validation = readValidation(valfile, src_w2i, trg_w2i)

    src_size = cutVocabulary(src_embeds, vocabulary_cutoff=20000)
    trg_size = cutVocabulary(src_embeds, vocabulary_cutoff=20000)

    if preSplitInfo.change
        @info "Singular Value Preprocessing:" preSplitInfo
        src_embeds = replaceSingulars(src_embeds |> cu |> permutedims, info=preSplitInfo) |> permutedims
        trg_embeds = replaceSingulars(trg_embeds |> cu |> permutedims, info=preSplitInfo) |> permutedims
    end

    for i in 1:numberOfExperiments # each language mapping is repeated 10 times
        # 1. normalize embeddings


        X = normalizeEmbedding(src_embeds) |> cu
        Y = normalizeEmbedding(trg_embeds) |> cu


        # 2 build seed dictionary
        @info "Buislding seed dictionary"
        src_idx, trg_idx = buildSeedDictionary(X, Y)

        src_size = cutVocabulary(X, vocabulary_cutoff=20000)
        trg_size = cutVocabulary(Y, vocabulary_cutoff=20000)

        @info "Starting Training"
        stochastic_interval   = 50
        stochastic_multiplier = 2.0
        stochastic_initial    = .1
        threshold = Float64(1e-6) # original threshold = Float64(1e-6)
        best_objective = objective = -100. # floating
        it = 1
        last_improvement = 0
        keep_prob = stochastic_initial
        stop = !true
        W = CUDA.zeros(size(X, 1), size(X,1))
        Wt_1 = CUDA.zeros(size(W))
        λ = Float32(1.)

        while true
            printstyled("Iteration : # ", it, "\n", color=:green)
            # increase the keep probability if we have not improved in stochastic_interval iterations
            if it - last_improvement > stochastic_interval
                if keep_prob >= 1.0
                    stop = true
                end
                keep_prob = min(1., stochastic_multiplier * keep_prob)
                println("Drop probability : ", 100 - 100 * keep_prob)
                last_improvement = it
            end

            if stop
                break
            end

            # updating training dictionary
            src_idx, trg_idx, objective, W = train(X, Y, Wt_1, src_idx, trg_idx, src_size, trg_size, keep_prob, objective; stop=stop, time=true, lambda=λ,
                                                   updateSV=trnSplitInfo)
            if objective - best_objective >= threshold
                last_improvement = it
                best_objective = objective
            end

            # validating
            if mod(it, 10) == 0
                accuracy, similarity = validate((W' * X), Y, validation)
                @info "Accuracy on validation set :", accuracy
                @info "Validation Similarity = " , similarity
            end
            it += 1
        end


        model = Dict{Symbol, Array{Int64}}(:src_idx => src_idx, :trg_idx => trg_idx)
        @save "../data/sims/FT/$(experiment)/en-$(trg)/deneme_model_$(i).bson" model

    end
end
