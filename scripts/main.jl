cd(@__DIR__)

include("../src/FluxVECMap.jl")

using .FluxVECMap
using LinearAlgebra
using Statistics
using Logging
using OhMyREPL
using CUDA
using BSON
using Distributed
using Logging
addprocs(nprocs())
using Test

# Zipf's Law
@everywhere begin zipfs(k::Int, N::Int, s::Int=1) = (1 / (k ^ s)) / reduce(+, 1 ./ (collect(1:N).^s)) end
""" returns the probabilities of word by using Zipfs' Law"""
getProbabilities(vsize::Int) = pmap(zipfs, 1:vsize, repeat([vsize], inner=vsize), on_error=ex->-10)

root, folders, Embedfiles = first(walkdir("../data/exp_raw/embeddings/"))

# for trg in ["de", "es", "fi", "it"]
for trg in ["es"]
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

    # for getting probs use: probs[src_w2i["word"]]
    # probs = getProbabilities(length(x_voc))

    for i in 1:1 # each language mapping is repeated 10 times
        io = open("../data/sims/en-$(trg)/log$(i)-zcore.txt", "a+")
        experiment_log  = SimpleLogger(io)
        with_logger(experiment_log) do
            @info "Experiment with Replacing Rare Singulars by skipping Ordinary Ones during alignment for EN-$(uppercase(trg))"
            @info "Replacement info : " freqs=Int(4e3) ordinary=Int(4e3) rares=Int(12e3)
        end

        global_logger(experiment_log)

        # 1. normalize embeddings
        X = src_embeds |> normalizeEmbedding |> cu
        Y = trg_embeds |> normalizeEmbedding |> cu


        # 2 build seed dictionary
        @info "Building seed dictionary"
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
            src_idx, trg_idx, objective, W = train(X, Y, Wt_1, src_idx, trg_idx, src_size, trg_size, keep_prob, objective; stop=stop, time=true, lambda=λ)

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

        flush(io)
        close(io)

        using BSON: @save, @load
        Wa = W |> Array
        @save "../../Wa.bson" Wa;
        s = src_idx |> Array ;
        t = trg_idx |> Array;
        @save "../../src.bson" s;
        @save "../../trg.bson" t;

        trg_idx

        XW, YW = advancedMapping(X |> permutedims, Y |> permutedims , src_idx, trg_idx)
        XW_replaced = replaceSingulars(XW |> permutedims) |> permutedims
        YW_replaced = replaceSingulars(YW |> permutedims) |> permutedims
        @time accuracy, similarity = validate(XW_replaced |> normalizeEmbedding, YW_replaced |> normalizeEmbedding, validation)
        printstyled("Ours : ", accuracy, color=:green)
        @time accuracy, similarity = validate(XW |> normalizeEmbedding, YW |> normalizeEmbedding, validation)
        printstyled("Their : ", accuracy, color=:green)

        # when writing to files need to transpose both embedding spaces!


        destfolder = "/run/media/PhD/PhD_Depo/VecMap.jl/Corr"
        writeEmbeds("$(destfolder)/en-$(trg)/SRC_MAPPED_zscore_$(i)", x_voc, XW |> Array)
        writeEmbeds("$(destfolder)/en-$(trg)/TRG_MAPPED_zscore_$(i)", y_voc, YW |> Array)

        writeEmbeds("$(destfolder)/en-$(trg)/SRC_MAPPED_OURS_zscore_$(i)", x_voc, XW_replaced)
        writeEmbeds("$(destfolder)/en-$(trg)/TRG_MAPPED_OURS_zscore_$(i)", y_voc, YW_replaced)



    end
end
