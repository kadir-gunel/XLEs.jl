using OhMyREPL
using XLEs
using LinearAlgebra
using Statistics
using OhMyREPL
using CUDA
using BSON
#using Distributed
using Logging
#addprocs(nprocs())
using Test
using Printf


# Zipf's Law
#@everywhere begin zipfs(k::Int, N::Int, s::Int=1) = (1 / (k ^ s)) / reduce(+, 1 ./ (collect(1:N).^s)) end
#""" returns the probabilities of word by using Zipfs' Law"""
#getProbabilities(vsize::Int) = pmap(zipfs, 1:vsize, repeat([vsize], inner=vsize), on_error=ex->-10)
function findLowestConditions2(E::L; n::Int=Int(20e3), rev::Bool=false) where {L}
    c, r = size(E)
    @views T = reshape(E, (c, n, 10));
    C = @views map(i -> log2(cond(T[:, :, i] * T[:, :, i]')), collect(1:10))
    C_sorted = sortperm(C, rev=rev)
    s = div(n, n)
    takens = collect(take(C_sorted, s)) |> sort;
    return reshape(T[:, :, takens], (c, n)) # makes a 20e3 Dictionary
end


function reorderEV(V::Array{String}, E::T1; splitSize::Int64=10000) where {T1} #, dictSize::Int64=Int(20e3)) where {T1}
    #first transform E into a Tensor
    c, r = size(E)
    n = div(r, splitSize)
    V = reshape(V, (splitSize, n))
    T = reshape(E, (c, splitSize, n))
    idx = map(i -> cond(T[:, :, i]), 1:n) |> sortperm

    return (vec(V[:, idx]), reshape(T[:, :, idx], (c, r)) )
    # rng = div(dictSize, splitSize)
    # idx = i[1:rng]
    # c, r, s =  size(T[:, :, i[idx]])
    #return reshape(T[:, :, i[idx]], (c, r * s ))
end

cd(@__DIR__)
root, folders, Embedfiles = first(walkdir("../data/exp_raw/embeddings/"))

src = "en";
trg = "es"
#for trg in ["es", "de",  "fi", "it"]
@info "Language Pair is :" trg
srcfile = root * src
trgfile = root * trg
valfile = "../data/exp_raw/dictionaries/en-$(trg).test.txt"

x_voc, src_embeds = readBinaryEmbeddings(srcfile)
y_voc, trg_embeds = readBinaryEmbeddings(trgfile)

# 1. normalize embeddings
X = src_embeds |> normalizeEmbedding |> cu
Y = trg_embeds |> normalizeEmbedding |> cu

#X = src_embeds |> unit |> zscore |> unit |> cu
#Y = trg_embeds |> unit |> zscore |> unit |> cu

#for n in [.05, .07, .1, .5, .7] # noises
#    @info "Perturbation NOISE $(n)"
#    Xp = add_gauss(src_embeds, n) |> normalizeEmbedding |> cu
#    Yp = add_gauss(trg_embeds, n) |> normalizeEmbedding |> cu

# x_voc, X = reorderEV(x_voc, X, splitSize=20000)
# y_voc, Y = reorderEV(y_voc, Y, splitSize=20000)

src_w2i = word2idx(x_voc);
trg_w2i = word2idx(y_voc);
validation = readValidation(valfile, src_w2i, trg_w2i)

# for getting probs use: probs[src_w2i["word"]]
# probs = getProbabilities(length(x_voc))

#for i in 1:1 # each language mapping is repeated 10 times

#    io = open("../data/sims/log_$(trg)-perturbation_results_$(n).txt", "a+")
#    experiment_log  = SimpleLogger(io)
#    with_logger(experiment_log) do
#        @info "Experiment with Replacing Rare Singulars by skipping Ordinary Ones during alignment for EN-$(uppercase(trg))"
#        @info "Replacement info : " freqs=Int(4e3) ordinary=Int(4e3) rares=Int(12e3)
#    end

#    global_logger(experiment_log)

    # calculate condiiton numbers
 #   @info "Calculating 𝝟s"
    # @time S = findLowestConditions2(X |> Array, rev=false) |> cu
    # @time T = findLowestConditions2(Y |> Array, rev=false)  |> cu

    # 2 build seed dictionary
  #  x_interval = 1:Int(40e3)
    #y_interval = 1:Int(20e3)
    # @time Xsub, x_idx = buildSubSpace(X |> Array)
    # @time Ysub, y_idx = buildSubSpace(Y |> Array, parts=10)

    # Ty = reshape(Y, (300, Int(20e3), 10));
    # T = Ty[:, :, x_idx]

    # sorted_x_idx = sort(x_idx[1:Int(20e3)])
    # sorted_y_idx = sort(y_idx[1:Int(20e3)])

    #=
    @info "𝜥 for X (first $(interval)) space = " cond(X[:, interval] * X[:, interval]')
    @info "𝜥 for X vocabulary space = " cond(X[:, 1:Int(4e3)] * X[:, 1:Int(4e3)]')
    @info "𝜥 for Xsub space = " cond(Xsub * Xsub')
    @info "𝜥 for Xsub vocabulary space = " cond(Xsub[:, 1:Int(4e3)] * Xsub[:, 1:Int(4e3)]')

    @info "𝑲 for Y (first $(interval))space = " cond(Y[:, interval] * Y[:, interval]')
    @info "𝜥 for Y vocabulary space = " cond(Y[:, 1:Int(4e3)] * Y[:, 1:Int(4e3)]')
    @info "𝜥 for Ysub space = " cond(Ysub * Ysub')
    @info "𝜥 for Ysub vocabulary space = " cond(Ysub[:, 1:Int(4e3)] * Ysub[:, 1:Int(4e3)]')
    @info "𝜥 for Ysub vocabulary space = " cond(Ysub[:, sorted_y_idx[1:Int(4e3)]] * Ysub[:, sorted_y_idx[1:Int(4e3)]]')
    =#

    # x_re = vcat(sort(x_idx[1:Int(4e3)]), sort(x_idx[Int(4e3+1):end]))
    #y_re = vcat(y_idx[1:Int(4e3)], sort(y_idx[Int(4e3+1):end]))

    # Xsub, Ysub = map(cu, [Xsub, Ysub])

    @info "Building seed dictionary"
    # @time src_idx, trg_idx = buildSeedDictionary(@view(Xsub[:, 1:Int(20e3)]), @view(T[:, :, 1])) #sub[:, 1:Int(20e3)]))
    src_idx, trg_idx = buildSeedDictionary(@view(X[:, 1:Int(20e3)]), @view(Y[:, 1:Int(20e3)]))
    #src_idx, trg_idx = softSeedDictionary(@view(X[:, 1:Int(20e3)]), @view(Y[:, 1:Int(20e3)]))
    src_size = cutVocabulary(X, vocabulary_cutoff=20000)
    trg_size = cutVocabulary(Y, vocabulary_cutoff=20000)

    @info "Starting Training"
    conditions = Float64[];
    validConditions = Float64[];
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
    λ = Float32(1)
    # X_sub = deepcopy(X[:, 1:Int(20e3)])
    # Y_sub = deepcopy(Y[:, 1:Int(20e3)])

    #         CUDA.GC.gc()
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
        src_idx, trg_idx, objective, W = train(@view(X[:, 1:Int(20e3)]), @view(Y[:, 1:Int(20e3)]), Wt_1, src_idx, trg_idx, src_size, trg_size, keep_prob, objective; stop=stop, time=true, lambda=λ)
        if objective - best_objective >= threshold
            last_improvement = it
            best_objective = objective
        end

        # validating
        if mod(it, 10) == 0
            accuracy, similarity = validate(W * X, Y, validation)
            @info "Accuracy on validation set :", accuracy
            @info "Validation Similarity = " , similarity
        end

        it += 1

    end

   # flush(io)
   # close(io)

    # @info src_idx
    # @info trg_idx




    XW, YW = advancedMapping(X |> permutedims, Y |> permutedims , src_idx, trg_idx)
    # XW_replaced = replaceSingulars(XW |> permutedims) |> permutedims
    #YW_replaced = replaceSingulars(YW |> permutedims) |> permutedims
     @time accuracy, similarity = validate(XW |> normalizeEmbedding, YW |> normalizeEmbedding, validation)
    # @time accuracy, similarity = validateCSLS(XW |> normalizeEmbedding, YW |> normalizeEmbedding, validation)
    #printstyled("Ours : ", accuracy, color=:green)
    #@time accuracy, similarity = validate(XW |> normalizeEmbedding, YW |> normalizeEmbedding, validation)
    #printstyled("Their : ", accuracy, color=:green)

    # when writing to files need to transpose both embedding spaces!


    # destfolder = "/run/media/PhD/PhD_Depo/VecMap.jl/Corr"
    # writeEmbeds("$(destfolder)/en-$(trg)/SRC_MAPPED_zscore_$(i)", x_voc, XW |> Array)
    # writeEmbeds("$(destfolder)/en-$(trg)/TRG_MAPPED_zscore_$(i)", y_voc, YW |> Array)

    # writeEmbeds("$(destfolder)/en-$(trg)/SRC_MAPPED_OURS_zscore_$(i)", x_voc, XW_replaced)
    # writeEmbeds("$(destfolder)/en-$(trg)/TRG_MAPPED_OURS_zscore_$(i)", y_voc, YW_replaced)



  #   end
# end
