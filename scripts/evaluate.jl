cd(@__DIR__)

include("../src/FluxVECMap.jl")
using .FluxVECMap
using BSON: @load, @save
using OhMyREPL
using CUDA
using LinearAlgebra
using Statistics
using StatsBase
using Base.Threads

"""
Splits the embedding space (E) into 2 equivalent subspaces.
Then transfers from one subspace to another described by
the splitSize.
"""
function oneToOneMapping(E::CuMatrix; splitSize::Int=200)
    c, r = size(E)
    searchOn = div(200, 2)
    samples = div(r, splitSize)
    @views Tx = reshape(E, (c, samples, splitSize))
    for i in 1:searchOn
        Σ = @view(Tx[:, :, i]) |> svdvals
        F = @view(Tx[:, :, i + searchOn]) |> svd;
        @views Tx[:, :, i + searchOn] .= F.U * cu(diagm(Σ)) * F.Vt
    end
    @views newX = reshape(Tx, (c, r))
    return newX;
end

function oneToManyMapping(E::Matrix; splitSize::Int=200, searchOn::Int=50)
    # 1'tanesini hepsine esitleme islemi
    c, r = size(E)
    condList = zeros(Float32, searchOn)
    samples = div(r, splitSize)
    @views Tx = reshape(E, (c, samples, splitSize))
    Fr = E[:, Int(60e3)+1:end] |> svd;
    @threads for i in 1:searchOn
        Σ  = reshape(@view(Tx[:, :, 1:i]), (c, i * samples)) |> svdvals
        Er = Fr.U * diagm(Σ) *  Fr.Vt;
        newE = hcat(E[:, 1:Int(60e3)], Er)
        condList[i] =  cond(newE * newE')
    end
    return condList
end

function getBestOneToManyMapping(E::Matrix, condList::Vector{Float32}; splitSize::Int=200)
    m = findmin(condList)[2]
    c, r = size(E)
    samples = div(r, splitSize)
    @views Tx = reshape(E, (c, samples, splitSize))
    Fr = E[:, Int(60e3)+1:end] |> svd;
    Σ  = reshape(@view(Tx[:, :, 1:m]), (c, m * samples)) |> svdvals;
    Er = Fr.U * diagm(Σ) *  Fr.Vt;
    newE = hcat(E[:, 1:Int(60e3)], Er)
    return newE
end




conds = Dict{String, Any}();

root, folders, Embedfiles = first(walkdir("../data/exp_raw/embeddings/"))

trg = "es"
@info "Language Pair is :" trg
srcfile = root * "en"
trgfile = root * trg
valfile = "../data/exp_raw/dictionaries/en-$(trg).test.txt"


@info "Reading Embeddings"
@time src_voc, src_embeds = readBinaryEmbeddings(srcfile);
@time trg_voc, trg_embeds = readBinaryEmbeddings(trgfile);

@info "Reading validation file"
src_w2i = word2idx(src_voc);
trg_w2i = word2idx(trg_voc);
@time validationSet = readValidation(valfile, src_w2i, trg_w2i)

@info "Applying Normalization"
X, Y = map(normalizeEmbedding, [src_embeds, trg_embeds]);



knn1to1 = []
knn1toM = []
csls1toM = []
csls1to1= []
condX1to1 = []
condY1to1 = []
condX1toM = []
condY1toM = []
for i in 1:10
    @info "Reading FT models"
    @load "../data/sims/FT/NU/en-$(trg)/model_$(i).bson" model
    NUmodel = deepcopy(model)
    src_idx = NUmodel[:src_idx]
    trg_idx = NUmodel[:trg_idx]
    # evalOriginal     = Postprocessing(X, Y, src_idx, trg_idx, x -> x, SplitInfo(change=false), validationSet, src_voc, trg_voc)
    # evalOriginalPost = Postprocessing(X, Y, src_idx, trg_idx, replaceSingulars, SplitInfo(change=true, freqs=Int(15e3), ordinary=0, rares=Int(200e3-15e3)), validationSet, src_voc, trg_voc)

    #@info "Original Model Evaluation : "
    # printstyled("Original Model - no post-processing: ", color=:red)
    # evalOriginal |> validateModel |> println
    # printstyled("Original Model w/ post-processing: ", color=:red)
    # evalOriginalPost |> validateModel |> println


    XW, YW = advancedMapping(X |> permutedims |> cu, Y |> permutedims |> cu, src_idx, trg_idx)
    XW, YW = map(normalizeEmbedding, [XW, YW])


    @time condListX = oneToManyMapping(XW |> Array)
    @time newX = getBestOneToManyMapping(XW |> Array, condListX) |> normalizeEmbedding |> cu

    @time condListY = oneToManyMapping(YW |> Array)
    @time newY = getBestOneToManyMapping(YW |> Array, condListY) |> normalizeEmbedding |> cu

    # push!(cond1toM, (findmin(condListX)[2], findmin(condListY)[2]))
    push!(condX1toM, cond((newX * newX') |> Array))
    push!(condY1toM, cond((newY * newY') |> Array))
    @time acc, sim = validate(newX, newY, validationSet)
    push!(knn1toM, acc)
    @time acc, sim = validate(newX, YW, validationSet)
    push!(knn1toM, acc)


    @time acc, sim = validateCSLS(newX, newY, validationSet)
    push!(csls1toM, acc)
    @time acc, sim = validateCSLS(newX, YW, validationSet)
    push!(csls1toM, acc)


    @time newX = oneToOneMapping(XW) |> normalizeEmbedding
    @time newY = oneToOneMapping(YW) |> normalizeEmbedding
    push!(condX1to1, cond((newX * newX') |> Array))
    push!(condY1to1, cond((newY * newY') |> Array))


    @time acc, sim = validate(newX, newY, validationSet)
    push!(knn1to1, acc)
    @time acc, sim = validate(XW, newY, validationSet)
    push!(knn1to1, acc)

    @time acc, sim = validateCSLS(newX, newY, validationSet)
    push!(csls1to1, acc)
    @time acc, sim = validateCSLS(XW, newY, validationSet)
    push!(csls1to1, acc)

end

p  = []

for list in [knn1to1, knn1toM, csls1to1, csls1toM]
 push!(p, reshape(knn1to1, (2, 10)))
end


map(f -> f(knn1to1[1:2:20]), [minimum, mean, maximum])
map(f -> f(knn1to1[2:2:20]), [minimum, mean, maximum])


map(f -> f(knn1toM[1:2:20]), [minimum, mean, maximum])
map(f -> f(knn1toM[2:2:20]), [minimum, mean, maximum])


map(f -> f(csls1toM[1:2:20]), [minimum, mean, maximum])
map(f -> f(csls1toM[2:2:20]), [minimum, mean, maximum])

map(f -> f(csls1to1[1:2:20]), [minimum, mean, maximum])
map(f -> f(csls1to1[2:2:20]), [minimum, mean, maximum])


map(f -> f(condX1to1[1:1:10]), [minimum, mean, maximum])
# map(f -> f(condX1to1[2:2:10]), [minimum, mean, maximum])

map(f -> f(condY1to1[1:1:10]), [minimum, mean, maximum])
# map(f -> f(condY1to1[2:2:20]), [minimum, mean, maximum])

map(f -> f(condX1toM[1:1:10]), [minimum, mean, maximum])
# map(f -> f(condX1toM[2:2:20]), [minimum, mean, maximum])

map(f -> f(condY1toM[1:1:10]), [minimum, mean, maximum])
# map(f -> f(condY1toM[2:2:20]), [minimum, mean, maximum])



# lang[trg] = [cond1to1, cond1toM, knn1to1, knn1toM, csls1toM, csls1to1]

push!(conds, trg => [condX1to1, condX1toM, condY1toM, condY1toM])

# push!(lang, trg => [cond1to1, cond1toM, knn1to1, knn1toM, csls1toM, csls1to1])

@save "../data/sims/conds.bson" conds
# @save "../data/sims/results.bson" lang

# @load "../data/sims/results.bson" lang
