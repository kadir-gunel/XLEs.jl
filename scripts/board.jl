cd(@__DIR__)

include("../src/FluxVECMap.jl")
using .FluxVECMap
using BSON: @load
using OhMyREPL
using CUDA
using LinearAlgebra
using Statistics


function correlationMatrix(X)
    F = CUDA.CUBLAS.svd(permutedims(X))
    C = F.V * cu(diagm(F.S .^ 2)) * F.Vt
    return C
end



root, folders, Embedfiles = first(walkdir("../data/exp_raw/embeddings/"))

trg = "it"

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

condX = []
condY = []
knn   = []
csls  = []
i = 1
@info i
@info "Reading FT models"
@load "../data/sims/FT/NU/en-$(trg)/model_$(i).bson" model
model = deepcopy(model)

src_idx = model[:src_idx]
trg_idx = model[:trg_idx]

XW, YW = advancedMapping(X |> permutedims |> cu, Y |> permutedims |> cu, src_idx, trg_idx)

newXW = replaceSingulars(XW) |> normalizeEmbedding
newYW = replaceSingulars(YW) |> normalizeEmbedding

XW = XW |> normalizeEmbedding
YW = YW |> normalizeEmbedding


corrXo = correlationMatrix(X |> cu) |> CUDA.CUBLAS.cond
corrYo = correlationMatrix(Y |> cu) |> CUDA.CUBLAS.cond

corrX  = correlationMatrix(XW) |> CUDA.CUBLAS.cond
corrY  = correlationMatrix(YW |> normalizeEmbedding) |> CUDA.CUBLAS.cond

corrXR = correlationMatrix(newXW) |> CUDA.CUBLAS.cond
corrYR  = correlationMatrix(newYW |> normalizeEmbedding) |> CUDA.CUBLAS.cond

push!(condX, [corrXo, corrX, corrXR])
push!(condY, [corrYo, corrY, corrYR])

nn, trg_ind = getIDX_NN(XW, YW , validationSet)
XY = mean(in.(nn, trg_ind))

nn, trg_ind = getIDX_NN(newXW , newYW , validationSet)
newXnewY = mean(in.(nn, trg_ind))


nn, trg_ind = getIDX_NN(newXW, YW , validationSet)
newXY = mean(in.(nn, trg_ind))
push!(knn, [XY, newXnewY, newXY])




nn, trg_ind = getIDX_CSLS(XW, YW, validationSet)
XY = mean(in.(nn, trg_ind))

nn, trg_ind = getIDX_CSLS(newXW, newYW, validationSet)
newXnewY = mean(in.(nn, trg_ind))


nn, trg_ind = getIDX_CSLS(newXW, YW, validationSet)
newXY = mean(in.(nn, trg_ind))
push!(csls, [XY, newXnewY, newXY])
