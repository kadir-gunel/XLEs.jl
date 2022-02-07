cd(@__DIR__)

include("../src/FluxVECMap.jl")

using .FluxVECMap
using LinearAlgebra
using StatsBase
using BSON: @load
using CUDA
using Printf

function fillVocabulary(dict::Dict{String, Int64}, test_voc::Array{String})
   test_idx = Int64[]
   for i in 1:length(test_voc)
      if in(test_voc[i], keys(dict))
         push!(test_idx, dict[test_voc[i]])
      else
         push!(test_idx, 0)
      end
   end
   zeroIdx = findall(test_idx .== 0)
   # replace all zeroed indices by a index number which is not inside the test_idx
   if maximum(sort(unique(test_idx))) != length(keys(dict))
      test_idx[zeroIdx] .= maximum(sort(unique(test_idx))) + 1
   else
      @error "Need to adjust the index value !"
   end
   return test_idx, zeroIdx
end

# 1. read embeddings
srcfile = "../data/exp_raw/embeddings/en"
trgfile = "../data/exp_raw/embeddings/it"
src_voc, X = readBinaryEmbeddings(srcfile)
trg_voc, Y = readBinaryEmbeddings(trgfile)

# 1.1 read mapped indices
@info "Reading FT models"
@load "../data/sims/FT/NU/en-it/model_1.bson" model
src_idx = model[:src_idx]
trg_idx = model[:trg_idx]

# 4. normalize lengths
X, Y = map(normalizeEmbedding, [X, Y]);
XW, YW = advancedMapping(X |> permutedims |> cu, Y |> permutedims |> cu, src_idx, trg_idx)
XW = replaceSingulars(XW)
YW = replaceSingulars(YW)

# 5. build word2Idx
src_w2i = word2idx(src_voc);
trg_w2i = word2idx(trg_voc);

# 2. read test files
lines = readlines("../../vecmap/data/similarity/en-it.mws353.txt")
STS = lowercase.(lines) .|> i -> split(i, '\t')
STS = STS|> k -> reduce(hcat, k) .|> String # source target score matrix
SRC_TRG = STS[1:2, :]
Golds = parse.(Float32, (STS[end, :]))

# 3. build vocabularies
src_test_voc = SRC_TRG[1, :]
trg_test_voc = SRC_TRG[2, :]

src_test_idx, src_zeroIdx = fillVocabulary(src_w2i, src_test_voc)
trg_test_idx, trg_zeroIdx = fillVocabulary(trg_w2i, trg_test_voc)

notIncluded = union(src_zeroIdx, trg_zeroIdx)
# excluding the oov words from both embedding spaces.
X_test = XW[:, src_test_idx] |> Array
X_test = X_test[:, setdiff(1:end, (notIncluded))]

Y_test = YW[:, trg_test_idx] |> Array
Y_test = Y_test[:, setdiff(1:end, (notIncluded))]

Golds = Golds[setdiff(1:end, notIncluded)]

X_test, Y_test = map(unit, [X_test, Y_test]);

cos = vec(sum(X_test .* Y_test, dims=1))
coverage = length(cos) / (length(cos) + length(notIncluded))
pearson  = cor(Golds, cos)
spearman = StatsBase.corspearman(Golds, cos)


@info "Correlation Similarities: test set coverage ($coverage)"
@printf "Pearson : %4.4f" pearson
@printf "Spearman : %4.4f" spearman
