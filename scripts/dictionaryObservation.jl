cd(@__DIR__)

using OhMyREPL

include("../src/FluxVECMap.jl")

using .FluxVECMap
using LinearAlgebra
using Statistics
using StatsBase
using Gnuplot
using CUDA


function cov2cor_proposing(C::AbstractMatrix)
    (nr, nc) = size(C)
    @assert nr == nc

    s = sqrt.(1.0 ./ LinearAlgebra.diag(C) )
    corr = transpose(s .* transpose(C) ) .* s
    corr[LinearAlgebra.diagind(corr) ] .= 1.0

    return corr
end

root, folders, Embedfiles = first(walkdir("../data/exp_raw/embeddings/"))

src = "en"
# trg = "es"

srcfile = root * src
#trgfile = root * trg

x_voc, src_embeds = readBinaryEmbeddings(srcfile)
# y_voc, trg_embeds = readBinaryEmbeddings(trgfile)

# 1. normalize embeddings
X = src_embeds |> normalizeEmbedding |> cu
Xsub, x_idx = buildSubSpace(X[:, 1:Int(20e3)]|> Array)

# Steps :
# 1. Create covariance Matrix
CovX, CovXsub = map(FluxVECMap.cudaCorrelationMatrix, [X[:, 1:Int(4e3)], Xsub|> cu])
CovX, CovXsub = map(Array, [CovX, CovXsub])
# 2. Create correlation Matrix
CorX, CorXsub = map(cov2cor_proposing, [CovX, CovXsub])
# 3. sort both matrices
CovX_sorted, CovXsub_sorted = map(i -> sort(i, dims=2) , [CovX, CovXsub])
CorX_sorted, CorXsub_sorted = map(i -> sort(i, dims=2) , [CorX, CorXsub])

# original
original = [CovX, CovX_sorted, CorX, CorX_sorted];
map(rank, [CovX, CovX_sorted, CorX, CorX_sorted])
# picked according to condition
condition = [CovXsub, CovXsub_sorted, CorXsub, CorXsub_sorted]
map(rank, [CovXsub, CovXsub_sorted, CorXsub, CorXsub_sorted])


O = map(svdvals, original)
C = map(svdvals, condition)

O = O .|> i-> log2.(i)
C = C .|> i-> log2.(i)


# 5. plot singular values for each.
@gp "set grid" "set key opaque" "set key top right"
@gp :- "set title 'Covariance and Correlation of Original 4k'"
@gp :- 1:length(O[1]) O[1] "w p pt 2.5 tit 'Σ_{CovX}'"
@gp :- 1:length(O[2]) O[2] "w p pt 2.5 tit 'Σ_{CovX_{sorted}}'"
@gp :- 1:length(O[3]) O[3] "w p pt 2.5 tit 'Σ_{CorX}'"
@gp :- 1:length(O[4]) O[4] "w p pt 2.5 tit 'Σ_{CorX_{sorted}}'"

@gp "set grid" "set key opaque" "set key top right"
@gp :- "set title 'Covariance and Correlation of SubX 4k'"
@gp :- 1:length(C[1]) C[1] "w p pt 2.5 tit 'Σ_{CovX}'"
@gp :- 1:length(C[2]) C[2] "w p pt 2.5 tit 'Σ_{CovX_{sorted}}'"
@gp :- 1:length(C[3]) C[3] "w p pt 2.5 tit 'Σ_{CorX}'"
@gp :- 1:length(C[4]) C[4] "w p pt 2.5 tit 'Σ_{CorX_{sorted}}'"




@gp :- "set title 'Correlation for English'"
@gp :- CorrX "w image tit 'Original'"
save(term="pngcairo size 1500,1100", output="../plots/correlation_En_1-4k.png")




sorted = sort(x_idx[1:Int(4e3)])
CovXsub  = FluxVECMap.cudaCorrelationMatrix(Xsub[:, sorted] |> cu) |> Array

@gp "set grid" "set key opaque" "set key top right"
@gp :- "set title 'Correlation for Sub English '"
@gp :- CorrX "w image tit 'sorted according to Condition Numbers'"
save(term="pngcairo size 1500,1100", output="../plots/correlation_Sub_En_1-4k.png")
