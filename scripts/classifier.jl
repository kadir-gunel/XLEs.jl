cd(@__DIR__)

include("../src/FluxVECMap.jl")

using .FluxVECMap
using LinearAlgebra
using Statistics
using Logging
using OhMyREPL
using Flux
using Flux.Data
using Flux: @epochs
using CUDA
using BSON: @load, @save
using DelimitedFiles
using Distributed
using WordTokenizers
using TextAnalysis

"""
lookup table creation according to minimum frequency limit
"""
function createFreqs(words::Array{String,1}, limit::Int64, fDict::Dict{String, Int64})
    for word in words
        fDict[word] = get(fDict, word, 0) + 1
    end
    return filter!(w -> w.second > limit, fDict)
end

"""
convert corpus to integer valued sentences
"""
function crp2int(corpus::Array, s2i::Dict)
    # convert each sentence to integer format
    # if not in dictionary then replace with UNK => integer
    intsents = Array{Int64}[];
    for line in corpus
        aux = Int64[]
        for word in line
            haskey(s2i, word) ? push!(aux, s2i[word]) : push!(aux, s2i["<UNK>"])
        end
        push!(intsents, aux)
    end
    return intsents
end


# if shorter than max length then insert <PAD>
function padding(corpus::Array, s2i::Dict; maxSeqLen::Int64=100)
    maxlen = maximum(length.(corpus))
    idx = findall(i -> length(i) < maxSeqLen , corpus)
    for sent in idx
        dif = maxlen - length(corpus[sent])
        for i in 1:dif
            insert!(corpus[sent], 1, s2i["<PAD>"])
        end
    end
end


Embedfile = "../data/exp_raw/embeddings/en"
trainfile = "../data/exp_raw/classification/Corona_NLP_train.csv"
testfile  = "../data/exp_raw/classification/Corona_NLP_test.csv"


x_voc, src_embeds = readBinaryEmbeddings(Embedfile)
X  = src_embeds |> normalizeEmbedding
src_w2i = word2idx(x_voc);

trainset = readdlm(trainfile, ',', String, header=true, use_mmap=true)[1]
testset  = readdlm(testfile,  ',', String, header=true, use_mmap=true)[1]

x_train, y_train = (trainset[:, 5]) .|> lowercase , (trainset[:, 6]) .|> lowercase
x_test ,  y_test = (testset[:, 5])  .|> lowercase , (testset[:, 6]) .|> lowercase

x_train = [filter(!ispunct, x_train[i]) for i in 1:length(x_train)] .|> tokenize .|> i -> join(i, ' ')
x_test = [filter(!ispunct, x_test[i]) for i in 1:length(x_test)] .|> tokenize .|> i -> join(i, ' ')



# create lookup table according to training files

words = string.(vcat((x_train .|> split)...))
fDict = Dict{String, Int64}()
wordfreqs = createFreqs(words, 0, fDict)
@info "Total Number of words:" length(words)
@info "Vocabulary Size:" length(unique(words))
@info "Reduced Vocabulary Size:" length(collesct(keys(fDict)))

vocabulary = collect(keys(fDict))

idx = Int64[]
for i in 1:length(vocabulary)
    if in(vocabulary[i], x_voc)
        push!(idx, src_w2i[vocabulary[i]])
    else
        push!(idx, 0)
    end
end

@info "Length, μ, Min, Max" length(idx) mean(idx) minimum(idx) maximum(idx)
# will consider only those elements inside the embedding, the rest will be described as UNK

stay = idx[idx .!= 0]
vocabulary = x_voc[stay]


s2i = Dict(term => i for (i, term) in enumerate(vocabulary))
s2i["<UNK>"] = get(s2i, "<UNK>", 0) + length(s2i) + 1 # adding UNK to the dictionary
s2i["<PAD>"] = get(s2i, "<PAD>", 0) + length(s2i) + 1 # adding PAD to the dictionary
push!(vocabulary, "<UNK>")
push!(vocabulary, "<PAD>")

trn_corpus  = corpus .|> i -> string.(i)
# first need to convert corpus to integer format
int_corpus = crp2int(trn_corpus, s2i)
# need padding
padding(int_corpus, s2i)

labels = Dict(l => i for (i, l) in enumerate(sort(unique(y_train))))
y_s = collect(keys(labels)) .== permutedims(y_train) # |> i -> Flux.onehotbatch(i, 1:length(collect(keys(labels))))

trn_corpus = reduce(hcat, int_corpus)
data = Flux.Data.DataLoader((trn_corpus, y_s), batchsize=128)

# need to add 2 more vectors to X space for padding and unk words
# for padding lets use 0 vector
# for unk just use a random vector

# we will use all embedding space
X = hcat(X, rand(300, 1), zeros(300,1))
# also need to add unk word and padding words to vocabulary
push!(x_voc, "<UNK>")
push!(x_voc, "<PAD>")

src_w2i["<UNK>"] = Int(200e3)+1
src_w2i["<UNK>"] = Int(200e3)+2

struct EmbeddingLayer
   W
   EmbeddingLayer(mf, vs) = new(param(Flux.glorot_normal(mf, vs)))
end

Flux.@functor EmbeddingLayer
doc_pad_size = 100

m = Chain(x -> X * Flux.onehotbatch(reshape(x, doc_pad_size*size(x,2)), 0:vocab_size-1),
          x -> reshape(x, max_features, doc_pad_size, trunc(Int64(size(x,2)/doc_pad_size))),
          x -> mean(x, dims=2),
          x -> reshape(x, max_features, :),
          Dense(max_features, 5),
          softmax
) |> gpu


param = Flux.params(model)

loss(x, y) = Flux.crossentropy(model(x), y)

loss(x_train, y_train)
loss(x_test, y_test)



opt = ADAM()
Flux.@epochs 50 Flux.train!(loss, param, data, opt)



acc = sum(Flux.onecold(model(x_test), 1:Int(32e3)) .== Flux.onecold(y_test, 1:Int(32e3))) / size(y, 2) # 1.0

W = W |> gpu
X = X |> gpu
Y = Y |> gpu
accuracy, similarity = validate(W2' * X |> normalizeEmbedding, Y |> normalizeEmbedding, validation)

F = svd(X[:, ]')
