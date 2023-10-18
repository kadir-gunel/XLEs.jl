using .GC
using .Threads
using .Iterators
using Mmap
using Test

using Parameters


@with_kw struct EmbeddingData
    datapath::String="/path/to/embeddings"
    srcLang::String="en"
    trgLang::String="es"
    validation::Bool=true
    reversed::Bool=false
    binary::Bool=true
end


function readData(data::EmbeddingData)
    datapath = data.datapath
    srcLang  = data.srcLang
    trgLang  = data.trgLang
    validation = data.validation
    reversed = data.reversed
    binary = data.binary
    root, folders, files = first(walkdir(datapath))
    embeddings, dictionaries = map(x -> folders[findfirst(isequal(x), folders)], ["embeddings", "dictionaries"])

    srcfile, trgfile = map(f -> reduce(*, [root, embeddings, "/", f]), [srcLang, trgLang])
    SRC, TRG = nothing, nothing
    if binary
        SRC = srcfile |> readBinaryEmbeddings;
        TRG = trgfile |> readBinaryEmbeddings;
    else
        SRC = srcfile |> readEmbeddings;
        TRG = trgfile |> readEmbeddings;
    end

    if validation
        VAL = reduce(*, [root, dictionaries, "/", srcLang, "-", trgLang, ".test.txt"])
        if reversed
            VAL = reduce(*, [root, dictionaries, "/", srcLang, "-", trgLang, ".reversed.txt"])
        end
        return (SRC, TRG, VAL)
    end
    return (SRC, TRG)::Tuple
end


function convertTxt2Bin(file::String)
    @info "Reading .txt file"
    V, E = file |> readEmbeddings
    @info "Writing to .bin file"
    writeBinaryEmbeddings(file::String, WE::Matrix, V::Array{String})
end


""" This method reads the .bin file and converts it to .txt format for python usage"""
function convertBin2Txt(file::String)
    @info "Reading .bin file"
    V, E = file |> readBinaryEmbeddings
    @info "Writing to .txt file"
    writeBinaryEmbeddings(file::String, V::Array{String}, E::Matrix)
end

function writeInitialDictionary(file::String, Vs::Array{String}, Vt::Array{String}; type::Symbol=:JS)
    @info "Writing Both source and target initial dictionaries to $file by adding the type of building $(string(type))"
    s = open(file * "_" * string(type) * ".txt", "w+")
    for (vs, vt) in zip(Vs, Vt)
        write(s, vs * " " * vt * "\n")
    end
    close(s)
end


function writeDictionary(file::String, dict::Array{String})
    @info "Writing Dictionary to $file as .txt file ..."
    s = open(file * "voc.txt", "w+")
   for word in dict
       write(s, word * "\n")
   end
   close(s)
   @info "Dictionary is written."
end
"""

"""
function writeBinaryEmbeddings(file::String, WE::Matrix, V::Array{String})
    d, w = size(WE);
    @info "Creating .bin for Word Embeddings"
    if d > w
        @warn "Permuting the Embedding matrix to raw major form"
        WE = Matrix(permutedims(WE))
    end

    s = open(file * "_WE.bin", "w+")
    write(s, d)
    write(s, w)
    write(s, WE)
    close(s)

    @info "Creating .txt for Vocabulary"
    s = open(file * "_voc.txt", "w+")
    for word in V
        write(s, word*"\n")
    end
    close(s)
    @info "Files are written by adding '_WE.bin' to the given file name $file "
end

"""
k represents the first k rows
returns Vocabulary and Word Embeddings
"""
function readBinaryEmbeddings(file::String; atype=Float32)
    @info "Reading Word Embedding file"
    s = open(file * "_WE.bin")   # default is read-only
    m = read(s, Int)
    n = read(s, Int)
    WE = Mmap.mmap(s, Matrix{atype}, (m,n))
    close(s)

    @info "Reading vocabulary file"
    V = readlines(file * "_voc.txt")
    return V, Matrix(permutedims(WE))
end


"""
This method reads the embedding file from virtual memory by using threads.
The speed of this method depends on the # of threads.
On a 4 core machine with 8 threads, a 200k x 300 embedding file is read in nearly 4 secs.
Returns vocabulary and Matrix of Float32
"""
function readEmbeddings(file::String)
    @warn "This method returns the embedding matrix X in column major format!"
    fsize = stat(file).size
    s = open(file)
    L = Mmap.mmap(s, Vector{UInt8}, fsize)
    newline = findfirst([0x0a], L)[1] - 1
    wcount, dims = map(x -> parse(Int64, x), split(String(L[1:newline]), ' '))
    vocabulary = Array{String}(undef, wcount)
    embeddings = Matrix{Float32}(undef, wcount, dims)
    I1, I2 = getIndices(L);
    createEmbeddings!(L, vocabulary, embeddings, I1, I2)
    close(s)
    return (vocabulary::Array{String}), Matrix(permutedims(embeddings::Matrix{Float32}))
end

function createEmbeddings!(L::Vector{UInt8}, vocabulary::AbstractArray{String}, embeddings::AbstractMatrix{Float32}, I1::Vector{Int64}, I2::Vector{Int64})
    @threads for idx in 1:length(I1)
        data = filter!(!isempty, split(String(L[I1[idx]:I2[idx]]), ' '))
        #data = split(String(L[I1[idx]:I2[idx]])) # ← This is the actual usage but VecMap's embedding files does not permit.
        vocabulary[idx] = data[1]
        embeddings[idx, :] .= parse.(Float32, data[2:end])
    end
    GC.gc()
    nothing
end

function getIndices(L::Vector{UInt8})
    n = convert(UInt8, '\n')
    lineidx = findall((L .== n) .== 1)
    I1 = similar(1:length(lineidx)-1)
    I2 = similar(I1)
    I1 .= lineidx[1:end-1] .+ 1
    I2 .= lineidx[2:end] .- 1
    return I1, I2
end

"""
This reading method is sequential and consumes too much memory compared to the threaded version.
"""
function readEmbeds(file; threshold=0, vocabulary=Nothing, dtype=Float32)
    @warn "This method reads the word embedding matrix in column major format"
    count, dims = parse.(Int64, split(readline(file), ' '))
    words  = String[]; # words in vocabulary
    matrix = isequal(vocabulary, Nothing) ?  Array{dtype}(undef, count, dims) : Array{Float32}[];

    # p = Progress(dims, 1, "Reading embeddig file: $file") # this is for progressbar
    for (i, line) in enumerate(drop(eachline(file), 1))
        mixed = split(chomp(line), " ")
        if vocabulary == Nothing
            push!(words, mixed[1])
            matrix[i, :] .= parse.(dtype, mixed[2:end])
        elseif in(mixed[1], vocabulary)
            push!(words, mixed[1])
            push!(matrix, parse.(dtype, mixed[2:end]))
        end
        # next!(p)
    end
    words, Matrix(permutedims(matrix))
end

""" this method is used for interoperability with the python vecmap"""
function writeEmbeds(file::String, voc::Array{String}, embed::M) where {M}
    if size(embed, 1) < size(embed, 2)
        @warn "Need to convert the embedding matrix format from column to raw major"
        embed = Matrix(permutedims(embed))
    end
    @info "Writing Embedding file as .txt - will consume too much space."
    s = open(file * ".txt", "w+")
    lines = length(voc)
    write(s, string(lines) * " " * string(size(embed, 2)) * "\n")
    for i in 1:lines
        write(s, voc[i] * " " * join(string.(embed[i, :]), " ") * "\n")
    end
    close(s)
end

function readValidation(valpath::String, src_w2i::Dict, trg_w2i::Dict)::Dict
    validation = Dict{Int64, Set}();
    oov   = Set{String}()
    vocab = Set{String}()
    for st in eachline(valpath) # st means source, target
        s, t = split(st)
        try
            src_ind = src_w2i[s]
            trg_ind = trg_w2i[t]
            !haskey(validation, src_ind) ? validation[src_ind] = Set(trg_ind) : push!(validation[src_ind], trg_ind)
            push!(vocab, s) # adding word to vocab
        catch KeyError
            push!(oov, s)
        end
    end
    setdiff!(oov, vocab)
    validation_coverage = length(validation) / (length(validation) + length(oov))
    @info "Validation Coverage:  $validation_coverage "
    @info "# of words inside validation set:  $(length(validation))"
    @info "# of out of vocabulary words: , $(length(oov))"
    return validation
end
