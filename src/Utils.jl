using .Iterators
using LinearAlgebra
using Statistics: mean, var, std
using Printf

using Parameters
using Distances

using CUDA
import CUDA.CUBLAS: gemm!, svd, axpy!, axpby!, ger!, max, maximum
import cuDNN: cudnnDropoutForward

isequal(CUDA.functional(), true) ? CUDA.allowscalar(true) : @error "No CUDA Functionality"


relu(E::CuMatrix) = max.(0, E);
dist2sim(M::Matrix) = 1 ./ exp.(M)
word2idx(voc::Vector{String}) = Dict(term => i for (i, term) in enumerate(voc))
simSize(X::T, Y::T; unsupervised_vocab::Int64=4000) where {T}  =
    unsupervised_vocab <= 0 ? min(size(X, 2) , size(Y, 2)) : min(size(X, 2) , size(Y, 2), unsupervised_vocab)
cutVocabulary(X::T; vocabulary_cutoff::Int64=20000) where {T} =
    vocabulary_cutoff <= 0 ? size(X, 2) : min(size(X, 2), vocabulary_cutoff)


function topk_mean(sim, k; inplace=false)
    n = size(sim, 1)
    ans_ = (zeros(eltype(sim), n, 1)) |> typeof(sim)
    if k <= 0
        return ans_
    end
    if !inplace
        sim = deepcopy(sim)
    end
    min_ = findmin(sim)[1]
    for i in 1:k
        vals_idx = findmax(sim, dims=2);
        ans_ += vals_idx[1]
        sim[vals_idx[2]] .= min_
    end
    return ans_ / k
end

function whitening(M::Matrix{Float32})
    F = svd(M)
    return F.V * diagm(1 ./ F.S) * (F.Vt)
end

function whitening(M::CuMatrix{Float32})
    F = svd(M)
    return F.V * cu(diagm(1 ./ F.S)) * (F.Vt)
end

function sqrt_eigen(subE)
    F = svd(subE)
    return F.U * diagm(sqrt.(F.S)) * F.Vt
end

function correlation(E::Matrix; ssize::Int64=Int(4e3))
    F = svd(E[:, 1:ssize])
    return F.V * diagm(F.S) * F.Vt
end

function cudaCorrelation(E::CuMatrix; sim_size::Int64=4000)
    F = svd(E[:, 1:sim_size]) # F is object of SVD
    return (F.V .* F.S') * F.Vt;
end

function csls(sim::CuMatrix; k::Int64=10)
    n, _ = size(sim) # sim is square matrix so no problem.
    knnfwd = topk_mean(sim, k) |> vec;
    knnbwd = topk_mean(permutedims(sim), k);

    axpby!(n, Float32(0.5), knnbwd, Float32(0.5), knnfwd); # fwd is overwritten
    A = CUDA.zeros(n, n); # creates zero matrix
    aux = CUDA.ones(n);

    ger!(Float32(1.), knnfwd, aux, A);
    axpby!(length(sim), -Float32(1.), A, Float32(1.), sim)
    return sim
end

function buildSeedDictionary(X::CuMatrix, Y::CuMatrix; sim_size::Int64=4000, k::Int64=10)
    # sims = map(cudaCorrelationMatrix, [X, Y])
    xsim = cudaCorrelation(X, sim_size=sim_size)
    ysim = cudaCorrelation(Y, sim_size=sim_size)
    sort!(ysim, dims=1)
    sort!(xsim, dims=1)
    # map(sim -> sort!(sim, dims=1), sims);
    xsim, ysim = map(normalizeEmbedding, [xsim, ysim])

    sim = CuMatrix{Float32}(undef, sim_size, sim_size);
    gemm!('T', 'N', cu(1.), xsim, ysim, cu(0.), sim);

    sim = csls(sim, k=k)

    src_idx = vec(vcat(collect(1:sim_size), permutedims(getindex.(argmax(sim, dims=1), 1))|> Array))
    trg_idx = vec(vcat((getindex.(argmax(sim, dims=2), 2))|> Array, collect(1:sim_size)))

    return src_idx, trg_idx
end

function update(from::CuMatrix, to::CuMatrix, keep_prob::Float64)
    d, n = size(from)
    S = CuMatrix{Float32}(undef, n, n)
    gemm!('T', 'N', Float32(1), from, to, Float32(0), S)

    revS  = permutedims(S)

    knnsim  = topk_mean(S, 10, inplace=true) |> vec
    bestsim = maximum(revS, dims=2) |> vec |> Array

    aux = CUDA.zeros(n) # this is zero vector
    axpy!(n, Float32(0.5), knnsim, aux);

    # the below code :     revS  = revS - (CUDA.ones(Float32, size(S)) .* (transpose(knnsim / 2)))
    A = CUDA.zeros(n, n); # creates zero matrix
    aux2 = CUDA.ones(n)
    ger!(Float32(1.), aux, aux2, A);
    axpby!(length(revS), -Float32(1.), A, Float32(1.), revS)

    idx = getindex.(argmax(cudnnDropoutForward(revS, dropout=1 - keep_prob), dims=2), 2) |> vec |> Array

    return idx, bestsim
end


function mapOrthogonal(X::CuMatrix, Y::CuMatrix)
    d, n = size(X)

    XY = CuMatrix{Float32}(undef, d, d)
    W = CuMatrix{Float32}(undef, d, d)

    gemm!('N', 'T', cu(Float32(1)), X, Y, cu(Float32(0)), XY)

    F = svd(XY)

    gemm!('N', 'N', cu(Float32(1)), F.U, F.Vt, cu(Float32(0)), W)

    return permutedims(W)
end

function train(X::CuMatrix, Y::CuMatrix, src_idx::L, trg_idx::L,
               src_size::Int64, trg_size::Int64, keep_prob::Float64,
               objective::T; stop::Bool=false) where {L, T}

    W = mapOrthogonal(X[:, src_idx], Y[:, trg_idx])

    if stop
        return src_idx, trg_idx, objective, W  # returning the results
    end

    WX = W  * X
    WY = W' * Y

    src_aligned_idx, best_sim_src = update(WX, Y, keep_prob)
    trg_aligned_idx, best_sim_trg = update(WY, X, keep_prob)

    src_idx = vcat(1:src_size, src_aligned_idx)
    trg_idx = vcat(trg_aligned_idx, 1:trg_size)

    objective = (mean(best_sim_src) + mean(best_sim_trg)) / 2
    return src_idx, trg_idx, objective, W
end


function main(X, Y, src_idx, trg_idx, validation; src_size=Int(20e3), trg_size=Int(20e3))
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
    d, n = X |> size
    W = cu(zeros(d, d))
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
        src_idx, trg_idx, objective, W = train(X[:, 1:src_size],
                                               Y[:, 1:trg_size],
                                               src_idx, trg_idx,
                                               src_size, trg_size,
                                               keep_prob, objective;
                                               stop=stop)

        if objective - best_objective >= threshold
            last_improvement = it
            best_objective = objective
        end

        # validating
        if mod(it, 10) == 0
            accuracy, similarity = validate((W * X), Y, validation)
            @printf "Validation Accuracy: %.4f, Similarity: %.4f \n" accuracy similarity
        end

        it += 1

    end
    return W, src_idx, trg_idx
end

function validate(XW::T, YW::T, validation::Dict) where {T}
    trg_indices = collect(values(validation));
    src_indices = collect(keys(validation));
    simval= XW[:, src_indices]' * YW;
    nn = getindex.(argmax(simval, dims=2), 2) |> Array
    accuracy = mean(in.(nn, trg_indices))
    j = collect(validation[src_indices[i]] for i in 1:length(src_indices))
    similarity = mean(maximum.(collect((simval[:, collect(flatten(j[i]))] for i in 1:length(j)))))
    return accuracy, similarity
end

function validateCSLS(XW::T, YW::T, validation::Dict) where {T}
    trg_indices = collect(values(validation));
    src_indices = collect(keys(validation));
    simbwd = zeros(eltype(YW), size(YW, 2))
    for i in 1:500:size(YW, 2)
        j = min(i + 500 - 1, size(YW, 2))
        simbwd[i:j] = topk_mean(permutedims(YW[:, i:j]) * XW, 10, inplace=true)
    end
    simvals = (2 .* permutedims(XW[:, src_indices]) * YW) .- CUDA.ones(Float32, (length(src_indices), Int(200e3))) .* cu(permutedims(simbwd));
    nn = getindex.(argmax(simvals, dims=2), 2) |> Array
    accuracy = mean(in.(nn, trg_indices))

    j = collect(validation[src_indices[i]] for i in 1:length(src_indices))
    similarity = mean(maximum.(collect((simvals[:, collect(flatten(j[i]))] for i in 1:length(j)))))

    return accuracy, similarity
end

function getIDX_NN(XW::T, YW::T, validation::Dict) where {T}
    trg_indices = collect(values(validation));
    src_indices = collect(keys(validation));
    simval= XW[:, src_indices]' * YW;
    nn = getindex.(argmax(simval, dims=2), 2) |> Array
    return nn, trg_indices
end

function getIDX_CSLS(XW::T, YW::T, validation::Dict) where {T}
    trg_indices = collect(values(validation));
    src_indices = collect(keys(validation));
    simbwd = zeros(eltype(YW), size(YW, 2))
    for i in 1:500:size(YW, 2)
        j = min(i + 500 - 1, size(YW, 2))
        simbwd[i:j] = topk_mean(permutedims(YW[:, i:j]) * XW, 10, inplace=true)
    end
    simvals = (2 .* permutedims(XW[:, src_indices]) * YW) .- CUDA.ones(Float32, (length(src_indices), Int(200e3))) .* cu(permutedims(simbwd));
    nn = getindex.(argmax(simvals, dims=2), 2) |> Array
    return nn, trg_indices
end
