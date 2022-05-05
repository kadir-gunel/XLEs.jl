function calcobjective(sim)
    bestfward = mean(CUDA.CUBLAS.maximum(sim, dims=2))
    bestbward = mean(CUDA.CUBLAS.maximum(permutedims(sim), dims=2))
    objective = (bestfward + bestbward) / 2
end


function mahalanobis(subx, suby; ssize::Int64=Int(2.2e2))
    d, w = size(subx)
    D = reshape(subx, (d, w, 1)) .- reshape(suby, (d, 1, w))
    D = reshape(D, (d, w * w))
    C = pinv(subx * suby') # 300x300
    sim = reshape(diag(D' * C' * D), (w, w))
    # src_idx = vec(vcat(collect(1:sim_size), permutedims(getindex.(argmax(sim, dims=1), 1))|> Array))
    # trg_idx = vec(vcat((getindex.(argmax(sim, dims=2),2))|> Array, collect(1:sim_size)))
    return sim #, src_idx, trg_idx
end

function mahalanobis1(subx, suby;sim_size::Int64=Int(4e3))
    d, w = size(subx)
    C = pinv(subx * suby')'
    distMah = zeros(w, w)
    for i in 1:w
        for j in 1:w
            dist = subx[:, i] - suby[:, j];
            distMah[i, j]  = dist' * C * dist
        end
    end
    return distMah
    src_idx = vec(vcat(collect(1:sim_size), permutedims(getindex.(argmax(distMah, dims=1), 1))|> Array))
    trg_idx = vec(vcat((getindex.(argmax(distMah, dims=2),2))|> Array, collect(1:sim_size)))
    return src_idx, trg_idx
end

function mahalanobisGPU(subx, suby)
    d, wx = size(subx)
    wy = size(suby, 2)
    data = hcat(subx, suby)
    C = cov(data |> permutedims)
    distMah = CUDA.zeros(wx, wy)
    for i in 1:wx
        dist = subx[:, i] .- suby;
        distMah[i, :] = sum((C \ dist) .* dist, dims=1)
    end
    @. distMah = sqrt(distMah)
    # sort!(distMah, dims=1)
    #src_idx = vec(vcat(collect(1:wx), permutedims(CUDA.CUBLAS.getindex.(CUDA.CUBLAS.argmax(distMah, dims=1), 1))|> Array));
    #trg_idx = vec(vcat((CUDA.CUBLAS.getindex.(CUDA.CUBLAS.argmax(distMah, dims=2),2))|> Array, collect(1:wy)));
    return distMah # , src_idx, trg_idx
end




function parallelMahalanobis1(subx, suby; sim_size::Int64=Int(4e3))
    d, w = size(subx)
    C = inv(subx * suby')
    distMah = similar(subx, w, w)
    @threads for j in 1:w
        for i in 1:w
            distMah[i, j] = @views ((subx[:, i] .- suby[:, j])' * C * (subx[:, i] .- suby[:, j]))
        end
    end
    #src_idx = vec(vcat(collect(1:sim_size), permutedims(getindex.(argmax(distMah, dims=1), 1))|> Array))
    #trg_idx = vec(vcat((getindex.(argmax(distMah, dims=2),2))|> Array, collect(1:sim_size)))
    return distMah #, src_idx, trg_idx
end


function calculatePseudoInverse(C)
    C = C |> Array;
    PI = pinv(C) |> cu;
    PI = PI * permutedims(PI) / 2 # hence making it symmetric again
    return PI |> Array
end

function updateMahalanobisDictionary(A, B, keep_prob::Float64)
    PI = calculatePseudoInverse(permutedims(A) * B)
    sim = pairwise(Mahalanobis(PI), A |> Array, B |> Array) |> cu
    revsim = sim |> permutedims |> cu

    knnsim  = topk_mean(sim, 10, inplace=true)
    bestsim = CUDA.CUBLAS.maximum(revsim, dims=2)

    revsim  = revsim - (CUDA.ones(Float32, size(sim)) .* (transpose(knnsim / 2)))
    idx = getindex.(argmax(CUDA.CUDNN.cudnnDropoutForward(revsim, dropout=1 - keep_prob), dims=2), 2)

    return vec(idx), vec(bestsim)
end

function findLowestConditions(E::L; n::Int=Int(20e3), rev::Bool=false) where {L}
    c, r = size(E)
    @views T = reshape(E, (c, 400, 500));
    C = @views map(i -> log2(cond(T[:, :, i] * T[:, :, i]')), collect(1:500))
    C_sorted = sortperm(C, rev=rev);
    s = div(n, 400)
    takens = collect(take(C_sorted, s)) |> sort;
    return reshape(T[:, :, takens], (c, n)) # makes a 20e3 Dictionary
end

function buildSubSpace(E::L; parts::Int=size(E, 2), rev::Bool=false) where {L}
    c, r = size(E)
    samples = div(r, parts)
    @views T = reshape(E, (c, samples, parts));
    C = Array{Float32}(undef, parts)
    @threads for i in 1:parts
        @views C[i] =  log2(cond(T[:, :, i] * permutedims(T[:, :, i])))
    end
    idx = sortperm(C, rev=rev) |> Array
    return reshape(T[:, :, idx], (c, samples * parts)), idx
end


function calcobjective(sim)
    bestfward = mean(CUDA.CUBLAS.maximum(sim, dims=2))
    bestbward = mean(CUDA.CUBLAS.maximum(permutedims(sim), dims=2))
    objective = (bestfward + bestbward) / 2
end


function mahalanobis(subx, suby; ssize::Int64=Int(2.2e2))
    d, w = size(subx)
    D = reshape(subx, (d, w, 1)) .- reshape(suby, (d, 1, w))
    D = reshape(D, (d, w * w))
    C = pinv(subx * suby') # 300x300
    sim = reshape(diag(D' * C' * D), (w, w))
    # src_idx = vec(vcat(collect(1:sim_size), permutedims(getindex.(argmax(sim, dims=1), 1))|> Array))
    # trg_idx = vec(vcat((getindex.(argmax(sim, dims=2),2))|> Array, collect(1:sim_size)))
    return sim #, src_idx, trg_idx
end

function mahalanobis1(subx, suby;sim_size::Int64=Int(4e3))
    d, w = size(subx)
    C = pinv(subx * suby')'
    distMah = zeros(w, w)
    for i in 1:w
        for j in 1:w
            dist = subx[:, i] - suby[:, j];
            distMah[i, j]  = dist' * C * dist
        end
    end
    return distMah
    src_idx = vec(vcat(collect(1:sim_size), permutedims(getindex.(argmax(distMah, dims=1), 1))|> Array))
    trg_idx = vec(vcat((getindex.(argmax(distMah, dims=2),2))|> Array, collect(1:sim_size)))
    return src_idx, trg_idx
end

function mahalanobisGPU(subx, suby)
    d, wx = size(subx)
    wy = size(suby, 2)
    data = hcat(subx, suby)
    C = cov(data |> permutedims)
    distMah = CUDA.zeros(wx, wy)
    for i in 1:wx
        dist = subx[:, i] .- suby;
        distMah[i, :] = sum((C \ dist) .* dist, dims=1)
    end
    @. distMah = sqrt(distMah)
    # sort!(distMah, dims=1)
    #src_idx = vec(vcat(collect(1:wx), permutedims(CUDA.CUBLAS.getindex.(CUDA.CUBLAS.argmax(distMah, dims=1), 1))|> Array));
    #trg_idx = vec(vcat((CUDA.CUBLAS.getindex.(CUDA.CUBLAS.argmax(distMah, dims=2),2))|> Array, collect(1:wy)));
    return distMah # , src_idx, trg_idx
end




function parallelMahalanobis1(subx, suby; sim_size::Int64=Int(4e3))
    d, w = size(subx)
    C = inv(subx * suby')
    distMah = similar(subx, w, w)
    @threads for j in 1:w
        for i in 1:w
            distMah[i, j] = @views ((subx[:, i] .- suby[:, j])' * C * (subx[:, i] .- suby[:, j]))
        end
    end
    #src_idx = vec(vcat(collect(1:sim_size), permutedims(getindex.(argmax(distMah, dims=1), 1))|> Array))
    #trg_idx = vec(vcat((getindex.(argmax(distMah, dims=2),2))|> Array, collect(1:sim_size)))
    return distMah #, src_idx, trg_idx
end

function buildDictionary(subx, suby; sim_size::Int64=Int(4e3))
    distXY = Mahalanobis(subx * suby', skipchecks=true)
    sim = pairwise(distXY, subx, suby) |> cu

    src_idx = vec(vcat(collect(1:sim_size), permutedims(getindex.(argmax(sim, dims=1), 1))|> Array))
    trg_idx = vec(vcat((getindex.(argmax(sim, dims=2),2))|> Array, collect(1:sim_size)))

    src_idx, trg_idx
end
