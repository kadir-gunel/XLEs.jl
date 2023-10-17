using LinearAlgebra
using Statistics: mean

function unit(matrix::T) where {T}
    norms = p_norm(matrix)
    norms[findall(isequal(0), norms)] .= 1
    return matrix ./ norms
end

function unitByVar(matrix::T) where {T}
    @warn "Use this function after center normalization !"
    d = sqrt(size(matrix, 1) - 1)
    norms = p_norm(matrix)
    norms[findall(isequal(0), norms)] .= 1
    return d * (matrix ./ norms)
end

center(matrix::T) where {T} = matrix .- (vec(mean(matrix, dims=2)))

# fast cosine similarity for Matrices
p_norm(M::T) where {T} = sqrt.(sum(real(M .* conj(M)), dims=1))
# cosine(X::T, Y::T) where {T} = diag((X ./ p_norm(X)) * (Y ./ p_norm(Y))
vecLength(X::T) where {T} = sqrt.(diag(X' * X))

normalizeEmbedding(E::T) where {T} = E |> unit |> center |> unit




function MMD(x::T, y::T; kernel::String="multiscale") where T
    atype = typeof(x)
    xx = x' * x
    yy = y' * y
    zz = x' * y

    rx, ry = map(x -> repeat(diag(x), outer=(1,size(x, 2))), [xx, yy])

    dxx = rx + rx' - 2xx
    dyy = ry + ry' - 2yy
    dxy = rx + ry' - 2zz

    if isequal(string(atype), "CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}")
        XX, YY, XY = map(xx -> cu(zeros(size(xx))), [xx, xx, xx])
    else
        XX, YY, XY = map(xx -> zeros(size(xx)), [xx, xx, xx])
    end

    if isequal(kernel, "multiscale")
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for b in bandwidth_range
            XX += b^2 * (b^2 .+ dxx).^(-1)
            YY += b^2 * (b^2 .+ dyy).^(-1)
            XY += b^2 * (b^2 .+ dxy).^(-1)
        end
    end

    if isequal(kernel, "rbf")
        bandwidth_range = [10, 15, 20, 50]
        for b in bandwidth_range
            XX += exp.(-.5 * dxx / b)
            YY += exp.(-.5 * dyy / b)
            XY += exp.(-.5 * dxy / b)
        end
    end

    return mean(XX + YY - 2 .* XY)
end


function freschet_distance(X::T, Y::T) where {T}
    μ1 = mean(X, dims=2)
    μ2 = mean(Y, dims=2)

    μ = sum((μ1 - μ2).^2)

    σ1 = cov(X, dims=2)
    σ2 = cov(Y, dims=2)

    σ_mean = sqrt.(σ1 .* σ2)

    σ_mean = isequal(σ_mean |> typeof, ComplexF32) ? real(σ_mean) : σ_mean

    return μ + tr(σ1 + σ2 - 2σ_mean)
end



function gram_rbf2(from::AbstractArray, to::AbstractArray; threshold=1.)
    F = svd(from)
    T = svd(to)
    newS = log.(F.S .+ T.S)
    dot_prod = F.V .* newS' * T.Vt
    sq_norms = diag(dot_prod)
    sq_dists =  -2 * dot_prod .+ sq_norms .+ permutedims(sq_norms)
    sq_median_distance = median(sq_dists)
    return exp.(-sq_dists / (2 * threshold^2 * sq_median_distance))
end



function center_gram(G::AbstractArray; unbiased=false)
    if !issymmetric(G)
        @error "Gram Matrix have to be symmetric"
    G = deepcopy(G)
    end
    if unbiased
        n, n = G |> size
        G[diagind(G)] .= 0.
        μs = sum(G, dims=2) / (n-1)
        μs = μs .- (sum(μs) / (2 * (n -1)))
        G = G .- μs .- permutedims(μs)
        G[diagind(G)] .= 0
    else
        μs = mean(G, dims=2)
        μs = μs .- mean(μs) / 2
        G = G .- μs .- permutedims(μs)
    end
    return G
end


function gram_rbf(X::AbstractArray; threshold=1.)
    F = svd(X)
    dot_prod = (F.V .* F.S') * F.Vt;
    sq_norms = diag(dot_prod)
    sq_dists =  -2 * dot_prod .+ sq_norms .+ permutedims(sq_norms)
    sq_median_distance = median(sq_dists)
    return exp.(-sq_dists / (2 * threshold^2 * sq_median_distance))
end




function corrAndCov(E::T, ssize::Int64=Int(4e3)) where {T}
    F = svd(E[:, 1:ssize]) # F is object of SVD
    C = (F.V .* F.S') * F.Vt;
    s = sqrt.(1.0 ./ LinearAlgebra.diag(C) )
    Corr = permutedims(s .* permutedims(C) ) .* s
    Corr[LinearAlgebra.diagind(Corr) ] .= 1.0
    C, Corr
end

function doubleCenter!(S::Matrix)
    d, w = size(S)
    if d != w
        @error "Input matrix is not square! Fatal Error!"
    end
    μ = fill(mean(S), w)
    S .= S .- mean(S, dims=1) .= mean(S, dims=2) .+ μ
    return S
end

