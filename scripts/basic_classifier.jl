using OhMyREPL
using Random
using Flux
Random.seed!(123)


x = Float32.(rand(300, 32000)) |> gpu
y = rand(1:20000, 32000) |> i -> Flux.onehotbatch(i, 1:20000) .|> Float32 |> gpu

nndata = Flux.Data.DataLoader((x, y), batchsize=20,shuffle=true)

Flux_nn = Chain(
    Dense(300, 300),
    Dense(300, 20000),    # no relu here
    softmax
) |> gpu
ps = Flux.params(Flux_nn)

loss(x, y) = Flux.crossentropy(Flux_nn(x), y)

opt = ADAM()
Flux.@epochs 20 Flux.train!(loss, ps, nndata, opt)

# acc = sum(Flux.onecold(Flux_nn(x), 1:2000) .== Flux.onecold(y, 1:2000)) / size(y, 2) # 1.0
