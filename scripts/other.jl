X2X = parallelCosine(X[:, 1:Int(70e3)]|> Array, X[:, 1:Int(70e3)] |> Array)
XW2XW = parallelCosine(XW[:, 1:Int(70e3)] |> Array, XW[:, 1:Int(70e3)] |> Array)
@time top_idxX  = parallelIdx(X2X, k = 5);
@time top_idxXW = parallelIdx(XW2XW, k = 5);

X2XW = mean(top_idxX .== top_idxXW)

X2X = nothing
XW2XW = nothing
GC.gc()



newXW2newXW = parallelCosine(newXW[:, 1:Int(70e3)] |> Array, newXW[:, 1:Int(70e3)] |> Array)
@time top_idxXW2newXW = parallelIdx(newXW2newXW, k = 5);

newXW2XW = mean(top_idxXW .== top_idxXW2newXW)
newXW2X = mean(top_idxX .== top_idxXW2newXW)


XW2XW = nothing;
newXW2newXW = nothing;

GC.gc()

# cheking middle 80

X2X = parallelCosine(X[:, Int(70e3):Int(140e3)]|> Array, X[:, Int(70e3):Int(140e3)] |> Array)
@time top_idxX  = parallelIdx(X2X, k = 5);
X2X = nothing
GC.gc()

XW2XW = parallelCosine(XW[:, Int(70e3):Int(140e3)]|> Array, XW[:, Int(70e3):Int(140e3)] |> Array)
@time top_idxXW  = parallelIdx(XW2XW, k = 5);
XW2XW = nothing
GC.gc()

X2XW = mean(top_idxX .== top_idxXW)

newXW2newXW = parallelCosine(newXW[:, Int(70e3):Int(140e3)]|> Array, newXW[:, Int(70e3):Int(140e3)] |> Array)
@time top_idxnewXW  = parallelIdx(newXW2newXW, k = 5);
XW2newXW = mean(top_idxXW .== top_idxnewXW)

X2newXW = mean(top_idxX .== top_idxnewXW)

newXW2newXW = nothing
GC.gc()

# checking last 60
X2X = parallelCosine(X[:, Int(140e3):end]|> Array, X[:, Int(140e3):end] |> Array)
XW2XW = parallelCosine(XW[:, Int(140e3):end] |> Array, XW[:, Int(140e3):end] |> Array)
@time top_idxX  = parallelIdx(X2X, k = 5);
@time top_idxXW = parallelIdx(XW2XW, k = 5);

X2XW = mean(top_idxX .== top_idxXW)

X2X = nothing
XW2XW = nothing
GC.gc()

newXW2newXW = parallelCosine(newXW[:, Int(140e3):end] |> Array, newXW[:, Int(140e3):end] |> Array)
@time top_idxXW2newXW = parallelIdx(newXW2newXW, k = 5);
newXW2XW = mean(top_idxXW .== top_idxXW2newXW)
newXW2X = mean(top_idxX .== top_idxXW2newXW)

newXW2newXW = nothing
GC.gc()
