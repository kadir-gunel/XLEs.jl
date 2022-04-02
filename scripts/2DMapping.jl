cd(@__DIR__)

using OhMyREPL
using SyntheticDatasets: make_moons
using XLEs
using Distances
using PGFPlotsX
using LaTeXStrings

# first creating the moon shape data

Moon = make_moons(n_samples=Int64(1e3), shuffle=false)
X, Y = Matrix(Moon[:, 1:2]), Moon[:, end]

C0 = X[Y .== 0, :] # samples belonging to only class 0
C1 = X[Y .== 1, :] # samples only belings to class 1


# lets plots them

