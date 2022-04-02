using OhMyREPL
using LinearAlgebra
using Statistics
using Random
using OptimalTransport


unnormCov(E) = E' * E
    



function convex_init(X, Y; niter=100, reg=.05, apply_sqrt=false)
    c, r = size(X)
    isequal(apply_sqrt, true) ?  map!(sqrt_eig, [X, Y]) : nothing
    covX = X' * X
    covY = Y' * Y 
    CovY = CovY * norm(covX) /  norm(covY)
    
    
end
