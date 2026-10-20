"""Base type for optimization problems."""
abstract type AbstractProblem end

"""Problem dimension n (rows of X)."""
function n(prob::AbstractProblem)
    throw(MethodError(n, (prob,)))
end

"""Problem dimension p (columns of X)."""
function p(prob::AbstractProblem)
    throw(MethodError(p, (prob,)))
end

"""Deterministic gradient, written in-place into `G`."""
function grad!(G, prob::AbstractProblem, X)
    throw(MethodError(grad!, (G, prob, X)))
end

"""Optional stochastic gradient; default falls back to deterministic `grad!`."""
function stoch_grad!(G, prob::AbstractProblem, X, ξ)
    grad!(G, prob, X)
    return nothing
end

"""Optional cost function; default is NaN."""
function cost(prob::AbstractProblem, X)
    return NaN
end

"""
Function-backed problem wrapper.

`grad!` must have signature `(G, X) -> nothing`.
Optional `stoch_grad!` must have signature `(G, X, ξ) -> nothing`.
Optional `cost` must have signature `(X) -> Real`.
"""
struct FunctionProblem{FG,FS,FC} <: AbstractProblem
    n::Int
    p::Int
    grad!_fun::FG
    stoch_grad!_fun::FS
    cost_fun::FC
end

function FunctionProblem(n::Int, p::Int, grad!_fun;
    stoch_grad!_fun=nothing,
    cost_fun=nothing,
)
    return FunctionProblem{typeof(grad!_fun),typeof(stoch_grad!_fun),typeof(cost_fun)}(
        n,
        p,
        grad!_fun,
        stoch_grad!_fun,
        cost_fun,
    )
end

n(prob::FunctionProblem) = prob.n
p(prob::FunctionProblem) = prob.p

grad!(G, prob::FunctionProblem, X) = (prob.grad!_fun(G, X); nothing)

function stoch_grad!(G, prob::FunctionProblem, X, ξ)
    if prob.stoch_grad!_fun === nothing
        grad!(G, prob, X)
    else
        prob.stoch_grad!_fun(G, X, ξ)
    end
    return nothing
end

function cost(prob::FunctionProblem, X)
    if prob.cost_fun === nothing
        return NaN
    end
    return prob.cost_fun(X)
end
