module SketchLanding

using LinearAlgebra
using Printf
using Random
using SparseArrays

include("types.jl")
include("problem_interface.jl")
include("output.jl")
include("utils.jl")
include("stepsizes.jl")
include("sketches.jl")
include("directions.jl")
include("estimators.jl")
include("algorithm.jl")

export AbstractProblem, n, p, grad!, stoch_grad!, cost, FunctionProblem
export AlgOutput, RecordOptions, create_recorder, init_output
export AbstractSketch, IdentitySketch, DenseMatrixSketch, GaussianSketch, HaarSketch, SubsampleSketch
export AbstractEstimator, PlainEstimator, SVRGEstimator
export State, init_state
export sample!, StX!, BX!, XtBX!, XtB1B2X!
export normal!, tangent!, direction!, update_estimator!
export sample_without_replacement_sorted!, intersect_positions!, compute_stepsize
export AbstractStepsizeState, ScalarStepsizeState, AdagradStepsizeState, AdamStepsizeState, BBStepsizeState, BBSplitStepsizeState, apply_step!
export sketchLanding

end
