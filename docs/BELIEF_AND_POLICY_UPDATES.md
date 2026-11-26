# Belief and policy updates

The continuous VFE implementation adds a Gaussian prediction penalty to the
KL divergence from the current diagonal Gaussian belief to its prior.
The observation model is evaluated at the belief mean. The observation
precision enters the likelihood, and the prior standard deviation enters
the KL term. The stored `precision_prior` parameter currently has no separate
effect on that computation.

The planner samples candidate actions and repeats each candidate across the
rollout horizon. It combines a predicted-state uncertainty term and a
preference likelihood using `epistemic_weight` and `pragmatic_weight`.
Scores have shape `[horizon, num_rollouts]` before summing over the horizon,
and `[num_rollouts]` afterward. The continuous path uses approximate
uncertainty estimates based on the current belief.

`plan()` refines policy parameters using the mean candidate score and logs the
index of the lowest-scoring candidate. `act(deterministic=True)` returns the
policy mean. The default action is sampled from the policy distribution.
See the [methods note](manuscript/02_computation.md) and
[numerical checks](OUTPUT_STRUCTURE.md).
