# Architecture

`Config` supplies dimensions, device, seed, inference iterations, and rollout
settings. The agent combines `BeliefState`, a generative observation model,
the free-energy calculators, and `PolicyEvaluator`. The environment owns
the simulated transition and the preferred observation.

1. `perceive` updates beliefs for the current observation.
2. `plan` evaluates candidate actions through internal model rollouts and
   updates the policy parameters.
3. `act` samples an action or selects the policy mean.
4. The caller applies the action to the environment. The returned observation
   enters the next inference step.

The [update guide](BELIEF_AND_POLICY_UPDATES.md) explains the loss terms.
`active_torchference/orchestrators/` adds experiment runners and plotting
around the agent. [OutputManager](OUTPUT_STRUCTURE.md) organizes artifacts
by run and category.
