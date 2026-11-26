## Free-energy computations

For perception, the implemented variational free energy is a negative
Gaussian observation log likelihood plus the KL divergence from the current
belief to its prior. The generative model is evaluated at the belief mean.
PyTorch gradients update the Gaussian parameters iteratively. The prior
standard deviation enters the divergence. The stored prior precision is
currently unused as a separate multiplier [@torchference].

During planning, each sampled candidate action is repeated across the
finite horizon. The model predicts the corresponding sequence of hidden states
and observations. A weighted uncertainty term and preference log likelihood
yield an expected free-energy score for each candidate at each timestep.
The horizon sum gives the per-candidate score. Planning uses the mean of
those scores as the gradient objective and records the smallest score as a
diagnostic. The returned deterministic action is the policy mean. The
default action is a draw from the policy, whose parameters have been
updated using the candidate scores [@torchference].
