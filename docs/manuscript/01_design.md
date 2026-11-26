## Action and perception

Active inference uses a generative model and preferences to formulate
perception and action [@friston2017]. Active Torchference implements
a small loop in PyTorch [@pytorch; @torchference]. A caller supplies the
current observation and the preferred observation. The agent adjusts its
belief, refines its policy parameters, and returns an action. The caller then
advances a simulated environment and obtains the next observation.

The belief state holds a mean and log standard deviation of a diagonal
Gaussian over hidden states. The generative observation model predicts
observations given hidden states. A transition model propagates candidate
actions through a finite horizon. The orchestration layer logs observations,
beliefs, actions, and free-energy diagnostics for inspection.
