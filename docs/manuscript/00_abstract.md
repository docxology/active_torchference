## Abstract

Active Torchference implements perception and action through separate
variational and expected free-energy objectives in a small PyTorch testbed.
This software note describes the Gaussian belief approximation, finite-horizon
candidate scoring, and action-selection policy. Two CPU diagnostic suites
exercise the implemented computations. A separately archived navigation
episode illustrates the shape of the recorded outputs. These artifacts
support inspection of the implementation on synthetic environments
[@torchference].
