# Active Torchference Documentation

Complete documentation for the Active Torchference framework.

## Documentation Index

### Getting Started

**[QUICKSTART.md](QUICKSTART.md)** - Start here  
Quick introduction with 5-minute example, common tasks, and troubleshooting.

**Topics covered**:
- Installation
- Basic usage
- Running examples
- Configuration tuning
- Common issues

**Audience**: New users, quick reference

---

### Core Concepts

**[AGENTS.md](AGENTS.md)** - Essential reading  
Comprehensive guide to Active Inference agents and the action-perception loop.

**Topics covered**:
- Active Inference framework
- Agent architecture
- Belief state management
- Policy evaluation
- Free energy computations
- Configuration
- Usage examples
- Validation and testing

**Audience**: All users, implementation details

---

### API Reference

**[API.md](API.md)** - Complete reference  
Detailed API documentation for all classes and methods.

**Topics covered**:
- Core classes (Agent, Config, Environment)
- Free energy classes (VFE, EFE)
- Belief management
- Policy evaluation
- Orchestrators (Runner, Logger, Visualizer, Animator)
- Utility functions
- Common patterns
- Best practices

**Audience**: Developers, API consumers

**[METHOD_REFERENCE.md](METHOD_REFERENCE.md)** - Quick method lookup  
Alphabetical reference of all public methods with signatures and descriptions.

**Audience**: Developers needing quick reference

---

### Architecture

**[ARCHITECTURE.md](ARCHITECTURE.md)** - System design  
Deep dive into framework architecture and design patterns.

**Topics covered**:
- Design principles
- System architecture
- Core components
- Data flow
- Testing strategy
- Extension points
- Performance considerations
- Best practices

**Audience**: Developers, contributors, system designers

---

### Technical Details

**[BELIEF_AND_POLICY_UPDATES.md](BELIEF_AND_POLICY_UPDATES.md)** - Implementation deep-dive  
Technical explanation of VFE minimization and EFE-based policy evaluation.

**Topics covered**:
- Belief updates via VFE
- Policy evaluation via EFE
- Iterative minimization
- Configuration parameters
- Mathematical details
- Integration examples

**Audience**: Advanced users, researchers, implementers

---

### Infrastructure

**[OUTPUT_STRUCTURE.md](OUTPUT_STRUCTURE.md)** - Output organization  
Guide to unified output directory structure and management.

**Topics covered**:
- Directory structure
- Output categories
- OutputManager usage
- Managing experiments
- Best practices
- Integration examples

**Audience**: All users, experiment management

---

## Documentation by User Type

### New Users
1. **[QUICKSTART.md](QUICKSTART.md)** - Get started fast
2. **[AGENTS.md](AGENTS.md)** - Understand core concepts
3. **[OUTPUT_STRUCTURE.md](OUTPUT_STRUCTURE.md)** - Organize experiments

### Developers
1. **[API.md](API.md)** - Complete API reference
2. **[METHOD_REFERENCE.md](METHOD_REFERENCE.md)** - Quick method lookup
3. **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design
4. **[BELIEF_AND_POLICY_UPDATES.md](BELIEF_AND_POLICY_UPDATES.md)** - Technical details

### Researchers
1. **[AGENTS.md](AGENTS.md)** - Theoretical foundation
2. **[BELIEF_AND_POLICY_UPDATES.md](BELIEF_AND_POLICY_UPDATES.md)** - Implementation details
3. **[ARCHITECTURE.md](ARCHITECTURE.md)** - Design decisions

### Contributors
1. **[ARCHITECTURE.md](ARCHITECTURE.md)** - System overview
2. **[API.md](API.md)** - Existing interfaces
3. **[BELIEF_AND_POLICY_UPDATES.md](BELIEF_AND_POLICY_UPDATES.md)** - Core algorithms

---

## Documentation by Topic

### Active Inference Theory
- **[AGENTS.md](AGENTS.md)** - Core concepts section
- **[BELIEF_AND_POLICY_UPDATES.md](BELIEF_AND_POLICY_UPDATES.md)** - Mathematical details

### Implementation
- **[AGENTS.md](AGENTS.md)** - Agent architecture
- **[API.md](API.md)** - Class interfaces
- **[METHOD_REFERENCE.md](METHOD_REFERENCE.md)** - Method signatures
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design

### Usage
- **[QUICKSTART.md](QUICKSTART.md)** - Getting started
- **[AGENTS.md](AGENTS.md)** - Usage examples
- **[API.md](API.md)** - Common patterns

### Configuration
- **[QUICKSTART.md](QUICKSTART.md)** - Configuration tuning
- **[AGENTS.md](AGENTS.md)** - Agent configuration
- **[BELIEF_AND_POLICY_UPDATES.md](BELIEF_AND_POLICY_UPDATES.md)** - Parameter details

### Testing
- **[AGENTS.md](AGENTS.md)** - Validation and testing
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Testing strategy
- **[../tests/README.md](../tests/README.md)** - Test suite

### Experiments
- **[OUTPUT_STRUCTURE.md](OUTPUT_STRUCTURE.md)** - Output organization
- **[API.md](API.md)** - Orchestrators
- **[../examples/README.md](../examples/README.md)** - Examples

---

## Quick Links

### Core Classes
- **ActiveInferenceAgent**: [API.md#activeinfererenceagent](API.md#activeinfererenceagent), [AGENTS.md#agent-architecture](AGENTS.md#agent-architecture)
- **Config**: [API.md#config](API.md#config), [QUICKSTART.md#configuration-tuning](QUICKSTART.md#configuration-tuning)
- **Environment**: [API.md#environment](API.md#environment), [AGENTS.md#custom-environments](AGENTS.md#custom-environments)

### Key Concepts
- **Action-Perception Loop**: [AGENTS.md#action-perception-loop](AGENTS.md#action-perception-loop), [QUICKSTART.md#the-action-perception-loop](QUICKSTART.md#the-action-perception-loop)
- **VFE (Variational Free Energy)**: [AGENTS.md#variational-free-energy-vfe](AGENTS.md#variational-free-energy-vfe), [BELIEF_AND_POLICY_UPDATES.md#belief-state-updates](BELIEF_AND_POLICY_UPDATES.md#belief-state-updates)
- **EFE (Expected Free Energy)**: [AGENTS.md#expected-free-energy-efe](AGENTS.md#expected-free-energy-efe), [BELIEF_AND_POLICY_UPDATES.md#policy-evaluation-efe-per-policy](BELIEF_AND_POLICY_UPDATES.md#policy-evaluation-efe-per-policy)

### Practical Guides
- **Installation**: [QUICKSTART.md#installation](QUICKSTART.md#installation)
- **First Example**: [QUICKSTART.md#5-minute-example](QUICKSTART.md#5-minute-example)
- **Troubleshooting**: [QUICKSTART.md#common-issues](QUICKSTART.md#common-issues), [AGENTS.md#troubleshooting](AGENTS.md#troubleshooting)
- **Custom Environments**: [AGENTS.md#custom-environments](AGENTS.md#custom-environments), [API.md#custom-environment](API.md#custom-environment)

---

## External Resources

### Framework Resources
- **Root README**: [../README.md](../README.md) - Project overview
- **Examples**: [../examples/README.md](../examples/README.md) - Example scripts
- **Tests**: [../tests/README.md](../tests/README.md) - Test suite
- **Validation**: [../validation/README.md](../validation/README.md) - Validation scripts
- **Package**: [../active_torchference/README.md](../active_torchference/README.md) - Package structure

### Theoretical Background
- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Friston, K., et al. (2017). Active inference: a process theory.
- Parr, T., Pezzulo, G., & Friston, K. (2022). Active Inference: The Free Energy Principle in Mind, Brain, and Behavior.

---

## Document Versions

All documents are up-to-date as of October 3, 2025.

### Recent Updates
- Added METHOD_REFERENCE.md for quick method lookup
- Added this README.md as documentation index
- Completed documentation at all directory levels
- All examples use unified output structure

---

## Contributing to Documentation

When contributing documentation:

1. **Clarity**: Write for understanding, not just completeness
2. **Examples**: Include code examples for all features
3. **Structure**: Use clear headings and TOC
4. **Cross-references**: Link to related documentation
5. **Consistency**: Follow existing style and format
6. **Testing**: Verify all code examples work
7. **Updates**: Keep examples synchronized with API

### Documentation Style Guide

- Use "show not tell" approach
- Include working code examples
- Explain "why" not just "what"
- Progressive disclosure (simple → complex)
- Clear section headings
- Table of contents for long documents
- Cross-references between docs

---

## Feedback

Documentation issues or suggestions?
1. Check if answer exists in other docs
2. Review examples for similar use cases
3. Check test files for usage patterns
4. Open GitHub issue for missing documentation

---

## Navigation Tips

### Finding Information

**"How do I...?"** → Start with [QUICKSTART.md](QUICKSTART.md)

**"What does X do?"** → Check [API.md](API.md) or [METHOD_REFERENCE.md](METHOD_REFERENCE.md)

**"Why does it work this way?"** → See [AGENTS.md](AGENTS.md) or [ARCHITECTURE.md](ARCHITECTURE.md)

**"How is X implemented?"** → Read [BELIEF_AND_POLICY_UPDATES.md](BELIEF_AND_POLICY_UPDATES.md)

**"Where are outputs saved?"** → See [OUTPUT_STRUCTURE.md](OUTPUT_STRUCTURE.md)

**"How do I customize X?"** → Check [AGENTS.md#advanced-topics](AGENTS.md#advanced-topics)

### Using This Documentation

1. **Sequential**: Read in order for complete understanding
2. **Reference**: Jump to specific topics as needed
3. **Examples**: Run examples while reading
4. **Testing**: Experiment with concepts in code
5. **Deep-dive**: Follow cross-references for details

---

## Documentation Completeness

✓ Installation guide  
✓ Quick start tutorial  
✓ Core concepts explanation  
✓ Complete API reference  
✓ Method lookup table  
✓ Architecture documentation  
✓ Technical deep-dives  
✓ Output management guide  
✓ Usage examples  
✓ Testing documentation  
✓ Validation scripts  
✓ Troubleshooting guides  
✓ Extension points  
✓ Best practices  

**Status**: Complete and up-to-date

