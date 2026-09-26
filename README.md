# SkillPoison

Code for **SkillPoison: Progressive Skill Poisoning via Successful Experiences** 

## Code Structure

```text
SkillPoison/
├── skillpoison/
│   ├── spin/
│   │   ├── hps/          # Hierarchical Task Selection
│   │   ├── egsa/         # Local Evidence-Grounded Success Attribution
│   │   ├── ceir/         # Global Cross-Experience Inductive Reinforcement
│   │   ├── execute/      # Task execution
│   │   ├── serialize/    # Formation-record serialization
│   │   ├── extract/      # Native Skill formation
│   │   ├── evaluate/     # Evaluation
│   │   └── registry/     # Datasets and experiment settings
│   └── ablation/         # Ablation studies
├── configs/              # Experiment configurations
├── docs/                 # Documentation
├── examples/             # Usage examples
├── scripts/              # Experiment scripts
├── tests/                # Tests
└── third_party/          # External framework references
```

The core method has three stages: `hps/` selects successful trajectories in which the target behavior is locally valid; `egsa/` links that behavior to verified success using task evidence; and `ceir/` organizes formation records across tasks. AutoSkill and Trace2Skill retain their native Skill extraction processes.

The `execute/`, `serialize/`, `extract/`, `evaluate/`, `registry/`, and `ablation/` directories are reserved for framework integration and evaluation; this release contains the three core method stages.
