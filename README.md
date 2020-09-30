# Multimodal Sensor Fusion with Differentiable Filters

Models, training, and eval scripts for our IROS 2020 conference paper:

<blockquote>
    Lee, M.*, Yi, B.*, Mart&iacute;n-Mart&iacute;n, R., Savarese, S., and Bohg, J.
    <strong>
        <a href="https://sites.google.com/view/multimodalfilter">
            Multimodal Sensor Fusion with Differentiable Filters.
        </a>
    </strong>
    Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), October 2020.
</blockquote>

This repository contains our train/eval scripts crossmodal/unimodal weighted
fusion architectures, and task-specific models and infrastructure. Filtering
interfaces, dataset utilities, and implementations of standard particle filters,
EKFs, and UKFs have been factored into a standalone library
(**[torchfilter](https://github.com/stanford-iprl-lab/torchfilter)**).

---

### Repository Overview

```
.
├── crossmodal
│   ├── base_models             - General implementations for crossmodal and
│   │                             unimodal weighted fusion models.
│   ├── door_models             - PF, EKF, and LSTM models for door tasks.
│   ├── push_models             - PF, EKF, and LSTM models for pushing tasks.
│   └── tasks                   - Task definitions & configuration.
│
└── scripts
    ├── bash_scripts            - Bash script helpers for training.
    ├── door_task               - Training & eval scripts for door tasks.
    │   └── data_collection     - Data collection scripts.
    └── push_task               - Training & eval scripts for pushing tasks.
```

**Additional code:**

- **[Surreal Robotics Suite](https://github.com/StanfordVL/robosuite)**: for
  Panda arm simulations in MuJoCo.
- **[fannypack](https://github.com/brentyi/fannypack)**: for experiment
  management.

---

### Installation

Package and dependencies can be installed in Python >=3.7 with `pip`:

```
git clone https://github.com/brentyi/multimodalfilter.git
cd multimodalfilter
pip install -e .
```
