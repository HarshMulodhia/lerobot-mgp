## Markov Generator Policy (MGP)

MGP is exposed as a first-class LeRobot policy type in this repository:

```bash
lerobot-train --policy.type=mgp ...
```

or with legacy shorthand:

```bash
lerobot-train --policy_type=mgp ...
```

For inference on real hardware, load a trained checkpoint with the regular rollout/eval flow:

```bash
lerobot-rollout --policy.path=outputs/train/<run>/checkpoints/last/pretrained_model ...
```

### Notes

- `mgp` currently uses the diffusion-policy training/inference backbone under the MGP policy interface.
- It integrates with the same dataset recording pipeline (`lerobot-record`) and deployment tooling used by other policies.
- This repository keeps compatibility with upstream LeRobot while prioritizing MGP workflows.
