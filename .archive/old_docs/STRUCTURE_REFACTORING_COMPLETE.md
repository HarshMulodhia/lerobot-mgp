# MGP Project Structure Refactoring Complete

## Final Structure (Matches LeRobot Standard)

```
src/lerobot/policies/mgp/
├── README.md                      ← Policy overview (like diffusion, smolvla, groot)
├── __init__.py                    ← Exports: MGPConfig, MGPPolicy, make_mgp_pre_post_processors
├── configuration_mgp.py           ← Config class (inherits DiffusionConfig)
├── modeling_mgp.py                ← Policy implementation + adapters
├── processor_mgp.py               ← Data pipeline (delegates to diffusion)
├── mgp_components.py              ← Core components: GM theory + training utilities
│
├── checks/                        ← Non-core validation/testing tools
│   ├── __init__.py
│   └── validate_lerobot_compatibility.py
│
└── docs/                          ← Comprehensive documentation
    ├── INDEX.md
    ├── mgp_guide.md
    ├── cross_robot_compatibility.md
    └── lerobot_pipeline_compatibility.md
```

## Files by Category

### Core Pipeline Files (5 files - REQUIRED)
These files are part of the training/inference pipeline:

1. **`__init__.py`** - Exports public API
   - `MGPConfig`
   - `MGPPolicy`
   - `make_mgp_pre_post_processors`

2. **`configuration_mgp.py`** - MGP configuration
   - Inherits from `DiffusionConfig`
   - 25+ optional parameters (all with defaults)
   - Registered with `@PreTrainedConfig.register_subclass("mgp")`

3. **`modeling_mgp.py`** - MGP policy implementation
   - `MGPPolicy` class inheriting from `DiffusionPolicy`
   - `DistributionShiftAdapter` - value + uncertainty estimation
   - `TrajectorySelector` - intelligent trajectory selection
   - Methods: `forward()`, `select_action()`, `apply_inference_time_alignment()`

4. **`processor_mgp.py`** - Data preprocessing
   - `make_mgp_pre_post_processors()` factory
   - Delegates to `make_diffusion_pre_post_processors()`
   - No processing changes (diffusion is suitable for MGP)

5. **`mgp_components.py`** - Core algorithmic components
   - `ProbabilityPath`, `GaussianCondOTPath` - probability interpolation
   - `GeneratorMatchingLoss` - training objective
   - `CurriculumScheduler` - progressive difficulty
   - `TrajectoryImportanceWeighter` - distribution shift handling
   - `SafetyConstrainedSampler` - hardware constraints
   - `EnergyBasedGeneratorMatching` - reward alignment

### Non-Core Files (Optional)

**Checks Directory** - Development/validation tools:
- `validate_lerobot_compatibility.py` - Tests MGP works with LeRobot pipelines
- Not needed for deployment, useful for development

**Docs Directory** - Comprehensive documentation:
- `mgp_guide.md` - User guide with examples
- `cross_robot_compatibility.md` - Robot-specific guides
- `lerobot_pipeline_compatibility.md` - Pipeline verification
- `INDEX.md` - Documentation index

### Removed/Deprecated

The following files were consolidated into the 5 core files:
- ✗ `generator_matching.py` → merged into `mgp_components.py`
- ✗ `mgp_training.py` → merged into `mgp_components.py`
- ✗ `lerobot_compatibility.py` → moved to `checks/validate_lerobot_compatibility.py`

## Comparison with Other Policies

### Diffusion
```
diffusion/
├── README.md
├── __init__.py
├── configuration_diffusion.py
├── modeling_diffusion.py
└── processor_diffusion.py
```

### SmolVLA
```
smolvla/
├── README.md
├── __init__.py
├── configuration_smolvla.py
├── modeling_smolvla.py
├── processor_smolvla.py
└── smolvlm_with_expert.py  ← Extra domain-specific file
```

### Groot
```
groot/
├── README.md
├── __init__.py
├── configuration_groot.py
├── modeling_groot.py
├── processor_groot.py
├── groot_n1.py              ← Extra domain-specific file
├── utils.py                 ← Extra helper file
├── action_head/             ← Extra domain-specific dir
└── eagle2_hg_model/         ← Extra domain-specific dir
```

### MGP (Refactored)
```
mgp/
├── README.md
├── __init__.py
├── configuration_mgp.py
├── modeling_mgp.py
├── processor_mgp.py
├── mgp_components.py        ← Extra: Core algorithmic components
├── checks/                  ← Extra: Validation (optional)
│   ├── __init__.py
│   └── validate_lerobot_compatibility.py
└── docs/                    ← Extra: Documentation (optional)
    ├── INDEX.md
    └── *.md
```

**MGP Structure Justified:**
- ✅ Has required 5 core files (like Diffusion)
- ✅ `mgp_components.py` is necessary because MGP introduces new algorithms (GM theory, curriculum, importance weighting) that aren't in Diffusion
- ✅ `checks/` and `docs/` are optional, non-critical, won't affect training/deployment
- ✅ Matches or exceeds complexity of Groot (which has action_head/ + eagle2_hg_model/ + utils.py)

## Files NOT Moved (Remain in Project Root)

These files document the overall project, not just the MGP policy:

```
lerobot-mgp/
├── MGP_DEPLOYMENT_GUIDE.md                    ← Project-level guides
├── MGP_LEROBOT_COMPATIBILITY_VERIFIED.md
├── IMPLEMENTATION_SUMMARY.md
├── WHATS_NEW.md
├── MGP_CROSS_ROBOT_COMPATIBILITY.md
├── validate_mgp.py                           ← Project-level validator
├── validate_lerobot_compatibility.py          ← Project-level validator
├── examples/
│   ├── train_mgp.py                          ← Project examples
│   └── inference_mgp_hardware.py
└── tests/
    └── test_mgp_policy.py                    ← Project tests
```

These stay at project root because they describe the overall lerobot-mgp fork, not just the MGP policy module.

## Usage

### For Users (Training/Deployment)
- Only need files in `src/lerobot/policies/mgp/` core (5 files)
- Works just like any other policy: `lerobot-train --policy.type=mgp ...`
- Everything else is optional

### For Developers
- Check `mgp_components.py` for algorithmic details
- Run `src/lerobot/policies/mgp/checks/validate_lerobot_compatibility.py` to verify
- Read docs in `src/lerobot/policies/mgp/docs/` for detailed guides

### For Project Understanding
- Read project-level docs in root folder
- Review examples in `examples/`
- Run tests in `tests/`

## Migration from Old Structure

**Old structure files to delete:**
- ✓ Old root-level validation scripts (duplicated in checks/)
- Old files were:
  - `generator_matching.py` (in mgp/)
  - `mgp_training.py` (in mgp/)
  - `lerobot_compatibility.py` (in mgp/)

**Files that were refactored:**
- ✓ Generator Matching code → `mgp_components.py`
- ✓ Training utilities → `mgp_components.py`
- ✓ Compatibility checks → `checks/validate_lerobot_compatibility.py`
- ✓ Extensive docs → `docs/` folder

## Verification

The new structure:
- ✅ Matches LeRobot standards (like Diffusion, SmolVLA, Groot)
- ✅ Has minimal core files (5 required for any policy)
- ✅ Extra `mgp_components.py` is justified (new algorithms)
- ✅ `checks/` and `docs/` are optional, well-organized
- ✅ Public API is clean (only 3 exports in __init__.py)
- ✅ No breaking changes to training/inference pipeline
- ✅ Fully compatible with `lerobot-train`, `lerobot-eval`, `lerobot-rollout`

## Summary

✅ **Complete** - MGP now follows LeRobot's standard policy structure with minimal, focused core files and well-organized supplementary documentation and tools.
