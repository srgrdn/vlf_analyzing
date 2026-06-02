Continue work in `/home/s_grudinin/code/university/vlf_analyzing/worktrees/2d_localization` on branch `feature/2d-sferics-localization`.

Before making changes:

1. Read `AGENTS.md`
2. Read `AGENT_STATUS.md`
3. Read `AGENT_HANDOFF.md`
4. If older project context is needed, inspect `docs/codex-history/FULL_CODEX_CONTEXT.md`
5. Run `git status --short --branch`

Branch scope is narrow:

- stay focused on the `spec_db -> 2D localization` path
- do not mix in envelope-model production pipeline code
- do not reintroduce unrelated monitoring/reporting scripts from other branches

Current completed milestones:

- localization dataset builder
- stratified review subset builder
- manual correction workflow

Current likely next milestone:

- help the user produce and use a first reviewed gold subset, or
- build the first `2D CNN` training skeleton if the user explicitly wants to move into training

Important current files:

- `build_localization_dataset.py`
- `build_localization_review_subset.py`
- `init_localization_corrections.py`
- `apply_localization_corrections.py`
- `README.md`

Required process:

- after every completed step, update `AGENT_STATUS.md`
- if the next agent needs context, update `AGENT_HANDOFF.md`

Useful validation commands:

```bash
python3 -m py_compile build_localization_dataset.py \
  build_localization_review_subset.py \
  init_localization_corrections.py \
  apply_localization_corrections.py \
  plot_broadband_spectrogram.py \
  detect_broadband_bursts.py
```

If working on review workflow, also verify:

```bash
python3 build_localization_dataset.py --help
python3 build_localization_review_subset.py --help
python3 init_localization_corrections.py --help
python3 apply_localization_corrections.py --help
```

When done, summarize:

- what changed
- what was validated
- what remains next
