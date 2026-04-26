# Repository Guidelines

## Project Structure & Module Organization
`src/final_project/` contains the shipped package. Keep data ingest and split logic in `data/`, training and inference flows in `engine/`, model code in `model/`, and reusable helpers in `utils/`. The CLI entrypoints live in `main.py` and `src/final_project/cli.py`; config parsing is centralized in `src/final_project/config.py`. Tests mirror the package layout under `tests/` (`tests/data`, `tests/engine`, `tests/model`) with repo-level checks in `tests/test_cli.py` and `tests/test_config.py`. Experiment YAMLs belong in `configs/`, research notes in `docs/`, and one-off utilities in `scripts/`.

## Build, Test, and Development Commands
Use `uv` for local workflows.

- `uv sync --group dev --extra train`: install test and training dependencies.
- `uv run python main.py --help`: list supported commands.
- `uv run pytest -q`: run the full test suite.
- `uv run python main.py run-cv --config configs/smoke.yaml`: run a lightweight cross-validation smoke path.
- `uv run python main.py blend --spec outputs/research/<spec>.json`: build a blend from saved run outputs.

## Coding Style & Naming Conventions
Target Python 3.12, use 4-space indentation, and keep type hints on public functions and dataclass fields. Follow existing patterns such as `Path`-based filesystem handling, frozen `@dataclass(slots=True)` config objects, and small helper functions for validation. Use `snake_case` for modules, functions, variables, test names, and YAML config files such as `baseline_mammonet32k_warmup_e4.yaml`. Keep CLI command names kebab-case, matching existing commands like `run-cv` and `warmup-external`.

## Testing Guidelines
Write tests with `pytest`, placing them beside the affected domain. Example: changes in `src/final_project/engine/blending.py` should usually add coverage in `tests/engine/test_blending.py`. Name files `test_<module>.py` and test functions `test_<behavior>`. Prefer fast unit tests with fakes or `monkeypatch` over real training runs, and guard GPU-only coverage with `pytest.mark.skipif`.

## Commit & Pull Request Guidelines
Recent history follows conventional prefixes such as `feat:`, `docs:`, `test:`, and `refactor:`. Keep commit subjects short, imperative, and scoped to one change. For pull requests, include the purpose, affected configs or commands, validation you ran (`uv run pytest -q`, smoke CLI calls, etc.), and any metric deltas for training, CV, or blend changes. Link the relevant issue or experiment note when available.

## Data & Artifact Hygiene
Large datasets and generated artifacts are intentionally local-only: `train_img/`, `test_img/`, `MammoNet32k_new/`, `outputs/`, `*.zip`, `*.csv`, and `*.docx` are gitignored. Do not commit derived checkpoints or experiment outputs unless the repo explicitly starts tracking a summarized report in `docs/`.
