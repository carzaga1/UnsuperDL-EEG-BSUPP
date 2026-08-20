# scripts/

Thin CLI entry points. Each script here should do argument parsing only and immediately
call into `bscarlos` (`src/bscarlos/`) for the real logic — no business logic lives here.

Populated starting at Task 15 (`preprocess_data`) and Task 20 (`train_vae`/`predict_vae`)
of `REFACTORING_PLAN.md`.
