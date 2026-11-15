# ENM156

A small causal speech denoiser (Python/PyTorch, 16 kHz mono), and under
`plugin/` a C++ port of it that runs as a VST3 effect (real time) and an ARA
plugin (whole recorded clips, offline). This is a fork of
`supersonic-lumina/ENM156`.

## Where the documentation is

Read these instead of rediscovering things; keep them up to date when
behaviour changes.

| | |
|---|---|
| `README.md` | the Python project: setup, every command, layout |
| `config.toml` | every setting, documented inline; personal overrides go in `config.local.toml` (git-ignored) |
| `legacy/README.md` | how the old single-file scripts map to today's settings |
| `plugin/README.md` | the plugin: decisions and licences, build and test commands, core API, latency, the real-time and ARA sections, limits, and the checklists for testing in a DAW |
| `plugin/tools/README.md` | the exported model file format, the golden test vectors, and the exact semantics a port must reproduce |
| issue 1 on the fork | the plan and what is still open |

## Keeping the fork close to upstream

- Plugin work is additive: new files under `plugin/` and
  `.github/workflows/plugin.yml`. Python additions for the plugin are new
  files in `plugin/tools/` that import `denoiser/` without changing it.
- Do not edit the existing Python project (`denoiser/`, `train*.py`,
  `evaluate.py`, `config.toml`, `tests/`, the root `README.md`) in the fork.
  A fix or change there goes upstream first. The owner made one exception:
  five review fixes (pull request 14), each its own commit so it can be
  offered upstream.

## Checks

```bash
# Python: every script in tests/, no dataset needed
for t in tests/test_*.py; do .venv/bin/python "$t" || break; done

# C++ core, command-line tool and tests (seconds)
cmake -S plugin -B plugin/build -G Ninja && cmake --build plugin/build && ctest --test-dir plugin/build --output-on-failure

# The plugin as well (minutes; fetches JUCE and the ARA SDK)
cmake -S plugin -B plugin/build-plugin -G Ninja -DDENOISER_BUILD_PLUGIN=ON && cmake --build plugin/build-plugin && ctest --test-dir plugin/build-plugin --output-on-failure
```

CI builds and tests the core and the plugin on Linux, Windows and macOS for
changes under `plugin/`.

## Working with workers

When work is split between several agents:

- Start workers **nested under the current session**, not as separate
  top-level sessions, so the session that started them hands over the brief,
  receives their report and stays responsible for the result.
- Each worker works on its own branch (in its own worktree) and opens **its
  own pull request**. Workers that depend on each other are started in
  dependency order, each from the branch it builds on.
- The session that started a worker reviews that worker's pull request:
  it rebuilds and re-runs the tests itself and reads the diff before
  approving, and says what it did not verify. Once its review passes and CI
  is green on Linux, Windows and macOS, that session **merges** the pull
  request; it does not wait for the owner. (GitHub does not accept a formal
  approval from the account that opened a pull request, so the approval is
  recorded as a review comment.)
- After its pull request is approved and merged, the worker **cleans up
  after itself**, so that a finished worker leaves nothing behind. In
  practice this takes three hands, because a worker cannot do all of it:
  - the worker removes its build folders and every scratch file it made;
  - the session that started it removes the worker's worktree and local
    branch from the main checkout, after checking the branch is fully merged
    (a worker's isolation does not let it touch anything outside its own
    worktree);
  - the remote branch is removed by GitHub's "Automatically delete head
    branches" setting, or by the owner. Cloud sessions are refused branch
    deletion (HTTP 403); do not work around that.

## Things that have already gone wrong once

- **Bit-exact comparisons across compilers.** Two compiled copies of the same
  float arithmetic can differ in the last bits (fused multiply-add; it broke
  a test on macOS arm64 only). Compare audio against a reference with a
  tolerance (1e-5 relative to `max(1, |ref|)`, as the golden tests do); keep
  exact equality for copied samples and integers. To reproduce an arm64
  failure on Linux, cross-compile with Clang for `aarch64-linux-gnu` and run
  the test under `qemu-aarch64`.
- **Nothing here has been checked with a trained model or in a DAW** unless a
  README says so explicitly. `*.pt` files are git-ignored and no dataset is
  in the repo; tests use untrained models and synthetic audio. Do not report
  a DAW or trained-model result that was not actually obtained.
- **GitHub only closes the first issue after a keyword.** "Resolves #2 and
  #3" closes #2 only: repeat the keyword for each issue.
