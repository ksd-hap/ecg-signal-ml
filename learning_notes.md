# Learning Notes

Personal log of what I actually learned at each stage of this project —
concepts, gotchas, and decisions I had to think through. Written after
finishing each major stage, in my own words.

This is separate from the project README: the README documents the project
for an outside reader, this file documents my own learning process.

---

## Stage 1 — Environment & Repository Setup

First time setting up a "real" project from scratch rather than a one-off
script, so this stage was mostly about the scaffolding around the code
rather than the code itself.

What I now understand:
- Why a project gets its own virtual environment (`venv`) instead of
  installing packages globally — keeps dependencies isolated and
  reproducible via `requirements.txt`.
- What `.gitignore` is for: keeping generated/environment-specific files
  (the venv, downloaded dataset files, editor cruft) out of version control,
  so the repo only tracks source of truth.
- The basic Git flow: `git init` → stage changes → commit, and that a
  commit message should describe *what* changed.
- That a README is written for an outside reader (what the project is, how
  to run it) and is a real deliverable, not an afterthought — separate from
  this file, which is my own record of what I learned.

Nothing unexpected went wrong here — this was straightforward once each
piece's purpose was explained.
