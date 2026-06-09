# SPDX-FileCopyrightText: 2026 Epic Games, Inc.
# SPDX-License-Identifier: MIT
"""Smoke tests that pin Lore's ``file diff`` output to git as an independent baseline.

The rest of ``test_diff.py`` asserts against hand-written expected diffs, which encode our
own understanding of what the diff *should* be. Here we instead generate the baseline with
``git diff --no-index`` and assert that Lore emits the same hunks, both with and without
the whitespace-ignoring options. That catches regressions in Lore's diff and whitespace
handling against a real, trusted implementation rather than against ourselves.

Two unavoidable, intended differences between the tools are normalised away so the test
checks the diff *content* rather than cosmetic formatting:

1. File headers differ (``--- file@1`` / ``+++ file`` versus git's ``diff --git`` /
   ``index`` / ``--- a/file``), and git appends a section heading to hunk headers
   (``@@ -2,7 +2,7 @@ Line 01``) that Lore never emits. We drop the header lines and trim
   the section heading to a bare ``@@ -a,b +c,d @@``.
2. For a line the whitespace-ignore options deem equal, the two tools display it as context
   using different whitespace: git uses the new side with whitespace collapsed/stripped,
   Lore preserves the old side verbatim. When an ignore flag is active we normalise the
   whitespace *within context lines only*. The context/added/removed structure and the
   verbatim ``+``/``-`` content stay strictly compared, so a real divergence (e.g. Lore
   classifying a line as changed where git calls it context) still fails the test.

Lore's ``--ignore-space-change`` is documented as collapsing *internal* whitespace runs,
whereas git's ``--ignore-space-change`` additionally ignores trailing whitespace. To avoid
encoding an assumption about that mismatch, scenarios whose only difference is trailing
whitespace are not paired with ``--ignore-space-change`` on its own (the matched flag is
``--ignore-space-at-eol``, and ``both`` covers the combined case).
"""

import re
import shutil
import subprocess

import pytest

from lore import Lore

# git produces the baseline; it is not the thing under test, so skip (don't fail) when it
# is unavailable. The whole module is gated together.
pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(
        shutil.which("git") is None,
        reason="git is required to generate the diff baseline",
    ),
]

# Flag combinations under test, keyed by a short id. Lore has no -w / --ignore-all-space
# equivalent, so that combination is out of scope.
LORE_FLAGS = {
    "default": {},
    "eol": {"ignore_space_at_eol": True},
    "change": {"ignore_space_change": True},
    "both": {"ignore_space_at_eol": True, "ignore_space_change": True},
}
GIT_FLAGS = {
    "default": (),
    "eol": ("--ignore-space-at-eol",),
    "change": ("--ignore-space-change",),
    "both": ("--ignore-space-at-eol", "--ignore-space-change"),
}


def _normalize_context_ws(line: str) -> str:
    """Collapse internal whitespace runs and drop trailing whitespace on a context line.

    ``line`` begins with the unified-diff context prefix (a single space). Used only when
    an ignore flag is active, to neutralise the intended difference in how git and Lore
    render the whitespace of a line they both treat as unchanged.
    """
    return " " + re.sub(r"\s+", " ", line[1:]).rstrip()


def _hunks(diff_text: str, *, normalize_context_ws: bool = False) -> str:
    """Reduce a unified diff to its hunk bodies for format-agnostic comparison.

    Walks the diff as a state machine. Everything before the first ``@@`` is file-header
    preamble (``diff --git``/``index``/``---``/``+++``/Lore's bare filename header/blank
    separators) and is skipped. From the first ``@@`` onward we are in the hunk body and
    keep every line -- crucially including blank lines. A well-formed unified diff renders
    a blank line with its prefix character (a bare ``" "`` for context, ``"+"``/``"-"`` for
    added/removed); a line that has lost its prefix arrives here as ``""`` and is kept as
    such, so the comparison surfaces the discrepancy instead of silently dropping it (which
    would also throw off the hunk line counts). ``+``/``-`` content is kept verbatim;
    context lines are whitespace-normalised only when ``normalize_context_ws`` is set. Hunk
    headers are canonicalised to a bare ``@@ -a,b +c,d @@`` (git appends a section heading
    Lore never emits).

    Lore terminates each patch with a trailing blank line that git does not emit, so
    trailing empty lines are stripped before comparing; interior blank lines are retained.
    """
    out = []
    in_body = False
    for line in diff_text.splitlines():
        if line.startswith("@@"):
            match = re.match(
                r"^(@@ .* @@)", line
            )  # trim git's trailing section heading
            out.append(match.group(1) if match else line)
            in_body = True
            continue
        if not in_body:
            continue  # file-header preamble before the first hunk
        if line.startswith("\\"):
            continue  # git-only "\ No newline at end of file" annotation
        if normalize_context_ws and line.startswith(" "):
            out.append(_normalize_context_ws(line))
        else:
            out.append(line)  # context / +/- / prefix-less blank lines kept verbatim
    while out and out[-1] == "":
        out.pop()  # drop Lore's trailing patch-terminator blank line(s)
    return "\n".join(out)


def _git_diff(tmp_path, old: str, new: str, flags=()) -> str:
    """Baseline diff: write old/new to temp files and run ``git diff --no-index``."""
    old_f = tmp_path / "old"
    new_f = tmp_path / "new"
    old_f.write_text(old)
    new_f.write_text(new)
    result = subprocess.run(
        [
            "git",
            "diff",
            "--no-index",
            "--unified=3",
            *flags,
            str(old_f),
            str(new_f),
        ],
        capture_output=True,
        text=True,
    )
    # --no-index exits 0 when identical and 1 when the files differ; anything higher is a
    # real git error.
    assert result.returncode in (0, 1), (
        f"git diff failed (exit {result.returncode}):\n{result.stderr}"
    )
    return result.stdout


# (old content, new content, applicable flag ids). Each scenario is paired only with the
# flag combinations whose semantics align between git and Lore (see module docstring).
_PLAIN = "".join(f"Line {i:02d}\n" for i in range(1, 11))
_PLAIN_EDIT = "".join(
    "Line 05 (modified)\n" if i == 5 else f"Line {i:02d}\n" for i in range(1, 11)
)
SCENARIOS = {
    # A non-whitespace edit: whitespace flags must not change the result.
    "plain": (_PLAIN, _PLAIN_EDIT, ["default", "eol", "change", "both"]),
    # Introducing a space where there was none is a real change under every flag.
    "introduce_ws": (
        "abc\nsecond\n",
        "a bc\nsecond\n",
        ["default", "eol", "change", "both"],
    ),
    # Trailing-whitespace-only: suppressed by --ignore-space-at-eol (and both). Not paired
    # with --ignore-space-change alone (see module docstring).
    "trailing_ws": (
        "foo   \nbar\nbaz\n",
        "foo\nbar\nbaz\n",
        ["default", "eol", "both"],
    ),
    # Internal-whitespace-run-only: suppressed by --ignore-space-change (and both); shown
    # by --ignore-space-at-eol (which only affects line ends).
    "internal_ws": (
        "a b c\nsecond\n",
        "a  b   c\nsecond\n",
        ["default", "eol", "change", "both"],
    ),
    # EOL-ws line + internal-ws line + a real change. Under an ignore flag the suppressed
    # line becomes context; the real change must still appear.
    "mixed": (
        "alpha   \nbeta  gamma\ndelta\n",
        "alpha\nbeta gamma\nDELTA\n",
        ["default", "eol", "both"],
    ),
    # A blank line sitting in the context window of a real change must render as a context
    # line (a bare " "), not be dropped. Guards the "blank line loses its prefix" bug.
    "blank_in_context": (
        "one\ntwo\n\nfour\nfive\n",
        "one\ntwo\n\nFOUR\nfive\n",
        ["default", "eol", "change", "both"],
    ),
    # Inserting / removing a blank line: the added/removed line is an empty "+"/"-" line.
    "add_blank_line": ("a\nb\n", "a\n\nb\n", ["default", "eol", "change", "both"]),
    "remove_blank_line": ("a\n\nb\n", "a\nb\n", ["default", "eol", "change", "both"]),
    # Indented code: leading whitespace on the changed +/- lines and on the context lines
    # must be preserved verbatim (with no ignore flag) and identically normalised (with).
    "leading_indent": (
        "def f():\n    return 1\n    pass\n",
        "def f():\n    return 2\n    pass\n",
        ["default", "eol", "change", "both"],
    ),
}

# NOTE: the `blank_in_context` cases currently FAIL on purpose. A blank line shown as
# context loses its leading " " prefix in Lore's output (emitted as a bare "\n"), whereas
# git renders it as a single space; downstream parsers then ignore the prefix-less line and
# the hunk line counts no longer add up. These are left failing as a proof of concept until
# the pending CR that restores the prefix is merged.
CASES = [
    (f"{name}-{flag_id}", name, old, new, flag_id)
    for name, (old, new, flag_ids) in SCENARIOS.items()
    for flag_id in flag_ids
]


@pytest.mark.parametrize(
    "case_id, scenario, old, new, flag_id",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_file_diff_matches_git_baseline(
    new_lore_repo, tmp_path, case_id, scenario, old, new, flag_id
):
    """Lore's file diff hunks match git's for each content/whitespace-flag combination."""
    repo: Lore = new_lore_repo()

    test_file = f"{scenario}.txt"
    repo.write_commit_push("Initial commit", {test_file: old}, offline=True)

    with repo.open_file(test_file, "w+") as output_file:
        output_file.write(new)

    lore_output = repo.file_diff(test_file, offline=True, **LORE_FLAGS[flag_id])
    git_output = _git_diff(tmp_path, old, new, flags=GIT_FLAGS[flag_id])

    # Normalise context-line whitespace only when an ignore flag is active.
    normalize = flag_id != "default"
    lore_hunks = _hunks(lore_output, normalize_context_ws=normalize)
    git_hunks = _hunks(git_output, normalize_context_ws=normalize)

    assert lore_hunks == git_hunks, (
        f"Lore diff hunks diverged from the git baseline (case={case_id})\n"
        f"--- git baseline ---\n{git_hunks}\n"
        f"--- lore output ---\n{lore_hunks}\n"
        f"--- raw lore output ---\n{lore_output}"
    )
