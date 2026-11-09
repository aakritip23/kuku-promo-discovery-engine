"""
screenplay_parser.py - Module 1 of the Promo Discovery Engine.

Turns timestamped microdrama screenplays (one .txt per episode) into structured JSON.

Design rules
  * Original timestamps are preserved verbatim as strings (start_ts / end_ts). A numeric
    copy (start_sec / end_sec) is derived from the string; it is never the other way round.
  * Nothing is invented. A missing or malformed timestamp stays None / its raw text and is
    reported in `issues`; it is never guessed or interpolated.
  * One bad episode never stops the run. Problems are collected as issues with file and line.

Expected source format (observed in the Kuku pack)
    EPISODE NUMBER: Episode 1
    EPISODE LOGLINE: ...
    CONTINUITY NOTES: ...
    --- SCRIPT START ---
    SCENE 01
    Scene heading: INT. CATHEDRAL - DAY
    Scene start: 00:00.00
    Scene end: 02:18.00
    **CAMERA:**                      (also **ACTION:**, **SOUND FX:**, **CHARACTER (delivery):**)
    Start: (00:00.00) | End: (00:04.80)
    text...
    --- SCRIPT END ---

CLI
    python screenplay_parser.py <input_dir> -o extracted_scenes.json [--episodes 1,2,3]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

PARSER_VERSION = "1.0"

# ----------------------------------------------------------------------------- patterns
TS_RE = re.compile(r"^(\d{1,3}):([0-5]\d)\.(\d{2})$")                      # mm:ss.cc
FILE_RE = re.compile(r"episode-(\d+)-(.*)\.txt$", re.IGNORECASE)
SCENE_RE = re.compile(r"^SCENE\s+(\d+)\s*$", re.IGNORECASE)
BLOCK_RE = re.compile(r"^\*\*(.+?):\*\*\s*$")
TIME_LINE_RE = re.compile(r"^Start:\s*\(([^)]*)\)\s*\|\s*End:\s*\(([^)]*)\)\s*$", re.IGNORECASE)
HEADER_RE = re.compile(r"^([A-Z][A-Z ]+):\s*(.*)$")
SCRIPT_START = "--- SCRIPT START ---"
SCRIPT_END = "--- SCRIPT END ---"

VOICE_MARKERS = {"O.S.": "O.S.", "V.O.": "V.O.", "OFF-SCREEN": "O.S.", "OFF SCREEN": "O.S."}
NON_DIALOGUE = {"CAMERA": "camera", "ACTION": "action", "SOUND FX": "sound"}
# Tolerance (seconds) for the sanity checks below; the source has sub-second jitter.
BOUND_TOLERANCE = 0.05


# ----------------------------------------------------------------------------- issues
def make_issue(severity: str, code: str, message: str, file: str,
               line: Optional[int] = None, ref: Optional[str] = None) -> Dict[str, Any]:
    """severity: 'error' (data unusable/missing) or 'warning' (data kept, but odd)."""
    return {"severity": severity, "code": code, "message": message,
            "file": file, "line": line, "ref": ref}


# ----------------------------------------------------------------------------- low-level helpers
def parse_timestamp(raw: Optional[str]) -> Optional[float]:
    """'01:06.80' -> 66.8. Returns None for anything that is not a valid mm:ss.cc stamp."""
    if raw is None:
        return None
    m = TS_RE.match(raw.strip())
    if not m:
        return None
    return round(int(m[1]) * 60 + int(m[2]) + int(m[3]) / 100, 2)


def parse_filename(path: str) -> Tuple[Optional[int], Optional[str]]:
    """'episode-007-Episode 1671709.txt' -> (7, 'Episode 1671709')."""
    m = FILE_RE.search(os.path.basename(path))
    return (int(m[1]), m[2]) if m else (None, None)


def read_lines(path: str, issues: List[dict]) -> List[str]:
    """Read a file as UTF-8 (BOM tolerated); fall back to latin-1 with a warning."""
    name = os.path.basename(path)
    try:
        with open(path, encoding="utf-8-sig") as fh:
            return fh.read().split("\n")
    except UnicodeDecodeError:
        issues.append(make_issue("warning", "encoding_fallback",
                                 "File is not valid UTF-8; decoded as latin-1.", name))
        with open(path, encoding="latin-1") as fh:
            return fh.read().split("\n")


def parse_block_label(label: str) -> Dict[str, Optional[str]]:
    """
    '**LADY JULIANA (V.O.) (Sad):**' label -> kind / character / voice_mode / delivery.
    CAMERA, ACTION and SOUND FX are non-speech blocks; every other label is a speaker.
    Parenthetical groups that are not voice markers are kept as `delivery` (nothing is dropped).
    """
    name = re.sub(r"\s*\([^)]*\)", "", label).strip()
    groups = [g.strip() for g in re.findall(r"\(([^)]*)\)", label)]
    if name.upper() in NON_DIALOGUE:
        return {"kind": NON_DIALOGUE[name.upper()], "character": None, "voice_mode": None,
                "delivery": ", ".join(groups) or None}
    voice, delivery = None, []
    for g in groups:
        if g.upper() in VOICE_MARKERS:
            voice = VOICE_MARKERS[g.upper()]
        else:
            delivery.append(g)
    return {"kind": "dialogue", "character": name, "voice_mode": voice,
            "delivery": ", ".join(delivery) or None}


def clean_dialogue(text: str) -> str:
    """Strip the surrounding straight/curly quotes only; the words are untouched."""
    return text.strip().strip('"“”').strip()


# ----------------------------------------------------------------------------- header
def parse_header(lines: List[str]) -> Tuple[Dict[str, str], Optional[int]]:
    """Read 'KEY: value' lines up to the SCRIPT START marker. Returns (header, marker_index)."""
    header: Dict[str, str] = {}
    for i, line in enumerate(lines):
        if line.strip() == SCRIPT_START:
            return header, i
        m = HEADER_RE.match(line.strip())
        if m:
            header[m[1].strip()] = m[2].strip()
    return header, None


def resolve_episode_id(header: Dict[str, str], file_index: Optional[int],
                       fname: str, issues: List[dict]) -> Optional[int]:
    """Episode id from 'EPISODE NUMBER: Episode N'; the filename index is the fallback."""
    m = re.search(r"(\d+)", header.get("EPISODE NUMBER", ""))
    from_header = int(m[1]) if m else None
    if from_header is None:
        issues.append(make_issue("warning" if file_index else "error", "episode_number_missing",
                                 "No parsable 'EPISODE NUMBER' in header; "
                                 + ("using filename index." if file_index else "no filename index either."),
                                 fname))
        return file_index
    if file_index is not None and file_index != from_header:
        issues.append(make_issue("warning", "episode_number_mismatch",
                                 f"Header says episode {from_header}, filename says {file_index}; using header.",
                                 fname))
    return from_header


# ----------------------------------------------------------------------------- scenes & blocks
def new_scene(number: int, line_no: int) -> Dict[str, Any]:
    return {"scene_number": number, "heading": None, "start_ts": None, "end_ts": None,
            "start_sec": None, "end_sec": None, "source_line": line_no, "items": []}


def set_scene_time(scene: dict, which: str, raw: str, fname: str, line_no: int, issues: List[dict]) -> None:
    """Store a 'Scene start/end' value verbatim; numeric copy only if it validates."""
    raw = raw.strip()
    scene[f"{which}_ts"] = raw or None
    sec = parse_timestamp(raw)
    scene[f"{which}_sec"] = sec
    if sec is None:
        issues.append(make_issue("error", "scene_timestamp_malformed",
                                 f"Scene {which} {raw!r} is not a valid mm:ss.cc timestamp.",
                                 fname, line_no, f"scene {scene['scene_number']}"))


def open_block(label: str, line_no: int) -> Dict[str, Any]:
    block = parse_block_label(label)
    block.update({"label_raw": label, "start_ts": None, "end_ts": None, "start_sec": None,
                  "end_sec": None, "text_lines": [], "source_line": line_no, "time_source_line": None})
    return block


def apply_time_line(block: dict, match: "re.Match[str]", fname: str, line_no: int,
                    issues: List[dict], ref: str) -> None:
    """Attach 'Start: (..) | End: (..)' to the open block, keeping raw strings."""
    block["time_source_line"] = line_no
    for which, raw in (("start", match[1].strip()), ("end", match[2].strip())):
        block[f"{which}_ts"] = raw or None
        block[f"{which}_sec"] = parse_timestamp(raw)
        if block[f"{which}_sec"] is None:
            issues.append(make_issue("error", "item_timestamp_malformed",
                                     f"{which.capitalize()} {raw!r} is not a valid mm:ss.cc timestamp.",
                                     fname, line_no, ref))


def finalize_block(block: dict, scene: dict, eid: int, fname: str, issues: List[dict]) -> None:
    """Close a block: build the item record, run per-item checks, append to the scene."""
    idx = len(scene["items"])
    item_id = f"E{eid:03d}_S{scene['scene_number']:02d}_{idx:03d}"
    text = " ".join(block["text_lines"]).strip()
    if block["kind"] == "dialogue":
        text = clean_dialogue(text)
    item = {
        "item_id": item_id, "kind": block["kind"], "character": block["character"],
        "voice_mode": block["voice_mode"], "delivery": block["delivery"],
        "start_ts": block["start_ts"], "end_ts": block["end_ts"],
        "start_sec": block["start_sec"], "end_sec": block["end_sec"],
        "text": text, "source_line": block["source_line"],
        "time_source_line": block["time_source_line"],
    }
    if block["time_source_line"] is None:
        issues.append(make_issue("error", "item_missing_timestamp",
                                 "Block has no 'Start: (..) | End: (..)' line; timestamps left empty.",
                                 fname, block["source_line"], item_id))
    elif item["start_sec"] is not None and item["end_sec"] is not None \
            and item["end_sec"] <= item["start_sec"]:
        issues.append(make_issue("warning", "item_end_not_after_start",
                                 f"End {item['end_ts']} is not after start {item['start_ts']}.",
                                 fname, block["time_source_line"], item_id))
    if not text:
        issues.append(make_issue("warning", "item_empty_text", "Block has no text.",
                                 fname, block["source_line"], item_id))
    scene["items"].append(item)


def parse_scenes(lines: List[str], first_line_idx: int, eid: int, fname: str,
                 issues: List[dict]) -> List[dict]:
    """State machine over the script body: SCENE -> scene fields -> blocks."""
    scenes: List[dict] = []
    scene: Optional[dict] = None
    block: Optional[dict] = None
    saw_end_marker = False

    def close() -> None:
        nonlocal block
        if block is not None and scene is not None:
            finalize_block(block, scene, eid, fname, issues)
        block = None

    for offset, raw in enumerate(lines[first_line_idx:], start=first_line_idx):
        line_no, s = offset + 1, raw.strip()
        if s == SCRIPT_END:
            close()
            saw_end_marker = True
            break
        m = SCENE_RE.match(s)
        if m:
            close()
            scene = new_scene(int(m[1]), line_no)
            scenes.append(scene)
            continue
        low = s.lower()
        if scene is not None and block is None and low.startswith("scene heading:"):
            scene["heading"] = s.split(":", 1)[1].strip() or None
            continue
        if scene is not None and block is None and low.startswith("scene start:"):
            set_scene_time(scene, "start", s.split(":", 1)[1], fname, line_no, issues)
            continue
        if scene is not None and block is None and low.startswith("scene end:"):
            set_scene_time(scene, "end", s.split(":", 1)[1], fname, line_no, issues)
            continue
        if low.startswith("actions and dialogue"):
            continue
        m = BLOCK_RE.match(s)
        if m:
            close()
            if scene is None:
                issues.append(make_issue("warning", "content_before_first_scene",
                                         f"Block {m[1]!r} appears before any SCENE marker; skipped.",
                                         fname, line_no))
                continue
            block = open_block(m[1].strip(), line_no)
            continue
        if block is not None:
            tm = TIME_LINE_RE.match(s)
            if tm and block["time_source_line"] is None:
                apply_time_line(block, tm, fname, line_no, issues, f"line {line_no}")
                continue
            if low.startswith("start:") and block["time_source_line"] is None:
                issues.append(make_issue("error", "time_line_malformed",
                                         f"Unreadable time line: {s[:80]!r}",
                                         fname, line_no, block["label_raw"]))
                block["time_source_line"] = line_no   # seen, but unusable; stamps stay None
                continue
            if s:
                block["text_lines"].append(s)
        elif s and scene is not None:
            issues.append(make_issue("warning", "orphan_text",
                                     f"Text outside any block in scene {scene['scene_number']}: {s[:60]!r}",
                                     fname, line_no))
    close()
    if not saw_end_marker:
        issues.append(make_issue("warning", "script_end_missing",
                                 "No '--- SCRIPT END ---' marker; parsed to end of file.", fname))
    return scenes


# ----------------------------------------------------------------------------- validation
def validate_scene_times(scenes: List[dict], eid: int, fname: str, issues: List[dict]) -> None:
    """Scene-level checks. Reports problems; never edits a timestamp."""
    prev: Optional[dict] = None
    for sc in scenes:
        ref = f"E{eid:03d}_S{sc['scene_number']:02d}"
        if not sc["heading"]:
            issues.append(make_issue("warning", "scene_heading_missing", "Scene has no heading.",
                                     fname, sc["source_line"], ref))
        for which in ("start", "end"):
            if sc[f"{which}_ts"] is None:
                issues.append(make_issue("error", "scene_timestamp_missing",
                                         f"Scene {which} is missing (left empty, not inferred).",
                                         fname, sc["source_line"], ref))
        s, e = sc["start_sec"], sc["end_sec"]
        if s is not None and e is not None and e <= s:
            issues.append(make_issue("error", "scene_end_not_after_start",
                                     f"Scene end {sc['end_ts']} <= start {sc['start_ts']}.",
                                     fname, sc["source_line"], ref))
        if s is not None and e is not None:
            for it in sc["items"]:
                if it["start_sec"] is not None and it["start_sec"] < s - BOUND_TOLERANCE or \
                   it["end_sec"] is not None and it["end_sec"] > e + BOUND_TOLERANCE:
                    issues.append(make_issue("warning", "item_outside_scene",
                                             f"Item {it['start_ts']}-{it['end_ts']} falls outside scene "
                                             f"{sc['start_ts']}-{sc['end_ts']}.",
                                             fname, it["source_line"], it["item_id"]))
        if prev and prev["end_sec"] is not None and s is not None and abs(s - prev["end_sec"]) > BOUND_TOLERANCE:
            issues.append(make_issue("warning", "scene_gap_or_overlap",
                                     f"Scene starts {sc['start_ts']} but previous ended {prev['end_ts']}.",
                                     fname, sc["source_line"], ref))
        prev = sc


# ----------------------------------------------------------------------------- episode / directory
def parse_episode_file(path: str) -> Dict[str, Any]:
    """Parse one file. Never raises: failures are returned as parse_status='failed' + issues."""
    fname = os.path.basename(path)
    issues: List[dict] = []
    file_index, source_id = parse_filename(path)
    episode: Dict[str, Any] = {
        "episode_id": file_index, "episode_label": None, "source_file": fname, "source_id": source_id,
        "logline": None, "continuity_notes": None, "duration_sec": None, "n_scenes": 0, "n_items": 0,
        "parse_status": "failed", "scenes": [], "issues": issues,
    }
    try:
        lines = read_lines(path, issues)
        if not any(l.strip() for l in lines):
            issues.append(make_issue("error", "empty_file", "File is empty.", fname))
            return episode
        header, marker = parse_header(lines)
        eid = resolve_episode_id(header, file_index, fname, issues)
        episode["episode_id"] = eid
        episode["episode_label"] = header.get("EPISODE NUMBER")
        episode["logline"] = header.get("EPISODE LOGLINE")
        episode["continuity_notes"] = header.get("CONTINUITY NOTES")
        for key in ("EPISODE LOGLINE", "CONTINUITY NOTES"):
            if key not in header:
                issues.append(make_issue("warning", "header_field_missing", f"Header field {key} missing.", fname))
        if marker is None:
            issues.append(make_issue("error", "script_start_missing",
                                     "No '--- SCRIPT START ---' marker; cannot locate the script body.", fname))
            return episode
        if eid is None:
            return episode
        scenes = parse_scenes(lines, marker + 1, eid, fname, issues)
        if not scenes:
            issues.append(make_issue("error", "no_scenes", "No SCENE blocks found.", fname))
        validate_scene_times(scenes, eid, fname, issues)
        for sc in scenes:
            sc["scene_id"] = f"E{eid:03d}_S{sc['scene_number']:02d}"
        episode["scenes"] = scenes
        episode["n_scenes"] = len(scenes)
        episode["n_items"] = sum(len(sc["items"]) for sc in scenes)
        ends = [sc["end_sec"] for sc in scenes if sc["end_sec"] is not None]
        episode["duration_sec"] = max(ends) if ends else None
        has_error = any(i["severity"] == "error" for i in issues)
        episode["parse_status"] = "ok_with_errors" if has_error else (
            "ok_with_warnings" if issues else "ok")
    except Exception as exc:  # defensive: a single malformed file must not stop the batch
        issues.append(make_issue("error", "unexpected_exception", f"{type(exc).__name__}: {exc}", fname))
        episode["parse_status"] = "failed"
    return episode


def natural_key(path: str) -> Tuple[int, str]:
    idx, _ = parse_filename(path)
    return (idx if idx is not None else 10 ** 9, os.path.basename(path))


def discover_files(input_dir: str) -> List[str]:
    """All episode .txt files under input_dir (recursive), in episode order."""
    files = glob.glob(os.path.join(input_dir, "**", "*.txt"), recursive=True)
    return sorted(files, key=natural_key)


def canonicalize_characters(episodes: List[dict]) -> None:
    """
    Add `character_canon` to dialogue items. Conservative and reversible: strips only the
    honorific 'LADY ' when the remainder is itself a speaker label elsewhere in the show.
    Real alias resolution (e.g. THE FORMER GROOM = Gerald) is left to human review.
    """
    speakers = {it["character"] for ep in episodes for sc in ep["scenes"]
                for it in sc["items"] if it["kind"] == "dialogue" and it["character"]}
    for ep in episodes:
        for sc in ep["scenes"]:
            for it in sc["items"]:
                c = it["character"]
                it["character_canon"] = c[5:] if c and c.startswith("LADY ") and c[5:] in speakers else c


def summarize(episodes: List[dict]) -> Dict[str, Any]:
    all_issues = [i for ep in episodes for i in ep["issues"]]
    by_code: Dict[str, int] = {}
    for i in all_issues:
        by_code[i["code"]] = by_code.get(i["code"], 0) + 1
    kinds: Dict[str, int] = {}
    for ep in episodes:
        for sc in ep["scenes"]:
            for it in sc["items"]:
                kinds[it["kind"]] = kinds.get(it["kind"], 0) + 1
    status: Dict[str, int] = {}
    for ep in episodes:
        status[ep["parse_status"]] = status.get(ep["parse_status"], 0) + 1
    return {"episodes": len(episodes), "scenes": sum(e["n_scenes"] for e in episodes),
            "items": sum(e["n_items"] for e in episodes), "items_by_kind": kinds,
            "episode_status": status, "issues_total": len(all_issues),
            "issues_by_severity": {s: sum(1 for i in all_issues if i["severity"] == s)
                                   for s in ("error", "warning")},
            "issues_by_code": by_code}


def parse_directory(input_dir: str, only_episodes: Optional[List[int]] = None) -> Dict[str, Any]:
    """Parse every episode file under input_dir and return the full document."""
    files = discover_files(input_dir)
    episodes = [parse_episode_file(p) for p in files]
    if only_episodes:
        wanted = set(only_episodes)
        episodes = [e for e in episodes if e["episode_id"] in wanted]
    canonicalize_characters(episodes)
    seen: Dict[int, str] = {}
    for ep in episodes:  # duplicate episode ids would silently merge downstream
        eid = ep["episode_id"]
        if eid is not None and eid in seen:
            ep["issues"].append(make_issue("error", "duplicate_episode_id",
                                           f"Episode {eid} also defined in {seen[eid]}.", ep["source_file"]))
        elif eid is not None:
            seen[eid] = ep["source_file"]
    return {"parser_version": PARSER_VERSION,
            "show": os.path.basename(os.path.normpath(input_dir)),
            "summary": summarize(episodes), "episodes": episodes}


def write_json(doc: Dict[str, Any], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Parse timestamped microdrama screenplays to JSON.")
    ap.add_argument("input_dir")
    ap.add_argument("-o", "--output", default="extracted_scenes.json")
    ap.add_argument("--episodes", help="comma-separated episode numbers to keep, e.g. 1,2,3")
    args = ap.parse_args(argv)
    only = [int(x) for x in args.episodes.split(",")] if args.episodes else None
    doc = parse_directory(args.input_dir, only)
    write_json(doc, args.output)
    s = doc["summary"]
    print(f"{s['episodes']} episodes, {s['scenes']} scenes, {s['items']} items; "
          f"issues: {s['issues_by_severity']}; status: {s['episode_status']}")
    print(f"wrote {args.output}")
    return 1 if s["episode_status"].get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
