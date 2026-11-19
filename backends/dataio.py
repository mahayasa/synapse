"""Read example data (CSV, TSV, pasted text, markdown tables, XLSX) into pandas tables.

Used by the few-shot builder and by profile_data.py. Pasted text is detected by content, so a
chat message saved to a .txt/.md file works whatever its delimiter.

  * folder  -> every .csv/.tsv/.xlsx/.xlsm inside it
  * file    -> one table (CSV/TSV/pasted text) or one table per sheet (XLSX)
  * legacy .xls is not supported: save it as .xlsx or .csv
"""
from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import pandas as pd

from .base import ConfigError

DIR_EXT = {".csv", ".tsv", ".xlsx", ".xlsm"}                    # picked up when a folder is given
EXCEL_EXT = {".xlsx", ".xlsm"}
FILE_EXT = DIR_EXT | {".txt", ".md", ".psv", ".tab", ".dat"}    # accepted when a file is named
_DELIMS = [("\t", "tab"), (",", "comma"), (";", "semicolon"), ("|", "pipe")]


def safe_name(text) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", str(text)).strip("_").lower()
    return s or "table"


def _unique(name: str, taken) -> str:
    key, i = name, 2
    while key in taken:
        key, i = f"{name}_{i}", i + 1
    return key


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    return df.reset_index(drop=True)


def _is_md_separator(line: str) -> bool:
    return bool(re.fullmatch(r"\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?", line.strip()))


def _sniff(lines: list[str]) -> str | None:
    """Pick the delimiter that splits most lines into the same number of columns."""
    sample = lines[:30]
    best, best_n = None, 0
    for d, _ in _DELIMS:
        counts = [ln.count(d) for ln in sample]
        n = counts[0]
        if n >= 1 and sum(c == n for c in counts) >= 0.8 * len(counts) and n > best_n:
            best, best_n = d, n
    if best:
        return best
    toks = [len(ln.split()) for ln in sample]
    if toks[0] >= 2 and sum(t == toks[0] for t in toks) >= 0.8 * len(toks):
        return "ws"
    return None


def parse_text_table(text: str) -> pd.DataFrame:
    """Parse pasted or file text: markdown table, TSV, CSV, semicolon, pipe or aligned columns."""
    lines = [ln.rstrip("\r") for ln in text.strip("\n").splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ConfigError("need a header line and at least one data row")
    if sum(ln.lstrip().startswith("|") for ln in lines) >= max(2, len(lines) // 2):
        rows = [[c.strip() for c in ln.strip().strip("|").split("|")]
                for ln in lines if not _is_md_separator(ln)]
        width = len(rows[0])
        rows = [(r + [""] * width)[:width] for r in rows]
        buf = io.StringIO()
        csv.writer(buf).writerows(rows)
        return _tidy(pd.read_csv(io.StringIO(buf.getvalue()), skipinitialspace=True))
    delim = _sniff(lines)
    if delim is None:
        raise ConfigError("could not detect columns; use a header row and consistent delimiters")
    body = "\n".join(lines)
    if delim == "ws":
        return _tidy(pd.read_csv(io.StringIO(body), sep=r"\s+", engine="python"))
    return _tidy(pd.read_csv(io.StringIO(body), sep=delim, skipinitialspace=True))


def _read_file(f: Path) -> dict[str, pd.DataFrame]:
    ext = f.suffix.lower()
    if ext == ".xls":
        raise ConfigError("legacy .xls is not supported; save the file as .xlsx or .csv")
    if ext in EXCEL_EXT:
        try:
            sheets = pd.read_excel(f, sheet_name=None)
        except ImportError as e:
            raise ConfigError("reading .xlsx needs openpyxl: pip install openpyxl") from e
        out: dict[str, pd.DataFrame] = {}
        for sheet, df in sheets.items():
            df = _tidy(df)
            if len(df.columns) and len(df):
                out[_unique(safe_name(sheet), out)] = df
        if not out:
            raise ConfigError(f"no data found in {f.name}")
        if len(out) == 1 and re.fullmatch(r"sheet\d*", next(iter(out))):
            return {safe_name(f.stem): next(iter(out.values()))}    # default sheet name: use the file name
        return out
    if ext not in FILE_EXT:
        raise ConfigError(f"unsupported file type {ext!r}; use CSV, TSV, XLSX or a pasted-text file")
    try:
        if ext == ".csv":
            df = _tidy(pd.read_csv(f, encoding="utf-8-sig"))
        elif ext == ".tsv":
            df = _tidy(pd.read_csv(f, sep="\t", encoding="utf-8-sig"))
        else:
            df = parse_text_table(f.read_text(encoding="utf-8-sig", errors="replace"))
    except ConfigError:
        raise
    except Exception as e:
        raise ConfigError(f"could not read {f.name}: {e}") from e
    if df.empty:
        raise ConfigError(f"no data rows found in {f.name}")
    return {safe_name(f.stem): df}


def read_tables(path) -> dict[str, pd.DataFrame]:
    """Return {table_name: DataFrame} from a folder or a single file."""
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"path not found: {p}")
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in DIR_EXT)
        if not files:
            raise ConfigError(f"no CSV, TSV or XLSX files in {p}")
        out: dict[str, pd.DataFrame] = {}
        for f in files:
            for name, df in _read_file(f).items():
                out[_unique(name, out)] = df
        return out
    return _read_file(p)
