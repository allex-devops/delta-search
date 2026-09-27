"""Turns a friendly filter like {"year": {"gte": 2023}, "tag": ["a", "b"]} into Chroma's where syntax."""
import re

RESERVED = {"doc_id", "dochash"}
COMPARE = {"gt", "gte", "lt", "lte"}
EQUALITY = {"eq", "ne"}
FIELD = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,40}$")


class FilterError(ValueError):
    pass


def _scalar(v):
    if isinstance(v, bool) or isinstance(v, (str, int, float)):
        return v
    raise FilterError(f"filter values must be strings, numbers or booleans, not {type(v).__name__}")


def _number(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise FilterError("gt, gte, lt and lte need a number")
    return v


def to_where(filt: dict | None) -> dict | None:
    if not filt:
        return None
    clauses = []
    for field, cond in filt.items():
        if not FIELD.match(field) or field in RESERVED:
            raise FilterError(f"can't filter on '{field}'")
        if isinstance(cond, dict):
            if not cond:
                raise FilterError(f"empty condition for '{field}'")
            for op, val in cond.items():
                if op in COMPARE:
                    clauses.append({field: {f"${op}": _number(val)}})
                elif op in EQUALITY:
                    clauses.append({field: {f"${op}": _scalar(val)}})
                elif op == "in":
                    clauses.append({field: {"$in": _members(val)}})
                else:
                    raise FilterError(f"unknown operator '{op}' for '{field}'")
        elif isinstance(cond, list):
            clauses.append({field: {"$in": _members(cond)}})
        else:
            clauses.append({field: {"$eq": _scalar(cond)}})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _members(v) -> list:
    if not isinstance(v, list) or not v:
        raise FilterError("'in' needs a non-empty list")
    return [_scalar(x) for x in v]
