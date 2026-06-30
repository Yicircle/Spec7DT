from __future__ import annotations

from typing import Iterable


def cigale_columns() -> dict[str, str]:
    """Return CIGALE column names for known observatory/filter pairs."""
    cols = {
        "GALEX.NUV": "galex.NUV",
        "GALEX.FUV": "galex.FUV",
        "SDSS.u": "sloan.sdss.u",
        "SDSS.g": "sloan.sdss.g",
        "SDSS.r": "sloan.sdss.r",
        "SDSS.i": "sloan.sdss.i",
        "SDSS.z": "sloan.sdss.z",
        "PanStarrs.y": "PAN-STARRS_y",
        "2MASS.J": "J_2mass",
        "2MASS.H": "H_2mass",
        "2MASS.Ks": "Ks_2mass",
        "Spitzer.ch1": "spitzer.irac.ch1",
        "Spitzer.ch2": "spitzer.irac.ch2",
        "Spitzer.ch3": "spitzer.irac.ch3",
        "Spitzer.ch4": "spitzer.irac.ch4",
        "WISE.w1": "wise.W1",
        "WISE.w2": "wise.W2",
        "WISE.w3": "wise.W3",
        "WISE.w4": "wise.W4",
        "WISE.W1": "wise.W1",
        "WISE.W2": "wise.W2",
        "WISE.W3": "wise.W3",
        "WISE.W4": "wise.W4",
        "F657N": "HST.UVIS1.F657N",
        "F658N": "HST.UVIS1.F658N",
        "PACS.blue": "herschel.pacs.blue",
        "PACS.green": "herschel.pacs.green",
        "PACS.red": "herschel.pacs.red",
        "SPIRE.PSW": "herschel.spire.PSW",
        "SPIRE.PMW": "herschel.spire.PMW",
        "SPIRE.PLW": "herschel.spire.PLW",
    }
    cols.update({f"{key}_err": f"{value}_err" for key, value in cols.items() if not key.endswith("_err")})
    return cols


def eazy_columns(col_names: Iterable[str]) -> dict[str, str]:
    """Return EAZY-style F_/E_ column names for generated catalog columns."""
    flux_dict = {name: f"F_{name}" for name in col_names if "_err" not in name}
    err_dict = {name: f"E_{name.removesuffix('_err')}" for name in col_names if "_err" in name}
    flux_dict.update(err_dict)
    return flux_dict


def lephare_columns() -> dict[str, str]:
    return {}


def ppxf_columns() -> dict[str, str]:
    return {}


def get_catalog_columns(cat_type: str, col_names: Iterable[str] = ()) -> dict[str, str]:
    """Return input-catalog column mappings for a supported fitting tool."""
    adapters = {
        "cigale": lambda: cigale_columns(),
        "eazy": lambda: eazy_columns(col_names),
        "lephare": lambda: lephare_columns(),
        "ppxf": lambda: ppxf_columns(),
        "goyangyi": lambda: cigale_columns(),
    }
    key = cat_type.lower()
    if key not in adapters:
        raise KeyError(f"Unknown catalog type '{cat_type}'. Available: {sorted(adapters)}")
    return adapters[key]()
