from typing import Literal

IOMethods = Literal[
    "FIR",
    "ARX",
    "ARMA",
    "ARMAX",
    "OE",
    "ARARX",
    "ARARMAX",
    "BJ",
    "GEN",
    "EARMAX",
    "EOE",
]

OLSimMethods = Literal["CVA", "MOESP", "N4SID"]
PARSIMMethods = Literal["ParsimK", "ParsimP", "ParsimS"]
# TODO: Use | if issue resolved https://github.com/python/typing/issues/1932
SSMethods = Literal["CVA", "MOESP", "N4SID", "ParsimK", "ParsimP", "ParsimS"]
# TODO: Use | if issue resolved https://github.com/python/typing/issues/1932
AvailableMethods = Literal[
    "FIR",
    "ARX",
    "ARMA",
    "ARMAX",
    "OE",
    "ARARX",
    "ARARMAX",
    "BJ",
    "GEN",
    "EARMAX",
    "EOE",
    "CVA",
    "MOESP",
    "N4SID",
    "ParsimK",
    "ParsimP",
    "ParsimS",
]

ArxMethods = Literal["FIR", "ARX", "ARMA", "ARARX", "ARARMAX"]
OptMethods = Literal[
    "OE", "BJ", "GEN", "ARARX", "ARARMAX", "ARMA", "ARMAX", "EARMAX", "EOE"
]

RLSMethods = Literal["ARX", "ARMA", "ARMAX", "ARARX", "ARARMAX", "FIR", "OE"]

ICMethods = Literal[
    "AIC",  # Akaike Information Criterion
    "AICc",  # Corrected Akaike Information Criterion
    "BIC",  # Bayesian Information Criterion
]

AvailableModes = Literal["lls", "ills", "rlls", "opt"]


METHOD_ORDERS: dict[AvailableMethods, list[str]] = {
    "FIR": ["na", "nb", "theta"],
    "ARX": ["na", "nb", "theta"],
    "ARMAX": ["na", "nb", "nc", "theta"],
    "OE": ["nb", "nf", "theta"],
    "ARMA": ["na", "nc", "theta"],
    "ARARX": ["na", "nb", "nd", "theta"],
    "ARARMAX": ["na", "nb", "nc", "nd", "theta"],
    "BJ": ["nb", "nc", "nd", "nf", "theta"],
    "GEN": ["na", "nb", "nc", "nd", "nf", "theta"],
    "EARMAX": ["na", "nb", "nc", "theta"],
    "EOE": ["nb", "nf", "theta"],
    "CVA": ["na"],
    "MOESP": ["na"],
    "N4SID": ["na"],
    "ParsimK": ["na"],
    "ParsimP": ["na"],
    "ParsimS": ["na"],
}
