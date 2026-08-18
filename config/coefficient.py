"""
Resistivity coefficients for copper alloying elements.

Each element maps to a dict with:
    coefficient : float  - resistivity increase coefficient in nΩ·m per wt%
                          (i.e. how much each 0.001 wt% of the element increases
                           the resistivity of pure copper)
"""

from __future__ import annotations


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------


def _r(
    coefficient: float,
) -> dict:
    """Build a single element resistivity coefficient dict."""
    return {
        "coefficient": coefficient
    }


# ------------------------------------------------------------------
# Resistivity coefficients [nΩ·m per wt%]
# Reference: Conductivity and resistivity values for copper alloys
# Pure copper baseline: ~17.0 nΩ·m at 20°C
# ------------------------------------------------------------------


# resistivity.py
# Resistivity reduction factors for 1 % element (nΩ·m)

RESISTIVITY_FACTORS = {
    "Zn": 3.4,
    "Sn": 29,
    "P":  130,
    "Mn": 33,
    "Fe": 110,
    "Ni": 14,
    "Si": 69,
    "Mg": 21,
    "Cr": 48,
    "As": 56,
    "Sb": 30, #tAKEN from study 1, as data unavialbale 
    "Cd": 1.6,
    "Bi": 17,
    "Ag": 2,
    "Co": 71,
    "Al": 26,
    "S":  160,
    "Be": 26, #No data available, since beryllium is chemically similar to aluminium (same charge to mass ratio, the values should be very close)
    "Zr": 0.5, #No dats available in the table, assigning a random value 
    "Au": 2, #No dats available in the table that we are using , but chemically similar  to silver so assigning a similar value. Cu,Ag, Au come in the same group of the periodic tablee 
    "B": 20, #No data available in the table, assigning similar value as silicon due to chemcial group. But studies don't show any impact of boron in electrical conducitvity at low concentrations. 
    "Ti": 200, 
    "Pt": 2, #No data available in the table. Assigning a value really low value, because  impact of Platinum in resistivity is very less at low concentrations.  as in our chart
    "Cu": 0.00,
    "Te": 34,
    "Pb" : 9
}

ENRICHMENT_FACTORS = {
    "Zn": 0.15, #mild or borderline
    "Sn": 0.68, # moderate
    "P":  0.58,
    "Mn": -0.05, #slight strengthener,
    "Fe": -0.1, #mild strengthener
    "Ni": -0.15, #grain boundary strengthener
    "Si": -0.05,
    "Mg": 0.1,
    "Cr": -0.1,
    "As": 0.1,
    "Sb": 0.82, #strong embrittler #tAKEN from study 1, as data unavialbale 
    "Cd": 0, #not reported anywhere for  embrittlement 
    "Bi": 1,
    "Ag": 0,
    "Co": 0,
    "Al": 0,
    "S":  0.65,
    "Be": -0.05, 
    "Zr": -0.15, 
    "B": -0.2,
    "Pt": 0,
    "Te": 0.88,
    "Cu": 0.00,
    "Pb": 0.95
}