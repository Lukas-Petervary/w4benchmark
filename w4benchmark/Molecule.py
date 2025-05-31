import base64
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Molecule:
    species: str
    spin: float
    charge: float
    """molecular geometry in the form of [(atom, (x, y, z)), ...]"""
    geom: list[(str, (float, float, float))]
    basis: "Basis"

    @staticmethod
    def parse_from_dict(name, geom: dict, b: dict) -> "Molecule":
        species = name
        spin = geom["spin"]
        charge = geom["charge"]
        geom = [(atom['element'], tuple(atom['position'])) for atom in geom["atoms"]]
        basis = Basis.parse_basis(b)
        return Molecule(species, spin, charge, geom, basis)

@dataclass(frozen=True)
class Basis:
    ecore: float
    ncas: int
    nelecas: tuple[int, int]
    h1e: np.ndarray
    h2e: np.ndarray
    cct2: np.ndarray | list[np.ndarray]

    @staticmethod
    def parse_basis(basis: dict) -> "Basis":
        return Basis(
            h1e=unpack_tensor(basis["h1e"]),
            h2e=unpack_tensor(basis["h2e"]),
            cct2=unpack_tensor(basis["cct2"]),
            ecore=float(basis["ecore"]),
            ncas=int(basis["ncas"]),
            nelecas=tuple(basis["nelecas"])
        )

def unpack_tensor(tensor: dict) -> np.ndarray | list[np.ndarray]:
    if isinstance(tensor, list): return [unpack_tensor(i) for i in tensor]
    return np.frombuffer(base64.b64decode(tensor["data"]), dtype=tensor["dtype"]).reshape(tensor["shape"])
