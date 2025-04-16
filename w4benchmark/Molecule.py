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
    basis: dict

    @staticmethod
    def parse_from_dict(name, d: dict) -> "Molecule":
        species = name
        spin = d["spin"]
        charge = d["charge"]
        geom = [(atom['element'], tuple(atom['position'][0])) for atom in d["atoms"]]
        basis = {k: Basis.parse_basis(v) for k, v in d["basis"].items()}
        return Molecule(species, spin, charge, geom, basis)

@dataclass(frozen=True)
class Basis:
    h1e: np.ndarray
    h2e: np.ndarray
    cct2: np.ndarray | list[np.ndarray]
    ecore: float
    ncas: int
    nelecas: tuple[int, int]

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
