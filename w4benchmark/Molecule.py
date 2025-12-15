import base64
from dataclasses import dataclass
from typing import List, Tuple, Union
import numpy as np

@dataclass(frozen=True)
class Molecule:
    """
    Represents a molecule with geometry and quantum chemistry basis data.

    Attributes:
        species (str): The molecule's identifier (e.g., chemical formula or name).
        spin (float): Spin multiplicity of the molecule.
        charge (float): Net charge of the molecule.
        geom (List[Tuple[str, Tuple[float, float, float]]]):
            List of atoms and their 3D coordinates. Each entry is a tuple of
            the atomic symbol and a position vector (x, y, z).
        basis (Basis): The basis set and associated tensors.
    """
    species: str
    spin: float
    charge: float
    geom: List[Tuple[str, Tuple[float, float, float]]]
    basis: "Basis"

    @staticmethod
    def parse_from_dict(name, mol: dict, b: dict) -> "Molecule":
        species = name
        spin = mol["spin"]
        charge = mol["charge"]
        geom = [(atom['element'], tuple(atom['position'])) for atom in mol["atoms"]]
        basis = Basis.parse_basis(b)
        return Molecule(species, spin, charge, geom, basis)

@dataclass(frozen=True)
class Basis:
    """
        Contains quantum chemistry basis set information and tensor data.

        Attributes:
            ecore (float): Core energy contribution.
            ncas (int): Number of active orbitals in the CAS (Complete Active Space).
            nelecas (Tuple[int, int]): Number of electrons in the CAS (alpha, beta).
            h1e (np.ndarray): One-electron integrals.
            h2e (np.ndarray): Two-electron integrals.
            cct2 (Union[np.ndarray, List[np.ndarray]]): Coupled cluster T2 amplitudes.
    """

    ecore: float
    ncas: int
    nelecas: tuple[int, int]
    h1e: np.ndarray
    h2e: np.ndarray
    cct2: Union[np.ndarray, List[np.ndarray]]

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
    return np.frombuffer(base64.b64decode(tensor["data"]), dtype="float64").reshape(tensor["shape"])
