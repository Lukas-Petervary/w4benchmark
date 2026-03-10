import base64, requests, os
import numpy as np
from w4benchmark import *
from dotenv import load_dotenv

def serialize_tensor(tensor: np.ndarray | list):
    if isinstance(tensor, list):
        return [s for s in (serialize_tensor(i) for i in tensor) if s is not None]
    if any(dim == 0 for dim in tensor.shape):
        return None
    return {
        "shape": tensor.shape,
        "data": base64.b64encode(tensor.tobytes()).decode("utf-8")
    }

load_dotenv()
@W4Decorators.process(basis="sto6g", api_key=os.getenv("API_KEY"), debug=10, post=True)
def db_post(name: str, mol: Molecule):
    _data = {
        "basis": W4.parameters.basis,
        "name": name,
        "ncas": mol.basis.ncas,
        "ecore": str(mol.basis.ecore),
        "nelecas": list(mol.basis.nelecas),
        "h1e": serialize_tensor(mol.basis.h1e),
        "h2e": serialize_tensor(mol.basis.h2e)
    }

    if W4.parameters.post:
        response = requests.post(
            W4.parameters.api_url + "/entries?token="+os.getenv("API_KEY"),
            headers={"Accept": "application/json"},
            json=_data
        )
        if response.status_code != 201:
            print(f"{name} Failed {response.status_code}: {response.reason}")
    else:
        print({k: v for k, v in _data.items() if k not in {"h1e", "h2e", "cct2"}})
