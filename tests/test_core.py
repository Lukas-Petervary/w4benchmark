import unittest
from w4benchmark import W4

class TestCore(unittest.TestCase):
    def setUp(self): W4.init()

    def test_self_mutation(self):
        with self.assertRaises(Exception):
            W4["acetaldehyde"].charge += 1.0

    def test_dereference_mutation(self):
        with self.assertRaises(Exception):
            data = W4["acetaldehyde"]
            data.charge += 1.0

    def test_nested_dereference(self):
        with self.assertRaises(Exception):
            data = W4
            charge = data["acetaldehyde"]
            charge.charge += 1.0

    def test_copied_immutability(self):
        with self.assertRaises(Exception):
            data = W4.copy()
            data["acetaldehyde"].charge += 1.0

    def test_dereferenced_data(self):
        try:
            data = W4["acetaldehyde"].charge
            data += 1.0
            pass
        except Exception as e:
            self.fail(f"Should not have raised exception: {e}")

    def test_print_data(self):
        def print_data(key, value):
            print(f"'{key}': {value}")
        for key, value in W4.items():
            print_data(key, value)

        print(W4)