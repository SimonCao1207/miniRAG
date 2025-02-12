import os

from setuptools import find_packages, setup


def get_version():
    version = {}
    init_path = os.path.join(os.path.dirname(__file__), "src", "miniRAG", "__init__.py")
    with open(init_path, "r", encoding="utf-8") as f:
        exec(f.read(), version)
    return version["__version__"]


setup(
    name="miniRAG",
    version=get_version(),
    package_dir={"": "src"},
    packages=find_packages(where="src"),  # Finds all packages inside src/
    install_requires=[],
)
