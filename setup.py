from setuptools import setup, find_packages

setup(
    name="hypercube-codehealer",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "redis>=4.5.0",
        "python-dotenv>=0.19.0"
    ],
    author="Hypercube AI",
    description="Autonomous code healing and vulnerability prevention system",
    license="AGPLv3",
    python_requires=">=3.9",
)
