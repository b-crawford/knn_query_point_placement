from setuptools import setup, find_packages

VERSION = "0.0.1"
DESCRIPTION = (
    "Build a system to minimise the number of knn queries to find all data points"
)

# Setting up
setup(
    name="knn_query_point_placement",
    version=VERSION,
    author="Billy Crawford",
    author_email="billy_crawford@hotmail.co.uk",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=["matplotlib", "scipy", "numpy", "pulp", "pandas"],
    keywords=["python", "k nearest neighbours", "knn", "optimisation"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)
