from setuptools import find_packages, setup

with open("../README.md") as readme_file:
    readme = readme_file.read()

setup(
    name="nslib",
    version="0.1.0",
    author="Gunnar Voet",
    author_email="gvoet@ucsd.edu",
    # url="https://github.com/gunnarvoet/northern_sill",
    license="GNU GPL v3",
    # Description
    description="Samoan Passage Northern Sill analysis",
    long_description=f"{readme}",
    long_description_content_type="text/markdown",
    # Requirements
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "gsw",
        "scipy",
        "xarray",
        "matplotlib",
        "seabird",
        "munch",
        "pandas",
        "IPython",
    ],
    extras_require={
        "cartopy": ["cartopy"],  # install these with: pip install gvpy[cartopy]
    },
    # Packaging
    packages=find_packages(include=["nslib", "nslib.*"]),
    zip_safe=False,
    platforms=["any"],  # or more specific, e.g. "win32", "cygwin", "osx"
    # Metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
