from setuptools import setup, find_packages

setup(
    name="crossmodal",
    version="0.0",
    description="Crossmodal filtering",
    url="http://github.com/brentyi/crossmodal_filtering2",
    author="brentyi",
    author_email="brentyi@berkeley.edu",
    license="BSD",
    packages=["crossmodal"],
    install_requires=[
        "fannypack",
        "diffbayes @ https://github.com/brentyi/diffbayes/tarball/master",
    ],
)
