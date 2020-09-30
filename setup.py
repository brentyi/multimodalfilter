from setuptools import find_packages, setup

setup(
    name="crossmodal",
    version="0.0",
    description="Crossmodal filtering",
    url="http://github.com/brentyi/multimodalfilter",
    author="brentyi",
    author_email="brentyi@berkeley.edu",
    license="BSD",
    packages=["crossmodal"],
    install_requires=[
        "fannypack",
        "torchfilter @ https://github.com/stanford-iprl-lab/torchfilter/tarball/master",
    ],
)
