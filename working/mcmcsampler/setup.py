from setuptools import setup, find_packages

setup(
    name="mysampler",
    description="Simple discrete-state MCMC sampler",
    author="Pietro Campana",
    author_email="campana.pietro@gmail.com",
    license="MIT",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=["numpy", "numba"],
)
