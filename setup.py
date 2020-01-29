from setuptools import setup, find_packages
import versioneer

requirements = [
    "numpy>=0.18",
    "pandas>=0.25",
    "xarray>=0.14.1",
    "scipy>=1.3",
    "pint>=0.10.1",
    "asdf>=2.5",
    "bottleneck>=1.3",
]

#entry_points = {}
#entry_points['asdf_extensions'] = [
#    'weldx = weldx.io.asdf.extension:WelDXExtension'
#]

setup(
    name="mypackage",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Cagtay Fabry",
    author_email="Cagtay.Fabry@bam.de",
    packages=find_packages(),
    url="www.bam.de/weldx",
    license="BSD License",
    description="WelDX Python API",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: Only"
        "Programming Language :: Python :: 3.7"
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Natural Language :: English",
    ],
    install_requires=requirements,
    #include_package_data=True, # include non-py files listed in MANIFEST.in
    #entry_points=entry_points, # register ASDF Extension entry_points
)
