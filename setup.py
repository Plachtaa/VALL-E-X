# coding=utf-8
import os
import sys
from pathlib import Path
from subprocess import DEVNULL, PIPE, run

from setuptools import find_packages, setup

project_root = Path(__file__).parent

# modified from https://github.com/lhotse-speech/lhotse/blob/master/setup.py

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
# NOTE: REMEMBER TO UPDATE THE FALLBACK VERSION IN valle/__init__.py WHEN RELEASING #
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
MAJOR_VERSION = 1
MINOR_VERSION = 0
PATCH_VERSION = 0
IS_DEV_VERSION = True  # False = public release, True = otherwise


if sys.version_info < (3,):
    # fmt: off
    print(
        "Python 2 has reached end-of-life and is no longer supported by valle."
    )
    # fmt: on
    sys.exit(-1)

if sys.version_info < (3, 7):
    print(
        "Python 3.6 has reached end-of-life on December 31st, 2021 "
        "and is no longer supported by valle."
    )
    sys.exit(-1)


def discover_valle_version() -> str:
    """
    Scans Valle source code to determine the current version.
    When development version is detected, it queries git for the commit hash
    to append it as a local version identifier.

    Ideally this function would have been imported from valle.version and
    re-used when valle is imported to set the version, but it introduces
    a circular dependency. To avoid this, we write the determined version
    into project_root / 'valle' / 'version.py' during setup and read it
    from there later. If it's not detected, the version will be 0.0.0.dev.
    """

    version = f"{MAJOR_VERSION}.{MINOR_VERSION}.{PATCH_VERSION}"
    if not IS_DEV_VERSION:
        # This is a PyPI public release -- return a clean version string.
        return version

    version = version + ".dev"

    # This is not a PyPI release -- try to read the git commit
    try:
        git_commit = (
            run(
                ["git", "rev-parse", "--short", "HEAD"],
                check=True,
                stdout=PIPE,
                stderr=DEVNULL,
            )
            .stdout.decode()
            .rstrip("\n")
            .strip()
        )
        dirty_commit = (
            len(
                run(
                    ["git", "diff", "--shortstat"],
                    check=True,
                    stdout=PIPE,
                    stderr=DEVNULL,
                )
                .stdout.decode()
                .rstrip("\n")
                .strip()
            )
            > 0
        )
        git_commit = (
            git_commit + ".dirty" if dirty_commit else git_commit + ".clean"
        )
        source_version = f"+git.{git_commit}"
    except Exception:
        source_version = ".unknownsource"
    # See the format:
    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#local-version-identifiers
    version = version + source_version

    return version


def mark_valle_version(version: str) -> None:
    (project_root / "valle" / "version.py").write_text(
        f'__version__ = "{version}"'
    )


VALLE_VERSION = discover_valle_version()
mark_valle_version(VALLE_VERSION)


install_requires = [
    "encodec",
    "phonemizer",
]

try:
    # If the user already installed PyTorch, make sure he has torchaudio too.
    # Otherwise, we'll just install the latest versions from PyPI for the user.
    import torch

    try:
        import torchaudio
    except ImportError:
        raise ValueError(
            "We detected that you have already installed PyTorch, but haven't installed torchaudio. "
            "Unfortunately we can't detect the compatible torchaudio version for you; "
            "you will have to install it manually. "
            "For instructions, please refer either to https://pytorch.org/get-started/locally/ "
            "or https://github.com/pytorch/audio#dependencies"
        )
except ImportError:
    install_requires.extend(["torch", "torchaudio"])

docs_require = (
    (project_root / "docs" / "requirements.txt").read_text().splitlines()
)
tests_require = [
    # "pytest==7.1.3",
    # "pytest-forked==1.4.0",
    # "pytest-xdist==2.5.0",
    # "pytest-cov==4.0.0",
]
workflow_requires = [""]
dev_requires = sorted(
    docs_require
    + tests_require
    + workflow_requires
    + ["jupyterlab", "matplotlib"]
)
all_requires = sorted(dev_requires)

if os.environ.get("READTHEDOCS", False):
    # When building documentation, omit torchaudio installation and mock it instead.
    # This works around the inability to install libsoundfile1 in read-the-docs env,
    # which caused the documentation builds to silently crash.
    install_requires = [
        req
        for req in install_requires
        if not any(req.startswith(dep) for dep in ["torchaudio", "SoundFile"])
    ]

setup(
    name="valle",
    version=VALLE_VERSION,
    python_requires=">=3.7.0",
    description="Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers",
    author="The Valle Development Team",
    author_email="lifeiteng0422@163.com",
    long_description=(project_root / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license="Apache-2.0 License",
    packages=find_packages(exclude=["test", "test.*"]),
    include_package_data=True,
    entry_points={},
    install_requires=install_requires,
    extras_require={
        "docs": docs_require,
        "tests": tests_require,
        "dev": dev_requires,
        "all": all_requires,
    },
    classifiers=[
        "Development Status :: 1 - Beta",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
)
