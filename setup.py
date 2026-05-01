"""
setup.py
────────
Package setup for RBC Infection Detection project.

Author  : C. Vaishnavi (22R91A7325)
Project : AI-Enhanced Microscopic Image Classification for RBC Infection
College : TKREC, Hyderabad
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name                          = "rbc-infection-detection",
    version                       = "1.0.0",
    author                        = "C. Vaishnavi",
    author_email                  = "vaishnavi@tkrec.ac.in",
    description                   = "AI-Enhanced Microscopic Image Classification for RBC Infection",
    long_description              = long_description,
    long_description_content_type = "text/markdown",
    url                           = "https://github.com/yourusername/rbc-infection-detection",
    packages                      = find_packages(),
    python_requires               = ">=3.9",
    install_requires              = requirements,
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    entry_points = {
        "console_scripts": [
            "rbc-train   = train:main",
            "rbc-predict = predict:main",
        ],
    },
)
