from setuptools import setup, find_packages

setup(
    name="spaced-repetition-scheduler",
    version="0.1.0",
    description=(
        "Personalized spaced-repetition scheduler powered by an LSTM "
        "trained with Reptile-style meta-learning"
    ),
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "flask>=3.0.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0"],
    },
)
