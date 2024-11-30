from setuptools import setup, find_packages

setup(
    name="tennis_analysis",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Dependencias más flexibles
    install_requires=[
        "numpy",
        "opencv-python",
        "yt-dlp",
        "scipy",
        "pandas",
        "matplotlib",
        "seaborn",
        "tqdm",
        "ultralytics"  # Este incluirá PyTorch automáticamente
    ],
    
    python_requires=">=3.8",
)