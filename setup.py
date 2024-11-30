from setuptools import setup, find_packages

setup(
    name="tennis_analysis",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Dependencias
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "torch>=1.9.0",
        "ultralytics>=8.0.0",  # YOLO
        "yt-dlp>=2023.3.4",    # Video download
        "scipy>=1.7.0",        # Interpolación
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
    ],
    
    # Scripts ejecutables
    entry_points={
        "console_scripts": [
            "tennis-download=tennis_analysis.examples.download_video:main",
            "tennis-preprocess=tennis_analysis.examples.preprocess_match:main",
            "tennis-process=tennis_analysis.examples.process_match:main",
        ],
    },
    
    # Metadatos
    author="Tu Nombre",
    author_email="tu@email.com",
    description="Framework para análisis de videos de tenis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords=["tennis", "computer-vision", "tracking", "YOLO", "sports-analysis"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    
    # Requerimientos de Python
    python_requires=">=3.8",
)