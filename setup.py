from setuptools import setup, find_packages

setup(
    name="ml_predictor",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'tensorflow',
        'scikit-learn',
        'PyYAML',
        'python-binance',
        'pydantic'
    ],
    extras_require={
        'optimized': ['numba>=0.53.0'],
        'full': [
            'numba>=0.53.0',
            'optuna',  # For hyperparameter optimization
            'shap',    # For model interpretability
            'plotly'   # For visualization
        ]
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="ML-based crypto trading predictor",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ml_predictor",
) 