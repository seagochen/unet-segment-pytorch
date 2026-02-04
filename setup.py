"""
UNet Segmentation Package Setup
"""

from pathlib import Path
from setuptools import setup, find_packages

# Read README for long description
readme_path = Path(__file__).parent / 'README.md'
long_description = readme_path.read_text(encoding='utf-8') if readme_path.exists() else ''

# Read requirements
requirements_path = Path(__file__).parent / 'requirements.txt'
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = [
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'numpy>=1.19.0',
        'pillow>=8.0.0',
        'pyyaml>=5.4.0',
        'tqdm>=4.60.0',
        'matplotlib>=3.3.0',
    ]

setup(
    name='unet-segment',
    version='0.1.0',
    author='',
    author_email='',
    description='UNet for medical image segmentation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='',
    packages=find_packages(exclude=['scripts', 'toolkits', 'configs', 'runs']),
    install_requires=requirements,
    extras_require={
        'augmentation': ['albumentations>=1.0.0'],
        'dev': ['pytest', 'black', 'isort', 'flake8'],
    },
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],
    entry_points={
        'console_scripts': [
            'unet-train=scripts.train:main',
            'unet-predict=scripts.predict:main',
        ],
    },
)
