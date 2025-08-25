from setuptools import setup, find_packages
import os
from pathlib import Path

# 从 requirements.txt 读取依赖
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    with open(requirements_path, 'r') as f:
        requirements = f.read().splitlines()
    return [req for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="xchrom",
    version="1.0.1",
    author="Yuanyuan Miao",
    author_email="miaoyuanyuan2022@sinh.ac.cn",
    description="Cross-cell chromatin accessibility prediction",
    long_description=Path("README.md").read_text('utf-8'),
    long_description_content_type="text/markdown",
    url="https://github.com/Miaoyuanyuan777/XChrom",
python_requires='>=3.9',
    # 自动发现所有包
    packages=find_packages(),
    
    # 包含非Python文件（如数据文件）
    include_package_data=True,
    
    # 从requirements.txt获取依赖
    install_requires=read_requirements()
)