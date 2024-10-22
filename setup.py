'''
Date: 2022-05-19 21:45:28
LastEditors: zhaoxuebin hzau_zxb@foxmail.com
LastEditTime: 2024-03-22 14:48:55
FilePath: /zhaoxuebin/easymodel/setup.py
Description: hzau_zxb@foxmail.com
'''
from setuptools import setup, find_packages


setup(
    name="guidance_ocr",
    version="0.1",
    author="zhaoxuebin",
    author_email="hzau_zxb@foxmail.com",
    description="guide vlm inference with ocr",
    url=" ", 
    install_requires = ["transformers>=4.45.0", 'torch>=2.3.0'],
    packages=find_packages()
)