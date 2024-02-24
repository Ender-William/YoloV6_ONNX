from setuptools import setup, find_packages

setup(
    name='YoloORT',
    version='1.0',
    packages=find_packages(),
    include_package_data=True,
    description='YoloV6 Onnxruntime Inference Pack',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='William Kuang',
    author_email='dakuang2002@126.com',
    url='https://github.com/Ender-William',
    install_requires=[
        'onnxruntime-gpu >= 1.12.3',
        'numpy >= 1.24.2',
        'opencv-python >=4.1.1',
    ],
    classifiers=[
        'License :: OSI Approved :: Apache License', 
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)
