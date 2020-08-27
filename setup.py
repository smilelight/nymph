from distutils.core import setup
import setuptools

with open('./version.txt', encoding='utf8') as f:
    version = f.read().strip()

with open('./README.md', 'r', encoding='utf8') as f:
    long_description = f.read()

with open('./requirements.txt', 'r', encoding='utf8') as f:
    install_requires = list(map(lambda x: x.strip(), f.readlines()))
print(setuptools.find_packages())
setup(
    name='nymph',
    version=version,
    description="General multi-feature classification library based on pytorch",
    author='lightsmile',
    author_email='iamlightsmile@gmail.com',
    url='https://github.com/smilelight/nymph',
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='Apache-2.0',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries'
    ],
)