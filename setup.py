#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import os
from setuptools import setup, find_packages


# This reads the __version__ variable from openfermion/_version.py
exec(open('src/gbstrategy/_version.py').read())

# Read in requirements.txt
requirements = open('requirements.txt').readlines()
requirements = [r.strip() for r in requirements]

setup(
    name='GrayBoxStrategy',
    version=__version__,
    author="The GrayBoxStrategy Developers"
           "(Chengyu Dai @ In8, Inc and University of Michigan,"
           "Brian Bullins @ In8, Inc and Princeton University)",
    author_email='jdaaph@gmail.com',
    url='http://www.graybox.ai',
    description=('GrayBox Deep Learning Hyperparameter Tuning Strategies Package'),
    install_requires=requirements,
    license='Apache 2',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=False,
)