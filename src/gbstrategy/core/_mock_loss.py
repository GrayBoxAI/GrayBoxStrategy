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

import random

import numpy


class LossFunc(object):
    dim = 0
    loss_name = ""

    @classmethod
    def epoch_loss(cls, epoch, hyperparams):
        final_loss = cls._final_loss(hyperparams)
        epoch_loss = cls._interpolation(final_loss, epoch)
        return epoch_loss

    @classmethod
    def _final_loss(cls, hyperparams):
        raise NotImplementedError

    @classmethod
    def _interpolation(cls, final_loss, epoch):
        raise NotImplementedError


class ExampleLoss1(LossFunc):
    loss_name = 'ExampleLoss1'
    dim = 1

    @classmethod
    def _final_loss(cls, hyperparams):
        learning_rate = hyperparams['hyperparams']['learning_rate']
        return 0.1 + 0.1*(numpy.log10(learning_rate)-0.004)**2

    @classmethod
    def _interpolation(cls, final_loss, epoch):
        outer_scale = max(0.7 + random.gauss(0., 0.02), 0.02)
        inner_scale = max(0.01 + random.gauss(0., 0.02), 0.1)
        return outer_scale*(inner_scale*epoch+1)**(-2)+final_loss+random.gauss(0., 0.005)
