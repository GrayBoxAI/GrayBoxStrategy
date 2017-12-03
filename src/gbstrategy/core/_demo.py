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


class DemoDriver(object):
    def __init__(self, interface, lossfunc):
        self.interface = interface
        self.lossfunc = lossfunc
        self.interface.register_driver(self)

        self._running_exps = []

    def next(self):
        not_finished_exps = [i for i in self._running_exps if not i.is_finished()]
        if not_finished_exps:
            i = random.choice(not_finished_exps)
            i.upload_training_loss(self.interface)

    def run_exp(self, exp_id, end_epoch, hyperparams):
        found_exp = self._grab_exp(exp_id)
        if found_exp:
            if end_epoch < found_exp.end_epoch:
                msg = 'The new end_epoch is even smaller than the previous setting for exp <{}>'
                msg.format(exp_id)
                raise ValueError(msg)
            else:
                found_exp.end_epoch = end_epoch
        else:
            self._register_exp(exp_id, end_epoch, hyperparams)
    
    def _register_exp(self, exp_id, end_epoch, hyperparams):
        exp = DemoExp(exp_id, self.lossfunc, end_epoch, hyperparams)
        self._running_exps.append(exp)

    def _grab_exp(self, exp_id):
        _running_exps_id = [exp.exp_id for exp in self._running_exps]
        if exp_id not in _running_exps_id:
            return None
        else:
            idx = _running_exps_id.index(exp_id)
        return self._running_exps[idx]


class DemoExp(object):
    def __init__(self, exp_id, lossfunc, end_epoch, hyperparams):
        self.exp_id = exp_id
        self.lossfunc = lossfunc
        self.end_epoch = end_epoch
        self.hyperparams = hyperparams

        self._curr_epoch = 0

    def is_finished(self):
        return self._curr_epoch >= self.end_epoch

    def upload_training_loss(self, interface):
        if self.is_finished():
            raise ValueError('This exp <{}> is finished'.format(self.exp_id))
        self._curr_epoch += 1
        loss = self.lossfunc.epoch_loss(self._curr_epoch, self.hyperparams)
        interface.upload_training_loss(self.exp_id, self._curr_epoch, self.lossfunc.loss_name, loss)


class LossFunc(object):
    dim = 0
    loss_name = "someLossName"

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
