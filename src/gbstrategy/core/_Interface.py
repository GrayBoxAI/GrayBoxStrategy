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

import copy


class Interface(object):
    "Interface base class for abstracting away the communication layer of the strategy"
    def __init__(self):
        self.driver = None
        self.strategy = None
        self.factory = None

    def register_strategy(self, strategy, factory):
        self.strategy = strategy
        self.factory = factory

    def register_driver(self, driver):
        self.driver = driver

    def run_exp(self, data):
        dic = copy.deepcopy(data)
        end_epoch = dic.pop('end_epoch')
        exp_id = dic.pop('exp_id')

        hyperparams = dic
        self.driver.run_exp(exp_id, end_epoch, hyperparams)

    def kill_exp(self, data):
        raise NotImplementedError

    def next_time_point(self):
        self.factory.generate_psm()
        self.driver.next()

    def upload_training_loss(self, exp_id, epoch, loss_name, loss_value):
        data = {
            'exp_id': exp_id,
            'epoch' : epoch,
            'loss_name' : loss_name,
            'loss_value': loss_value,
        }
        self.strategy.ReceiveTrainingLoss(**data)
