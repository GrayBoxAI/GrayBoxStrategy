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
import uuid

from psm.components import EnterState

from gbstrategy.components.actions import RunExp
from gbstrategy.components.enterstates import End, HyperparamsSet, Init, StrategyHyperparamsSet
from gbstrategy.components.triggers import (ReceiveHyperparams, ReceiveRandomSearchHyperparams,
                                            ReceiveTime, ReceiveTrainingLoss,
                                           )

from gbstrategy.core import Strategy


class HalvingStage(EnterState):
    name = 'HalvingStage'


class SuccessiveHalvingStrategy(Strategy):
    # Capitalized components are built in in the base class `Strategy
    _psm_states = ['Init', 'StrategyHyperparamsSet', 'HyperparamsSet', 'HalvingStage', 'End']
    _psm_transitions = [{
        'source'    : 'Init',
        'dest'      : 'StrategyHyperparamsSet',
        'trigger'   : ReceiveRandomSearchHyperparams(),
        'conditions': lambda self: True,
        'before'    : 'set_num_exp',
    }, {
        'source'    : 'StrategyHyperparamsSet',
        'dest'      : 'HyperparamsSet',
        'trigger'   : ReceiveHyperparams(),
        'conditions': lambda self: True,
        'before'    : 'run_rand_search',
    }, {
        'source'    : 'HyperparamsSet',
        'dest'      : 'HalvingStage',
        'trigger'   : ReceiveTrainingLoss(),
        'conditions': 'rand_exp_finished',
        'before'    : 'enter_half_search',
    }, {
        'source'    : 'HalvingStage',
        'dest'      : 'HalvingStage',
        'trigger'   : ReceiveTrainingLoss(),
        'conditions': 'exp_finished',
        'before'    : 'run_half_search',
    }]

    def do_nothing(self, event):
        return []

    def set_num_exp(self, event):
        self._psm_data['state']['num_exp'] = self._psm_data['strategy']['num_exp']
        self._psm_data['state']['total_num_epochs'] = self._psm_data['strategy']['epoch']
        return []

    def run_rand_search(self, event):
        self._psm_data['state']['num_epochs'] = 2
        num_epochs = self._psm_data['state']['num_epochs']
        num_exp = self._psm_data['strategy']['num_exp']
        self._psm_data['state']['num_exp'] = num_exp

        lr_set = self._psm_data['hyperparams']['learning_rate']
        actions = []
        for i in range(num_exp):
            data = {
                'exp_id': uuid.uuid4(),
                'end_epoch' : num_epochs,
                'hyperparams':   {
                    'learning_rate': random.uniform(*lr_set),
                }
            }
            actions.append(RunExp(data=data))
        return actions

    def run_half_search(self, event):
        state_data = self._psm_data['state']
        state_data['num_exp'] //= 2

        # halve the num of exp
        top_exps = self.get_top_exps(state_data['num_exp'], state_data['total_num_epochs'])

        # run more till new epoch
        state_data['total_num_epochs'] += state_data['num_epochs']
        state_data['num_epochs'] *= 2
        actions = []
        for exp_id in top_exps:
            data = {
                'exp_id': exp_id,
                'end_epoch' : state_data['total_num_epochs'],
                'hyperparams':   {
                    'learning_rate': None,
                }
            }
            actions.append(RunExp(data=data))
        return actions

    def enter_half_search(self, event):
        state_data = self._psm_data['state']
        state_data['num_exp'] //= 2
        top_exps = self.get_top_exps(state_data['num_exp'], self._psm_data['strategy']['epoch'])

        state_data['num_epochs'] *= 2
        state_data['total_num_epochs'] = state_data['num_epochs'] + \
                                         self._psm_data['strategy']['epoch']

        actions = []
        for exp_id in top_exps:
            data = {
                'exp_id': exp_id,
                'end_epoch' : state_data['total_num_epochs'],
                'hyperparams':   {
                    'learning_rate': None,
                }
            }
            actions.append(RunExp(data=data))
        return actions

    def exp_finished(self, event):
        loss_log = self._psm_data['trainingloss']
        desired_epoch = self._psm_data['state']['total_num_epochs']
        desired_num = self._psm_data['state']['num_exp']
        loss_log = [l for l in loss_log if l['epoch'] == desired_epoch]
        cond = len(loss_log) >= desired_num
        return cond

    def rand_exp_finished(self, event):
        loss_log = self._psm_data['trainingloss']
        desired_epoch = self._psm_data['strategy']['epoch']
        desired_num = self._psm_data['state']['num_exp']
        loss_log = [l for l in loss_log if l['epoch'] == desired_epoch]
        cond = len(loss_log) >= desired_num
        return cond

    def get_top_exps(self, num, epoch):
        loss_log = self._psm_data['trainingloss']
        loss_log = [l for l in loss_log if l['epoch'] == epoch]
        exps_id = [(l['exp_id'], l['loss_value']) for l in loss_log]
        top_exps_id = sorted(exps_id, key=lambda x: x[1])[:num]
        top_exps_id = [_[0] for _ in top_exps_id]
        return top_exps_id
