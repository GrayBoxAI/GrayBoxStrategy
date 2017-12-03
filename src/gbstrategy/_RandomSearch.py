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
from psm import PersistentStateMachine

from gbstrategy.components.actions import RunExp
from gbstrategy.components.triggers import ReceiveHyperparams, ReceiveRandomSearchHyperparams
from gbstrategy.core import Strategy



class RandomSearchStrategy(Strategy):
    # Capitalized components are built in in the base class `Strategy
    _psm_states = ['Init', 'StrategyHyperparamsSet', 'HyperparamsSet', 'End']
    _psm_transitions = [{
        'source'    : 'Init',
        'dest'      : 'StrategyHyperparamsSet',
        'trigger'   : ReceiveRandomSearchHyperparams(),
        'conditions': lambda self: True,
        'before'    : 'do_nothing',
    }, {
        'source'    : 'StrategyHyperparamsSet',
        'dest'      : 'HyperparamsSet',
        'trigger'   : ReceiveHyperparams(),
        'conditions': lambda self: True,
        'before'    : 'run_rand_search',
    }]

    def do_nothing(self, event):
        return []

    def run_rand_search(self, event):
        data = {
            'exp_id': 'whatever_hashed',
            'end_epoch' : 1,
            'hyperparams':   {
                'lr'    : 0.01,
                'lambda': 0.002,
            }
        }
        actions = [RunExp(data=data)]
        return actions
