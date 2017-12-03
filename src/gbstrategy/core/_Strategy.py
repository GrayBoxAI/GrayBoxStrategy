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
from functools import partial

from psm import PersistentStateMachine
from psm.components import Action, CounterAction, EnterState, Trigger


class StrategyMachineFactory(object):
    def __init__(self, strategy, logger, interface):
        self._psm = None
        self.strategy = strategy
        self.logger = logger
        self.interface = interface

    def generate_psm(self):
        if self.logger.empty():
            state = 'Init'
            EnterState.logInit(self.logger)
        else:
            state, trigger, data = PersistentStateMachine.srp(self.interface, self.logger)
            state = state.name
            self.strategy._psm_data = data
            # print(state, trigger, data)

        # register interface
        self.interface.register_strategy(self.strategy, self)

        tmp_transition = copy.deepcopy(self.strategy._psm_transitions)
        for idx, t in enumerate(self.strategy._psm_transitions):
            # register triggers
            trigger_func_name = self.strategy.helper_register_trigger(t['trigger'], idx, self.logger)
            tmp_transition[idx]['trigger'] = t['trigger'].name
            tmp_transition[idx]['prepare'] = trigger_func_name

            # register action issuers and add transition
            act_issuer_name = self.strategy.helper_register_action_issuer(t['before'], idx, self.interface, self.logger)
            tmp_transition[idx]['before'] = act_issuer_name

        logEnterState_withdata = partial(self.logger.logEnterState,
                                         self.strategy.get_state_data)
        self._psm = Machine(model=self.strategy,
                            states=self.strategy._psm_states,
                            after_state_change=logEnterState_withdata,
                            initial=state,
                            send_event=True,
                            transitions=tmp_transition,
                            # ignore_invalid_triggers=True,
                           )
        return self._psm


class Strategy(object):
    # Please do not set _psm_data when subclassing!
    _psm_transitions = []
    _psm_states = []
    _psm_data = {'state':{}}

    def get_state_data(self):
        return self._psm_data['state']

    def issue_actions(self, actions):
        for a in actions:
            a.issue(self.interface, self.logger)

    def helper_register_trigger(self, trigger, idx, logger):
        def get_trigger_func(trigger):
            def func(self, eventdata):
                trigger.store(eventdata.kwargs)
                trigger.log(logger)
                trigger.aggregate_data(self._psm_data)
            return func

        # Use partial to add trigger function to strategy instance as bound method
        func_trigger = partial(get_trigger_func(trigger), self)
        func_name = '_{}_{}'.format(trigger.__class__.__name__, idx)
        setattr(self, func_name, func_trigger)
        return func_name

    def helper_register_action_issuer(self, act_issuer_name, idx, interf, logger):
        def get_func_act():
            act_issuer = getattr(self, act_issuer_name)
            def func(newself, eventdata):
                actions = act_issuer(eventdata)
                for a in actions:
                    a.issue(interf, logger)
            return func

        func_act = partial(get_func_act(), self)
        act_issuer_name = act_issuer_name + '_{}'.format(idx)
        setattr(self, act_issuer_name, func_act)
        return act_issuer_name
