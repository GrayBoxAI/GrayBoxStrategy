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

import datetime

from psm.components import Trigger

def clean_time(time):
    if not time.__class__ is datetime:
        raise ValueError('Time should be in python datetime format')
    return time


class ReceiveTrainingLoss(Trigger):
    _psm_data_prefix = 'trainingloss'
    fields = {
        'exp_id'    : str,
        'epoch'     : int,
        'loss_name' : str,
        'loss_value': float,
    }


class ReceiveTime(Trigger):
    _psm_data_prefix = 'time'
    fields = {
        'time': clean_time,
    }


class ReceiveRandomSearchHyperparams(Trigger):
    _psm_data_prefix = 'strategy'
    fields = {
        'num_exp'   : int,
        'epoch'     : int,
    }


class ReceiveHyperparams(Trigger):
    _psm_data_prefix = 'hyperparams'
    fields = {}


class FailureRecovery(Trigger):
    fields = {}
