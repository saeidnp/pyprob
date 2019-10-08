import torch
import sys
import opcode
import random
import time
from termcolor import colored

from .distributions import Normal, Categorical, Uniform, TruncatedNormal
from .trace import Variable, Trace
from . import util, TraceMode, PriorInflation, InferenceEngine, ImportanceWeighting
from .nn import InferenceNetworkLSTM


_trace_mode = TraceMode.PRIOR
_inference_engine = InferenceEngine.IMPORTANCE_SAMPLING
_prior_inflation = PriorInflation.DISABLED
_likelihood_importance = 1.
_importance_weighting = ImportanceWeighting.IW0
_current_trace = None
_current_trace_root_function_name = None
_current_trace_inference_network = None
_current_trace_inference_network_proposal_min_train_iterations = None
_current_trace_previous_variable = None
_current_trace_replaced_variable_proposal_distributions = {}
_current_trace_observed_variables = None
_current_trace_proposals = None
_current_trace_execution_start = None
_metropolis_hastings_trace = None
_metropolis_hastings_site_address = None
_metropolis_hastings_site_transition_log_prob = 0
_address_dictionary = None
_rejection_sampling_stack = []
_current_partial_trace = None
_target_rejection_address = None


class RejectionSamplingStack:
    def __init__(self):
        '''
        Stores a stack of tuples (rejection sampling variabe, LSTM hidden state) if a network is present and has a hidden state,
        otherwise, the (rejection sampling variabe, None)
        '''
        self._stack = []
    
    def push(self, variable):
        hidden_state = None
        if _current_trace_inference_network is not None and isinstance(_current_trace_inference_network, InferenceNetworkLSTM):
            hidden_state = _current_trace_inference_network._infer_lstm_state
        self._stack.append([variable, hidden_state])

    def updateTopVariable(self, variable):
        self._stack[-1][0] = variable

    def pop(self):
        self._stack.pop()

    @property
    def top_variable(self):
        return self._stack[-1][0]

    @property
    def top_hidden(self):
        return self._stack[-1][1]

    def size(self):
        return len(self._stack)

    def isempty(self):
        return self.size() == 0


class PartialTrace:
    def __init__(self, trace, target_address):
        self.trace = trace
        self.target_address = target_address
        self.index = 0


class RejectionEndException(Exception):
    def __init__(self, length):
        self.length = length


# _extract_address and _extract_target_of_assignment code by Tobias Kohn (kohnt@tobiaskohn.ch)
def _extract_address(root_function_name, user_specified_name, append_rejectoin=True):
    # Retun an address in the format:
    # 'instruction pointer' __ 'qualified function name'
    if user_specified_name is not None and user_specified_name.startswith('__'):
        user_specified_name = None # TODO: temporary for mini sherpa experimetns.
    frame = sys._getframe(2)
    ip = frame.f_lasti
    names = []
    var_name = _extract_target_of_assignment()
    if var_name is None:
        names.append('?')
    else:
        names.append(var_name)
    while frame is not None:
        n = frame.f_code.co_name
        if n.startswith('<') and not n == '<listcomp>':
            break
        names.append(n)
        if n == root_function_name:
            break
        frame = frame.f_back
    address_base_noname = '{}__{}'.format(ip, '__'.join(reversed(names)))
    if append_rejectoin:
        if not _rejection_sampling_stack.isempty():
            address_base_noname = '{}_RS_{}'.format(address_base_noname, _rejection_sampling_stack.top_variable.address)
    return '{}__{}'.format(address_base_noname, user_specified_name)


def _extract_target_of_assignment():
    frame = sys._getframe(3)
    code = frame.f_code
    next_instruction = code.co_code[frame.f_lasti+2]
    instruction_arg = code.co_code[frame.f_lasti+3]
    instruction_name = opcode.opname[next_instruction]
    if instruction_name == 'STORE_FAST':
        return code.co_varnames[instruction_arg]
    elif instruction_name in ['STORE_NAME', 'STORE_GLOBAL']:
        return code.co_names[instruction_arg]
    elif instruction_name in ['LOAD_FAST', 'LOAD_NAME', 'LOAD_GLOBAL'] and \
            opcode.opname[code.co_code[frame.f_lasti+4]] in ['LOAD_CONST', 'LOAD_FAST'] and \
            opcode.opname[code.co_code[frame.f_lasti+6]] == 'STORE_SUBSCR':
        base_name = (code.co_varnames if instruction_name == 'LOAD_FAST' else code.co_names)[instruction_arg]
        second_instruction = opcode.opname[code.co_code[frame.f_lasti+4]]
        second_arg = code.co_code[frame.f_lasti+5]
        if second_instruction == 'LOAD_CONST':
            value = code.co_consts[second_arg]
        elif second_instruction == 'LOAD_FAST':
            var_name = code.co_varnames[second_arg]
            value = frame.f_locals[var_name]
        else:
            value = None
        if type(value) is int:
            index_name = str(value)
            return base_name + '[' + index_name + ']'
        else:
            return None
    elif instruction_name == 'RETURN_VALUE':
        return 'return'
    else:
        return None


def _inflate(distribution):
    if _prior_inflation == PriorInflation.ENABLED:
        if isinstance(distribution, Categorical):
            return Categorical(util.to_tensor(torch.zeros(distribution.num_categories).fill_(1./distribution.num_categories)))
        elif isinstance(distribution, Normal):
            return Normal(distribution.mean, distribution.stddev * 3)
    return None


def tag(value, name=None, address=None):
    global _current_trace
    if address is None:
        address_base = _extract_address(_current_trace_root_function_name, name) + '__None'
    else:
        address_base = address + '__None'
    if _address_dictionary is not None:
        address_base = _address_dictionary.address_to_id(address_base)
    instance = _current_trace.last_instance(address_base) + 1
    address = address_base + '__' + str(instance)

    value = util.to_tensor(value)

    variable = Variable(distribution=None, value=value, address_base=address_base, address=address, instance=instance, log_prob=0., tagged=True, name=name)
    _current_trace.add(variable)


def _get_variable_from_partial_trace(address):
    # Returns the next variable from the partial trace
    # Verifies the variable address to be the same as the given address
    # Updates the partial trace (advances the index and sets it to null if reached the target address)
    global _current_partial_trace
    variable = _current_partial_trace.trace.variables[_current_partial_trace.index]
    assert variable.address == address


    # Update partial trace
    if variable.address == _current_partial_trace.target_address:
        _current_partial_trace = None
    else:
        _current_partial_trace.index += 1

    return variable


def observe(distribution, value=None, name=None, address=None):
    global _current_trace
    global RejectionSamplingStack

    rejection_address = None
    if not _rejection_sampling_stack.isempty():
        rejection_address = _rejection_sampling_stack.top_variable.address

    if address is None:
        address_base = _extract_address(_current_trace_root_function_name, name) + '__' + distribution._address_suffix
    else:
        address_base = address + '__' + distribution._address_suffix
    if _address_dictionary is not None:
        address_base = _address_dictionary.address_to_id(address_base)
    instance = _current_trace.last_instance(address_base) + 1
    address = address_base + '__' + str(instance)

    if _current_partial_trace is not None:
        variable = _get_variable_from_partial_trace(address)
    else:
        if name in _current_trace_observed_variables:
            # Override observed value
            value = _current_trace_observed_variables[name]
        elif value is not None:
            value = util.to_tensor(value)
        elif distribution is not None:
            value = distribution.sample()

        log_prob = _likelihood_importance * distribution.log_prob(value, sum=True)
        if _inference_engine == InferenceEngine.IMPORTANCE_SAMPLING or _inference_engine == InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK:
            log_importance_weight = float(log_prob)
        else:
            log_importance_weight = None  # TODO: Check the reason/behavior for this
        variable = Variable(distribution=distribution, value=value, address_base=address_base, address=address, instance=instance, log_prob=log_prob, log_importance_weight=log_importance_weight, observed=True, name=name)
        variable.rejection_address = rejection_address
    _current_trace.add(variable)
    

def sample(distribution, control=True, replace=False, name=None, address=None):
    global _current_trace
    global _current_trace_previous_variable
    global _current_trace_replaced_variable_proposal_distributions
    global _rejection_sampling_stack
    global _current_partial_trace

    replace = False
    rejection_address = None
    # If there is not active rejection sampling, the variable is not "replaced"
    if (not _rejection_sampling_stack.isempty()):
        #TODO: problematic conditions. _importance_weighting != ImportanceWeighting.IW2 needs to be fixed. Not sure about _trace_mode != TraceMode.POSTERIOR
        replace = True
        rejection_address = _rejection_sampling_stack.top_variable.address
        if _rejection_sampling_stack.top_variable.control == False:
            control = False

    # Only replace if controlled
    if not control:
        replace = False # TODO: why?

    if _inference_engine == InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS or _inference_engine == InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS:
        control = True
        replace = False

    if address is None:
        address_base = _extract_address(_current_trace_root_function_name, name) + '__' + distribution._address_suffix
    else:
        address_base = address + '__' + distribution._address_suffix
    if _address_dictionary is not None:
        address_base = _address_dictionary.address_to_id(address_base)

    instance = _current_trace.last_instance(address_base) + 1

    if _current_partial_trace is not None:
        address = address_base + '__' + str(instance)
        variable = _get_variable_from_partial_trace(address)
    else:
        # Partial trace not given
        if name in _current_trace_observed_variables:
            # Variable is observed
            address = address_base + '__' + str(instance)
            value = _current_trace_observed_variables[name]
            log_prob = _likelihood_importance * distribution.log_prob(value, sum=True)
            if _inference_engine == InferenceEngine.IMPORTANCE_SAMPLING or _inference_engine == InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK:
                log_importance_weight = float(log_prob)
            else:
                log_importance_weight = None  # TODO: Check the reason/behavior for this
            variable = Variable(distribution=distribution, value=value, address_base=address_base, address=address, instance=instance, log_prob=log_prob, log_importance_weight=log_importance_weight, observed=True, name=name)
        else:
            # Variable is sampled
            reused = False
            observed = False
            if _trace_mode == TraceMode.POSTERIOR:
                if _inference_engine == InferenceEngine.IMPORTANCE_SAMPLING:
                    address = address_base + '__' + str(instance)
                    inflated_distribution = _inflate(distribution)
                    if inflated_distribution is None:
                        if name in _current_trace_proposals and control and _importance_weighting != ImportanceWeighting.IW0:
                            proposal_distribution = _current_trace_proposals[name]
                        else:
                            proposal_distribution = distribution
                        value = proposal_distribution.sample()
                        log_prob = distribution.log_prob(value, sum=True)
                        proposal_log_prob = proposal_distribution.log_prob(value, sum=True)
                        log_importance_weight = float(log_prob) - float(proposal_log_prob)
                    else:
                        value = inflated_distribution.sample()
                        log_prob = distribution.log_prob(value, sum=True)
                        log_importance_weight = float(log_prob) - float(inflated_distribution.log_prob(value, sum=True))  # To account for prior inflation
                elif _inference_engine == InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK:
                    address = address_base + '__' + ('replaced' if replace else str(instance))  # Address seen by inference network
                    if control:
                        variable = Variable(distribution=distribution, value=None, address_base=address_base, address=address, instance=instance, log_prob=0., control=control, replace=replace, name=name, observed=observed, reused=reused)
                        update_previous_variable = False
                        if replace:
                            # TODO: address not in _current_trace_replaced_variable_proposal_distributions might not be sufficient to discover a new replace loop instance. Implement better.
                            if address not in _current_trace_replaced_variable_proposal_distributions:
                                _current_trace_replaced_variable_proposal_distributions[address] = _current_trace_inference_network._infer_step(variable, prev_variable=_current_trace_previous_variable, proposal_min_train_iterations=_current_trace_inference_network_proposal_min_train_iterations)
                                update_previous_variable = True
                            if _importance_weighting == ImportanceWeighting.IW0: # use prior as proposal for all replace=True addresses
                                proposal_distribution = distribution
                            else: # _importance_weighting == ImportanceWeighting.IW1
                                proposal_distribution = _current_trace_replaced_variable_proposal_distributions[address]
                        else:
                            proposal_distribution = _current_trace_inference_network._infer_step(variable, prev_variable=_current_trace_previous_variable, proposal_min_train_iterations=_current_trace_inference_network_proposal_min_train_iterations)
                            update_previous_variable = True
                        value = proposal_distribution.sample()
                        if value.dim() > 0:
                            value = value[0]
                        log_prob = distribution.log_prob(value, sum=True)
                        proposal_log_prob = proposal_distribution.log_prob(value, sum=True)
                        if util.has_nan_or_inf(log_prob):
                            print(colored('Warning: prior log_prob has NaN, inf, or -inf.', 'red', attrs=['bold']))
                            print('distribution', distribution)
                            print('value', value)
                            print('log_prob', log_prob)
                        if util.has_nan_or_inf(proposal_log_prob):
                            print(colored('Warning: proposal log_prob has NaN, inf, or -inf.', 'red', attrs=['bold']))
                            print('distribution', proposal_distribution)
                            print('value', value)
                            print('log_prob', proposal_log_prob)
                        log_importance_weight = float(log_prob) - float(proposal_log_prob)
                        if update_previous_variable:
                            variable = Variable(distribution=distribution, value=value, address_base=address_base, address=address, instance=instance, log_prob=log_prob, log_importance_weight=log_importance_weight, control=control, replace=replace, name=name, observed=observed, reused=reused)
                            _current_trace_previous_variable = variable
                            # print('prev_var address {}'.format(variable.address))
                    else:
                        value = distribution.sample()
                        log_prob = distribution.log_prob(value, sum=True)
                        log_importance_weight = None
                    address = address_base + '__' + str(instance)  # Address seen by everyone except the inference network
                else:  # _inference_engine == InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS or _inference_engine == InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS
                    address = address_base + '__' + str(instance)
                    log_importance_weight = None
                    if _metropolis_hastings_trace is None:
                        value = distribution.sample()
                        log_prob = distribution.log_prob(value, sum=True)
                    else:
                        if address == _metropolis_hastings_site_address:
                            global _metropolis_hastings_site_transition_log_prob
                            _metropolis_hastings_site_transition_log_prob = util.to_tensor(0.)
                            if _inference_engine == InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS:
                                if isinstance(distribution, Normal):
                                    proposal_kernel_func = lambda x: Normal(x, distribution.stddev)
                                elif isinstance(distribution, Uniform):
                                    proposal_kernel_func = lambda x: TruncatedNormal(x, 0.1*(distribution.high - distribution.low), low=distribution.low, high=distribution.high)
                                else:
                                    proposal_kernel_func = None

                                if proposal_kernel_func is not None:
                                    _metropolis_hastings_site_value = _metropolis_hastings_trace.variables_dict_address[address].value
                                    _metropolis_hastings_site_log_prob = _metropolis_hastings_trace.variables_dict_address[address].log_prob
                                    proposal_kernel_forward = proposal_kernel_func(_metropolis_hastings_site_value)
                                    alpha = 0.5
                                    if random.random() < alpha:
                                        value = proposal_kernel_forward.sample()
                                    else:
                                        value = distribution.sample()
                                    log_prob = distribution.log_prob(value, sum=True)
                                    proposal_kernel_reverse = proposal_kernel_func(value)

                                    _metropolis_hastings_site_transition_log_prob = torch.log(alpha * torch.exp(proposal_kernel_reverse.log_prob(_metropolis_hastings_site_value, sum=True)) + (1 - alpha) * torch.exp(_metropolis_hastings_site_log_prob)) + log_prob
                                    _metropolis_hastings_site_transition_log_prob -= torch.log(alpha * torch.exp(proposal_kernel_forward.log_prob(value, sum=True)) + (1 - alpha) * torch.exp(log_prob)) + _metropolis_hastings_site_log_prob
                                else:
                                    value = distribution.sample()
                                    log_prob = distribution.log_prob(value, sum=True)
                            else:
                                value = distribution.sample()
                                log_prob = distribution.log_prob(value, sum=True)
                            reused = False
                        elif address not in _metropolis_hastings_trace.variables_dict_address:
                            value = distribution.sample()
                            log_prob = distribution.log_prob(value, sum=True)
                            reused = False
                        else:
                            value = _metropolis_hastings_trace.variables_dict_address[address].value
                            reused = True
                            try:  # Takes care of issues such as changed distribution parameters (e.g., batch size) that prevent a rescoring of a reused value under this distribution.
                                log_prob = distribution.log_prob(value, sum=True)
                            except:
                                value = distribution.sample()
                                log_prob = distribution.log_prob(value, sum=True)
                                reused = False

            else:  # _trace_mode == TraceMode.PRIOR or _trace_mode == TraceMode.PRIOR_FOR_INFERENCE_NETWORK:
                if _trace_mode == TraceMode.PRIOR:
                    address = address_base + '__' + str(instance)
                elif _trace_mode == TraceMode.PRIOR_FOR_INFERENCE_NETWORK:
                    address = address_base + '__' + ('replaced' if replace else str(instance))
                inflated_distribution = _inflate(distribution)
                if inflated_distribution is None:
                    value = distribution.sample()
                    log_prob = distribution.log_prob(value, sum=True)
                    log_importance_weight = None
                else:
                    value = inflated_distribution.sample()
                    log_prob = distribution.log_prob(value, sum=True)
                    log_importance_weight = float(log_prob) - float(inflated_distribution.log_prob(value, sum=True))  # To account for prior inflation

            if _trace_mode == TraceMode.POSTERIOR and _importance_weighting == ImportanceWeighting.IW2:
                # IW2 should take all the weights into account => no replaced variable
                replace = False
            variable = Variable(distribution=distribution, value=value, address_base=address_base, address=address, instance=instance, log_prob=log_prob, log_importance_weight=log_importance_weight, control=control, replace=replace, name=name, observed=observed, reused=reused)
            variable.rejection_address = rejection_address
    _current_trace.add(variable)
    return variable.value


def rejection_sampling(control=True, name=None, address=None):
    global _current_trace
    global _current_trace_previous_variable
    global _current_trace_replaced_variable_proposal_distributions
    global _rejection_sampling_stack

    rejection_sampling_suffix = 'rejsmp'

    if address is None:
        address_base = _extract_address(_current_trace_root_function_name, name, append_rejectoin=False) + '__' + rejection_sampling_suffix
        # Prblematic for nested rejection sampling!
    else:
        address_base = address + '__' + rejection_sampling_suffix
    if _address_dictionary is not None:
        address_base = _address_dictionary.address_to_id(address_base)
    if (not _rejection_sampling_stack.isempty()) and _rejection_sampling_stack.top_variable.address_base == address_base:
        # It is not a new rejection sampling. Rather, it's retrying sampling
        # We use the same instance number in such cases
        instance = _current_trace.last_instance(address_base)
        value = _current_trace.variables_dict_address_base[address_base].value + 1
    else:
        instance = _current_trace.last_instance(address_base) + 1
        value = util.to_tensor(1)

    address = address_base + '__' + str(instance)

    if _current_partial_trace is not None:
        variable = _get_variable_from_partial_trace(address)
        assert value == variable.value
    else:
        variable = Variable(distribution=None, value=value, address_base=address_base, address=address, instance=instance, log_prob=0., log_importance_weight=None, rejsmp=True, name=name, control=control)
        # Value shows the number of retries

    _current_trace.add(variable)
    if value == 1:
        # Start of a new rejection sampling
        _rejection_sampling_stack.push(variable)
    else:
        # Retrying the same rejection sampling loop
        # -> Replace the active rejectoin sampling variable
        # -> Restore LSTM's hidden state (if exists)
        hidden_state = _rejection_sampling_stack.top_hidden
        if hidden_state is not None:
            _current_trace_inference_network._infer_lstm_state = hidden_state
        _rejection_sampling_stack.updateTopVariable(variable)


def rejection_sampling_end():
    global _target_rejection_address
    if _target_rejection_address == _rejection_sampling_stack.top_variable.address:
        raise RejectionEndException(int(_rejection_sampling_stack.top_variable.value.item()))
    _rejection_sampling_stack.pop()

    # TODO: add a dummy variable or a tag to the trace?


def _init_traces(func, trace_mode=TraceMode.PRIOR, prior_inflation=PriorInflation.DISABLED, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING, inference_network=None, observe=None, proposal=None, metropolis_hastings_trace=None, address_dictionary=None, likelihood_importance=1., importance_weighting=ImportanceWeighting.IW0):
    global _trace_mode
    global _inference_engine
    global _prior_inflation
    global _likelihood_importance
    global _importance_weighting

    _trace_mode = trace_mode
    _inference_engine = inference_engine
    _prior_inflation = prior_inflation
    _likelihood_importance = likelihood_importance
    _importance_weighting = importance_weighting
    global _current_trace_root_function_name
    global _current_trace_inference_network
    global _current_trace_inference_network_proposal_min_train_iterations
    global _current_trace_observed_variables
    global _current_trace_proposals
    global _address_dictionary
    _address_dictionary = address_dictionary
    _current_trace_root_function_name = func.__code__.co_name
    if observe is None:
        _current_trace_observed_variables = {}
    else:
        _current_trace_observed_variables = observe
    if proposal is None:
        _current_trace_proposals = {}
    else:
        _current_trace_proposals = proposal
    _current_trace_inference_network = inference_network
    if _current_trace_inference_network is None:
        if _inference_engine == InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK:
            raise ValueError('Cannot run trace with IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK without an inference network.')
    else:
        _current_trace_inference_network.eval()
        _current_trace_inference_network._infer_init(_current_trace_observed_variables)
        # _current_trace_inference_network_proposal_min_train_iterations = int(_current_trace_inference_network._total_train_iterations / 10)
        _current_trace_inference_network_proposal_min_train_iterations = None

    if _inference_engine == InferenceEngine.LIGHTWEIGHT_METROPOLIS_HASTINGS or _inference_engine == InferenceEngine.RANDOM_WALK_METROPOLIS_HASTINGS:
        global _metropolis_hastings_trace
        global _metropolis_hastings_site_transition_log_prob
        _metropolis_hastings_trace = metropolis_hastings_trace
        _metropolis_hastings_site_transition_log_prob = None
        if _metropolis_hastings_trace is not None:
            global _metropolis_hastings_site_address
            variable = random.choice(_metropolis_hastings_trace.variables_controlled)
            _metropolis_hastings_site_address = variable.address


def _begin_trace(partial_trace=None, target_rejection_address=None):
    global _current_trace
    global _current_trace_previous_variable
    global _current_trace_replaced_variable_proposal_distributions
    global _current_trace_execution_start
    global _current_partial_trace
    global _rejection_sampling_stack
    global _target_rejection_address

    _rejection_sampling_stack = RejectionSamplingStack()
    _current_partial_trace = partial_trace
    _target_rejection_address = target_rejection_address

    _current_trace_execution_start = time.time()
    _current_trace = Trace()
    _current_trace_previous_variable = None
    _current_trace_replaced_variable_proposal_distributions = {}


def _end_trace(result):
    # Make sure there is no non-ended rejection sampling.
    global _rejection_sampling_stack
    if not _rejection_sampling_stack.isempty():
        print(f'{_rejection_sampling_stack.size()}, {_rejection_sampling_stack.top_variable.address}')
    assert _rejection_sampling_stack.isempty()

    execution_time_sec = time.time() - _current_trace_execution_start
    _current_trace.end(result, execution_time_sec)
    return _current_trace
