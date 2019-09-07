import time

_update_delay=0
_t_0 = 0

def update_sacred_run(sacred_run, loss, total_train_traces,
                      total_train_seconds=None, valid=False):
    global _t_0
    global _update_delay
    if sacred_run is not None:
        if time.time() - _t_0 < _update_delay:
            return
        _t_0 = time.time()
        if not valid:
            sacred_run.info['traces'] = total_train_traces
            sacred_run.info['train_time'] = total_train_seconds
            sacred_run.info['traces_per_sec'] = total_train_traces / total_train_seconds

            # for omniboard plotting (metrics)
            sacred_run.log_scalar('training.loss', loss,
                                  total_train_traces)
        else:
            # for omniboard plotting (metrics)
            sacred_run.log_scalar('validation.loss', loss,
                                  total_train_traces)