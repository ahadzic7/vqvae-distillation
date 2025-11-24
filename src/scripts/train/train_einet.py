import torch
import os
import time
import filelock
import pickle
import traceback
import logging
from tqdm import tqdm
from src.utilities import get_device, rb_path, get_dist_family
from src.metrics import eval_ll
from datasets.data import data_loaders, cluster_data
from src.EinsumNetwork import EinsumNetwork
from src.EinsumNetwork.graph.poon_domingos_structure import poon_domingos_structure
from src.scripts.eval.eval_einet import eval_einet


def graph(config, dims): 
    structure = config["structure"]["type"]
    if structure == 'poon_domingos':
        structure_param = config["structure"]["hyperparams"]["pcs"]
        pd_delta = [[dims[0] / d, dims[1] / d] for d in structure_param]
        rg = poon_domingos_structure(shape=dims[:2], delta=pd_delta)
    elif structure == 'poon_domingos_vertical' or structure == 'poon_domingos_horizontal':
        if structure == 'poon_domingos_vertical':
            axes = [1]
        else:
            axes = [0]
        structure_param = config["structure"]["hyperparams"]["pcs"]
        pd_delta = [[dims[axes[0]] / d] for d in structure_param]
        rg = poon_domingos_structure(shape=dims[:2], axes=axes, delta=pd_delta)
    else:
        raise AssertionError
    print("------------------------>", pd_delta)
    return rg
    # elif structure == 'vtree_vertical' or structure == 'vtree_horizontal':
    #     vtree_delta = [8]
    #     structure_param = vtree_delta
    #     axis = 1 if structure == 'vtree_vertical' else 0
    #     vtree_mode = 'balanced' # 'sliced'   
    #     if vtree_mode == 'sliced':
    #         rg = Graph.vtree_sliced_structure(shape=dims, axis=axis, delta=structure_param)
    #     elif vtree_mode == 'balanced':
    #         rg = Graph.vtree_balanced_structure(shape=dims, axis=axis, delta=structure_param)

def fwd_pass(spn, trainl, target_shape, device):
    for batch in tqdm(trainl):
        batch = batch.permute(0, 2, 3, 1).reshape(target_shape).to(torch.float)
        log_likelihood = spn.forward(batch.to(device)).sum()
        log_likelihood.backward()
        spn.em_process_batch()
    spn.em_update()

def early_stoppage(spn, model_file, record, valid_ll, epoch_count, counter, warmup=0, n_patience_epochs=15):
    if record['best_validation_ll'] is None or valid_ll > record['best_validation_ll']:
        record['best_validation_ll'] = valid_ll
        torch.save(spn, model_file)
        counter = 0
    else:
        counter = counter + 1

    if epoch_count < warmup:
        counter = 0
    if counter > n_patience_epochs:
        print('Early stopping --- break')
        return True, None # break
    
    return False, counter

def continue_training(record_file, model_file):
    if os.path.isfile(model_file) and os.path.isfile(record_file):
        spn = torch.load(model_file, weights_only=False)
        record = pickle.load(open(record_file, 'rb'))
        print("Loaded model")
    else:
        spn = None
        record = {
            'valid_ll': [],
            'test_ll': [],
            'epoch_count': 0,
            'epoch_times': [],
            'elapsed_time': 0.0,
            'best_validation_ll': None
        }
    return spn, record

def check_time_limits(time_limits, enter_time, record, _epc):
    train_time_limit, worker_time_limit = time_limits
    ##### check if enough training time
    if record['elapsed_time'] > train_time_limit:
        print("train timeout --- break")
        return True, None

    ##### check if enough worker time
    elapsed_time = time.time() - enter_time
    if worker_time_limit - elapsed_time < 1.05 * (elapsed_time / (_epc + 1)):
        print("short of worker time --- break")
        return True, -1
    return False, None


def train_fun(config, spn, record, files, datals, time_limits, dims, device):
    family = get_dist_family(config)
    record_file, model_file = files
    trainl, validl, testl = datals
    enter_time, time_mark = time.time(), time.time()
    
    if family != EinsumNetwork.NormalArray:
        target_shape = (-1, dims[0] * dims[1])
    else:
        target_shape = (-1, dims[0] * dims[1], dims[2])
    counter, ecc = 0, 0
    for _epc, epoch_count in enumerate(range(record['epoch_count'], config["epochs"])):
        ecc = epoch_count
        epoch_start = time.time()
        fwd_pass(spn, trainl, target_shape, device)
        epoch_time = time.time() - epoch_start

        valid_ll, valid_bpd = eval_ll(spn, validl, family, dims, device)
        test_ll, test_bpd = eval_ll(spn, testl, family, dims, device)

        record['valid_ll'].append(valid_ll)
        record['test_ll'].append(test_ll)
        record['epoch_count'] = epoch_count + 1
        record['epoch_times'].append(epoch_time)
        record['elapsed_time'] += time.time() - time_mark
        time_mark = time.time()

        pickle.dump(record, open(record_file, 'wb'))
        log_str = f"[{epoch_count}, {record['elapsed_time']:.1f}] valid LL {valid_ll:.1f} (valid | test bpd) ({valid_bpd:.2f}|{test_bpd:.2f})"
        print(log_str)

        stop, counter = early_stoppage(spn, model_file, record, valid_ll, epoch_count, counter)
        if stop:
            break

        # stop, rval = check_time_limits(time_limits, enter_time, record, _epc)
        # if stop:
        #     return rval

    return ecc

def time_limiting():
    train_time_limit=3600 * 6
    worker_start=time.time()
    worker_time_limit=42600
    ttt = worker_time_limit - (time.time() - worker_start)
    return train_time_limit, ttt



def training_einet(config, models_dir):
    device = get_device()

    trainl, validl, testl = data_loaders(config["input_data"])
    clustering_config = config.get("clustering", False)
    
    if clustering_config and clustering_config["use_clustering"]:
        models_dir, datals = cluster_data(config, [trainl, validl, testl])
        trainl, validl, testl = datals

    shp = config["input_data"]["input_shape"]
    dims = (shp["height"], shp["width"], shp["channels"])

    result_path, _ = rb_path(models_dir, config)
    print(result_path)
    print()

    os.makedirs(result_path, exist_ok=True)
    lock_file = result_path + "/file.lock"
    done_file = result_path + "/file.done"
    failed_file = result_path + "/file.failed"
    lock = filelock.FileLock(lock_file)

    if os.path.isfile(done_file):
        print('Model is trained')
        return None

    model_file = os.path.join(result_path, 'einet.mdl')
    record_file = os.path.join(result_path, 'record.pkl')

    spn, record = continue_training(record_file, model_file)
    if spn is None:
        rg = graph(config, dims)

        args = EinsumNetwork.Args(
            num_var=dims[0]*dims[1],
            num_dims=dims[2],
            num_classes=1,
            num_sums=config["K"],
            num_input_distributions=config["I"],
            exponential_family=get_dist_family(config),
            exponential_family_args=config["family_args"],
            online_em_frequency=config["em"]["online_em_frequency"],
            online_em_stepsize=config["em"]["online_em_stepsize"]
        )
        spn = EinsumNetwork.EinsumNetwork(rg, args)
        spn.initialize()
    
    spn.to(device)
    print('NUMBER OF PARAMETERS: ', spn.get_n_params())
    print(spn)

    time_limits = time_limiting()
    datals = (trainl, validl, testl)
    files = (record_file, model_file)
    del_config = False
    try:
        lock.acquire(timeout=0.1)
        try:
            ret = train_fun(config, spn, record, files, datals, time_limits, dims, device)
            os.system(f"touch {done_file}")
            epoch_count = result_path + f"/epoch_{ret}.txt"
            os.system(f"touch {epoch_count}")
            del_config = True
        except Exception as exc:
            msg = traceback.format_exc()
            logging.error(msg)
            print('Failure!')
            print(msg)
            os.system(f"touch {failed_file}")
        lock.release()
    except filelock.Timeout:
        print('filelock timeout')
    print(type(spn))
    return spn

def train_einet(config, models_dir, rdir=None):
    spn = training_einet(config, models_dir)
    eval_einet(config, models_dir, rdir=rdir, einet=spn)