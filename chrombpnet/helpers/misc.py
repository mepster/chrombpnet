from subprocess import Popen, PIPE, run
from contextlib import ExitStack
import os
import tensorflow as tf

def run_parallel_cmds2(cmds):
    os.environ["PYTHONUNBUFFERED"] = "1"
    procs = []
    for cmd in cmds:
        p = Popen(cmd, shell=True)
        procs.append(p)
    for p in procs:
        p.wait()

def run_parallel_cmds(cmds, fnames=None, chunk_output=False, raise_exception=False, serial=False):
    # if you set fnames to a list of fnames:
    #   stdout and stderr will pipe (together) to a file called f"{fname}", for each cmd
    #   chunk_output is ignored.
    # else:
    #   if chunk_output is False, then we print stdout and stderr in interleaved fashion as we go.
    #   if chunk_output is True, then we buffer stdout and stderr to print at the end.
    # if serial is True, then we don't run in parallel; we just run serially. For convenience.
    # if raise_exception is True, then we raise an exception if any command fails (return code != 0), after all finish

    os.environ["PYTHONUNBUFFERED"] = "1"
    assert(fnames == None or (len(cmds) == len(fnames)))

    processes = []
    failed = False
    for idx, cmd in enumerate(cmds):
        with ExitStack() as stack:
            if fnames:
                out = stack.enter_context(open(fnames[idx], "w"))
                print(cmd, file=out)
            elif chunk_output:
                out = PIPE
                # we print cmd later right before we print stdout and stderr
            else:
                out = None
                print(cmd)

            p = Popen(cmd, shell=True, stdout=out, stderr=out, text=True) # mixes stdout and stderr
            processes.append(p)
            if serial:
                # wait for each process before starting the next
                if p.wait():
                    failed = True

    for (p) in processes:
        if not serial:
            # wait for all processes together at the end
            if p.wait():
                failed = True

        if chunk_output and not fnames:
            print(p.args)
            print(p.stdout.read(), end="")
            print(p.stderr.read(), end="")

        if failed:
            print(f"run_parallel_cmds: command failed: {p.args}")

    if raise_exception and failed:
        raise Exception(f"run_parallel_cmds: at least one command failed")

def get_gpu_scope(args):
    """
    Get a tf strategy.scope to either run on single, or multiple GPUs.
    Use it like this:
        with get_gpu_scope(args):
            // load model here
    If --multiGPU is not set, it returns an empty scope that won't change anything.
    You can still set e.g., CUDA_VISIBLE_DEVICES=1,2 in your environment and it will work.
    """
    if vars(args).get('multiGPU') and args.multiGPU:
        # run the model in "data parallel" mode on multiple GPU devices (on one machine).
        strategy = tf.distribute.MirroredStrategy()
        print('Number of GPU devices: {}'.format(strategy.num_replicas_in_sync))

        # workaround to explicitly close strategy. https://github.com/tensorflow/tensorflow/issues/50487
        # this will supposedly be fixed in tensorflow 2.10
        version = tf.__version__.split(".")
        if (int(version[0]) < 2 or int(version[1]) < 10):
            import atexit
            atexit.register(strategy._extended._collective_ops._pool.close)

        return strategy.scope()
    else:
        print('Single GPU device')
        class EmptyScope(object):
            def __enter__(self): pass
            def __exit__(self, exc_type, exc_val, exc_tb): pass
        return EmptyScope()

def get_strategy(args):
    print("get_strategy() is deprecated! Use get_gpu_scope()")
    # get tf strategy to either run on single, or multiple GPUs
    # you can also do this for cpu only: return tf.distribute.OneDeviceStrategy(device="/cpu:0")

    if vars(args).get('multiGPU') and args.multiGPU:
        # run the model in "data parallel" mode on multiple GPU devices (on one machine).
        strategy = tf.distribute.MirroredStrategy()
        print('Number of GPU devices: {}'.format(strategy.num_replicas_in_sync))

        # workaround to explicitly close strategy. https://github.com/tensorflow/tensorflow/issues/50487
        # this will supposedly be fixed in tensorflow 2.10
        version = tf.__version__.split(".")
        if (int(version[0]) < 2 or int(version[1]) < 10):
            import atexit
            atexit.register(strategy._extended._collective_ops._pool.close)
    else:
        strategy = tf.distribute.get_strategy()
        print('Single GPU device')

    return strategy

if __name__ == '__main__':
    cmds, fnames = [], []
    for i in (1, 2, 3, 4):
        cmds.append(f'echo "hi {i}" ; sleep {i} ; echo "bye {i}"')
        fnames.append(f"foo.{i}.out")
    #run_parallel_cmds(cmds, fnames=fnames, chunk_output=False, raise_exception=False)
    run_parallel_cmds2(cmds)

