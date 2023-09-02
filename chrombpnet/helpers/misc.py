from subprocess import Popen, PIPE, run
from contextlib import ExitStack
import os

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

def get_strategy(args):
    import tensorflow as tf

    # get tf strategy to either run on single, or multiple GPUs
    if vars(args).get('multiGPU') and args.multiGPU:
        # run the model in "data parallel" mode on multiple GPU devices (on one machine).
        strategy = tf.distribute.MirroredStrategy()
        print('Number of GPU devices: {}'.format(strategy.num_replicas_in_sync))
    else:
        strategy = tf.distribute.get_strategy()
        print('Single device')

    return strategy


if __name__ == '__main__':
    cmds, fnames = [], []
    for i in (1, 2, 3, 4):
        cmds.append(f'echo "hi {i}" ; sleep {i} ; echo "bye {i}"')
        fnames.append(f"foo.{i}.out")
    #run_parallel_cmds(cmds, fnames=fnames, chunk_output=False, raise_exception=False)
    run_parallel_cmds2(cmds)

