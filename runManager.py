import os
import glob
import errno


def getRun(directory):
    
    
    directory = os.path.join(directory, "tensorboard")
    if not os.path.isdir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise e

    run_names = list(map(lambda st : st.split("/")[-1], glob.glob(os.path.join(directory, "*"))))
    if len(run_names) > 0:
        run_names = sorted([int(run) for run in run_names])
        j = run_names[len(run_names) - 1] + 1
    else:
        j = 0
    return "./+" + directory + str(j)