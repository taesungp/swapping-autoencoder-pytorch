import os
import importlib
from .__main__ import find_launcher_using_name




if __name__ == "__main__":
    import sys
    import pickle

    assert len(sys.argv) >= 3
    
    name = sys.argv[1]
    Launcher = find_launcher_using_name(name)

    cache = "/tmp/tmux_launcher/{}".format(name)
    if os.path.isfile(cache):
        instance = pickle.load(open(cache, 'r'))
    else:
        instance = Launcher()

    cmd = sys.argv[2]
    if cmd == "launch":
        instance.launch()
    elif cmd == "stop":
        instance.stop()
    elif cmd == "send":
        expid = int(sys.argv[3])
        cmd = int(sys.argv[4])
        instance.send_command(expid, cmd)

    os.makedirs("/tmp/tmux_launcher/", exist_ok=True)
    pickle.dump(instance, open(cache, 'w'))
