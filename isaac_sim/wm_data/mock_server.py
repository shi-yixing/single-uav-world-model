import sys, numpy as np
from multiprocessing.connection import Listener, wait
N, port = int(sys.argv[1]), int(sys.argv[2])
L = Listener(("127.0.0.1", port), authkey=b"airdreamer", backlog=64)
conns = {}
while len(conns) < N:
    c = L.accept(); m = c.recv(); conns[m[1]] = c
t = [0] * N; rng = np.random.default_rng(0)
while True:
    msgs = {}
    for i, c in conns.items():
        try: msgs[i] = c.recv()
        except EOFError: raise SystemExit
    for i, c in conns.items():
        if msgs[i][0] == "reset": t[i] = 0
        last = t[i] >= 40
        o = {"depth": rng.integers(0, 255, (48, 80, 1), dtype=np.uint8), "state": rng.normal(size=15).astype(np.float32),
             "reward": np.float32(0), "is_first": t[i] == 0, "is_last": last, "is_terminal": last,
             "log/goal": np.float32(last), "log/collision": np.float32(0), "log/height": np.float32(0), "log/timeout": np.float32(0),
             "occ": np.zeros((64, 64), bool), "seen": np.zeros((64, 64), bool), "pos": np.zeros(3, np.float32), "yaw": np.float32(0)}
        t[i] += 1; c.send(o)
