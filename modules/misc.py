import os
import time
import requests
import json
import getpass
import psutil
from IPython import get_ipython


class TensorboardHandler:
    def __init__(self, ngrok_authtoken=None):
        if ngrok_authtoken is None:
            self._ngrok_authtoken = getpass.getpass('Enter ngrok authtoken: ')
        else:
            self._ngrok_authtoken = ngrok_authtoken

    def start(self, logdir):
        os.makedirs(logdir, exist_ok=True)
        get_ipython().system_raw(f'tensorboard --logdir {logdir} --host 0.0.0.0 --port 6006 &')
        get_ipython().system_raw(f'./ngrok http --authtoken {self._ngrok_authtoken} --region eu 6006 &')
        time.sleep(1)
        response = requests.get("http://localhost:4040/api/tunnels")
        self.public_url = json.loads(response.text)['tunnels'][0]['public_url']
        print("Tensorboard:", self.public_url)

    def stop(self):
        for p in psutil.process_iter():
            if 'ngrok' in p.name() or 'tensorboard' in p.name():
                try:
                    # p.send_signal(signal.CTRL_C_EVENT)
                    p.terminate()
                    p.wait(timeout=3)
                except psutil.TimeoutExpired:
                    print(f"Process didn't terminated: {p.name}")
                    return False
        return True

    def status(self):
        processes = {}
        for p in psutil.process_iter():
            if 'ngrok' in p.name() or 'tensorboard' in p.name():
                processes[p.name()] = p.status()
        return processes

    def __del__(self):
        self.stop()
