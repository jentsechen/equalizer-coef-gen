import sys

PROJ_ROOT = __file__[0 : (__file__.find("source"))]
sys.path.append(PROJ_ROOT + "source/pylib")
EXEC_ROOT = f"{PROJ_ROOT}/build/release/bin"


from dataclasses import dataclass
import subprocess
from typing import Callable, Optional, Protocol
from util.JsonSerializable import JsonIO
import os


class SimRun(Protocol):
    exec_fn: str
    sim_setting: Optional[JsonIO] = None
    simset_fn: str = "sim_setting.json"
    log_path: str = "sim.log"
    preprocess: Optional[Callable] = None
    postprocess: Optional[Callable] = None

    def run(self, simset_fn: Optional[str] = None, log_path: Optional[str] = None, sim_setting: Optional[JsonIO] = None) -> None:
        simset_fn, log_path = self._prepare_filenames(simset_fn, log_path)
        self._pre_process()
        self._remove_old_files(simset_fn, log_path)
        self._save_simset_file(simset_fn, sim_setting)
        self._run_sim(self.exec_fn, simset_fn, log_path)
        self._post_process()

    @staticmethod
    def _run_sim(exec_fn: str, simset_fn: str, log_path: str):
        exec_full_path = os.path.join(EXEC_ROOT, exec_fn)
        command = [exec_full_path, simset_fn]
        with open(log_path, "a") as log_file:
            result = subprocess.run(command, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
                log_file.write(result.stdout)

            if result.stderr:
                print(f"Error: {result.stderr}")
                log_file.write(f"Error: {result.stderr}")

    def _pre_process(self):
        if self.preprocess:
            self.preprocess()

    def _post_process(self):
        if self.postprocess:
            self.postprocess()

    def _remove_old_files(self, simset_fn: str, log_path: str) -> None:
        if os.path.exists(simset_fn):
            os.remove(simset_fn)
        if os.path.exists(log_path):
            os.remove(log_path)

    def _prepare_filenames(self, simset_fn: Optional[str], log_path: Optional[str]) -> "tuple[str, str]":
        simset_fn = simset_fn or self.simset_fn
        log_path = log_path or self.log_path
        if not simset_fn.endswith(".json"):
            simset_fn += ".json"
        if not log_path.endswith(".log"):
            log_path += ".log"
        return simset_fn, log_path

    def _save_simset_file(self, simset_fn: str, sim_setting: Optional[JsonIO]) -> None:
        if sim_setting is not None:
            sim_setting.save_json(simset_fn)
        elif self.sim_setting is not None:
            self.sim_setting.save_json(simset_fn)


if __name__ == "__main__":

    @dataclass
    class AAA(JsonIO):
        a: int = 1
        b: str = "bbb"

    @dataclass
    class MySimRun(SimRun):
        exec_fn = "test_aaaa"
        par: AAA = AAA()

    aaa = AAA(a=4, b="ccc")
    MySimRun(par=aaa).run()
    # MySimRun().run()
