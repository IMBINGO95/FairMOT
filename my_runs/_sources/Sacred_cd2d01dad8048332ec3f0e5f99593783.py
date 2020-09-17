from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment()
ex.observers.append(FileStorageObserver.create("my_runs"))

@ex.config
def my_config():
    message = "hello world"

@ex.automain
def my_main(message):
    print(message)