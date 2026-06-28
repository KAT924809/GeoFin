from abc import ABC, abstractmethod 

class BaseCollector(ABC):

    @abstractmethod
    def search(self):
        pass

    @abstractmethod
    def download(self):
        pass

    @abstractmethod
    def save_metadata(self):
        pass