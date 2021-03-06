import json


class Config:

    def __init__(self, model_name, epochs, batch_size, optimizer, decoder, debug, validate):
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.decoder = decoder
        self.debug = debug
        self.validate = validate


class ConfigReader:

    def __init__(self, path):
        self.path = path

    def read(self):
        with open(self.path, 'r') as json_file:
            json_string = json_file.read()
            json_data = json.loads(json_string)
            return Config(**json_data)
