import yaml

VARS_PATH_NOTEBOOK = '../../config/variables.yaml'
VARS_PATH_SCRIPT = 'config/variables.yaml'
VARS_PATH = {
    "notebook": VARS_PATH_NOTEBOOK,
    "script": VARS_PATH_SCRIPT,
}


class Variables(object):
    """Class responsible for loading variables.yaml and returning those
    values as attributes, self-refreshing each time one attribute is
    called.

    Note: When calling __repr__ the original object is not refreshed.
    """

    def __init__(self, d=VARS_PATH["notebook"]):
        self.refresh(d)

    def refresh(self, d=VARS_PATH["notebook"]):
        if type(d) is str:
            with open(d) as yaml_file:
                d = yaml.safe_load(yaml_file)
        self.from_dict(d)

    def from_dict(self, d):
        self.__dict__ = {}
        for key, value in d.items():
            if type(value) is dict:
                value = Variables(value)
            self.__dict__[key] = value

    def to_dict(self):
        d = {}
        for key, value in self.__dict__.items():
            if type(value) is Variables:
                value = value.to_dict()
            d[key] = value
        return d

    def __repr__(self):
        return str(self.to_dict())

    def __getattribute__(self, __name: str):
        with open(VARS_PATH["notebook"]) as yaml_file:
            d = yaml.safe_load(yaml_file)
        if __name in d.keys():
            if isinstance(d[__name], dict):
                object.__getattribute__(self, "__setattr__")(
                    __name,
                    Variables(d[__name]),
                )
            else:
                object.__getattribute__(self, "__setattr__")(
                    __name, d[__name]
                )
        return object.__getattribute__(self, __name)


class VariablesScript(object):
    def __init__(self, d=VARS_PATH["script"]):
        self.refresh(d)

    def refresh(self, d=VARS_PATH["script"]):
        if type(d) is str:
            with open(d) as yaml_file:
                d = yaml.safe_load(yaml_file)
        self.from_dict(d)

    def from_dict(self, d):
        self.__dict__ = {}
        for key, value in d.items():
            if type(value) is dict:
                value = VariablesScript(value)
            self.__dict__[key] = value

    def to_dict(self):
        d = {}
        for key, value in self.__dict__.items():
            if type(value) is VariablesScript:
                value = value.to_dict()
            d[key] = value
        return d

    def __repr__(self):
        self.refresh()
        return str(self.to_dict())

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]
