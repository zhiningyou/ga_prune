_ENVS = dict()


def register(cls):
  global _ENVS
  _ENVS[cls.__name__.lower()] = cls()
  return cls


def get_env(name):
  return _ENVS[name]


@register
class GCP(object):
  data_dir = "/path/to/your/data"
  output_dir = "/path/to/your/output"


@register
class TPU(object):
  data_dir = "/path/to/your/data"
  output_dir = "/path/to/your/output"


@register
class Local(object):
  data_dir = "e:\\coding\\tmp\\data"
  output_dir = "e:\\coding\\tmp\\runs"
