import numpy as np
class featuremap_read:
    def __init__(self, data=None, range=None, tag=None):
        self.data = data
        self.range = range
        self.tag = tag
    def get_data(self):
        if self.data:
            result = []
            _data = self.data
            img_len = len(_data[0]['value'])
            over_len = img_len - (self.range+16)
            if over_len >= 0:
                values = {'wall_time': _data[0]['wall_time'],
                          'step': _data[0]['step'],
                          'Remaining_pictures': over_len,
                          'value':  _data[0]['value'][self.range:(self.range+16), :, :]
                          }
                result.append(values)

            else:
                values = {'wall_time': _data[0]['wall_time'],
                          'step': _data[0]['step'],
                          'Remaining_pictures': 0,
                          'value': _data[0]['value'][self.range:, :, :]
                          }
                result.append(values)
            return result
        else:
            return None