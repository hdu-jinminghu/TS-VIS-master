import numpy as np
class featuremap_read:
    def __init__(self, data=None, range=None, tag=None, sorce_data=None):
        self.data = data
        self.range = range
        self.tag = tag
        self.sorce = sorce_data
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
                          'value':  _data[0]['value'][self.range:(self.range+16), :, :],
                          'sorce_index': self.sorce[0]['value'][0].tolist(),
                          'sorce_data': self.sorce[0]['value'][1].tolist()
                          }
                result.append(values)

            else:
                values = {'wall_time': _data[0]['wall_time'],
                          'step': _data[0]['step'],
                          'Remaining_pictures': 0,
                          'value': _data[0]['value'][self.range:, :, :],
                          'sorce_index': self.sorce[0]['value'][0].tolist(),
                          'sorce_data': self.sorce[0]['value'][1].tolist()
                          }
                result.append(values)
            return result
        else:
            return None