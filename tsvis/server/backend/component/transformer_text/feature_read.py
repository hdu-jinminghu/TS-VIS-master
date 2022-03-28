class transformer_text_read:
    def __init__(self, data=None, tag=None):
        self.data = data
        self.tag = tag
    def get_data(self):
        _data = self.data
        for kid_data in _data[0]['value']:
            if type(_data[0]['value'][kid_data]) == dict:
                for kid in _data[0]['value'][kid_data]:
                    _data[0]['value'][kid_data][kid] = _data[0]['value'][kid_data][kid].tolist()


        value = {
            'wall_time': _data[0]['wall_time'],
            'step': _data[0]['step'],
            'data': _data[0]['value']
        }
        return value
