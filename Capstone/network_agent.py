class NetworkAgent():

    def __init__(self):
        raise NotImplementedError
    
    def msg(self):
        raise NotImplementedError
    
    def add_msg(self):
        raise NotImplementedError

    def stf(self):
        raise NotImplementedError
    
    def clear_msgs(self):
        raise NotImplementedError
    

class DynamicAgent(NetworkAgent):

    def __init__(self):
        raise NotImplementedError
    
    def step(self):
        raise NotImplementedError
    
    def ctl(self):
        raise NotImplementedError