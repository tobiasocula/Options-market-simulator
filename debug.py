

class Debugger:

    def __init__(self, mode=None):
        self.mode = mode

    def debug(self, msg, mode):
        if self.mode is None or self.mode != mode:
            return
        
        print(msg)
        

        