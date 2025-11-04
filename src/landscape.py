class LossSurface():
    def __init__(self, function):
        self.function = function

    def compute_gradient(function, x, y):
        h = 1e-6
        d_x = (function(x+h, y) - function(x-h, y)) / (2*h) 
        d_y = (function(x, y+h) - function(x, y-h)) / (2*h) 
        return d_x,d_y


