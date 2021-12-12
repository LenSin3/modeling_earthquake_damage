from pandas.core.frame import DataFrame


class InvalidDataFrame(Exception):
    def __init__(self, df: DataFrame) -> None: 
        self.df = df
    
    def __str__(self):
        return "Object supplied is not a valid DataFrame"

class InvalidColumn(Exception):
    def __init__(self, col: str) -> None:
        self.col = col
    
    def __str__(self):
        return "{} is not a column in DataFrame object supplied.".format(self.col)

class InvalidFilePath(Exception):
    def __init__(self, pth_str: str) -> None:
        self.pth_str = pth_str
    
    def __str__(self):
        return "{} is not a valid file path.".format(self.pth_str)

class UnexpectedDataFrame(Exception):
    def __init__(self, df: DataFrame) -> None: 
        self.df = df
    
    def __str__(self):
        return "Object supplied is not the expected DataFrame"

class InvalidDataType(Exception):
    def __init__(self, col: object) -> None:
        self.col = col
    
    def __str__(self):
        return "{} is not the appropriate data type".format(self.col)

class InvalidList(Exception):
    def __init__(self, obj: object) -> None:
        self.obj = obj
    
    def __str__(self):
        return "Object must be a list."
