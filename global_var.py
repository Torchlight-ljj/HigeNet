def _init():  
    global _global_dict
    _global_dict = {"embed":False, "encoder":False, "prob":False}

def set_value(key,value):
    _global_dict[key] = value

def get_value():
    try:
        return _global_dict
    except:
        print('error!!')