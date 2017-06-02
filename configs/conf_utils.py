
def get_conf_base(param_dict):
    try:
        conf_seed = param_dict['conf_seed']
    except:
        conf_seed = None
    try: 
        conf_var = param_dict['conf_var']
    except:
        conf_var = None
    return conf_seed, conf_var