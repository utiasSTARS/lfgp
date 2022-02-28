import os


def get_transfer_params(load_existing_dir, load_model, load_buffer, load_transfer_exp_settings, load_aux_old_removal):
    if load_model == "":
        load_model = False
    else:
        load_model = os.path.join(load_existing_dir, load_model)

    if load_buffer == "":
        load_buffer = False
    else:
        load_buffer = os.path.join(load_existing_dir, load_buffer)

    # transfer
    if load_transfer_exp_settings != "":
        assert load_buffer and load_model
        load_transfer_exp_settings = os.path.join(load_existing_dir, load_transfer_exp_settings)
    else:
        load_transfer_exp_settings = False
    
    if load_aux_old_removal != "":
        load_aux_old_removal = load_aux_old_removal.split(',')
    else:
        load_aux_old_removal = None
    
    return load_model, load_buffer, load_transfer_exp_settings, load_aux_old_removal