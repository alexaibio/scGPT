

def _compare_model_and_checkpoint(model, checkpoint):
    chk_keys = checkpoint.keys()
    mdl_keys = model.state_dict().keys()

    print('\n>>> Model defined in code:')
    [print(k) for k in mdl_keys]

    print('\n>>>> saved checkpoints: ')
    [print(k) for k in chk_keys]

    print('\n--------  projection of saved checkpoints to Model defined in code')
    if set(chk_keys) != set(mdl_keys):
        print('----- WARNING! model and checkpoint mismatch!')
        larger_dict = chk_keys if len(chk_keys) >= len(mdl_keys) else mdl_keys
        for key in larger_dict:
            key1 = key if key in chk_keys else "N/A"
            key2 = key if key in mdl_keys else "N/A"
            print(f"{key2}  \  {key1}")  # model keys first


