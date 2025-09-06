import torch

# load model from model path and extract the llm part, then save it to the save path
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Extract LLM from model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the extracted LLM')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    model_path = args.model_path
    save_path = args.save_path
    # load model
    model = torch.load(model_path)
    # print('model loaded')
    # # show model keys
    # print('model keys:', model.keys())
    # print(model['state_dict'].keys())
    # extract llm part
    llm = {
        'model': {},
        'config': {'w_bias':True},
    }
    for k in model['state_dict'].keys():
        # level_of_models.0.
        if 'level_of_models.0.' in k:
            # remove level_of_models.0. from the key
            new_k = k.replace('level_of_models.0.', '')
            llm['model'][new_k] = model['state_dict'][k]
    print('new model state dict:', llm['model'].keys())
    # save llm part
    torch.save(llm, save_path)
    print(f'new model saved to: {save_path}')