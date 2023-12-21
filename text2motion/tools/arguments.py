from argparse import ArgumentParser, Namespace

GEN_NS = Namespace()
GEN_NS

GEN_PARSER = ArgumentParser()
GEN_PARSER.add_argument('--opt_path', type=str, help='Opt path')
GEN_PARSER.add_argument('--text', type=str, default="", help='Text description for motion generation')
GEN_PARSER.add_argument('--motion_length', type=int, default=60, help='Number of frames for motion generation')
GEN_PARSER.add_argument('--result_path', type=str, default="test_sample.gif", help='Path to save generation result')
GEN_PARSER.add_argument('--npy_path', type=str, default="", help='Path to save 3D keypoints sequence')
GEN_PARSER.add_argument('--gpu_id', type=int, default=-1, help="which gpu to use")
GEN_PARSER.add_argument('--seed', type=int, default=0, help="random seed")


GEN_PARSER.add_argument("-dm", "--display_mesh", action='store_true', required=False, default=False, help="Display mesh if this flag is present")
# for now just specifies file name (with spaces) made by inference
GEN_PARSER.add_argument("-p", "--prompt", type=str, required=False, default="", help="Prompt for inference display",)
GEN_PARSER.add_argument("-sf", "--seq_file", type=str, required=False, default="", help="file for non-inference display",)
# add model_path arg
GEN_PARSER.add_argument("-m", "--model_path", type=str, required=False, default="", help="Path to model directory e.g. ./checkpoints/grab/grab_baseline_dp_2gpu_8layers_1000",)
GEN_PARSER.add_argument("-sg", "--save_gif", action='store_true', required=False, default=False, help="Saves gif")

def get_case_arguments(case_:str): 
    """
    Returns the parsed GEN_PARSER for the indicated case
    cases can be: 'train','evaluation' and 'generation'
    """
    args = {
        'generation': GEN_PARSER.parse_args("")
    }

    if case_ in ('train','evaluation'):
        raise NotImplementedError(f'{case_} arguments have not been included yet, \
                                  you can find them in their script file in tools/{case_}.py')

    return args.get(case_, None)