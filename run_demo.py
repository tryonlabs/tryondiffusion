import argparse

if __name__ == '__main__':
    argp = argparse.ArgumentParser(description="Gradio demo")
    argp.add_argument('-n',
                      '--name',
                      type=str, default="data", help='Name of the gradio demo to launch')
    args = argp.parse_args()

    if args.name == "extract_garment":
        from demos import extract_garment_demo as demo
        demo.launch()
    elif args.name == "model_swap":
        from demos import model_swap_demo as demo
        demo.launch()
