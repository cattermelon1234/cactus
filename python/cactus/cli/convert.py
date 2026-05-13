from .common import (
    get_weights_dir,
    print_color,
    RED, GREEN, YELLOW,
)


def cmd_convert(args):
    """Convert a HuggingFace model to CQ format using the convert pipeline."""
    model_id = args.model_name
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(get_weights_dir(model_id))
    bits = getattr(args, 'bits', 4) or 4
    token = getattr(args, 'token', None)
    cache_dir = getattr(args, 'cache_dir', None)

    try:
        from ..convert.cli import main as cq_main
        cq_args = ['convert', '--model', model_id, '--out', str(output_dir), '--bits', str(bits)]
        if token:
            cq_args.extend(['--token', token])
        if cache_dir:
            cq_args.extend(['--cache-dir', cache_dir])
        cq_args.append('--force')
        cq_main(cq_args)
        print_color(GREEN, f"Model converted to {output_dir}")
        return 0
    except SystemExit as e:
        return e.code if e.code else 0
    except Exception as e:
        print_color(RED, f"Conversion error: {e}")
        return 1
