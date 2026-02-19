import argparse


def main():
    parser = argparse.ArgumentParser(
        description="CLI application for stacking astrophotography images."
    )

    parser.add_argument(
        "--input", "-i", required=True, help="Path to input image directory"
    )

    parser.add_argument(
        "--output", "-o", required=True, help="Path to output stacked image"
    )

    parser.add_argument(
        "--method",
        "-m",
        choices=["average", "median"],
        default="average",
        help="Stacking method to use (default: average or median)",
    )

    parser.add_argument(
        "--align",
        "-a",
        action="store_true",
        default=False,
        help="Align images before stacking (default: False)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable verbose output (default: False)",
    )

    args = parser.parse_args()
    # Here you would add the logic to process the images based on the provided arguments
    if args.verbose:
        print(f"Input directory: {args.input}")
        print(f"Output file: {args.output}")
        print(f"Stacking method: {args.method}")
        print(f"Align images: {args.align}")
