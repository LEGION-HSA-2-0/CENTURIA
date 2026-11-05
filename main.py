import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="CENTURIA pipeline")
    commands = parser.add_subparsers(dest="command", required=True)

    finetune = commands.add_parser("finetune", help="LoRA fine-tuning of an OCR model.")
    finetune.add_argument("model")
    finetune.add_argument("--epochs", type=int, default=2)
    finetune.add_argument("--train-size", type=int, help="Use only N training images.")

    transcribe = commands.add_parser("transcribe", help="Transcribe text in an image.")
    transcribe.add_argument("model")
    transcribe.add_argument("--prompt", type=Path)
    transcribe.add_argument("--fewshot", type=Path)
    transcribe.add_argument("--finetuned", type=Path)

    extract = commands.add_parser("extract", help="Extract field values from text.")
    extract.add_argument("extractor", choices=["regex", "llm"])
    extract.add_argument("input", help="Transcription file or write ground_truth.")
    extract.add_argument("--prompt", type=Path, default=Path("data/prompts/extraction_prompt.txt"))
    extract.add_argument("--fewshot", type=Path, default=Path("data/fewshot/extraction_examples.json"))

    evaluate = commands.add_parser("evaluate", help="Compare a results file with the ground truth.")
    evaluate.add_argument("results", type=Path)

    args = parser.parse_args()

    if args.command == "finetune":
        from training.lora import train
        train(args.model, epochs=args.epochs, train_size=args.train_size)

    elif args.command == "transcribe":
        from transcription.transcribe import transcribe
        transcribe(args.model, prompt=args.prompt, fewshot=args.fewshot, finetuned=args.finetuned)

    elif args.command == "extract":
        from extraction.extract import extract
        extract(args.extractor, args.input, prompt=args.prompt, fewshot=args.fewshot)

    elif args.command == "evaluate":
        from evaluation.evaluate import evaluate
        evaluate(args.results)

if __name__ == "__main__":
    main()
                                     