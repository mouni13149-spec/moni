from __future__ import annotations

import argparse


PROMPT_TEMPLATE = """You are an expert technical recruiter.
Score the resume against the job description from 0 to 100.

Job:
{job}

Resume:
{resume}

Return only the numeric fit score."""


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA-3 with LoRA on resume-job scoring pairs.")
    parser.add_argument("--train", required=True, help="Training JSONL file.")
    parser.add_argument("--eval", required=True, help="Evaluation JSONL file.")
    parser.add_argument("--base-model", required=True, help="Base HF model, such as meta-llama/Meta-Llama-3-8B-Instruct.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "Missing optional ML dependencies. Install them with: pip install -e \".[llm]\""
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto")

    dataset = load_dataset("json", data_files={"train": args.train, "eval": args.eval})
    dataset = dataset.map(lambda row: {"text": format_example(row)})

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=25,
        bf16=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        peft_config=lora_config,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


def format_example(row: dict) -> str:
    job = row["job"]
    resume = row["resume"]
    prompt = PROMPT_TEMPLATE.format(job=job["description"], resume=resume["text"])
    return f"{prompt}\n{float(row['human_score']):.0f}"


if __name__ == "__main__":
    main()

