import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from tqdm import tqdm
from contextlib import redirect_stdout, nullcontext

QUESTION_MAP = {
    1: "Comprehensible",
    2: "Repetition",
    3: "Grammar",
    4: "Attribution",
    5: "Main ideas",
    6: "Conciseness"
}

def evaluate_with_seahorse_custom(summaries_df, question_num, batch_size=4, device="cuda", output_log_file=None):
    
    context = open(output_log_file, "w") if output_log_file else nullcontext()

    with context as file, (redirect_stdout(file) if output_log_file else nullcontext()):

        # Take only first 10 samples
        summaries_df = summaries_df.head(10)

        model_name = f"google/seahorse-large-q{question_num}"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        template = "premise: {premise} hypothesis: {hypothesis}"

        # Create pairs of texts and print them for inspection
        print(f"--- SEAHORSE ---")
        print(f"\nEvaluating for {QUESTION_MAP[question_num]}:")
        for i, row in summaries_df.iterrows():
            print(f"Sample {i+1}:")
            print(f"Gold: {row['gold'][:200]}...")
            print(f"Generated: {row['best_rsa_summary'][:200]}...")

        metrics = {
            f"SHMetric/{QUESTION_MAP[question_num]}/proba_1": [],
            f"SHMetric/{QUESTION_MAP[question_num]}/proba_0": [],
            f"SHMetric/{QUESTION_MAP[question_num]}/guess": []
        }

        # Process in batches
        for i in tqdm(range(0, len(summaries_df), batch_size)):
            batch = summaries_df.iloc[i:i+batch_size]
            texts = [template.format(premise=text[:20*1024], hypothesis=summary)
                    for text, summary in zip(batch['gold'], batch['best_rsa_summary'])]

            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                N_inputs = inputs["input_ids"].shape[0]
                decoder_input_ids = torch.full(
                    (N_inputs, 1),
                    tokenizer.pad_token_id,
                    dtype=torch.long,
                    device=model.device
                )

                outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
                logits = outputs.logits[:, -1, [497, 333]]
                probs = torch.softmax(logits, dim=-1)
                guess = probs.argmax(dim=-1)

                metrics[f"SHMetric/{QUESTION_MAP[question_num]}/proba_1"].extend(probs[:, 1].tolist())
                metrics[f"SHMetric/{QUESTION_MAP[question_num]}/proba_0"].extend(probs[:, 0].tolist())
                metrics[f"SHMetric/{QUESTION_MAP[question_num]}/guess"].extend(guess.tolist())

        # Print probabilities for each sample
        df_metrics = pd.DataFrame(metrics)
        print("Probabilities for each sample:")
        print(df_metrics)
        print(f"--- END SEAHORSE ---\n\n")

        return df_metrics