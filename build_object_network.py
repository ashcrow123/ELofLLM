import asyncio
from data_loader.BRM_loader import load_object_feature_pairs_v2
from llm_methods.run_gpt_prompt import run_gpt_prompt_speaker_retrieval_async
import json
from tqdm import tqdm
from copy import deepcopy
import argparse


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--max-concurrency", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    model = args.model
    max_concurrency = args.max_concurrency
    batch_size = args.batch_size

    all_pairs = load_object_feature_pairs_v2()
    copy_loader = deepcopy(all_pairs)
    BR_labels = [
        "smell",
        "sound",
        "tactile",
        "taste",
        "visual_colour",
        "visual_form_and_surface",
        "visual_motion",
        "function",
        "encyclopaedic",
        "taxonomic",
    ]
    for concept, features in copy_loader.items():
        for label in features:
            if not (label in BR_labels):
                del all_pairs[concept][label]
    copy_loader = deepcopy(all_pairs)
    for concept, features in copy_loader.items():
        if all([features[label] == [] for label in BR_labels]):
            del all_pairs[concept]

    all_words = list(all_pairs.keys())
    results = {key: [] for key in all_words}
    semaphore = asyncio.Semaphore(max_concurrency)

    async def process_one_batch(object_features, batch_features, batch_words):
        async with semaphore:
            output = await run_gpt_prompt_speaker_retrieval_async(
                object_features=object_features,
                features_list=batch_features,
                model=model,
                verbose=False,
            )
        return [batch_words[num] for num in output.num_list]

    async def process_one_word(word):
        object_features = all_pairs[word]
        other_words = [w for w in all_words if w != word]
        other_features = [all_pairs[w] for w in other_words]

        batches = []
        for i in range(0, len(other_features), batch_size):
            batch_features = other_features[i:i + batch_size]
            batch_words = other_words[i:i + batch_size]
            batches.append(
                process_one_batch(object_features, batch_features, batch_words)
            )

        if not batches:
            return word, []

        batch_results = await asyncio.gather(*batches)
        related = [w for batch in batch_results for w in batch]
        return word, related

    tasks = [process_one_word(w) for w in all_words]
    pbar = tqdm(total=len(tasks))
    for coro in asyncio.as_completed(tasks):
        word, related = await coro
        for rw in related:
            results[word].append(rw)
            results[rw].append(word)
        pbar.update(1)
    pbar.close()

    with open(f"./data/{model.replace('/', '-')}_network.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    asyncio.run(main())
