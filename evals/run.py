"""Run the evaluation suites against LangSmith.

    poetry run python -m evals.run --suite all --sync
    poetry run python -m evals.run --suite tools --label prompt-v3

Scores land in LangSmith as "experiments". Comparing two of them example by
example is what the UI is for, so nothing is tallied here.
"""

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import aevaluate

from evals.datasets import SUITES, Suite
from evals.targets import EvalTenants, make_target
from langgraph_chatbot.ai.bootstrap import build_runtime
from langgraph_chatbot.config import get_settings
from langgraph_chatbot.logger import setup_logging


def _upload(client: Client, suite: Suite, *, sync: bool) -> None:
    """Make sure LangSmith holds this suite's examples.

    Re-running without --sync leaves the dataset alone, so the same examples are
    reused and experiments stay comparable.
    """
    if not client.has_dataset(dataset_name=suite.dataset):
        dataset = client.create_dataset(
            dataset_name=suite.dataset, description=suite.description
        )
    else:
        dataset = client.read_dataset(dataset_name=suite.dataset)
        existing = [e.id for e in client.list_examples(dataset_id=dataset.id)]
        if existing and not sync:
            return
        if existing:
            client.delete_examples(example_ids=existing)

    client.create_examples(dataset_id=dataset.id, examples=suite.examples)


async def main() -> int:
    load_dotenv()
    setup_logging()

    parser = argparse.ArgumentParser(description="Run LangSmith evaluations.")
    parser.add_argument("--suite", default="all", choices=[*SUITES, "all"])
    parser.add_argument("--label", default="", help="tag the experiment, e.g. prompt-v3")
    parser.add_argument("--sync", action="store_true", help="re-upload the examples")
    args = parser.parse_args()

    if not os.environ.get("LANGSMITH_API_KEY"):
        print("LANGSMITH_API_KEY is not set - nothing to report to.", file=sys.stderr)
        return 1

    client = Client()
    # Fresh ids for this run: Qdrant and Mem0 keep what a run writes, and reusing
    # ids would let this run recall the previous one's facts.
    tenants = EvalTenants()
    print(f"eval run {tenants.run_id}")

    runtime = await build_runtime(get_settings())
    try:
        for name in SUITES if args.suite == "all" else [args.suite]:
            suite = SUITES[name]
            _upload(client, suite, sync=args.sync)
            print(f"{suite.dataset}: {len(suite.examples)} examples")
            await aevaluate(
                make_target(runtime, tenants, voice=suite.voice),
                data=suite.dataset,
                evaluators=suite.evaluators,
                experiment_prefix="-".join(filter(None, [suite.dataset, args.label])),
                max_concurrency=4,
            )
    finally:
        await runtime.aclose()

    print("\nScores: https://smith.langchain.com")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
