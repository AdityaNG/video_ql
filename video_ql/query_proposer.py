"""
video_ql query proposer module.
"""

import argparse
import json
from typing import List

import yaml
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .models import Query


def generate_system_prompt() -> str:
    """Generate the system prompt for query generation."""
    return """You are an AI specializing in video analysis. Your task is to generate relevant queries for analyzing video content based on a provided context.

For each query, you should provide:
1. A detailed question (query)
2. A shorter version of the question (short_question)
3. A list of possible answers (options)
4. A shorter version of the options (short_options)

Your queries should be focused, specific, and highly relevant to the provided context. They should help extract meaningful information from video frames.

Examples:
- For a security camera context, you might ask about presence of people, suspicious activities, or environmental conditions
- For a dashcam context, you might ask about road conditions, traffic signals, or driver behavior
- For a manufacturing context, you might ask about equipment status, safety compliance, or process stages

Format your response as a JSON array of query objects.
"""


def generate_queries_from_context(
    context: str, model_name: str = "gpt-4o-mini", num_queries: int = 5
) -> List[Query]:
    """
    Generate relevant queries for video analysis based on the provided context.

    Args:
        context: A description of the video content and analysis goals
        model_name: The name of the LLM to use
        num_queries: Number of queries to generate

    Returns:
        A list of Query objects
    """
    system_prompt = generate_system_prompt()

    # Create the human message prompt
    human_prompt = f"""Please generate {num_queries} relevant queries for analyzing video frames based on this context:

Context: "{context}"

The queries should help identify and extract key information from the video frames according to the context.
"""

    # Initialize the appropriate model
    if "gpt" in model_name.lower():
        model = ChatOpenAI(temperature=0.7, model=model_name)  # type: ignore
    elif "claude" in model_name.lower():
        model = ChatAnthropic(temperature=0.7, model=model_name)  # type: ignore
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Format instructions to ensure proper JSON output
    format_instructions = """Return your response as a JSON array of objects with the following structure:
[
  {
    "query": "Detailed question text",
    "short_question": "Short question text",
    "options": ["Option 1", "Option 2", ...],
    "short_options": ["Opt1", "Opt2", ...]
  },
  ...
]"""

    # Get the model response
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt + "\n\n" + format_instructions),
    ]

    response = model.invoke(messages)

    # Extract and parse the JSON response
    try:
        # Try to extract JSON from the response if it's wrapped in markdown code blocks
        content = response.content
        assert isinstance(content, str)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        # Parse the JSON
        queries_data = json.loads(content)  # type: ignore

        # Convert to Query objects
        queries = [Query(**query_data) for query_data in queries_data]
        return queries

    except Exception as e:
        # If JSON parsing fails, try to extract structured data more carefully
        print(f"Error parsing model response: {e}")
        print(f"Raw response: {response.content}")

        # Return an empty list as fallback
        return []


def save_queries_to_yaml(queries: List[Query], output_path: str) -> None:
    """Save generated queries to a YAML file."""
    queries_data = [query.model_dump() for query in queries]
    with open(output_path, "w") as f:
        yaml.dump(
            {"queries": queries_data},
            f,
            sort_keys=False,
            default_flow_style=False,
        )
    print(f"Queries saved to {output_path}")


def main() -> None:
    """Command-line interface for query generation."""
    parser = argparse.ArgumentParser(
        description="Generate queries for video analysis"
    )
    parser.add_argument(
        "--context", required=True, help="Context description for the video"
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini", help="LLM model to use"
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=5,
        help="Number of queries to generate",
    )
    parser.add_argument(
        "--output", default="queries.yaml", help="Output file path"
    )

    args = parser.parse_args()

    print(f"Generating queries using {args.model}...")
    queries = generate_queries_from_context(
        context=args.context,
        model_name=args.model,
        num_queries=args.num_queries,
    )

    if queries:
        print(f"Generated {len(queries)} queries:")
        for i, query in enumerate(queries, 1):
            print(f"\n{i}. {query.query}")
            print(f"   Short: {query.short_question}")
            print(f"   Options: {query.options}")
            print(f"   Short options: {query.short_options}")

        save_queries_to_yaml(queries, args.output)
    else:
        print("Failed to generate queries.")


if __name__ == "__main__":
    main()
