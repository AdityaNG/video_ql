"""Interactive CLI interface for video_ql project."""

import argparse
import concurrent.futures
import os
import time
from typing import List, Optional

import cv2
import yaml
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Confirm, Prompt

from .base import VideoQL
from .models import Query, VideoProcessorConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive video analysis and querying"
    )
    parser.add_argument(
        "--video", type=str, required=True, help="Path to the video file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for query generation (supported: gpt-4o-mini, claude-3-haiku-20240307, gemini-2.0-flash, gemini-2.0-flash-lite-001, moondream-v1)",  # noqa
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Optional path to existing config YAML file",
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Disable caching of results",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads to use for processing",
    )
    return parser.parse_args()


def display_cost_estimate(video_ql: VideoQL):
    """Display estimated cost for processing the video."""
    console = Console()

    # Get cost estimate
    estimate = video_ql.estimate_processing_cost()

    console.print("\n[bold]Estimated Processing Cost[/bold]")
    console.print(f"Video model: [cyan]{estimate['model']}[/cyan]")
    console.print(
        f"Total frames to process: [cyan]{estimate['total_frames']:,}[/cyan]"
    )
    console.print(
        f"Estimated API calls: [cyan]{estimate['api_calls']:,}[/cyan]"
    )
    console.print(
        f"Estimated token usage: [cyan]{estimate['estimated_tokens']:,}[/cyan]"
    )
    console.print(
        f"Estimated cost: [cyan]${estimate['estimated_cost_usd']:.4f} USD[/cyan]"  # noqa
    )

    # Ask for confirmation if cost is high
    if estimate["estimated_cost_usd"] > 1.0:
        return Confirm.ask(
            "[yellow]Processing cost may be significant. Continue?[/yellow]"
        )

    return True


def display_final_cost(video_ql: VideoQL):
    """Display the actual cost of processing the video."""
    console = Console()

    # Get actual cost data
    cost_data = video_ql.get_processing_cost()

    console.print("\n[bold]Processing Cost Summary[/bold]")
    console.print(f"Video model: [cyan]{cost_data['model']}[/cyan]")
    console.print(
        f"Processed frames: [cyan]{cost_data['processed_frames']:,}/{cost_data['total_frames']:,} ({cost_data['completion_percentage']:.1f}%)[/cyan]"  # noqa
    )
    console.print(f"API calls made: [cyan]{cost_data['api_calls']:,}[/cyan]")
    console.print(
        f"Total tokens used: [cyan]{cost_data['total_tokens']:,}[/cyan]"
    )
    console.print(
        f"  - Prompt tokens: [cyan]{cost_data['prompt_tokens']:,}[/cyan]"
    )
    console.print(
        f"  - Completion tokens: [cyan]{cost_data['completion_tokens']:,}[/cyan]"  # noqa
    )
    console.print(
        f"Total cost: [cyan]${cost_data['total_cost_usd']:.4f} USD[/cyan]"
    )


def process_frames_in_chunks(video_ql: VideoQL, num_threads: int = 4):
    """Process video frames in parallel chunks to build the cache."""
    console = Console()
    total_frames = len(video_ql)

    # Display cost estimate before processing
    console.print("\n[bold]Estimating processing cost...[/bold]")
    estimate = video_ql.estimate_processing_cost()
    console.print(f"Model: [cyan]{estimate['model']}[/cyan]")
    console.print(
        f"Estimated API calls: [cyan]{estimate['api_calls']:,}[/cyan]"
    )
    console.print(
        f"Estimated token usage: [cyan]{estimate['estimated_tokens']:,}[/cyan]"
    )
    console.print(
        f"Estimated cost: [cyan]${estimate['estimated_cost_usd']:.4f} USD[/cyan]"  # noqa
    )

    # Ask for confirmation if cost is high
    if estimate["estimated_cost_usd"] > 1.0:
        if not Confirm.ask(
            "[yellow]Processing cost may be significant. Continue?[/yellow]"
        ):
            console.print("[yellow]Processing canceled by user.[/yellow]")
            return

    console.print(f"\n[bold]Processing {total_frames} frames...[/bold]")

    # Determine number of threads to use
    max_threads = min(num_threads, total_frames)

    # Split frames into chunks for parallel processing
    chunk_size = max(1, total_frames // max_threads)
    chunks = []

    for i in range(0, total_frames, chunk_size):
        end = min(i + chunk_size, total_frames)
        chunks.append(list(range(i, end)))

    # Track overall progress
    processed = 0

    # Define the worker function for each thread
    def process_chunk(chunk_indices):
        nonlocal processed
        for idx in chunk_indices:
            try:
                video_ql[idx]  # This will compute and cache the frame analysis
                processed += 1
            except Exception as e:
                console.print(f"[red]Error processing frame {idx}: {e}[/red]")

    # Use a thread pool to process chunks in parallel
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_threads
    ) as executor:
        futures = []
        for chunk in chunks:
            future = executor.submit(process_chunk, chunk)
            futures.append(future)

        # Show progress while threads are working
        last_update_time = time.time()
        while processed < total_frames:
            progress = processed / total_frames
            progress_bar = "█" * int(20 * progress) + "░" * (
                20 - int(20 * progress)
            )

            # Update the progress bar
            console.print(
                f"\r[bold blue]{progress_bar}[/bold blue] {processed}/{total_frames}",  # noqa
                end="\r",
            )

            # Show cost update every 10 seconds
            current_time = time.time()
            if current_time - last_update_time > 10:
                cost_data = video_ql.get_processing_cost()
                console.print(
                    f"\nToken usage so far: {cost_data['total_tokens']:,} (${cost_data['total_cost_usd']:.4f} USD)"  # noqa
                )
                last_update_time = current_time

            time.sleep(0.5)  # Update progress every half second

            # Check if all futures are done
            if all(future.done() for future in futures):
                break

        # Final progress update
        progress_bar = "█" * 20
        console.print(
            f"\r[bold blue]{progress_bar}[/bold blue] {total_frames}/{total_frames}"  # noqa
        )

        # Check for any exceptions in the futures
        for future in futures:
            try:
                future.result()
            except Exception as e:
                console.print(f"[red]Thread error: {e}[/red]")

    # Display final cost information
    display_final_cost(video_ql)

    console.print("\n[bold green]✓ Processing complete![/bold green]")


def load_or_generate_context_and_queries(
    video_path: str, model_name: str, config_path: Optional[str] = None
) -> tuple[str, List[Query], VideoProcessorConfig]:
    """
    Either load context and queries from a config file,
    or interactively generate them with the user.
    """
    console = Console()

    # Default video processor config
    video_processor_config = VideoProcessorConfig()

    if config_path and os.path.exists(config_path):
        # Load from config file
        console.print(f"Loading configuration from [bold]{config_path}[/bold]")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        context = config.get("context", "Answer the following")
        queries = [Query(**q) for q in config.get("queries", [])]

        # Update video processor config if present
        if "fps" in config:
            video_processor_config.fps = config["fps"]
        if "tile_frames" in config:
            video_processor_config.tile_frames = tuple(config["tile_frames"])
        if "frame_stride" in config:
            video_processor_config.frame_stride = config["frame_stride"]
        if "max_resolution" in config:
            video_processor_config.max_resolution = tuple(
                config["max_resolution"]
            )

        console.print(
            f'Loaded [bold]{len(queries)}[/bold] queries and context: [italic]"{context}"[/italic]'  # noqa
        )

        # Ask if user wants to use these or generate new ones
        if not Confirm.ask("Do you want to use this configuration?"):
            return generate_new_context_and_queries(video_path, model_name)

        return context, queries, video_processor_config

    # No config or user wants to generate new queries
    return generate_new_context_and_queries(video_path, model_name)


def generate_new_context_and_queries(
    video_path: str, model_name: str
) -> tuple[str, List[Query], VideoProcessorConfig]:
    """Interactively generate context and queries with user."""
    console = Console()

    # Get initial context from user
    console.print(
        "\n[bold]First, let's create a context for your video analysis.[/bold]"
    )
    console.print(
        "Describe the video content and what you're interested in tracking or analyzing."  # noqa
    )

    context = Prompt.ask("[bold]Video context[/bold]")

    video_ql = VideoQL(
        video_path=video_path,
        queries=[],  # no queries yet
        context=context,
    )

    # Generate initial queries
    console.print(
        "\n[bold]Generating queries based on your description...[/bold]"
    )
    queries = video_ql.generate_queries(
        context=context, model_name=model_name, num_queries=5
    )

    # Show and confirm queries
    while True:
        console.print("\n[bold]Generated queries:[/bold]")
        for i, query in enumerate(queries, 1):
            console.print(f"[bold]{i}.[/bold] {query.query}")
            if query.options:
                console.print(f"   Options: {', '.join(query.options)}")

        if Confirm.ask("\nAre these queries suitable for your analysis?"):
            break

        # Regenerate with refined context
        console.print(
            "\n[bold]Let's refine the context to generate better queries.[/bold]"  # noqa
        )
        console.print(
            "Please provide additional details or be more specific about what you want to track."  # noqa
        )
        context = Prompt.ask("[bold]Refined context[/bold]", default=context)

        # Ask how many queries to generate
        num_queries = int(
            Prompt.ask(
                "[bold]How many queries would you like to generate?[/bold]",
                default="5",
            )
        )

        console.print("\n[bold]Generating refined queries...[/bold]")
        queries = video_ql.generate_queries(
            context=context, model_name=model_name, num_queries=num_queries
        )

    # Let user customize video processing parameters
    console.print("\n[bold]Video Processing Parameters[/bold]")
    console.print(
        "These parameters affect processing speed, quality, and detail level."
    )

    # Use default VideoProcessorConfig values as starting point
    default_config = VideoProcessorConfig()

    fps = float(
        Prompt.ask(
            "[bold]Frames per second to analyze[/bold] (lower = faster)",
            default=str(default_config.fps),
        )
    )

    tile_height = int(
        Prompt.ask(
            "[bold]Tile height[/bold] (rows of frames to analyze together)",
            default=str(default_config.tile_frames[0]),
        )
    )

    tile_width = int(
        Prompt.ask(
            "[bold]Tile width[/bold] (columns of frames to analyze together)",
            default=str(default_config.tile_frames[1]),
        )
    )

    frame_stride = int(
        Prompt.ask(
            "[bold]Frame stride[/bold] (gap between analyzed frames, higher = faster)",  # noqa
            default=str(default_config.frame_stride),
        )
    )

    # Create the config
    video_processor_config = VideoProcessorConfig(
        fps=fps,
        tile_frames=(tile_height, tile_width),
        frame_stride=frame_stride,
        max_resolution=default_config.max_resolution,
        context=context,
    )

    return context, queries, video_processor_config


def save_configuration(
    context: str,
    queries: List[Query],
    video_processor_config: VideoProcessorConfig,
    output_dir: str,
):
    """Save the current configuration to a YAML file."""
    console = Console()

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config_path = os.path.join(output_dir, "video_ql_config.yaml")

    # Prepare configuration data
    config_data = {
        "context": context,
        "queries": [q.model_dump() for q in queries],
        "fps": video_processor_config.fps,
        "tile_frames": video_processor_config.tile_frames,
        "frame_stride": video_processor_config.frame_stride,
        "max_resolution": video_processor_config.max_resolution,
    }

    # Save to file
    with open(config_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False)

    console.print(
        f"\n[bold green]Configuration saved to:[/bold green] {config_path}"
    )


def display_matching_frames(
    video_ql: VideoQL, matching_frames: List[int], output_dir: str
):
    """Display and save matching frames."""
    console = Console()

    if not matching_frames:
        console.print("[yellow]No frames matched your query.[/yellow]")
        return

    # Create output directory if needed
    results_dir = os.path.join(output_dir, "query_results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    console.print(
        f"\n[bold green]Found {len(matching_frames)} matching frames![/bold green]"  # noqa
    )

    # Generate a unique timestamp for this result set
    timestamp = int(time.time())

    # Save and display frames
    for i, idx in enumerate(matching_frames):
        try:
            # Get nearest analysis result
            effective_stride = video_ql.effective_stride
            analysis_idx = idx // effective_stride
            analysis = video_ql[analysis_idx]

            # Extract the actual frame
            frames = video_ql.extract_frames(idx, 1)
            if not frames:
                continue

            frame = frames[0]["frame"]
            vis_frame = video_ql._visualize_results(frame, analysis)

            # Save the frame
            frame_path = os.path.join(
                results_dir,
                f"match_{timestamp}_{i:03d}_frame_{idx:04d}_{analysis.timestamp:.2f}s.jpg",  # noqa
            )
            cv2.imwrite(frame_path, vis_frame)

            # Display frame
            cv2.imshow("Query Results", vis_frame)
            key = cv2.waitKey(500) & 0xFF  # Show each frame for 500ms
            if key == 27:  # ESC key
                break

        except Exception as e:
            console.print(f"[red]Error displaying frame {idx}: {e}[/red]")

    # Keep the last frame visible until user presses a key
    if matching_frames:
        console.print(
            "[italic]Press any key in the image window to continue...[/italic]"
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    console.print(f"[bold]Results saved to:[/bold] {results_dir}")


def interactive_query_loop(
    video_ql: VideoQL, output_dir: str, model_name: str
):
    """Run the interactive query loop."""
    console = Console()

    console.print(
        "\n[bold]Video processing complete! You can now ask questions about the video.[/bold]"  # noqa
    )
    console.print("For example:")
    console.print('- "Show me all scenes where a person is visible"')
    console.print('- "Find moments when the car stops at a traffic light"')
    console.print('- "When does the subject look directly at the camera?"')

    # Continue asking questions until user wants to exit
    while True:
        question = Prompt.ask(
            "\n[bold]Ask a question about the video[/bold] (or type 'exit' to quit)"  # noqa
        )

        if question.lower() in ["exit", "quit", "q"]:
            break

        console.print("[bold]Generating query from your question...[/bold]")

        try:
            # Generate a QueryConfig from the natural language question
            query_config = video_ql.generate_query_config(
                question=question,
                model_name=model_name,
            )

            # Show the generated query config
            console.print("\n[bold]Generated query:[/bold]")
            console.print(
                Markdown(
                    f"```yaml\n{yaml.dump(query_config.model_dump(), default_flow_style=False)}```"  # noqa
                )
            )

            # Ask if this looks correct
            if not Confirm.ask("Does this query look correct?"):
                console.print(
                    "[yellow]Let's try a different question.[/yellow]"
                )
                continue

            # Run the query and display results
            console.print("[bold]Running query...[/bold]")
            matching_frames = video_ql.query_video(query_config)

            # Display matching frames
            display_matching_frames(video_ql, matching_frames, output_dir)

        except Exception as e:
            console.print(f"[red]Error processing your question: {e}[/red]")


def main():
    """Run the interactive CLI."""
    args = parse_args()
    console = Console()

    # Print welcome message
    console.print("[bold]=====================================[/bold]")
    console.print(
        "[bold blue]   Welcome to Interactive VideoQL   [/bold blue]"
    )
    console.print("[bold]=====================================[/bold]")

    # Check if video file exists
    if not os.path.exists(args.video):
        console.print(
            f"[bold red]Error:[/bold red] Video file not found: {args.video}"
        )
        return

    # Either load or generate context and queries
    context, queries, video_processor_config = (
        load_or_generate_context_and_queries(
            args.video, args.model, args.config
        )
    )

    # Save the configuration
    save_configuration(
        context, queries, video_processor_config, args.output_dir
    )

    # Initialize VideoQL
    console.print("\n[bold]Initializing video analysis...[/bold]")
    video_ql = VideoQL(
        video_path=args.video,
        queries=queries,
        context=context,
        video_processor_config=video_processor_config,
        disable_cache=args.disable_cache,
        model_name=args.model,
    )

    # Process video frames to build cache
    process_frames_in_chunks(video_ql, args.threads)

    # Start interactive query loop
    interactive_query_loop(video_ql, args.output_dir, args.model)

    # Display final cost information
    display_final_cost(video_ql)

    # Goodbye message
    console.print(
        "\n[bold green]Thanks for using VideoQL! Goodbye.[/bold green]"
    )


if __name__ == "__main__":
    main()
