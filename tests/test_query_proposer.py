import os
import pytest
from unittest.mock import MagicMock, patch

from video_ql.query_proposer import (
    generate_system_prompt,
    generate_queries_from_context,
)
from video_ql.models import Query

# Check if OpenAI API key is available
has_openai_key = os.environ.get("OPENAI_API_KEY", "").startswith("sk-")


def test_generate_system_prompt():
    """Test that the system prompt is generated correctly"""
    prompt = generate_system_prompt()

    # Check that the prompt is a non-empty string
    assert isinstance(prompt, str)
    assert len(prompt) > 0

    # Check that the prompt contains key instructional phrases
    assert "video analysis" in prompt.lower()
    assert "query" in prompt.lower()
    assert "options" in prompt.lower()
    assert "short_question" in prompt.lower()
    assert "short_options" in prompt.lower()

    # Check that the prompt includes formatting instructions
    assert "json" in prompt.lower()


@pytest.mark.skipif(not has_openai_key, reason="OpenAI API key not available")
def test_generate_queries_from_context_real_api():
    """Test generating queries using the real API"""
    context = "You are a forklift dashcam and you are to inspect the driver presence and the forklift status"

    queries = generate_queries_from_context(
        context=context, model_name="gpt-4o-mini", num_queries=3
    )

    # Check that we got Query objects back
    assert len(queries) > 0
    assert all(isinstance(q, Query) for q in queries)

    # Check structure of the first query
    first_query = queries[0]
    assert isinstance(first_query.query, str)
    assert isinstance(first_query.short_question, str)
    assert isinstance(first_query.options, list)
    assert isinstance(first_query.short_options, list)

    # Check that the queries are relevant to the context
    relevant_terms = ["forklift", "driver", "present", "status"]
    all_query_text = " ".join([q.query.lower() for q in queries])
    assert any(term in all_query_text for term in relevant_terms)


@patch("video_ql.query_proposer.ChatOpenAI")
def test_generate_queries_from_context_mock(mock_chat):
    """Test generating queries with a mocked API response"""
    # Setup mock
    mock_model = MagicMock()
    mock_chat.return_value = mock_model

    # Create a mock response with properly formatted content
    mock_response = MagicMock()
    mock_response.content = """```json
[
  {
    "query": "Is the driver present in the forklift?",
    "short_question": "Driver present",
    "options": ["yes", "no"],
    "short_options": ["yes", "no"]
  },
  {
    "query": "Where is the forklift currently at?",
    "short_question": "Forklift position",
    "options": ["Truck", "Warehouse", "Charging"],
    "short_options": ["Truck", "WH", "Charging"]
  }
]
```"""

    mock_model.invoke.return_value = mock_response

    # Run the function
    context = "You are a forklift dashcam and you are to inspect the driver presence and the forklift status"
    queries = generate_queries_from_context(context=context)

    # Verify results
    assert len(queries) == 2
    assert queries[0].query == "Is the driver present in the forklift?"
    assert queries[0].short_question == "Driver present"
    assert queries[0].options == ["yes", "no"]
    assert queries[0].short_options == ["yes", "no"]

    assert queries[1].query == "Where is the forklift currently at?"
    assert queries[1].short_question == "Forklift position"
    assert queries[1].options == ["Truck", "Warehouse", "Charging"]
    assert queries[1].short_options == ["Truck", "WH", "Charging"]


@patch("video_ql.query_proposer.ChatOpenAI")
def test_generate_queries_from_context_invalid_response(mock_chat):
    """Test handling of invalid API responses"""
    # Setup mock with invalid JSON response
    mock_model = MagicMock()
    mock_chat.return_value = mock_model

    mock_response = MagicMock()
    mock_response.content = "This is not valid JSON"

    mock_model.invoke.return_value = mock_response

    # Run the function
    context = "Test context"
    queries = generate_queries_from_context(context=context)

    # Should handle error gracefully
    assert len(queries) == 0


@patch("video_ql.query_proposer.ChatAnthropic")
def test_generate_queries_claude_model(mock_chat):
    """Test using Claude model instead of GPT"""
    # Setup mock
    mock_model = MagicMock()
    mock_chat.return_value = mock_model

    mock_response = MagicMock()
    mock_response.content = """[
      {
        "query": "Is the driver present in the forklift?",
        "short_question": "Driver present",
        "options": ["yes", "no"],
        "short_options": ["yes", "no"]
      }
    ]"""

    mock_model.invoke.return_value = mock_response

    # Run with Claude model
    context = "Test context"
    queries = generate_queries_from_context(
        context=context, model_name="claude-3-haiku-20240307"
    )

    # Verify Claude model was used
    mock_chat.assert_called_once()
    assert len(queries) == 1


def test_invalid_model_name():
    """Test that an invalid model name raises an error"""
    with pytest.raises(ValueError):
        generate_queries_from_context(
            "Test context", model_name="invalid-model"
        )
