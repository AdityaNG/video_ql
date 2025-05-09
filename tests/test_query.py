import pytest
from video_ql.models import Label, QueryCondition, AndCondition, OrCondition
from video_ql.query import matches_query, matches_subquery


@pytest.fixture
def sample_analysis():
    return Label(
        timestamp=1.0,
        results={
            "person_present": True,
            "car_visible": False,
            "weather": "sunny",
            "scene_type": "outdoor",
            "empty": "",
        },
    )


def test_matches_subquery_basic():
    analysis = Label(
        timestamp=1.0,
        results={"person_present": True, "car_visible": False},
    )

    # Test basic truthy value check
    query = QueryCondition(query="person present?")
    assert matches_subquery(analysis, query) is True

    query = QueryCondition(query="car visible?")
    assert matches_subquery(analysis, query) is False

    # Test non-existent field
    query = QueryCondition(query="bicycle visible?")
    assert matches_subquery(analysis, query) is False


def test_matches_subquery_with_options(sample_analysis):
    # Test with options where value matches
    query = QueryCondition(query="weather?", options=["sunny", "cloudy"])
    assert matches_subquery(sample_analysis, query) is True

    # Test with options where value doesn't match
    query = QueryCondition(query="weather?", options=["rainy", "cloudy"])
    assert matches_subquery(sample_analysis, query) is False

    # Test with options for truthy value
    query = QueryCondition(query="person present?", options=["True"])
    assert (
        matches_subquery(sample_analysis, query) is False
    )  # Won't match as True != "True"


def test_matches_subquery_empty_values(sample_analysis):
    # Test with empty field
    query = QueryCondition(query="empty?")
    assert matches_subquery(sample_analysis, query) is False


def test_matches_query_simple():
    analysis = Label(
        timestamp=1.0,
        results={"person_present": True, "scene_type": "outdoor"},
    )

    queries = [QueryCondition(query="person present?")]
    assert matches_query(analysis, queries) is True

    queries = [QueryCondition(query="car visible?")]
    assert matches_query(analysis, queries) is False


def test_matches_query_and_condition():
    analysis = Label(
        timestamp=1.0,
        results={"person_present": True, "scene_type": "outdoor"},
    )

    # Test AND where all conditions match
    queries = [
        AndCondition(
            AND=[
                QueryCondition(query="person present?"),
                QueryCondition(query="scene type?", options=["outdoor"]),
            ]
        )
    ]
    assert matches_query(analysis, queries) is True

    # Test AND where not all conditions match
    queries = [
        AndCondition(
            AND=[
                QueryCondition(query="person present?"),
                QueryCondition(query="scene type?", options=["indoor"]),
            ]
        )
    ]
    assert matches_query(analysis, queries) is False


def test_matches_query_or_condition():
    analysis = Label(
        timestamp=1.0,
        results={"person_present": True, "car_visible": False},
    )

    # Test OR where one condition matches
    queries = [
        OrCondition(
            OR=[
                QueryCondition(query="person present?"),
                QueryCondition(query="car visible?"),
            ]
        )
    ]
    assert matches_query(analysis, queries) is True

    # Test OR where no conditions match
    queries = [
        OrCondition(
            OR=[
                QueryCondition(query="bicycle visible?"),
                QueryCondition(query="truck visible?"),
            ]
        )
    ]
    assert matches_query(analysis, queries) is False


def test_matches_query_nested():
    analysis = Label(
        timestamp=1.0,
        results={
            "person_present": True,
            "car_visible": False,
            "weather": "sunny",
            "scene_type": "outdoor",
        },
    )

    # Test OR with nested AND
    queries = [
        OrCondition(
            OR=[
                AndCondition(
                    AND=[
                        QueryCondition(query="person present?"),
                        QueryCondition(query="weather?", options=["rainy"]),
                    ]
                ),
                AndCondition(
                    AND=[
                        QueryCondition(
                            query="scene type?", options=["outdoor"]
                        ),
                        QueryCondition(query="weather?", options=["sunny"]),
                    ]
                ),
            ]
        )
    ]
    assert matches_query(analysis, queries) is True

    # Test where nothing matches
    queries = [
        OrCondition(
            OR=[
                AndCondition(
                    AND=[
                        QueryCondition(query="person present?"),
                        QueryCondition(query="weather?", options=["rainy"]),
                    ]
                ),
                AndCondition(
                    AND=[
                        QueryCondition(
                            query="scene type?", options=["indoor"]
                        ),
                        QueryCondition(query="weather?", options=["sunny"]),
                    ]
                ),
            ]
        )
    ]
    assert matches_query(analysis, queries) is False


def test_matches_query_multiple_top_level():
    analysis = Label(
        timestamp=1.0,
        results={"person_present": True, "car_visible": False},
    )

    # Test with multiple top-level queries (only one needs to match)
    queries = [
        QueryCondition(query="bicycle visible?"),
        QueryCondition(query="person present?"),
    ]
    assert matches_query(analysis, queries) is True

    # Test with multiple top-level queries where none match
    queries = [
        QueryCondition(query="bicycle visible?"),
        QueryCondition(query="truck visible?"),
    ]
    assert matches_query(analysis, queries) is False
