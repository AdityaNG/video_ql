from typing import List, Union

from .models import AndCondition, Label, OrCondition, QueryCondition


def matches_query(
    analysis: Label,
    queries: List[Union[QueryCondition, AndCondition, OrCondition]],
) -> bool:
    """Check if an analysis matches a query or set of queries"""
    for query in queries:
        # Check if it's an AND query
        if isinstance(query, AndCondition):
            if all(
                matches_subquery(analysis, subquery) for subquery in query.AND
            ):
                return True

        # Check if it's an OR query
        elif isinstance(query, OrCondition):
            for subquery in query.OR:
                if isinstance(subquery, AndCondition):
                    # Handle nested AND within OR
                    if all(
                        matches_subquery(analysis, sub) for sub in subquery.AND
                    ):
                        return True
                else:
                    # Simple subquery
                    if matches_subquery(analysis, subquery):
                        return True

        # Simple query
        elif isinstance(query, QueryCondition) and matches_subquery(
            analysis, query
        ):
            return True

    return False


def matches_subquery(analysis: Label, subquery: QueryCondition) -> bool:
    """
    Check if an analysis matches a single subquery with
    improved handling.
    """
    # Get the query key (field name)
    query_text = subquery.query
    field_name = query_text.lower().replace("?", "").replace(" ", "_")

    # Get the options to match
    options = subquery.options or []

    # Check if the field exists in the analysis results
    if field_name in analysis.results:
        # If options are specified, check if the analysis value
        # is in the options
        if options:
            return analysis.results[field_name] in options
        # Otherwise, just check if the field has a truthy value
        return bool(analysis.results[field_name])

    return False
