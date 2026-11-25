def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Fraction of the top-k results that are relevant."""
    if k <= 0:
        raise ValueError("k must be greater than zero")

    relevant_found = sum(
        1 for chunk_id in retrieved_ids if chunk_id in relevant_ids
    )

    return relevant_found / k

def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Fraction of all relevant chunks found in the top-k results."""
    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not relevant_ids:
        raise ValueError("relevant_ids must not be empty")

    top_k = retrieved_ids[:k]
    relevant_found = sum(
        1 for chunk_id in top_k if chunk_id in relevant_ids
    )

    return relevant_found / len(relevant_ids)

def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """Reciprocal rank of the first relevant result."""
    if not relevant_ids:
        raise ValueError("relevant_ids must not be empty")

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0