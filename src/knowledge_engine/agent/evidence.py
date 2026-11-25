from knowledge_engine.models import RetrievedEvidence

def deduplicate_evidence(
        evidence: list[RetrievedEvidence]
)-> list[RetrievedEvidence]:
    unique_evidence: dict[str , RetrievedEvidence] = {}

    for item in evidence:
        existing = unique_evidence.get(item.chunk_id)

        if existing is None or item.score > existing.score:
            unique_evidence[item.chunk_id] = item

    return list(unique_evidence.values())