from src.defenses import OutlierFilterDefense, MetadataTrustDefense, MajorityAgreementDefense
from src.embeddings import FakeEmbeddingModel
from src.models import Item, Query, RetrievedDocument


def make_query():
    return Query(query_id="q1", text="keyboard")


def test_outlier_filter_keeps_scores_within_z_threshold():
    docs = [
        RetrievedDocument(item=Item(item_id="i1", title="A", description="d"), score=0.9),
        RetrievedDocument(item=Item(item_id="i2", title="B", description="d"), score=0.88),
        RetrievedDocument(item=Item(item_id="i3", title="C", description="d"), score=0.05),
    ]
    defense = OutlierFilterDefense(max_z_score=1.0)
    filtered_ids = {doc.item.item_id for doc in defense.filter(make_query(), docs)}
    assert "i3" not in filtered_ids


def test_outlier_filter_passes_through_short_lists_unchanged():
    docs = [RetrievedDocument(item=Item(item_id="i1", title="A", description="d"), score=0.9)]
    defense = OutlierFilterDefense()
    assert defense.filter(make_query(), docs) == docs


def test_metadata_trust_defense_removes_flagged_items():
    docs = [
        RetrievedDocument(item=Item(item_id="i1", title="A", description="d"), score=0.9),
        RetrievedDocument(item=Item(item_id="poison-i1", title="B", description="d", metadata={"is_adversarial": True}), score=0.95),
    ]
    defense = MetadataTrustDefense()
    filtered_ids = {doc.item.item_id for doc in defense.filter(make_query(), docs)}
    assert filtered_ids == {"i1"}


def test_majority_agreement_defense_drops_isolated_outlier_text():
    docs = [
        RetrievedDocument(item=Item(item_id="i1", title="Mechanical Keyboard", description="Quiet mechanical keyboard"), score=0.9),
        RetrievedDocument(item=Item(item_id="i2", title="Mechanical Keyboard", description="Quiet mechanical keyboard"), score=0.89),
        RetrievedDocument(item=Item(item_id="i3", title="Mechanical Keyboard", description="Quiet mechanical keyboard"), score=0.88),
        RetrievedDocument(item=Item(item_id="poison", title="Totally unrelated payload", description="Buy now discount offer"), score=0.95),
    ]
    defense = MajorityAgreementDefense(FakeEmbeddingModel(dim=8), min_neighbors=2, similarity_threshold=0.9)
    kept_ids = {doc.item.item_id for doc in defense.filter(make_query(), docs)}
    assert "poison" not in kept_ids
    assert {"i1", "i2", "i3"}.issubset(kept_ids)
