from src.models import Item, Query, RetrievedDocument, Recommendation


def test_item_holds_id_title_description_and_metadata():
    item = Item(item_id="i1", title="Wireless Mouse", description="Ergonomic mouse", metadata={"category": "electronics"})
    assert item.item_id == "i1"
    assert item.title == "Wireless Mouse"
    assert item.metadata["category"] == "electronics"


def test_item_metadata_defaults_to_empty_dict():
    item = Item(item_id="i2", title="Keyboard", description="Mechanical keyboard")
    assert item.metadata == {}


def test_query_user_id_defaults_to_none():
    query = Query(query_id="q1", text="quiet mechanical keyboard")
    assert query.user_id is None


def test_retrieved_document_pairs_item_with_score():
    item = Item(item_id="i1", title="Mouse", description="desc")
    doc = RetrievedDocument(item=item, score=0.87)
    assert doc.item is item
    assert doc.score == 0.87


def test_recommendation_holds_query_ranked_items_and_explanation():
    query = Query(query_id="q1", text="keyboard")
    item = Item(item_id="i1", title="Keyboard", description="desc")
    rec = Recommendation(query=query, ranked_items=[item], explanation="Because it matches.")
    assert rec.query is query
    assert rec.ranked_items == [item]
    assert rec.explanation == "Because it matches."
