from zotero_mcp import server


class DummyContext:
    def info(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None

    def warn(self, *_args, **_kwargs):
        return None


class FakeWriteClient:
    def __init__(self):
        self.created = []
        self.updated_items = []
        self._collections = [{"key": "COLL1234", "data": {"name": "PhD Research"}}]
        self._item = {
            "data": {
                "key": "ITEM1234",
                "title": "Original Title",
                "tags": [{"tag": "old"}],
                "collections": [],
            }
        }

    def collections(self):
        return self._collections

    def create_items(self, items):
        self.created.extend(items)
        return {"success": {"0": "ITEM5678"}}

    def item(self, _item_key):
        return self._item

    def update_item(self, item):
        self.updated_items.append(item)
        return {"success": True}


def test_create_item_parses_json_tags_and_collection_names(monkeypatch):
    fake_client = FakeWriteClient()
    monkeypatch.setattr(server, "_get_write_client", lambda _ctx: (fake_client, None))

    result = server.create_item(
        item_type="journalArticle",
        title="A New Paper",
        creators='[{"creatorType":"author","firstName":"Ada","lastName":"Lovelace"}]',
        tags='["ml","ai"]',
        collection_names=["PhD Research"],
        ctx=DummyContext(),
    )

    assert "Successfully created item" in result
    assert len(fake_client.created) == 1
    payload = fake_client.created[0]
    assert payload["itemType"] == "journalArticle"
    assert payload["tags"] == [{"tag": "ml"}, {"tag": "ai"}]
    assert payload["collections"] == ["COLL1234"]


def test_update_item_add_and_remove_tags(monkeypatch):
    fake_client = FakeWriteClient()
    monkeypatch.setattr(server, "_get_write_client", lambda _ctx: (fake_client, None))

    result = server.update_item(
        item_key="ITEM1234",
        add_tags=["new"],
        remove_tags=["old"],
        extra_fields={"language": "en"},
        ctx=DummyContext(),
    )

    assert "Successfully updated item" in result
    assert fake_client.updated_items
    updated_data = fake_client.updated_items[0]["data"]
    assert updated_data["tags"] == [{"tag": "new"}]
    assert updated_data["language"] == "en"


def test_update_item_without_changes_returns_message(monkeypatch):
    fake_client = FakeWriteClient()
    monkeypatch.setattr(server, "_get_write_client", lambda _ctx: (fake_client, None))

    result = server.update_item(item_key="ITEM1234", ctx=DummyContext())

    assert "No updates specified" in result
