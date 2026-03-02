from zotero_mcp import server


class DummyContext:
    def info(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None

    def warn(self, *_args, **_kwargs):
        return None


class FakeCollectionsClient:
    def __init__(self):
        self._collections = [
            {"key": "ROOT1234", "data": {"name": "PhD Research"}},
            {
                "key": "SUBA1234",
                "data": {"name": "Machine Learning", "parentCollection": "ROOT1234"},
            },
        ]
        self.added = []

    def collections(self):
        return self._collections

    def collection(self, key):
        for coll in self._collections:
            if coll["key"] == key:
                return coll
        raise KeyError(key)

    def create_collections(self, _payload):
        return {"success": {"0": "NEWC1234"}}

    def addto_collection(self, collection_key, item_keys):
        self.added.append((collection_key, item_keys))
        return True


def test_search_collections_returns_matches(monkeypatch):
    fake_client = FakeCollectionsClient()
    monkeypatch.setattr(server, "get_zotero_client", lambda: fake_client)

    result = server.search_collections(query="machine", ctx=DummyContext())

    assert "Collections matching 'machine'" in result
    assert "Machine Learning" in result
    assert "SUBA1234" in result


def test_create_collection_respects_write_client_error(monkeypatch):
    monkeypatch.setattr(
        server,
        "_get_write_client",
        lambda _ctx: (None, "Error: write operations need web credentials"),
    )

    result = server.create_collection(name="Drafts", ctx=DummyContext())

    assert "Error: write operations need web credentials" in result


def test_add_items_to_collection_accepts_csv_item_keys(monkeypatch):
    fake_client = FakeCollectionsClient()
    monkeypatch.setattr(server, "_get_write_client", lambda _ctx: (fake_client, None))

    result = server.add_items_to_collection(
        collection_key="ROOT1234",
        item_keys="AAA11111, BBB22222",
        ctx=DummyContext(),
    )

    assert "Successfully added 2 item(s)" in result
    assert fake_client.added == [("ROOT1234", ["AAA11111", "BBB22222"])]
