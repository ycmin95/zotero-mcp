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
        self._collections = [{"key": "COLL1234", "data": {"name": "Reading List"}}]
        self.created_items = []
        self.created_attachments = []

    def collections(self):
        return self._collections

    def create_items(self, items):
        if items and items[0].get("itemType") == "attachment":
            self.created_attachments.extend(items)
            return {"success": {"0": "ATTACH123"}}
        self.created_items.extend(items)
        return {"success": {"0": "ITEM12345"}}


class FakeHeadResponse:
    ok = True
    headers = {"Content-Type": "application/pdf"}


def test_add_by_identifier_creates_item_and_attaches_pdf(monkeypatch):
    fake_client = FakeWriteClient()
    monkeypatch.setattr(server, "_get_write_client", lambda _ctx: (fake_client, None))
    monkeypatch.setattr(
        server,
        "_fetch_crossref",
        lambda _doi: (
            {
                "itemType": "journalArticle",
                "title": "Crossref Title",
                "creators": [],
                "DOI": "10.1234/example",
            },
            "https://example.org/paper.pdf",
        ),
    )
    monkeypatch.setattr(server.requests, "head", lambda *_args, **_kwargs: FakeHeadResponse())

    result = server.add_by_identifier(
        identifiers=["10.1234/example"],
        collection_names=["Reading List"],
        tags=["important"],
        attach_pdfs=True,
        ctx=DummyContext(),
    )

    assert "Created 1 item(s)." in result
    assert len(fake_client.created_items) == 1
    created = fake_client.created_items[0]
    assert created["collections"] == ["COLL1234"]
    assert created["tags"] == [{"tag": "important"}]
    assert len(fake_client.created_attachments) == 1


def test_add_by_identifier_reports_unsupported_format(monkeypatch):
    fake_client = FakeWriteClient()
    monkeypatch.setattr(server, "_get_write_client", lambda _ctx: (fake_client, None))

    result = server.add_by_identifier(
        identifiers=["not-an-identifier"],
        ctx=DummyContext(),
    )

    assert "Unsupported or unrecognized identifier format" in result
    assert "No items were created." in result
