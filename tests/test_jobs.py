from forge_mcp.jobs import JobStore


def test_create_update_get(tmp_path):
    store = JobStore(str(tmp_path / "jobs.json"))
    job = store.create(kind="veo", model="veo-3.0-fast-generate-001", prompt="a cat",
                       project="Proj", subpath=None, filename="cat")
    assert job["status"] == "running"
    store.update(job["id"], status="done", results=[{"url": "https://x/y.mp4"}])
    got = store.get(job["id"])
    assert got["status"] == "done" and got["results"][0]["url"] == "https://x/y.mp4"


def test_persistence_roundtrip(tmp_path):
    path = str(tmp_path / "jobs.json")
    s1 = JobStore(path)
    job = s1.create(kind="veo", model="m", prompt="p", project=None, subpath=None, filename=None)
    s1.update(job["id"], operation_name="operations/abc")
    s2 = JobStore(path)  # fresh load from disk
    got = s2.get(job["id"])
    assert got["operation_name"] == "operations/abc"
    assert got["status"] == "running"


def test_mark_interrupted_without_operation(tmp_path):
    path = str(tmp_path / "jobs.json")
    s1 = JobStore(path)
    a = s1.create(kind="veo", model="m", prompt="p", project=None, subpath=None, filename=None)
    b = s1.create(kind="veo", model="m", prompt="p2", project=None, subpath=None, filename=None)
    s1.update(b["id"], operation_name="operations/xyz")
    s2 = JobStore(path)
    resumable = s2.mark_interrupted()
    assert s2.get(a["id"])["status"] == "failed"          # no operation name -> lost
    assert [j["id"] for j in resumable] == [b["id"]]       # has operation -> caller resumes


def test_running_count(tmp_path):
    store = JobStore(str(tmp_path / "jobs.json"))
    a = store.create(kind="veo", model="m", prompt="p", project=None, subpath=None, filename=None)
    store.create(kind="veo", model="m", prompt="p2", project=None, subpath=None, filename=None)
    assert store.running_count() == 2
    store.update(a["id"], status="done")
    assert store.running_count() == 1


def test_get_unknown(tmp_path):
    store = JobStore(str(tmp_path / "jobs.json"))
    assert store.get("nope") is None
