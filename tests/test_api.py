import io

from PIL import Image

from utils import storage


def post_restore(client, png, reference=None, headers=None, name="photo.png"):
    files = {"image": (name, png, "image/png")}
    if reference is not None:
        files["reference"] = ("ref.png", reference, "image/png")
    return client.post("/api/restore", files=files, data={"engine": "unet"}, headers=headers or {})


def signup(client, username="maya_01", password="correct-horse"):
    r = client.post("/api/auth/signup", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def test_home_page_status_and_health(client):
    assert "Clarity" in client.get("/").text
    assert client.get("/healthz").json() == {"ok": True}
    status = client.get("/api/status").json()
    assert status["engines"][0]["id"] == "unet"


def test_guest_restore_with_reference_scores(client, sample_png, sample_clean_png):
    r = post_restore(client, sample_png, sample_clean_png)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["metrics"]["psnr"] > body["metrics"]["input_psnr"] + 3
    assert body["history_id"] is None and body["input_url"] is None
    png = client.get(body["output_url"])
    assert png.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(png.content)).size == (body["width"], body["height"])
    assert "attachment" in client.get(body["download_url"]).headers["content-disposition"]
    assert not list(storage.images_dir().rglob("*.png")), "guest photos must not be written to disk"


def test_bad_uploads_get_friendly_errors(client):
    r = client.post("/api/restore", files={"image": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400 and "couldn't be opened" in r.json()["detail"]
    tiny = io.BytesIO()
    Image.new("RGB", (8, 8)).save(tiny, "PNG")
    r = client.post("/api/restore", files={"image": ("t.png", tiny.getvalue(), "image/png")})
    assert r.status_code == 400 and "too small" in r.json()["detail"]
    assert client.get("/api/result/doesnotexist1234").status_code == 404


def test_large_photo_is_downscaled_and_input_returned(client):
    from app import MAX_SIDE

    buf = io.BytesIO()
    Image.new("RGB", (MAX_SIDE + 800, 900), "teal").save(buf, "PNG")
    body = post_restore(client, buf.getvalue()).json()
    assert body["resized"] and max(body["width"], body["height"]) == MAX_SIDE
    assert client.get(body["input_url"]).status_code == 200


def test_transparent_png_is_flattened_on_white(client):
    buf = io.BytesIO()
    Image.new("RGBA", (64, 64), (0, 0, 0, 0)).save(buf, "PNG")
    r = post_restore(client, buf.getvalue())
    assert r.status_code == 200


def test_accounts_history_and_isolation(client, sample_png):
    alice = signup(client, "alice_1")
    bob = signup(client, "bob_22")
    body = post_restore(client, sample_png, headers=alice).json()
    assert body["history_id"]

    items = client.get("/api/history", headers=alice).json()["items"]
    assert len(items) == 1 and client.get("/api/history", headers=bob).json()["items"] == []
    hid = items[0]["id"]
    for kind in ("input", "output", "thumb"):
        assert client.get(f"/api/history/{hid}/{kind}", headers=alice).status_code == 200
        assert client.get(f"/api/history/{hid}/{kind}", headers=bob).status_code == 404
    assert client.delete(f"/api/history/{hid}", headers=bob).status_code == 404
    assert client.delete(f"/api/history/{hid}", headers=alice).status_code == 200
    assert client.get("/api/history", headers=alice).json()["items"] == []


def test_auth_rules(client):
    assert client.get("/api/history").status_code == 401
    assert client.post("/api/auth/signup", json={"username": "ab", "password": "longenough"}).status_code == 400
    assert client.post("/api/auth/signup", json={"username": "okname", "password": "short"}).status_code == 400
    headers = signup(client, "Maya_01")
    assert client.post("/api/auth/signup", json={"username": "maya_01", "password": "another-pass"}).status_code == 400
    assert client.post("/api/auth/login", json={"username": "MAYA_01", "password": "correct-horse"}).status_code == 200
    assert client.post("/api/auth/login", json={"username": "maya_01", "password": "wrong-pass"}).status_code == 401
    assert client.get("/api/me", headers=headers).json() == {"username": "maya_01"}
    client.post("/api/auth/logout", headers=headers)
    assert client.get("/api/me", headers=headers).status_code == 401


def test_login_attempts_are_rate_limited(client):
    codes = [client.post("/api/auth/login", json={"username": "ghost_1", "password": "whatever1"}).status_code
             for _ in range(12)]
    assert codes[:10] == [401] * 10 and codes[10:] == [429, 429]


def test_history_is_capped(client, sample_png, monkeypatch):
    monkeypatch.setattr(storage, "MAX_HISTORY", 3)
    headers = signup(client, "capped_1")
    for _ in range(5):
        assert post_restore(client, sample_png, headers=headers).status_code == 200
    assert len(client.get("/api/history", headers=headers).json()["items"]) == 3
    assert len(list((storage.images_dir() / "capped_1").glob("*_output.png"))) == 3


def test_database_recreated_if_deleted(client):
    signup(client, "first_user")
    storage.db_path().unlink()
    signup(client, "second_user")
