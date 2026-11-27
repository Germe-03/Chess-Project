import pytest
from pathlib import Path
from ops.hashes import pruefen, sha256_datei
import hashlib

ROOT = Path(__file__).resolve().parent.parent

def create_fake_repo(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "a.npz").write_bytes(b'abc')
    # Erstellen des registrieren JSON mit der Version a.npz
    (tmp_path / "models" / "registry.json").write_text('{"versions": [{"file": "a.npz"}]}')
    # Erstellen der SHA256SUMS Datei mit dem berechneten Hash von a.npz
    sha256 = hashlib.sha256(b'abc').hexdigest()
    (tmp_path / "models" / "SHA256SUMS").write_text(f'{sha256}  models/a.npz\n')

def test_echtes_repo():
    assert pruefen(ROOT) == []

def test_abweichung(tmp_path):
    create_fake_repo(tmp_path)
    assert pruefen(tmp_path) == []
    # Überschreiben der Datei a.npz mit b'abd'
    (tmp_path / "models" / "a.npz").write_bytes(b'abd')
    fehler = pruefen(tmp_path)
    assert len(fehler) == 1 and fehler[0].startswith('ABWEICHUNG models/a.npz')

def test_nicht_im_manifest(tmp_path):
    create_fake_repo(tmp_path)
    # Aktualisieren des registrieren JSON mit einer weiteren Version b.npz
    (tmp_path / "models" / "registry.json").write_text('{"versions": [{"file": "a.npz"}, {"file": "b.npz"}]}')
    assert pruefen(tmp_path) == ['NICHT IM MANIFEST models/b.npz']

def test_fehlt(tmp_path):
    create_fake_repo(tmp_path)
    # Löschen der Datei a.npz
    (tmp_path / "models" / "a.npz").unlink()
    assert pruefen(tmp_path) == ['FEHLT models/a.npz']
