from pathlib import Path

from improve import yamlutil
from improve.run import cmd_one_task, cmd_propose


def test_yaml_roundtrip(tmp_path):
    path = tmp_path / "bugs.yaml"
    doc = {
        "items": [
            {
                "id": "bug-x",
                "title": "example",
                "severity": "low",
                "status": "open",
                "evidence": "tests/foo.py",
                "notes": "n",
                "source": "seed",
            }
        ]
    }
    yamlutil.dump_doc(path, doc)
    loaded = yamlutil.load_doc(path)
    assert loaded["items"][0]["id"] == "bug-x"
    yamlutil.upsert_item(loaded, {"id": "bug-x", "status": "fixed"})
    assert loaded["items"][0]["status"] == "fixed"


def test_propose_dry_run(capsys):
    assert cmd_propose(dry_run=True) == 0
    out = capsys.readouterr().out
    assert "bug hunter prompt" in out


def test_one_task_dry_run(capsys):
    assert cmd_one_task(dry_run=True) == 0
    out = capsys.readouterr().out
    assert "Current task" in out or "No open" in out
