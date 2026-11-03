"""Tests for the file_discovery submodule of the resurfemg library."""

from pathlib import Path, PurePath

import pytest

from resurfemg.data_connector import file_discovery

TEST_DATA_PATH = Path(__file__).resolve().parents[2] / "test_data"

# patient/date layout, plus a few files at the wrong depth on purpose.
TREE_FILES = [
    "a.Poly5",
    "b.txt",
    "p1/d1/x.Poly5",
    "p1/d1/y.txt",
    "p1/d2/z.Poly5",
    "p2/notes.Poly5",  # too shallow for ['patient', 'date']
    "p2/d1/w.Poly5",
    "p2/d1/deep/v.Poly5",  # too deep for ['patient', 'date']
]


def _posix(value) -> str:
    """Normalise str/Path cells so results compare the same on any OS."""
    return PurePath(value).as_posix()


def _rows(df) -> set[tuple[str, ...]]:
    return {tuple(_posix(v) for v in row) for row in df.itertuples(index=False)}


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    for rel in TREE_FILES:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return tmp_path


# --- find_files -------------------------------------------------------------


def test_find_files_no_levels_finds_all_matches_recursively(tree):
    files = file_discovery.find_files(
        tree, "*", "Poly5", folder_levels=None, verbose=False
    )
    assert list(files.columns) == ["files"]
    assert {_posix(f) for f in files["files"]} == {
        f for f in TREE_FILES if f.endswith(".Poly5")
    }


def test_find_files_default_args_find_every_file(tree):
    files = file_discovery.find_files(tree, verbose=False)
    assert {_posix(f) for f in files["files"]} == set(TREE_FILES)


def test_find_files_extension_filter(tree):
    files = file_discovery.find_files(tree, extension_regex="txt", verbose=False)
    assert {_posix(f) for f in files["files"]} == {"b.txt", "p1/d1/y.txt"}


def test_find_files_file_name_filter(tree):
    files = file_discovery.find_files(
        tree, file_name_regex="x", extension_regex="Poly5", verbose=False
    )
    assert {_posix(f) for f in files["files"]} == {"p1/d1/x.Poly5"}


def test_find_files_accepts_str_and_path(tree):
    from_str = file_discovery.find_files(str(tree), verbose=False)
    from_path = file_discovery.find_files(tree, verbose=False)
    assert _rows(from_str) == _rows(from_path)


def test_find_files_levels_split_path_into_columns(tree):
    files = file_discovery.find_files(
        tree, "*", "Poly5", folder_levels=["patient", "date"], verbose=False
    )
    assert list(files.columns) == ["patient", "date", "files"]
    assert _rows(files) == {
        ("p1", "d1", "x.Poly5"),
        ("p1", "d2", "z.Poly5"),
        ("p2", "d1", "w.Poly5"),
    }


def test_find_files_str_level_means_one_level(tree):
    files = file_discovery.find_files(
        tree, "*", "Poly5", folder_levels="patient", verbose=False
    )
    assert list(files.columns) == ["patient", "files"]
    assert _rows(files) == {("p2", "notes.Poly5")}


def test_find_files_does_not_mutate_folder_levels(tree):
    levels = ["patient", "date"]
    file_discovery.find_files(tree, "*", "Poly5", folder_levels=levels, verbose=False)
    assert levels == ["patient", "date"]


def test_find_files_no_matches_returns_empty_frame(tree):
    files = file_discovery.find_files(
        tree, "*", "nope", folder_levels=["patient"], verbose=False
    )
    assert files.empty
    assert list(files.columns) == ["patient", "files"]


def test_find_files_verbose_reports_skipped_files(tree, capsys):
    file_discovery.find_files(tree, "*", "Poly5", folder_levels=["patient", "date"])
    out = capsys.readouterr().out
    assert "did not match" in out
    assert "notes.Poly5" in out


def test_find_files_not_verbose_is_silent(tree, capsys):
    file_discovery.find_files(
        tree, "*", "Poly5", folder_levels=["patient", "date"], verbose=False
    )
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_path": "missing"},
        {"base_path": 123},
        {"file_name_regex": 1},
        {"extension_regex": 1},
        {"folder_levels": 1},
    ],
    ids=["missing_path", "bad_path_type", "bad_name", "bad_ext", "bad_levels"],
)
def test_find_files_invalid_inputs_raise(tree, kwargs):
    if kwargs.get("base_path") == "missing":
        kwargs = {"base_path": tree / "missing"}
        error_type = ValueError
    else:
        error_type = TypeError
    with pytest.raises(error_type):
        file_discovery.find_files(**{"base_path": tree, **kwargs})


# --- find_folders -----------------------------------------------------------


def test_find_folders_no_levels_returns_top_level_folders(tree):
    folders = file_discovery.find_folders(tree, folder_levels=None, verbose=False)
    assert list(folders.columns) == ["destination"]
    assert {_posix(f) for f in folders["destination"]} == {"p1", "p2"}


def test_find_folders_one_level(tree):
    folders = file_discovery.find_folders(
        tree, folder_levels=["patient"], verbose=False
    )
    assert list(folders.columns) == ["patient"]
    assert _rows(folders) == {("p1",), ("p2",)}
    assert len(folders) == 2  # no duplicates from subfolders


def test_find_folders_str_level_equals_list_level(tree):
    as_str = file_discovery.find_folders(tree, folder_levels="patient", verbose=False)
    as_list = file_discovery.find_folders(
        tree, folder_levels=["patient"], verbose=False
    )
    assert _rows(as_str) == _rows(as_list)


def test_find_folders_two_levels_split_path_into_columns(tree):
    folders = file_discovery.find_folders(
        tree, folder_levels=["patient", "date"], verbose=False
    )
    assert list(folders.columns) == ["patient", "date"]
    assert _rows(folders) == {("p1", "d1"), ("p1", "d2"), ("p2", "d1")}
    assert len(folders) == 3  # p2/d1/deep must not add a row


def test_find_folders_not_verbose_is_silent(tree, capsys):
    file_discovery.find_folders(tree, folder_levels=["patient", "date"], verbose=False)
    assert capsys.readouterr().out == ""


def test_find_folders_missing_path_raises(tree):
    with pytest.raises(ValueError):
        file_discovery.find_folders(tree / "missing")


def test_find_folders_bad_levels_raise(tree):
    with pytest.raises(TypeError):
        file_discovery.find_folders(tree, folder_levels=1)


# --- sanity checks against the repo's test_data -----------------------------

needs_test_data = pytest.mark.skipif(
    not TEST_DATA_PATH.is_dir(), reason="test_data folder not found"
)


@needs_test_data
def test_find_files_on_test_data():
    found = file_discovery.find_files(
        TEST_DATA_PATH, "*", "Poly5", folder_levels=None, verbose=False
    )
    assert len(found) == len(list(TEST_DATA_PATH.rglob("*.Poly5")))


@needs_test_data
def test_find_folders_on_test_data():
    found = file_discovery.find_folders(
        TEST_DATA_PATH, folder_levels=None, verbose=False
    )
    assert len(found) == len([p for p in TEST_DATA_PATH.iterdir() if p.is_dir()])
