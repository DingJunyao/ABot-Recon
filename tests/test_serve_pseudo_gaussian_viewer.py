from scripts.serve_pseudo_gaussian_viewer import repository_root, viewer_url


def test_repository_root_is_project_parent_of_scripts():
    root = repository_root()

    assert (root / "tools" / "pseudo_gaussian_viewer.html").is_file()
    assert (root / "outputs").is_dir()


def test_viewer_url_points_to_project_tool():
    assert viewer_url(8765) == (
        "http://127.0.0.1:8765/tools/pseudo_gaussian_viewer.html"
    )
