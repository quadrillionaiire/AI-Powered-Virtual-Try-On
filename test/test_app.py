from streamlit.testing.v1 import AppTest

def test_app_runs():
    at = AppTest.from_file("src/app.py")
    at.run()
    assert "Upload your test image" in at.html  # Example text check
