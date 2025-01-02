from streamlit.testing.v1 import AppTest

def test_app_url():
    at = AppTest.from_file("src/app.py")
    at.run()
    assert "http://localhost:8501" in at.url

