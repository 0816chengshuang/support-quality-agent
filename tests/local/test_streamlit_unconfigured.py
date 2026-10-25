"""The real upstream UI must remain usable when no model is configured."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from openchatbi.config_loader import ConfigLoader

UI = Path(__file__).resolve().parents[2] / "sample_ui" / "streamlit_ui.py"


def test_missing_model_disables_chat_without_errors():
    ConfigLoader._config = None
    app = AppTest.from_file(str(UI), default_timeout=60).run()

    assert len(app.exception) == 0
    assert len(app.error) == 0
    assert len(app.warning) == 1
    assert "模型尚未配置" in app.warning[0].value
    assert app.chat_input[0].disabled
    assert not app.session_state["graph_manager"]._initialized
    assert app.title[0].value == "工单质量分析助手"
    assert app.sidebar.text_input[0].label == "用户标识"
    assert app.sidebar.text_input[1].label == "会话编号"
    assert app.chat_input[0].placeholder == "请输入工单统计问题，或补充查询条件…"
    assert any(heading.value == "历史对话" for heading in app.subheader)


def test_switching_session_without_model_does_not_start_graph():
    ConfigLoader._config = None
    app = AppTest.from_file(str(UI), default_timeout=60).run()
    app.sidebar.text_input[1].set_value("unconfigured-session").run()

    assert len(app.exception) == 0
    assert len(app.error) == 0
    assert app.chat_input[0].disabled
    assert not app.session_state["graph_manager"]._initialized
