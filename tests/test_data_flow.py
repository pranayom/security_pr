"""Tests for the data flow / taint analysis."""

from pathlib import Path

from oss_maintainer_toolkit.analysis.data_flow import trace_data_flow


FIXTURES = Path(__file__).parent / "fixtures"


class TestTraceDataFlow:
    def test_finds_sql_injection_flow(self):
        result = trace_data_flow(str(FIXTURES / "sample_taint.py"))
        sql_flows = [f for f in result.flows if "SQL" in f.description]
        assert len(sql_flows) >= 1

    def test_finds_command_injection_flow(self):
        result = trace_data_flow(str(FIXTURES / "sample_taint.py"))
        cmd_flows = [f for f in result.flows if "command" in f.description.lower()]
        assert len(cmd_flows) >= 1

    def test_finds_eval_injection_flow(self):
        result = trace_data_flow(str(FIXTURES / "sample_taint.py"))
        eval_flows = [f for f in result.flows if "code" in f.description.lower()]
        assert len(eval_flows) >= 1

    def test_finds_taint_propagation(self):
        """Taint should propagate through variable reassignment."""
        result = trace_data_flow(str(FIXTURES / "sample_taint.py"))
        prop_flows = [f for f in result.flows if f.variable == "query"]
        assert len(prop_flows) >= 1

    def test_finds_pickle_injection(self):
        result = trace_data_flow(str(FIXTURES / "sample_taint.py"))
        pickle_flows = [f for f in result.flows if "deserialization" in f.description.lower()]
        assert len(pickle_flows) >= 1

    def test_finds_input_to_system(self):
        result = trace_data_flow(str(FIXTURES / "sample_taint.py"))
        input_flows = [f for f in result.flows if "input" in f.taint_type.lower()]
        assert len(input_flows) >= 1

    def test_flow_has_correct_structure(self):
        result = trace_data_flow(str(FIXTURES / "sample_taint.py"))
        assert result.total_flows > 0
        flow = result.flows[0]
        assert flow.file.endswith("sample_taint.py")
        assert flow.source_line > 0
        assert flow.sink_line > 0
        assert flow.source_line != flow.sink_line
        assert flow.taint_type
        assert flow.variable
        assert flow.description

    def test_analyzes_directory(self):
        result = trace_data_flow(str(FIXTURES))
        assert result.files_analyzed >= 1

    def test_nonexistent_target(self):
        result = trace_data_flow("/nonexistent/path")
        assert result.files_analyzed == 0
        assert len(result.errors) == 1

    def test_clean_file_has_no_flows(self):
        result = trace_data_flow(str(FIXTURES / "sample_clean.py"))
        assert result.total_flows == 0
