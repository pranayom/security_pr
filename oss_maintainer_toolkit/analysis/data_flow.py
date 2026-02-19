"""AST-based taint analysis for Python source files."""

import ast
from pathlib import Path

from oss_maintainer_toolkit.models import DataFlowResult, TaintFlow


# Sources of tainted (user-controlled) data
TAINT_SOURCES: dict[str, str] = {
    # Flask
    "request.args": "Flask query parameters",
    "request.form": "Flask form data",
    "request.data": "Flask raw request data",
    "request.json": "Flask JSON body",
    "request.values": "Flask combined args+form",
    "request.headers": "Flask request headers",
    "request.cookies": "Flask cookies",
    "request.files": "Flask uploaded files",
    # Django
    "request.GET": "Django GET parameters",
    "request.POST": "Django POST parameters",
    "request.body": "Django raw request body",
    # Built-in
    "input": "user input via input()",
    "sys.argv": "command-line arguments",
}

# Dangerous sinks where tainted data should not flow without sanitization
TAINT_SINKS: dict[str, str] = {
    "cursor.execute": "SQL execution (SQL injection)",
    "cursor.executemany": "SQL execution (SQL injection)",
    "os.system": "OS command execution (command injection)",
    "os.popen": "OS command execution (command injection)",
    "subprocess.call": "subprocess execution (command injection)",
    "subprocess.run": "subprocess execution (command injection)",
    "subprocess.Popen": "subprocess execution (command injection)",
    "subprocess.check_output": "subprocess execution (command injection)",
    "subprocess.check_call": "subprocess execution (command injection)",
    "eval": "code evaluation (code injection)",
    "exec": "code execution (code injection)",
    "pickle.loads": "deserialization (insecure deserialization)",
    "pickle.load": "deserialization (insecure deserialization)",
    "yaml.load": "YAML deserialization (insecure deserialization)",
    "open": "file operations (path traversal)",
}


def _get_attr_str(node: ast.AST) -> str | None:
    """Convert an AST attribute chain to a dotted string like 'request.args'."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _get_attr_str(node.value)
        if parent:
            return f"{parent}.{node.attr}"
    return None


def _get_call_name(node: ast.Call) -> str | None:
    """Get the dotted name of a function call."""
    return _get_attr_str(node.func)


def _get_source_line(lines: list[str], lineno: int) -> str:
    """Get the source line (1-indexed) safely."""
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1].strip()
    return ""


class TaintTracker(ast.NodeVisitor):
    """Intraprocedural taint tracker for a single function body.

    Tracks tainted variables through assignments, string formatting,
    and f-strings. Flags when a tainted variable flows to a sink.
    """

    def __init__(self, file_path: str, source_lines: list[str]):
        self.file_path = file_path
        self.source_lines = source_lines
        self.flows: list[TaintFlow] = []
        # Per-function state: var_name -> (source_line, source_type)
        self._tainted: dict[str, tuple[int, str]] = {}

    def _check_tainted(self, node: ast.AST) -> tuple[str, int, str] | None:
        """Check if an expression contains tainted data. Returns (var_name, source_line, taint_type) or None."""
        if isinstance(node, ast.Name) and node.id in self._tainted:
            src_line, taint_type = self._tainted[node.id]
            return node.id, src_line, taint_type

        if isinstance(node, ast.Subscript):
            return self._check_tainted(node.value)

        attr_str = _get_attr_str(node)
        if attr_str and attr_str in self._tainted:
            src_line, taint_type = self._tainted[attr_str]
            return attr_str, src_line, taint_type

        # Check if any sub-expression is tainted (BinOp for string concat, JoinedStr for f-strings)
        if isinstance(node, ast.BinOp):
            left = self._check_tainted(node.left)
            if left:
                return left
            return self._check_tainted(node.right)

        if isinstance(node, ast.JoinedStr):
            for val in node.values:
                if isinstance(val, ast.FormattedValue):
                    tainted = self._check_tainted(val.value)
                    if tainted:
                        return tainted

        if isinstance(node, ast.Call):
            call_name = _get_call_name(node)
            if call_name:
                # Check if method is called on a tainted object (e.g., data.get(...))
                if isinstance(node.func, ast.Attribute):
                    obj_tainted = self._check_tainted(node.func.value)
                    if obj_tainted:
                        return obj_tainted
                # Check if .format() is called with tainted args
                if call_name.endswith(".format"):
                    for arg in node.args:
                        tainted = self._check_tainted(arg)
                        if tainted:
                            return tainted
            # Check call arguments
            for arg in node.args:
                tainted = self._check_tainted(arg)
                if tainted:
                    return tainted

        return None

    def _process_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Process a function definition, tracking taint within its scope."""
        old_tainted = self._tainted.copy()
        self._tainted = {}

        # Mark parameters that come from taint sources (Flask/Django route params)
        # Also check for default values and annotations that hint at request objects
        for arg in node.args.args:
            if arg.arg == "request":
                # If a parameter is named "request", any access to its attrs is tainted
                pass  # handled by attribute access below

        # Walk the function body
        for child in ast.walk(node):
            self._visit_node(child)

        self._tainted = old_tainted

    def _visit_node(self, node: ast.AST) -> None:
        """Visit a single node looking for taint introductions and sink calls."""
        # Assignment: track taint propagation
        if isinstance(node, ast.Assign):
            self._handle_assign(node)
        elif isinstance(node, ast.AnnAssign) and node.value:
            if isinstance(node.target, ast.Name):
                self._handle_assign_target(node.target.id, node.value)

        # Calls: check if tainted data flows to a sink
        if isinstance(node, ast.Call):
            self._handle_call(node)

    def _handle_assign(self, node: ast.Assign) -> None:
        """Handle assignment statements for taint tracking."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._handle_assign_target(target.id, node.value)
            elif isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple):
                for t, v in zip(target.elts, node.value.elts):
                    if isinstance(t, ast.Name):
                        self._handle_assign_target(t.id, v)

    def _handle_assign_target(self, var_name: str, value: ast.AST) -> None:
        """Track whether an assigned value introduces or propagates taint."""
        # Direct taint source: x = request.args.get(...)
        if isinstance(value, ast.Call):
            call_name = _get_call_name(value)
            if call_name:
                for source, desc in TAINT_SOURCES.items():
                    if call_name.startswith(source):
                        self._tainted[var_name] = (
                            getattr(value, "lineno", 0), desc
                        )
                        return
                # input() is a taint source
                if call_name == "input":
                    self._tainted[var_name] = (
                        getattr(value, "lineno", 0),
                        TAINT_SOURCES["input"],
                    )
                    return

        # Direct attribute taint: x = request.args
        attr_str = _get_attr_str(value)
        if attr_str:
            for source, desc in TAINT_SOURCES.items():
                if attr_str == source or attr_str.startswith(source + ".") or attr_str.startswith(source + "["):
                    self._tainted[var_name] = (
                        getattr(value, "lineno", 0), desc
                    )
                    return

        # Subscript: x = request.args["key"] or sys.argv[1]
        if isinstance(value, ast.Subscript):
            sub_attr = _get_attr_str(value.value)
            if sub_attr:
                for source, desc in TAINT_SOURCES.items():
                    if sub_attr == source or sub_attr.startswith(source):
                        self._tainted[var_name] = (
                            getattr(value, "lineno", 0), desc
                        )
                        return

        # Propagation: if RHS is tainted, LHS becomes tainted
        tainted = self._check_tainted(value)
        if tainted:
            _, src_line, taint_type = tainted
            self._tainted[var_name] = (src_line, taint_type)

    def _handle_call(self, node: ast.Call) -> None:
        """Check if a call sends tainted data to a sink."""
        call_name = _get_call_name(node)
        if not call_name:
            return

        sink_desc = TAINT_SINKS.get(call_name)
        if not sink_desc:
            return

        for arg in node.args:
            tainted = self._check_tainted(arg)
            if tainted:
                var_name, src_line, taint_type = tainted
                self.flows.append(TaintFlow(
                    file=self.file_path,
                    source_line=src_line,
                    source_code=_get_source_line(self.source_lines, src_line),
                    sink_line=node.lineno,
                    sink_code=_get_source_line(self.source_lines, node.lineno),
                    taint_type=taint_type,
                    variable=var_name,
                    description=f"Tainted data ({taint_type}) flows to {sink_desc} via variable '{var_name}'",
                ))
                return  # one flow per call is enough

    def analyze(self, tree: ast.AST) -> list[TaintFlow]:
        """Analyze an AST tree for taint flows."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._process_function(node)
        return self.flows


def trace_data_flow(target: str) -> DataFlowResult:
    """Analyze Python files for tainted data flows from sources to sinks.

    Args:
        target: Path to a Python file or directory to analyze.

    Returns:
        DataFlowResult with all identified taint flows.
    """
    target_path = Path(target)
    flows: list[TaintFlow] = []
    files_analyzed = 0
    errors: list[str] = []

    if target_path.is_file():
        files = [target_path]
    elif target_path.is_dir():
        files = sorted(target_path.rglob("*.py"))
    else:
        return DataFlowResult(
            files_analyzed=0,
            total_flows=0,
            flows=[],
            errors=[f"Target not found: {target}"],
        )

    for file_path in files:
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(file_path))
            source_lines = source.splitlines()

            tracker = TaintTracker(str(file_path), source_lines)
            file_flows = tracker.analyze(tree)
            flows.extend(file_flows)
            files_analyzed += 1
        except SyntaxError as e:
            errors.append(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            errors.append(f"Error analyzing {file_path}: {e}")

    return DataFlowResult(
        files_analyzed=files_analyzed,
        total_flows=len(flows),
        flows=flows,
        errors=errors,
    )
