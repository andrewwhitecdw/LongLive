import ast
import pathlib

INFERENCE_PY = pathlib.Path(__file__).parents[1] / "inference.py"


def _is_inside_if(node, parent_map):
    while node is not None:
        if isinstance(node, ast.If):
            return True
        node = parent_map.get(node)
    return False


def test_lora_imports_are_lazy():
    tree = ast.parse(INFERENCE_PY.read_text())
    parent_map = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_map[child] = parent

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "peft" and not _is_inside_if(node, parent_map):
                    raise AssertionError("peft is imported at module level")
        elif isinstance(node, ast.ImportFrom):
            if node.module == "utils.lora_utils":
                for alias in node.names:
                    if alias.name == "configure_lora_for_model" and not _is_inside_if(node, parent_map):
                        raise AssertionError("configure_lora_for_model is imported at module level")

    peft_imports = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Import) and any(a.name == "peft" for a in n.names)
    ]
    lora_utils_imports = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and n.module == "utils.lora_utils"
    ]
    assert peft_imports, "import peft is missing"
