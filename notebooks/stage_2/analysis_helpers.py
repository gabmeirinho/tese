import re, time

#count the unique column in the expression, like ,((total_phenols + ((malic_acid - flavanoids) + (alcohol - color_intensity))) - hue)
def count_unique_columns(expression: str) -> int:
    unique_features = set()

    #feature pattern to match column names, all start with column_ followed by digits or letters
    feature_patterns = r'column_[A-Za-z0-9_]+'
    matches = re.findall(feature_patterns, expression, re.IGNORECASE)

    for match in matches:
        unique_features.add(match)
    return len(unique_features)

def count_operators(expression: str) -> int:
    pattern = "(?<=\w|\))\s*([+\-*/])\s*(?=\w|\()"
    return len(re.findall(pattern, expression))

def calculate_tree_depth(expression: str) -> int:
    max_depth = 0
    current_depth = 0

    for char in expression:
        if char == '(':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == ')':
            current_depth -= 1

    return max_depth

def calculate_expression_complexity(expression: str, weight_config: dict) -> float:
    n_unique_features = count_unique_columns(expression)
    n_operators = count_operators(expression)
    tree_depth = calculate_tree_depth(expression)

    expression_complexity = (weight_config["w_operator"] * n_operators)+ \
                            (weight_config["w_column"] * n_unique_features)+ \
                            (weight_config['w_depth'] * tree_depth)
    
    return expression_complexity