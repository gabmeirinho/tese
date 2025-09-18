def count_operators(feature: Feature) -> int:
    if isinstance(feature, PrimitiveFeature):
        return 0
    count = 1
    for field in fields(feature):
        child = getattr(feature, field.name)
        if isinstance(child, Feature)
            count += count_operators(child)
    return count

def get_unique_columns_rec(feature: Feature, unique_indices: set):
    if isinstance(feature, PrimitiveFeature):
        unique_indices.add(feature.top_feature_idx)
        return
    for field in fields(feature):
        child = getattr(feature, field.name)
        if isinstance(child, Feature):
            get_unique_columns_rec(child, unique_indices)

def count_unique_columns(feature: Feature)->int:
    unique_indices = set()
    get_unique_columns_rec(feature,unique_indices)
    return len(unique_indices)

def calculate_depth(feature: Feature) -> int:
    if isinstance(feature, PrimitiveFeature):
        return 1
    max_child_depth = 0
    for field in fields(feature):
        child = getattr(feature, field.name)
        if isinstance(child, feature):
            max_child_depth = max(max_child_depth, calculate_depth(child))
    return 1 + max_child_depth
