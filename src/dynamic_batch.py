def get_dynamic_batch_size(allocated: float, total: float) -> int:
    """Dynamically adjust batch size based on GPU memory"""
    if total - allocated < 2:  # Less than 2GB free
        return 1
    ratio = (total - allocated) / total
    return max(1, int(4 * ratio))  # 4x scaling factor
