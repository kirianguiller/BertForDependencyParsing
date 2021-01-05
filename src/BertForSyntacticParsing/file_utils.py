def is_transformers_available():
    try:
        import transformers
        return True
    except:
        return False
