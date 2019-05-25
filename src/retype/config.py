class ReApplyFlags:

    __slots__ = ("replace_any", "incremental")

    def __init__(self, *, replace_any=True, incremental=False):
        self.replace_any = replace_any
        self.incremental = incremental
