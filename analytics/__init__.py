__all__ = ["ProjectedCourt", "DataAnalytics"]


def __getattr__(name):
    if name == "ProjectedCourt":
        from .projected_court import ProjectedCourt

        return ProjectedCourt
    if name == "DataAnalytics":
        from .data_analytics import DataAnalytics

        return DataAnalytics
    raise AttributeError(f"module 'analytics' has no attribute {name!r}")
