import logging


def setup_logging(level: str = "INFO") -> None:
    # key=value style so log lines stay grep-able and parseable
    logging.basicConfig(
        level=level,
        format="ts=%(asctime)s level=%(levelname)s logger=%(name)s msg=%(message)s",
    )
