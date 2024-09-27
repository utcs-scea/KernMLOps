"""Library for maintaining, manipulating, and using schemas."""

from data_schema.quanta_runtime import QuantaRuntimeTable
from data_schema.schema import (
    CollectionData,
    CollectionGraph,
    CollectionTable,
    SystemInfoTable,
    collection_id_column,
)

table_types = [
    SystemInfoTable,
    QuantaRuntimeTable,
]

__all__ = [
    "collection_id_column",
    "table_types",
    "CollectionTable",
    "CollectionData",
    "CollectionGraph",
    "SystemInfoTable",
]
