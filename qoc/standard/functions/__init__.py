"""
functions - a directory for exposing common operations
"""

from qoc.standard.functions.convenience import (commutator,
                                                conjugate_transpose,
                                                krons,
                                                matmuls,
                                                rms_norm,
                                                column_vector_list_to_matrix,
                                                matrix_to_column_vector_list,)

from qoc.standard.functions.generatingHamiltonian import (corner_terms,
                                                plaquette_terms,
                                                single_qubit_terms,
                                                single_qubit_terms_custom,
                                                corner_terms_2,
                                                plaquette_terms_2,
                                                single_qubit_terms_2,
                                                single_qubit_terms_custom_2,
                                                corner_terms_3,
                                                plaquette_terms_3,
                                                single_qubit_terms_3,
                                                single_qubit_terms_custom_3,)

__all__ = [
    "commutator", "conjugate_transpose", "krons", "matmuls",
    "rms_norm",
    "column_vector_list_to_matrix", "matrix_to_column_vector_list",
    "corner_terms", "plaquette_terms", "single_qubit_terms", "single_qubit_terms_custom",
    "corner_terms_2", "plaquette_terms_2", "single_qubit_terms_2", "single_qubit_terms_custom_2",
    "corner_terms_3", "plaquette_terms_3", "single_qubit_terms_3", "single_qubit_terms_custom_3",
]
