"""
standard - a directory for standard definitions
"""

from .constants import (get_annihilation_operator,
                        get_creation_operator,
                        get_eij,SIGMA_X, SIGMA_Y, SIGMA_Z,
                        SIGMA_MINUS, SIGMA_PLUS,)

from .costs import (ControlArea,
                    ControlBandwidthMax,
                    ControlNorm,
                    ControlVariation,
                    ForbidDensities,
                    ForbidStates,
                    TargetDensityInfidelity,
                    TargetDensityInfidelityTime,
                    TargetStateInfidelity,
                    TargetStateInfidelityTime,
                    TargetUnitaryInfidelity,
                    TargetUnitaryInfidelityTime,)

from .functions import (commutator, conjugate_transpose,
                        krons, matmuls,
                        rms_norm,
                        column_vector_list_to_matrix,
                        matrix_to_column_vector_list,
                        corner_terms,
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

from .optimizers import (Adam, LBFGSB, SGD,)

from .plot import (plot_controls, plot_density_population,
                   plot_state_population, plot_error)

from .utils import (ans_jacobian, generate_save_file_path, CustomJSONEncoder,)

__all__ = [
    "get_annihilation_operator", "get_creation_operator",
    "get_eij", "SIGMA_X", "SIGMA_Y", "SIGMA_Z", "SIGMA_MINUS",
    "SIGMA_PLUS",
    "ControlArea", "ControlBandwidthMax",
    "ControlNorm", "ControlVariation", "ForbidDensities",
    "ForbidStates",
    "TargetDensityInfidelity", "TargetDensityInfidelityTime",
    "TargetStateInfidelity", "TargetStateInfidelityTime",
    "TargetUnitaryInfidelity", "TargetUnitaryInfidelityTime",
    "commutator", "conjugate_transpose", "expm", "krons",
    "rms_norm",
    "matmuls", "column_vector_list_to_matrix", "matrix_to_column_vector_list",
    "corner_terms", "plaquette_terms", "single_qubit_terms", "single_qubit_terms_custom",
    "corner_terms_2", "plaquette_terms_2", "single_qubit_terms_2", "single_qubit_terms_custom_2",
    "corner_terms_3", "plaquette_terms_3", "single_qubit_terms_3", "single_qubit_terms_custom_3",
    "Adam", "LBFGSB", "SGD",
    "plot_controls", "plot_density_population", "plot_state_population",
    "ans_jacobian", "generate_save_file_path", "CustomJSONEncoder",
]
