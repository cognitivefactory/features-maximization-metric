# -*- coding: utf-8 -*-

"""
* Name:         cognitivefactory/features_maximization_metric/tests/test_py
* Description:  Unittests for the `cognitivefactory.features_maximization_metric.fmc` module.
* Author:       Erwan Schild
* Created:      23/11/2022
* Licence:      CeCILL (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORT PYTHON DEPENDENCIES
# ==============================================================================

import re
from typing import List

import pytest
from scipy.sparse import csr_matrix

from cognitivefactory.features_maximization_metric.fmc import FeaturesMaximizationMetric

# ==============================================================================
# test_FeaturesMaximizationMetric_init_error_data_size
# ==============================================================================


def test_FeaturesMaximizationMetric_init_error_data_size():
    """
    In `FeaturesMaximizationMetric`, test `__init__` method with data size error.
    """

    # Invalid case with inconsistencies in number of data.
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The vectors `data_vectors` and the list of classes `data_classes` have inconsistent shapes (currently: '3' vs '5')."
        ),
    ):
        FeaturesMaximizationMetric(
            data_vectors=csr_matrix(
                [
                    [11, 12],
                    [21, 22],
                    [31, 32],
                ]
            ),
            data_classes=[
                "classe_data_1",
                "classe_data_2",
                "classe_data_3",
                "classe_data_4",
                "classe_data_5",
            ],
            list_of_possible_features=["feature_1", "feature_2"],
        )


# ==============================================================================
# test_FeaturesMaximizationMetric_init_error_features_size
# ==============================================================================


def test_FeaturesMaximizationMetric_init_error_features_size():
    """
    In `FeaturesMaximizationMetric`, test `__init__` method with features size error.
    """

    # Invalid case with inconsistencies in number of features.
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The vectors `data_vectors` and the list of features `list_of_possible_features` have inconsistent shapes (currently: '4' vs '2')."
        ),
    ):
        FeaturesMaximizationMetric(
            data_vectors=csr_matrix(
                [
                    [11, 12, 13, 14],
                    [21, 22, 23, 24],
                    [31, 32, 33, 34],
                    [41, 42, 43, 44],
                    [51, 52, 53, 54],
                ]
            ),
            data_classes=[
                "classe_data_1",
                "classe_data_2",
                "classe_data_3",
                "classe_data_4",
                "classe_data_5",
            ],
            list_of_possible_features=["feature_1", "feature_2"],
        )


# ==============================================================================
# test_FeaturesMaximizationMetric_init_error_amplification
# ==============================================================================


def test_FeaturesMaximizationMetric_init_error_amplification():
    """
    In `FeaturesMaximizationMetric`, test `__init__` method with amplification factor error.
    """

    # Invalid case with not integer value.
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The amplification factor `amplification_factor` has to be a positive integer (currently: 'a')."
        ),
    ):
        FeaturesMaximizationMetric(
            data_vectors=csr_matrix(
                [
                    [11, 12],
                    [21, 22],
                    [31, 32],
                    [41, 42],
                ]
            ),
            data_classes=[
                "classe_A",
                "classe_A",
                "classe_B",
                "classe_B",
            ],
            list_of_possible_features=["feature_1", "feature_2"],
            amplification_factor="a",
        )

    # Invalid case with not positive integer value.
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The amplification factor `amplification_factor` has to be a positive integer (currently: '-1')."
        ),
    ):
        FeaturesMaximizationMetric(
            data_vectors=csr_matrix(
                [
                    [11, 12],
                    [21, 22],
                    [31, 32],
                    [41, 42],
                ]
            ),
            data_classes=[
                "classe_A",
                "classe_A",
                "classe_B",
                "classe_B",
            ],
            list_of_possible_features=["feature_1", "feature_2"],
            amplification_factor=-1,
        )


# ==============================================================================
# test_FeaturesMaximizationMetric_default_values
# ==============================================================================


def test_FeaturesMaximizationMetric_init_default_values(capsys):
    """
    In `FeaturesMaximizationMetric`, test `__init__` method with default values.

    Args:
        capsys: Fixture capturing the system output.
    """

    ###
    ### Data.
    ###

    # Define people characteristics that will be studied.
    characteristics_studied: List[str] = [
        "Shoes size",
        "Hair size",
        "Nose size",
    ]

    # Get people characteristics.
    people_characteristics: csr_matrix = csr_matrix(
        [
            [9, 5, 5],
            [9, 10, 5],
            [9, 20, 6],
            [5, 15, 5],
            [6, 25, 6],
            [5, 25, 5],
        ]
    )

    # Get people genders.
    people_genders: List[str] = [
        "Man",
        "Man",
        "Man",
        "Woman",
        "Woman",
        "Woman",
    ]

    ###
    ### Feature Maximization Metrics.
    ###

    # Computation.
    fmc_computer: FeaturesMaximizationMetric = FeaturesMaximizationMetric(
        data_vectors=people_characteristics,
        data_classes=people_genders,
        list_of_possible_features=characteristics_studied,
    )

    # Check verbose output.
    assert capsys.readouterr().out == ""

    ###
    ### Check parameters.
    ###

    # Check data vectors (by checking the difference is never True).
    vectors_differences = (
        fmc_computer.data_vectors
        != csr_matrix(
            [
                [9, 5, 5],
                [9, 10, 5],
                [9, 20, 6],
                [5, 15, 5],
                [6, 25, 6],
                [5, 25, 5],
            ]
        )
    ).todense()
    assert not vectors_differences.any()

    # Check data classes.
    assert fmc_computer.data_classes == [
        "Man",
        "Man",
        "Man",
        "Woman",
        "Woman",
        "Woman",
    ]

    # Check list of possible features.
    assert fmc_computer.list_of_possible_features == [
        "Shoes size",
        "Hair size",
        "Nose size",
    ]

    # Check list of possible classes.
    assert fmc_computer.list_of_possible_classes == [
        "Man",
        "Woman",
    ]

    # Check amplification factor.
    assert fmc_computer.amplification_factor == 1

    ###
    ### Check computations.
    ###

    # Check Features F-Measure.
    assert fmc_computer.features_fmeasure == {
        "Shoes size": {
            "Man": 0.4462809917355372,
            "Woman": 0.2285714285714286,
        },
        "Hair size": {
            "Man": 0.3932584269662921,
            "Woman": 0.6598984771573604,
        },
        "Nose size": {
            "Man": 0.29090909090909095,
            "Woman": 0.24806201550387597,
        },
    }

    # Check Features Overall Average.
    assert fmc_computer.features_overall_average == 0.3778300718072642  # noqa: WPS459

    # Check Features Selection.
    assert fmc_computer.features_selection == {
        "Shoes size": True,
        "Hair size": True,
        "Nose size": False,
    }

    # Check Features Marginal Average.
    assert fmc_computer.features_marginal_averages == {
        "Shoes size": 0.3374262101534829,
        "Hair size": 0.5265784520618262,
        "Nose size": 0.2694855532064835,
    }

    # Check Features Contrast.
    assert fmc_computer.features_contrast == {
        "Shoes size": {
            "Man": 1.3226032190342898,
            "Woman": 0.6773967809657103,
        },
        "Hair size": {
            "Man": 0.7468183049011644,
            "Woman": 1.2531816950988357,
        },
        "Nose size": {
            "Man": 0.0,  # 1.07949791
            "Woman": 0.0,  # 0.92050209
        },
    }

    # Check Features Activation.
    assert fmc_computer.features_activation == {
        "Shoes size": {
            "Man": True,
            "Woman": False,
        },
        "Hair size": {
            "Man": False,
            "Woman": True,
        },
        "Nose size": {
            "Man": False,
            "Woman": False,
        },
    }


# ==============================================================================
# test_FeaturesMaximizationMetric_verbose_output
# ==============================================================================


def test_FeaturesMaximizationMetric_verbose_output(capsys):
    """
    In `FeaturesMaximizationMetric`, test `__init__` method with verbose output .

    Args:
        capsys: Fixture capturing the system output.
    """

    ###
    ### Data.
    ###

    # Define people characteristics that will be studied.
    characteristics_studied: List[str] = [
        "Shoes size",
        "Hair size",
        "Nose size",
    ]

    # Get people characteristics.
    people_characteristics: csr_matrix = csr_matrix(
        [
            [9, 5, 5],
            [9, 10, 5],
            [9, 20, 6],
            [5, 15, 5],
            [6, 25, 6],
            [5, 25, 5],
        ]
    )

    # Get people genders.
    people_genders: List[str] = [
        "Man",
        "Man",
        "Man",
        "Woman",
        "Woman",
        "Woman",
    ]

    ###
    ### Feature Maximization Metrics.
    ###

    # Computation.
    FeaturesMaximizationMetric(
        data_vectors=people_characteristics,
        data_classes=people_genders,
        list_of_possible_features=characteristics_studied,
        verbose=True,
    )

    # Check verbose output.
    assert capsys.readouterr().out.split("\n") == [
        "`FeaturesMaximizationMetric.__init__` : Check parameters.",
        "`FeaturesMaximizationMetric.__init__` : Store parameters.",
        "`FeaturesMaximizationMetric.__init__` : Start computations.",
        "`FeaturesMaximizationMetric.__init__` : Compute Features F-Measure.",
        "`FeaturesMaximizationMetric.__init__` : Compute Features Selection.",
        "`FeaturesMaximizationMetric.__init__` : Compute Features Contrast.",
        "`FeaturesMaximizationMetric.__init__` : Computations done.",
        "",
    ]


# ==============================================================================
# test_FeaturesMaximizationMetric_get_most_activated_classes_by_a_feature_error_feature
# ==============================================================================


def test_FeaturesMaximizationMetric_get_most_activated_classes_by_a_feature_error_feature():
    """
    In `FeaturesMaximizationMetric`, test `get_most_activated_classes_by_a_feature` method with unknown feature.
    """

    ###
    ### Data.
    ###

    # Define people characteristics that will be studied.
    characteristics_studied: List[str] = [
        "Shoes size",
        "Hair size",
        "Nose size",
    ]

    # Get people characteristics.
    people_characteristics: csr_matrix = csr_matrix(
        [
            [9, 5, 5],
            [9, 10, 5],
            [9, 20, 6],
            [5, 15, 5],
            [6, 25, 6],
            [5, 25, 5],
        ]
    )

    # Get people genders.
    people_genders: List[str] = [
        "Man",
        "Man",
        "Man",
        "Woman",
        "Woman",
        "Woman",
    ]

    ###
    ### Feature Maximization Metrics.
    ###

    # Computation.
    fmc_computer: FeaturesMaximizationMetric = FeaturesMaximizationMetric(
        data_vectors=people_characteristics,
        data_classes=people_genders,
        list_of_possible_features=characteristics_studied,
    )

    ###
    ### Get with bad parameters.
    ###

    # Invalid case with unknown feature.
    with pytest.raises(
        ValueError,
        match=re.escape("The requested feature `'UNKNOWN'` is unknown."),
    ):
        fmc_computer.get_most_activated_classes_by_a_feature(
            feature="UNKNOWN",
        )


# ==============================================================================
# test_FeaturesMaximizationMetric_get_most_activated_classes_by_a_feature_error_sort_by
# ==============================================================================


def test_FeaturesMaximizationMetric_get_most_activated_classes_by_a_feature_error_sort_by():
    """
    In `FeaturesMaximizationMetric`, test `get_most_activated_classes_by_a_feature` method with unknown sort option.
    """

    ###
    ### Data.
    ###

    # Define people characteristics that will be studied.
    characteristics_studied: List[str] = [
        "Shoes size",
        "Hair size",
        "Nose size",
    ]

    # Get people characteristics.
    people_characteristics: csr_matrix = csr_matrix(
        [
            [9, 5, 5],
            [9, 10, 5],
            [9, 20, 6],
            [5, 15, 5],
            [6, 25, 6],
            [5, 25, 5],
        ]
    )

    # Get people genders.
    people_genders: List[str] = [
        "Man",
        "Man",
        "Man",
        "Woman",
        "Woman",
        "Woman",
    ]

    ###
    ### Feature Maximization Metrics.
    ###

    # Computation.
    fmc_computer: FeaturesMaximizationMetric = FeaturesMaximizationMetric(
        data_vectors=people_characteristics,
        data_classes=people_genders,
        list_of_possible_features=characteristics_studied,
    )

    ###
    ### Get with bad parameters.
    ###

    # Invalid case with invalid sort option.
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The sort option factor `sort_by` has to be in the following values: `{'contrast', 'fmeasure'}` (currently: 'UNKNOWN')."
        ),
    ):
        fmc_computer.get_most_activated_classes_by_a_feature(
            feature="Shoes size",
            sort_by="UNKNOWN",
        )


# ==============================================================================
# test_FeaturesMaximizationMetric_get_most_activated_classes_by_a_feature_default
# ==============================================================================


def test_FeaturesMaximizationMetric_get_most_activated_classes_by_a_feature_default():
    """
    In `FeaturesMaximizationMetric`, test `get_most_activated_classes_by_a_feature` method.
    """

    ###
    ### Data.
    ###

    # Define people characteristics that will be studied.
    characteristics_studied: List[str] = [
        "Shoes size",
        "Hair size",
        "Nose size",
    ]

    # Get people characteristics.
    people_characteristics: csr_matrix = csr_matrix(
        [
            [9, 5, 5],
            [9, 10, 5],
            [9, 20, 6],
            [5, 15, 5],
            [6, 25, 6],
            [5, 25, 5],
        ]
    )

    # Get people genders.
    people_genders: List[str] = [
        "Man",
        "Man",
        "Man",
        "Woman",
        "Woman",
        "Woman",
    ]

    ###
    ### Feature Maximization Metrics.
    ###

    # Computation.
    fmc_computer: FeaturesMaximizationMetric = FeaturesMaximizationMetric(
        data_vectors=people_characteristics,
        data_classes=people_genders,
        list_of_possible_features=characteristics_studied,
    )

    ###
    ### Get with correct parameters.
    ###

    # Default case.
    assert fmc_computer.get_most_activated_classes_by_a_feature(
        feature="Shoes size",
    ) == ["Man"]
    assert fmc_computer.get_most_activated_classes_by_a_feature(
        feature="Hair size",
    ) == ["Woman"]
    assert (
        fmc_computer.get_most_activated_classes_by_a_feature(
            feature="Nose size",
        )
        == []  # noqa: WPS520 (falsy constant)
    )


# ==============================================================================
# test_FeaturesMaximizationMetric_get_most_activated_classes_by_a_feature_only_activated_sorted_by_contrast
# ==============================================================================


def test_FeaturesMaximizationMetric_get_most_activated_classes_by_a_feature_only_activated_sorted_by_constrast():
    """
    In `FeaturesMaximizationMetric`, test `get_most_activated_classes_by_a_feature` method, only activated, sorted by contrast.
    """

    ###
    ### Data.
    ###

    # Define people characteristics that will be studied.
    characteristics_studied: List[str] = [
        "Shoes size",
        "Hair size",
        "Nose size",
    ]

    # Get people characteristics.
    people_characteristics: csr_matrix = csr_matrix(
        [
            [9, 5, 5],
            [9, 10, 5],
            [9, 20, 6],
            [5, 15, 5],
            [6, 25, 6],
            [5, 25, 5],
        ]
    )

    # Get people genders.
    people_genders: List[str] = [
        "Man",
        "Man",
        "Man",
        "Woman",
        "Woman",
        "Woman",
    ]

    ###
    ### Feature Maximization Metrics.
    ###

    # Computation.
    fmc_computer: FeaturesMaximizationMetric = FeaturesMaximizationMetric(
        data_vectors=people_characteristics,
        data_classes=people_genders,
        list_of_possible_features=characteristics_studied,
    )

    ###
    ### Get with correct parameters.
    ###

    # Case with only activated, sort by contrast.
    assert fmc_computer.get_most_activated_classes_by_a_feature(
        feature="Shoes size",
        activation_only=True,
        sort_by="contrast",
        max_number=None,
    ) == ["Man"]
    assert fmc_computer.get_most_activated_classes_by_a_feature(
        feature="Hair size",
        activation_only=True,
        sort_by="contrast",
        max_number=None,
    ) == ["Woman"]
    assert (
        fmc_computer.get_most_activated_classes_by_a_feature(
            feature="Nose size",
            activation_only=True,
            sort_by="contrast",
            max_number=None,
        )
        == []  # noqa: WPS520 (falsy constant)
    )


# ==============================================================================
# test_FeaturesMaximizationMetric_get_most_activated_classes_by_a_feature_sorted_by_contrast
# ==============================================================================


def test_FeaturesMaximizationMetric_get_most_activated_classes_by_a_feature_sorted_by_contrast():
    """
    In `FeaturesMaximizationMetric`, test `get_most_activated_classes_by_a_feature` method, sorted_by_contrast.
    """

    ###
    ### Data.
    ###

    # Define people characteristics that will be studied.
    characteristics_studied: List[str] = [
        "Shoes size",
        "Hair size",
        "Nose size",
    ]

    # Get people characteristics.
    people_characteristics: csr_matrix = csr_matrix(
        [
            [9, 5, 5],
            [9, 10, 5],
            [9, 20, 6],
            [5, 15, 5],
            [6, 25, 6],
            [5, 25, 5],
        ]
    )

    # Get people genders.
    people_genders: List[str] = [
        "Man",
        "Man",
        "Man",
        "Woman",
        "Woman",
        "Woman",
    ]

    ###
    ### Feature Maximization Metrics.
    ###

    # Computation.
    fmc_computer: FeaturesMaximizationMetric = FeaturesMaximizationMetric(
        data_vectors=people_characteristics,
        data_classes=people_genders,
        list_of_possible_features=characteristics_studied,
    )

    ###
    ### Get with correct parameters.
    ###

    # Case with all, sort by contrast.
    assert fmc_computer.get_most_activated_classes_by_a_feature(
        feature="Shoes size",
        activation_only=False,
        sort_by="contrast",
        max_number=None,
    ) == ["Man", "Woman"]
    assert fmc_computer.get_most_activated_classes_by_a_feature(
        feature="Hair size",
        activation_only=False,
        sort_by="contrast",
        max_number=None,
    ) == ["Woman", "Man"]
    assert fmc_computer.get_most_activated_classes_by_a_feature(
        feature="Nose size",
        activation_only=False,
        sort_by="contrast",
        max_number=None,
    ) == ["Woman", "Man"]


# ==============================================================================
# test_FeaturesMaximizationMetric_get_most_activated_classes_by_a_feature_sorted_by_fmeasure
# ==============================================================================


def test_FeaturesMaximizationMetric_get_most_activated_classes_by_a_feature_sorted_by_fmeasure():
    """
    In `FeaturesMaximizationMetric`, test `get_most_activated_classes_by_a_feature` method, sorted by fmeasure.
    """

    ###
    ### Data.
    ###

    # Define people characteristics that will be studied.
    characteristics_studied: List[str] = [
        "Shoes size",
        "Hair size",
        "Nose size",
    ]

    # Get people characteristics.
    people_characteristics: csr_matrix = csr_matrix(
        [
            [9, 5, 5],
            [9, 10, 5],
            [9, 20, 6],
            [5, 15, 5],
            [6, 25, 6],
            [5, 25, 5],
        ]
    )

    # Get people genders.
    people_genders: List[str] = [
        "Man",
        "Man",
        "Man",
        "Woman",
        "Woman",
        "Woman",
    ]

    ###
    ### Feature Maximization Metrics.
    ###

    # Computation.
    fmc_computer: FeaturesMaximizationMetric = FeaturesMaximizationMetric(
        data_vectors=people_characteristics,
        data_classes=people_genders,
        list_of_possible_features=characteristics_studied,
    )

    ###
    ### Get with correct parameters.
    ###

    # Case with all, sort by fmeasure.
    assert fmc_computer.get_most_activated_classes_by_a_feature(
        feature="Shoes size",
        activation_only=False,
        sort_by="fmeasure",
        max_number=None,
    ) == ["Man", "Woman"]
    assert fmc_computer.get_most_activated_classes_by_a_feature(
        feature="Hair size",
        activation_only=False,
        sort_by="fmeasure",
        max_number=None,
    ) == ["Woman", "Man"]
    assert fmc_computer.get_most_activated_classes_by_a_feature(
        feature="Nose size",
        activation_only=False,
        sort_by="fmeasure",
        max_number=None,
    ) == ["Man", "Woman"]


# ==============================================================================
# test_FeaturesMaximizationMetric_get_most_activated_classes_by_a_feature_sorted_by_fmeasure_limited_to_1
# ==============================================================================


def test_FeaturesMaximizationMetric_get_most_activated_classes_by_a_feature_sorted_by_fmeasure_limited_to_1():
    """
    In `FeaturesMaximizationMetric`, test `get_most_activated_classes_by_a_feature` method, sorted by fmeasure, limited to 1.
    """

    ###
    ### Data.
    ###

    # Define people characteristics that will be studied.
    characteristics_studied: List[str] = [
        "Shoes size",
        "Hair size",
        "Nose size",
    ]

    # Get people characteristics.
    people_characteristics: csr_matrix = csr_matrix(
        [
            [9, 5, 5],
            [9, 10, 5],
            [9, 20, 6],
            [5, 15, 5],
            [6, 25, 6],
            [5, 25, 5],
        ]
    )

    # Get people genders.
    people_genders: List[str] = [
        "Man",
        "Man",
        "Man",
        "Woman",
        "Woman",
        "Woman",
    ]

    ###
    ### Feature Maximization Metrics.
    ###

    # Computation.
    fmc_computer: FeaturesMaximizationMetric = FeaturesMaximizationMetric(
        data_vectors=people_characteristics,
        data_classes=people_genders,
        list_of_possible_features=characteristics_studied,
    )

    ###
    ### Get with correct parameters.
    ###

    # Case with all, sort by fmeasure, limit to 1.
    assert fmc_computer.get_most_activated_classes_by_a_feature(
        feature="Shoes size",
        activation_only=False,
        sort_by="fmeasure",
        max_number=1,
    ) == ["Man"]
    assert fmc_computer.get_most_activated_classes_by_a_feature(
        feature="Hair size",
        activation_only=False,
        sort_by="fmeasure",
        max_number=1,
    ) == ["Woman"]
    assert fmc_computer.get_most_activated_classes_by_a_feature(
        feature="Nose size",
        activation_only=False,
        sort_by="fmeasure",
        max_number=1,
    ) == ["Man"]


# ==============================================================================
# test_FeaturesMaximizationMetric_get_most_active_features_by_a_classe_error_classe
# ==============================================================================


def test_FeaturesMaximizationMetric_get_most_active_features_by_a_classe_error_classe():
    """
    In `FeaturesMaximizationMetric`, test `get_most_active_features_by_a_classe` method with unknown classe.
    """

    ###
    ### Data.
    ###

    # Define people characteristics that will be studied.
    characteristics_studied: List[str] = [
        "Shoes size",
        "Hair size",
        "Nose size",
    ]

    # Get people characteristics.
    people_characteristics: csr_matrix = csr_matrix(
        [
            [9, 5, 5],
            [9, 10, 5],
            [9, 20, 6],
            [5, 15, 5],
            [6, 25, 6],
            [5, 25, 5],
        ]
    )

    # Get people genders.
    people_genders: List[str] = [
        "Man",
        "Man",
        "Man",
        "Woman",
        "Woman",
        "Woman",
    ]

    ###
    ### Feature Maximization Metrics.
    ###

    # Computation.
    fmc_computer: FeaturesMaximizationMetric = FeaturesMaximizationMetric(
        data_vectors=people_characteristics,
        data_classes=people_genders,
        list_of_possible_features=characteristics_studied,
    )

    ###
    ### Get with bad parameters.
    ###

    # Invalid case with unknown feature.
    with pytest.raises(
        ValueError,
        match=re.escape("The requested classe `'UNKNOWN'` is unknown."),
    ):
        fmc_computer.get_most_active_features_by_a_classe(
            classe="UNKNOWN",
        )


# ==============================================================================
# test_FeaturesMaximizationMetric_get_most_active_features_by_a_classe_error_sort_by
# ==============================================================================


def test_FeaturesMaximizationMetric_get_most_active_features_by_a_classe_error_sort_by():
    """
    In `FeaturesMaximizationMetric`, test `get_most_active_features_by_a_classe` method with unknown sort option.
    """

    ###
    ### Data.
    ###

    # Define people characteristics that will be studied.
    characteristics_studied: List[str] = [
        "Shoes size",
        "Hair size",
        "Nose size",
    ]

    # Get people characteristics.
    people_characteristics: csr_matrix = csr_matrix(
        [
            [9, 5, 5],
            [9, 10, 5],
            [9, 20, 6],
            [5, 15, 5],
            [6, 25, 6],
            [5, 25, 5],
        ]
    )

    # Get people genders.
    people_genders: List[str] = [
        "Man",
        "Man",
        "Man",
        "Woman",
        "Woman",
        "Woman",
    ]

    ###
    ### Feature Maximization Metrics.
    ###

    # Computation.
    fmc_computer: FeaturesMaximizationMetric = FeaturesMaximizationMetric(
        data_vectors=people_characteristics,
        data_classes=people_genders,
        list_of_possible_features=characteristics_studied,
    )

    ###
    ### Get with bad parameters.
    ###

    # Invalid case with invalid sort option.
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The sort option factor `sort_by` has to be in the following values: `{'contrast', 'fmeasure'}` (currently: 'UNKNOWN')."
        ),
    ):
        fmc_computer.get_most_active_features_by_a_classe(
            classe="Man",
            sort_by="UNKNOWN",
        )


# ==============================================================================
# test_FeaturesMaximizationMetric_get_most_active_features_by_a_classe_default
# ==============================================================================


def test_FeaturesMaximizationMetric_get_most_active_features_by_a_classe_default():
    """
    In `FeaturesMaximizationMetric`, test `get_most_active_features_by_a_classe` method.
    """

    ###
    ### Data.
    ###

    # Define people characteristics that will be studied.
    characteristics_studied: List[str] = [
        "Shoes size",
        "Hair size",
        "Nose size",
    ]

    # Get people characteristics.
    people_characteristics: csr_matrix = csr_matrix(
        [
            [9, 5, 5],
            [9, 10, 5],
            [9, 20, 6],
            [5, 15, 5],
            [6, 25, 6],
            [5, 25, 5],
        ]
    )

    # Get people genders.
    people_genders: List[str] = [
        "Man",
        "Man",
        "Man",
        "Woman",
        "Woman",
        "Woman",
    ]

    ###
    ### Feature Maximization Metrics.
    ###

    # Computation.
    fmc_computer: FeaturesMaximizationMetric = FeaturesMaximizationMetric(
        data_vectors=people_characteristics,
        data_classes=people_genders,
        list_of_possible_features=characteristics_studied,
    )

    ###
    ### Get with correct parameters.
    ###

    # Default case.
    assert fmc_computer.get_most_active_features_by_a_classe(
        classe="Man",
    ) == ["Shoes size"]
    assert fmc_computer.get_most_active_features_by_a_classe(
        classe="Woman",
    ) == ["Hair size"]


# ==============================================================================
# test_FeaturesMaximizationMetric_get_most_active_features_by_a_classe_only_activated_sorted_by_contrast
# ==============================================================================


def test_FeaturesMaximizationMetric_get_most_active_features_by_a_classe_only_activated_sorted_by_contrast():
    """
    In `FeaturesMaximizationMetric`, test `get_most_active_features_by_a_classe` method, only activated, sort by contrast.
    """

    ###
    ### Data.
    ###

    # Define people characteristics that will be studied.
    characteristics_studied: List[str] = [
        "Shoes size",
        "Hair size",
        "Nose size",
    ]

    # Get people characteristics.
    people_characteristics: csr_matrix = csr_matrix(
        [
            [9, 5, 5],
            [9, 10, 5],
            [9, 20, 6],
            [5, 15, 5],
            [6, 25, 6],
            [5, 25, 5],
        ]
    )

    # Get people genders.
    people_genders: List[str] = [
        "Man",
        "Man",
        "Man",
        "Woman",
        "Woman",
        "Woman",
    ]

    ###
    ### Feature Maximization Metrics.
    ###

    # Computation.
    fmc_computer: FeaturesMaximizationMetric = FeaturesMaximizationMetric(
        data_vectors=people_characteristics,
        data_classes=people_genders,
        list_of_possible_features=characteristics_studied,
    )

    ###
    ### Get with correct parameters.
    ###

    # Case with only activated, sort by contrast.
    assert fmc_computer.get_most_active_features_by_a_classe(
        classe="Man",
        activation_only=True,
        sort_by="contrast",
        max_number=None,
    ) == ["Shoes size"]
    assert fmc_computer.get_most_active_features_by_a_classe(
        classe="Woman",
        activation_only=True,
        sort_by="contrast",
        max_number=None,
    ) == ["Hair size"]


# ==============================================================================
# test_FeaturesMaximizationMetric_get_most_active_features_by_a_classe_sorted_by_contrast
# ==============================================================================


def test_FeaturesMaximizationMetric_get_most_active_features_by_a_classe_sorted_by_contrast():
    """
    In `FeaturesMaximizationMetric`, test `get_most_active_features_by_a_classe` method, sorted by contrast.
    """

    ###
    ### Data.
    ###

    # Define people characteristics that will be studied.
    characteristics_studied: List[str] = [
        "Shoes size",
        "Hair size",
        "Nose size",
    ]

    # Get people characteristics.
    people_characteristics: csr_matrix = csr_matrix(
        [
            [9, 5, 5],
            [9, 10, 5],
            [9, 20, 6],
            [5, 15, 5],
            [6, 25, 6],
            [5, 25, 5],
        ]
    )

    # Get people genders.
    people_genders: List[str] = [
        "Man",
        "Man",
        "Man",
        "Woman",
        "Woman",
        "Woman",
    ]

    ###
    ### Feature Maximization Metrics.
    ###

    # Computation.
    fmc_computer: FeaturesMaximizationMetric = FeaturesMaximizationMetric(
        data_vectors=people_characteristics,
        data_classes=people_genders,
        list_of_possible_features=characteristics_studied,
    )

    ###
    ### Get with correct parameters.
    ###

    # Case with all, sort by contrast.
    assert fmc_computer.get_most_active_features_by_a_classe(
        classe="Man",
        activation_only=False,
        sort_by="contrast",
        max_number=None,
    ) == ["Shoes size", "Hair size", "Nose size"]
    assert fmc_computer.get_most_active_features_by_a_classe(
        classe="Woman",
        activation_only=False,
        sort_by="contrast",
        max_number=None,
    ) == ["Hair size", "Shoes size", "Nose size"]


# ==============================================================================
# test_FeaturesMaximizationMetric_get_most_active_features_by_a_classe_sorted_by_fmeasure
# ==============================================================================


def test_FeaturesMaximizationMetric_get_most_active_features_by_a_classe_sorted_by_fmeasure():
    """
    In `FeaturesMaximizationMetric`, test `get_most_active_features_by_a_classe` method, sorted by fmeasure.
    """

    ###
    ### Data.
    ###

    # Define people characteristics that will be studied.
    characteristics_studied: List[str] = [
        "Shoes size",
        "Hair size",
        "Nose size",
    ]

    # Get people characteristics.
    people_characteristics: csr_matrix = csr_matrix(
        [
            [9, 5, 5],
            [9, 10, 5],
            [9, 20, 6],
            [5, 15, 5],
            [6, 25, 6],
            [5, 25, 5],
        ]
    )

    # Get people genders.
    people_genders: List[str] = [
        "Man",
        "Man",
        "Man",
        "Woman",
        "Woman",
        "Woman",
    ]

    ###
    ### Feature Maximization Metrics.
    ###

    # Computation.
    fmc_computer: FeaturesMaximizationMetric = FeaturesMaximizationMetric(
        data_vectors=people_characteristics,
        data_classes=people_genders,
        list_of_possible_features=characteristics_studied,
    )

    ###
    ### Get with correct parameters.
    ###

    # Case with all, sort by fmeasure.
    assert fmc_computer.get_most_active_features_by_a_classe(
        classe="Man",
        activation_only=False,
        sort_by="fmeasure",
        max_number=None,
    ) == ["Shoes size", "Hair size", "Nose size"]
    assert fmc_computer.get_most_active_features_by_a_classe(
        classe="Woman",
        activation_only=False,
        sort_by="fmeasure",
        max_number=None,
    ) == ["Hair size", "Nose size", "Shoes size"]


# ==============================================================================
# test_FeaturesMaximizationMetric_get_most_active_features_by_a_classe_sorted_by_fmeasure_limited_to_1
# ==============================================================================


def test_FeaturesMaximizationMetric_get_most_active_features_by_a_classe_sorted_by_fmeasure_limited_to_1():
    """
    In `FeaturesMaximizationMetric`, test `get_most_active_features_by_a_classe` method, sorted by fmeasure, limited to 1.
    """

    ###
    ### Data.
    ###

    # Define people characteristics that will be studied.
    characteristics_studied: List[str] = [
        "Shoes size",
        "Hair size",
        "Nose size",
    ]

    # Get people characteristics.
    people_characteristics: csr_matrix = csr_matrix(
        [
            [9, 5, 5],
            [9, 10, 5],
            [9, 20, 6],
            [5, 15, 5],
            [6, 25, 6],
            [5, 25, 5],
        ]
    )

    # Get people genders.
    people_genders: List[str] = [
        "Man",
        "Man",
        "Man",
        "Woman",
        "Woman",
        "Woman",
    ]

    ###
    ### Feature Maximization Metrics.
    ###

    # Computation.
    fmc_computer: FeaturesMaximizationMetric = FeaturesMaximizationMetric(
        data_vectors=people_characteristics,
        data_classes=people_genders,
        list_of_possible_features=characteristics_studied,
    )

    ###
    ### Get with correct parameters.
    ###

    # Case with all, sort by fmeasure, limit to 1.
    assert fmc_computer.get_most_active_features_by_a_classe(
        classe="Man",
        activation_only=False,
        sort_by="fmeasure",
        max_number=1,
    ) == ["Shoes size"]
    assert fmc_computer.get_most_active_features_by_a_classe(
        classe="Woman",
        activation_only=False,
        sort_by="fmeasure",
        max_number=1,
    ) == ["Hair size"]


# ==============================================================================
# test_FeaturesMaximizationMetric_compare_itself
# ==============================================================================


def test_FeaturesMaximizationMetric_compare_itself():
    """
    In `FeaturesMaximizationMetric`, test `compare` method on itself.
    """

    ###
    ### Data.
    ###

    # Define people characteristics that will be studied.
    characteristics_studied: List[str] = [
        "Shoes size",
        "Hair size",
        "Nose size",
    ]

    # Get people characteristics.
    people_characteristics: csr_matrix = csr_matrix(
        [
            [9, 5, 5],
            [9, 10, 5],
            [9, 20, 6],
            [5, 15, 5],
            [6, 25, 6],
            [5, 25, 5],
        ]
    )

    # Get people genders.
    people_genders: List[str] = [
        "Man",
        "Man",
        "Man",
        "Woman",
        "Woman",
        "Woman",
    ]

    ###
    ### Feature Maximization Metrics.
    ###

    # Computation.
    fmc_computer: FeaturesMaximizationMetric = FeaturesMaximizationMetric(
        data_vectors=people_characteristics,
        data_classes=people_genders,
        list_of_possible_features=characteristics_studied,
    )

    ###
    ### Comparison.
    ###

    assert fmc_computer.compare(
        fmc_reference=fmc_computer,
    ) == (1.0, 1.0, 1.0)


# ==============================================================================
# test_FeaturesMaximizationMetric_compare_error_list_of_possible_features
# ==============================================================================


def test_FeaturesMaximizationMetric_compare_error_list_of_possible_features():
    """
    In `FeaturesMaximizationMetric`, test `compare` method with error in list_of_possible_features.
    """

    ###
    ### Data.
    ###

    # Define people characteristics that will be studied.
    characteristics_studied_1: List[str] = [
        "Shoes size",
        "Hair size",
        "Nose size",
    ]
    characteristics_studied_2: List[str] = [
        "Shoes size",
        "Hair size",
        "Tongue size",
        "Ear size",
    ]

    # Get people characteristics.
    people_characteristics_1: csr_matrix = csr_matrix(
        [
            [9, 5, 5],
            [9, 10, 5],
            [9, 20, 6],
            [5, 15, 5],
            [6, 25, 6],
            [5, 25, 5],
        ]
    )
    people_characteristics_2: csr_matrix = csr_matrix(
        [
            [9, 5, 4, 7],
            [9, 10, 3, 6],
            [9, 20, 5, 7],
            [5, 15, 5, 7],
            [6, 25, 4, 7],
            [5, 25, 3, 6],
        ]
    )

    # Get people genders.
    people_genders: List[str] = [
        "Man",
        "Man",
        "Man",
        "Woman",
        "Woman",
        "Woman",
    ]

    ###
    ### Feature Maximization Metrics.
    ###

    # Computation.
    fmc_computer_1: FeaturesMaximizationMetric = FeaturesMaximizationMetric(
        data_vectors=people_characteristics_1,
        data_classes=people_genders,
        list_of_possible_features=characteristics_studied_1,
    )

    # Computation.
    fmc_computer_2: FeaturesMaximizationMetric = FeaturesMaximizationMetric(
        data_vectors=people_characteristics_2,
        data_classes=people_genders,
        list_of_possible_features=characteristics_studied_2,
    )

    ###
    ### Comparison.
    ###

    # Invalid case with inconsistencies in list_of_possible_features.
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The list of features `list_of_possible_features` must be the same for both FMC modelization. +: ['Nose size'], -: ['Tongue size', 'Ear size']"
        ),
    ):
        fmc_computer_1.compare(
            fmc_reference=fmc_computer_2,
        )


# ==============================================================================
# test_FeaturesMaximizationMetric_compare_default
# ==============================================================================


def test_FeaturesMaximizationMetric_compare_default():
    """
    In `FeaturesMaximizationMetric`, test `compare` method.
    """

    ###
    ### Data.
    ###

    # Define people characteristics that will be studied.
    characteristics_studied: List[str] = [
        "Shoes size",
        "Hair size",
        "Nose size",
    ]

    # Get people characteristics.
    people_characteristics: csr_matrix = csr_matrix(
        [
            [9, 5, 5],
            [9, 10, 5],
            [9, 10, 6],
            [9, 15, 6],
            [9, 20, 6],
            [5, 15, 5],
            [6, 20, 6],
            [5, 25, 5],
            [6, 25, 5],
            [5, 25, 5],
            [3, 5, 15],
            [4, 10, 15],
            [2, 5, 15],
            [3, 10, 14],
            [3, 15, 14],
        ]
    )

    # Get people genders.
    people_genders_1: List[str] = [
        "Man",
        "Man",
        "Man",
        "Man",
        "Man",
        "Woman",
        "Woman",
        "Woman",
        "Woman",
        "Woman",
        "Child",
        "Child",
        "Child",
        "Child",
        "Child",
    ]
    people_genders_2: List[str] = [
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "1",
        "1",
        "1",
        "1",
        "1",
    ]

    ###
    ### Feature Maximization Metrics.
    ###

    # Computation.
    fmc_reference: FeaturesMaximizationMetric = FeaturesMaximizationMetric(
        data_vectors=people_characteristics,
        data_classes=people_genders_1,
        list_of_possible_features=characteristics_studied,
    )

    # Computation.
    fmc_computer: FeaturesMaximizationMetric = FeaturesMaximizationMetric(
        data_vectors=people_characteristics,
        data_classes=people_genders_2,
        list_of_possible_features=characteristics_studied,
    )

    ###
    ### Comparison.
    ###

    # Default case.
    assert fmc_computer.compare(
        fmc_reference=fmc_reference,
    ) == (0.5793801642856952, 1.0, 0.7336804366512111)


# ==============================================================================
# test_FeaturesMaximizationMetric_compare_rounded
# ==============================================================================


def test_FeaturesMaximizationMetric_compare_rounded():
    """
    In `FeaturesMaximizationMetric`, test `compare` method with rounded option.
    """

    ###
    ### Data.
    ###

    # Define people characteristics that will be studied.
    characteristics_studied: List[str] = [
        "Shoes size",
        "Hair size",
        "Nose size",
    ]

    # Get people characteristics.
    people_characteristics: csr_matrix = csr_matrix(
        [
            [9, 5, 5],
            [9, 10, 5],
            [9, 10, 6],
            [9, 15, 6],
            [9, 20, 6],
            [5, 15, 5],
            [6, 20, 6],
            [5, 25, 5],
            [6, 25, 5],
            [5, 25, 5],
            [3, 5, 15],
            [4, 10, 15],
            [2, 5, 15],
            [3, 10, 14],
            [3, 15, 14],
        ]
    )

    # Get people genders.
    people_genders_1: List[str] = [
        "Man",
        "Man",
        "Man",
        "Man",
        "Man",
        "Woman",
        "Woman",
        "Woman",
        "Woman",
        "Woman",
        "Child",
        "Child",
        "Child",
        "Child",
        "Child",
    ]
    people_genders_2: List[str] = [
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "1",
        "1",
        "1",
        "1",
        "1",
    ]

    ###
    ### Feature Maximization Metrics.
    ###

    # Computation.
    fmc_reference: FeaturesMaximizationMetric = FeaturesMaximizationMetric(
        data_vectors=people_characteristics,
        data_classes=people_genders_1,
        list_of_possible_features=characteristics_studied,
    )

    # Computation.
    fmc_computer: FeaturesMaximizationMetric = FeaturesMaximizationMetric(
        data_vectors=people_characteristics,
        data_classes=people_genders_2,
        list_of_possible_features=characteristics_studied,
    )

    ###
    ### Comparison.
    ###

    # Rounded case.
    assert fmc_computer.compare(
        fmc_reference=fmc_reference,
        rounded=4,
    ) == (0.5794, 1.0, 0.7337)
