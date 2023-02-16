# -*- coding: utf-8 -*-

"""
* Name:         cognitivefactory.features_maximization_metric.fmc
* Description:  Implementation of Features Maximization Metrics.
* Author:       Erwan SCHILD
* Created:      23/11/2022
* Licence:      CeCILL-C License v1.0 (https://cecill.info/licences.fr.html)
"""

# ==============================================================================
# IMPORTS :
# ==============================================================================

from typing import Dict, List, Literal, Optional, Tuple

from scipy.sparse import csr_matrix
from sklearn.metrics.cluster import homogeneity_completeness_v_measure

# ==============================================================================
# FEATURES MAXIMIZATION METRIC
# ==============================================================================


class FeaturesMaximizationMetric:
    r"""
    This class implements the ***Features Maximization Metric***.
    It's a dataset modelization based on vectors features and data labels:
    for each couple `(feature, classe)`, it gives a score (called **F-Measure**) that describe the power of identification and distinction of the feature for this classe.

    This metric is computed by applying the following steps:

        1. Compute the ***Features F-Measure*** metric (based on ***Features Recall*** and ***Features Predominance*** metrics).

            > (a) The ***Features Recall*** `FR[f][c]` for a given class `c` and a given feature `f` is the ratio between
            > the sum of the vectors weights of the feature `f` for data in class `c`
            > and the sum of all vectors weights of feature `f` for all data.
            > It answers the question: "_Can the feature `f` distinguish the class `c` from other classes `c'` ?_"

            > (b) The ***Features Predominance*** `FP[f][c]` for a given class `c` and a given feature `f` is the ratio between
            > the sum of the vectors weights of the feature `f` for data in class `c`
            > and the sum of all vectors weights of all feature `f'` for data in class `c`.
            > It answers the question: "_Can the feature `f` better identify the class `c` than the other features `f'` ?_"

            > (c) The ***Features F-Measure*** `FM[f][c]` for a given class `c` and a given feature `f` is
            > the harmonic mean of the ***Features Recall*** (a) and the ***Features Predominance*** (c).
            > It answers the question: "_How much information does the feature `f` contain about the class `c` ?_"

        2. Compute the ***Features Selection*** (based on ***F-Measure Overall Average*** comparison).

            > (d) The ***F-Measure Overall Average*** is the average of ***Features F-Measure*** (c) for all classes `c` and for all features `f`.
            > It answers the question: "_What are the mean of information contained by features in all classes ?_"

            > (e) A feature `f` is ***Selected*** if and only if it exist at least one class `c` for which the ***Features F-Measure*** (c) `FM[f][c]` is bigger than the ***F-Measure Overall Average*** (d).
            > It answers the question: "_What are the features which contain more information than the mean of information in the dataset ?_"

            > (f) A Feature `f` is ***Deleted*** if and only if the ***Features F-Measure*** (c) `FM[f][c]` is always lower than the ***F-Measure Overall Average*** (d) for each class `c`.
            > It answers the question: "_What are the features which do not contain more information than the mean of information in the dataset ?_"

        3. Compute the ***Features Contrast*** and ***Features Activation*** (based on ***F-Measure Marginal Averages*** comparison).

            > (g) The ***F-Measure Marginal Averages*** for a given feature `f` is the average of ***Features F-Measure*** (c) for all classes `c` and for the given feature `f`.
            > It answers the question: "_What are the mean of information contained by the feature `f` in all classes ?_"

            > (h) The ***Features Contrast*** `FC[f][c]` for a given class `c` and a given selected feature `f` is the ratio between
            > the ***Features F-Measure*** (c) `FM[f][c]`
            > and the ***F-Measure Marginal Averages*** (g) for selected feature f
            > put to the power of an ***Amplification Factor***.
            > It answers the question: "_How relevant is the feature `f` to distinguish the class `c` ?_"

            > (i) A selected Feature `f` is ***Active*** for a given class `c` if and only if the ***Features Contrast*** (h) `FC[f][c]` is bigger than `1.0`.
            > It answers the question : "_For which classes a selected feature `f` is relevant ?_"

    In order to ***evaluate it according to a reference***, a FMC modelization is represented by the Features Activation of its vector features,
    and a similarity score to the reference is computed, based on common metrics on clustering (homogeneity, completeness, v_measure).

    Attributes:
        data_vectors (csr_matrix): The sparse matrix representing the vector of each data (i.e. `data_vectors[d,f]` is the weight of data `d` for feature `f`).
        data_classes (List[str]): The list representing the class of each data (i.e. `data_classes[d]` is the class of data `d`).
        list_of_possible_features (List[str]): The list of existing vectors features.
        list_of_possible_classes (List[str]):  The list of existing data classes.
        amplification_factor (int): The positive integer called "amplification factor" aimed at emphasize the feature contrast. Usually at `1`.
        features_frecall (Dict[str, Dict[str, float]]): The computation of *Features Recall* (_Can the feature `f` distinguish the class `c` from other classes `l'` ?_).
        features_fpredominance (Dict[str, Dict[str, float]]): The computation of *Features Predominance* (_Can the feature `f` better identify the class `c` than the other features `f'` ?_).
        features_fmeasure (Dict[str, Dict[str, float]]): The computation of *Features F-Measure* (_How much information does the feature `f` contain about the class `c` ?_).
        features_overall_average (float): The computation of *Overall Average of Features F-Measure* (_What are the mean of information contained by features in all classes ?_).
        features_selection (Dict[str, bool]): The computation of *Features Selected* (_What are the features which contain more information than the mean of information in the dataset ?_).
        features_marginal_averages (Dict[str, float]):  The computation of *Marginal Averages of Features F-Measure* (_What are the mean of information contained by the feature `f` in all classes ?_).
        features_contrast (Dict[str, Dict[str, float]]): The computation of *Features Contrast* (_How important is the feature `f` to distinguish the class `c` ?_).
        features_activation (Dict[str, Dict[str, bool]]): The computation of *Features Activation* (_For which classes a selected feature `f` is relevant ?_).

    Example:
        - Basic usecase: "_What are the physical characteristics that most distinguish men from women ?_"
        ```python

        # Problem to solve.
        print(">> What are the physical characteristics that most distinguish men from women ?")

        ###
        ### Python dependencies.
        ###

        from cognitivefactory.features_maximization_metric.fmc import FeaturesMaximizationMetric
        from scipy.sparse import csr_matrix
        from typing import List

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

        # Main computation.
        fmc_computer: FeaturesMaximizationMetric = FeaturesMaximizationMetric(
            data_vectors=people_characteristics,
            data_classes=people_genders,
            list_of_possible_features=characteristics_studied,
            amplification_factor=1,
        )

        ###
        ### Analysis 1: Delete characteristics that aren't relevant.
        ###

        print(
            "\n",
            "1. Which characteristic seems not relevant to distinguish men from women ?",
        )
        for characteristic in characteristics_studied:
            if not fmc_computer.features_selection[characteristic]:
                print(
                    "    - '{0}' seems not relevant.".format(characteristic)
                )

        ###
        ### Analysis 2: Describe gender by relevant characteristics.
        ###

        print(
            "\n",
            "2. According to remaining characteristics:",
        )
        for gender in sorted(set(people_genders)):
            print(
                "    - Which characteristic seems important to recognize a '{0}' ?".format(gender)
            )

            for characteristic in fmc_computer.get_most_active_features_by_a_classe(
                classe=gender,
            ):
                print(
                    "        - '{0}' seems important (fmeasure of '{1:.2f}', contrast of '{2:.2f}').".format(
                        characteristic,
                        fmc_computer.features_fmeasure[characteristic][gender],
                        fmc_computer.features_contrast[characteristic][gender],
                    )
                )
        ```

    References:
        - Features Maximization Metric: `Lamirel J.-C., Cuxac P., Hajlaoui K., A new approach for feature selection based on quality metric, Advances in Knowledge Discovery and Management, 6 (665), Springer.`
    """

    # =========================================================================================
    # INITIALIZATION
    # =========================================================================================

    def __init__(
        self,
        data_vectors: csr_matrix,
        data_classes: List[str],
        list_of_possible_features: List[str],
        amplification_factor: int = 1,
        verbose: bool = False,
    ):
        """
        The constructor for `FeaturesMaximizationMetric` class.
        It applies the several steps of ***Feature Maximization***:
            1. Compute the ***Features F-Measure*** metric (based on ***Features Recall*** and ***Features Predominance*** metrics).
            2. Compute the ***Features Selection*** (based on ***F-Measure Overall Average*** comparison).
            3. Compute the ***Features Contrast*** and ***Features Activation*** (based on ***F-Measure Marginal Averages*** comparison).

        Args:
            data_vectors (scipy.sparse.csr_matrix): A sparse matrix representing the vector of each data (i.e. `data_vectors[d,f]` is the weight of data `d` for feature `f`).
            data_classes (List[str]): A list representing the class of each data (i.e. `data_classes[d]` is the class of data `d`).
            list_of_possible_features (List[str]): A list of existing vectors features.
            amplification_factor (int, optional): A positive integer called "amplification factor" aimed at emphasize the feature contrast. Defaults to `1`.
            verbose (bool): An option to display progress status of computations. Defaults to `False`.

        Raises:
            ValueError: if `data_vectors` and `data_classes` have inconsistent shapes.
            ValueError: if `data_vectors` and `list_of_possible_features` have inconsistent shapes.
            ValueError: if `amplification_factor` is not a positive integer.
        """

        ###
        ### Check parameters.
        ###

        # Display progress status if requested.
        if verbose:
            print("`FeaturesMaximizationMetric.__init__`", ":", "Check parameters.")

        # Check data size.
        if data_vectors.shape[0] != len(data_classes):
            raise ValueError(
                "The vectors `data_vectors` and the list of classes `data_classes` have inconsistent shapes (currently: '{0}' vs '{1}').".format(
                    data_vectors.shape[0],
                    len(data_classes),
                )
            )

        # Check features size.
        if data_vectors.shape[1] != len(list_of_possible_features):
            raise ValueError(
                "The vectors `data_vectors` and the list of features `list_of_possible_features` have inconsistent shapes (currently: '{0}' vs '{1}').".format(
                    data_vectors.shape[1],
                    len(list_of_possible_features),
                )
            )

        # Check amplification factor.
        if (not isinstance(amplification_factor, int)) or amplification_factor < 1:
            raise ValueError(
                "The amplification factor `amplification_factor` has to be a positive integer (currently: '{0}').".format(
                    amplification_factor,
                )
            )

        ###
        ### Store parameters.
        ###

        # Display progress status if requested.
        if verbose:
            print("`FeaturesMaximizationMetric.__init__`", ":", "Store parameters.")

        # Store data information.
        self.data_vectors: csr_matrix = data_vectors
        self.data_classes: List[str] = data_classes
        # Store features and classes lists.
        self.list_of_possible_features: List[str] = list_of_possible_features
        self.list_of_possible_classes: List[str] = sorted(set(data_classes))
        # Store amplification factor.
        self.amplification_factor: int = amplification_factor

        ###
        ### Compute Features Maximization Metric.
        ###

        # Display progress status if requested.
        if verbose:
            print("`FeaturesMaximizationMetric.__init__`", ":", "Start computations.")

        # 1. Compute the *Features F-Measure* metric (based on *Features Recall* and *Features Predominance* metrics).

        # Display progress status if requested.
        if verbose:
            print("`FeaturesMaximizationMetric.__init__`", ":", "Compute Features F-Measure.")

        # Initialize variables.
        self.features_frecall: Dict[str, Dict[str, float]]
        self.features_fpredominance: Dict[str, Dict[str, float]]
        self.features_fmeasure: Dict[str, Dict[str, float]]
        # Compute variables.
        self._compute_features_frecall_fpredominance_fmeasure()

        # 2. Perform a *Features Selection* (based on *F-Measure Overall Average* comparison).

        # Display progress status if requested.
        if verbose:
            print("`FeaturesMaximizationMetric.__init__`", ":", "Compute Features Selection.")

        # Initialize variables.
        self.features_overall_average: float
        self.features_selection: Dict[str, bool]
        # Compute variables.
        self._compute_features_selection()

        # 3. Compute the *Features Contrast* and *Features Activation* (based on *F-Measure Marginal Averages* comparison).

        # Display progress status if requested.
        if verbose:
            print("`FeaturesMaximizationMetric.__init__`", ":", "Compute Features Contrast.")

        # Initialize variables.
        self.features_marginal_averages: Dict[str, float]
        self.features_contrast: Dict[str, Dict[str, float]]
        self.features_activation: Dict[str, Dict[str, bool]]
        # Compute variables.
        self._compute_features_contrast_and_activation()

        # Display progress status if requested.
        if verbose:
            print("`FeaturesMaximizationMetric.__init__`", ":", "Computations done.")

    # ==============================================================================
    # COMPUTE FEATURES F-MEASURE
    # ==============================================================================

    def _compute_features_frecall_fpredominance_fmeasure(
        self,
    ) -> None:
        """
        Compute:
            (a) the ***Features Recall*** (cf. `self.features_frecall`),
            (b) the ***Features Predominance*** (cf. `self.features_fpredominance`), and
            (c) the ***Features F-Measure*** (cf. `self.features_fmeasure`).
        """

        ###
        ### Temporary computations.
        ###

        # Temporary variable used to store sums of all vectors weights for a given feature `f` and a given class `c`.
        # Needed for both Features Recall and Features Predominance computations.
        sum_by_feature_and_classe: Dict[str, Dict[str, float]] = {
            feature: {classe: 0.0 for classe in self.list_of_possible_classes}
            for feature in self.list_of_possible_features
        }

        # Temporary variable used to store sums of all vectors weights for a given feature `f` and all classes.
        # Needed for Features Recall computation.
        sum_by_features: Dict[str, float] = {feature: 0.0 for feature in self.list_of_possible_features}

        # Temporary variable used to store sums of all vectors weights for all features and a given class `c`.
        # Needed for Features Predominance computation.
        sum_by_classe: Dict[str, float] = {classe: 0.0 for classe in self.list_of_possible_classes}

        # Index used to get non zero elements in the sparse matrix weights.
        indices_x, indices_y = self.data_vectors.nonzero()

        # Browse non zero weights in vectors to compute all the needed sums.
        for index in range(self.data_vectors.nnz):
            # Get needed information (data, class/ classe, feature, vectors weight)
            data_index: int = indices_x[index]
            data_classe: str = self.data_classes[data_index]
            feature_index: int = indices_y[index]
            data_feature: str = self.list_of_possible_features[feature_index]
            weight: float = self.data_vectors[data_index, feature_index]  # TODO: check if np.nan ?

            # Update the several sums.
            sum_by_feature_and_classe[data_feature][data_classe] += weight
            sum_by_features[data_feature] += weight
            sum_by_classe[data_classe] += weight

        ###
        ### Features F-Measure computation.
        ###

        # Compute Features Recall.
        self.features_frecall = {
            feature: {
                classe: (
                    0.0  # TODO: set to np.nan ?
                    if sum_by_features[feature] == 0
                    else sum_by_feature_and_classe[feature][classe] / sum_by_features[feature]
                )
                for classe in self.list_of_possible_classes
            }
            for feature in self.list_of_possible_features
        }

        # Compute Features Predominance.
        self.features_fpredominance = {
            feature: {
                classe: (
                    0.0  # TODO: set to np.nan ?
                    if sum_by_classe[classe] == 0
                    else sum_by_feature_and_classe[feature][classe] / sum_by_classe[classe]
                )
                for classe in self.list_of_possible_classes
            }
            for feature in self.list_of_possible_features
        }

        # Compute Features F-Measure.
        self.features_fmeasure = {
            feature: {
                classe: (
                    0.0  # TODO: set to np.nan ?
                    if (self.features_frecall[feature][classe] + self.features_fpredominance[feature][classe] == 0)
                    else (
                        2
                        * (self.features_frecall[feature][classe] * self.features_fpredominance[feature][classe])
                        / (self.features_frecall[feature][classe] + self.features_fpredominance[feature][classe])
                    )
                )
                for classe in self.list_of_possible_classes
            }
            for feature in self.list_of_possible_features
        }

    # =============================================================================================
    # COMPUTE FEATURES SELECTION
    # =============================================================================================

    def _compute_features_selection(
        self,
    ) -> None:
        """
        Compute:
            (d) the ***F-Measure Overall Average*** (cf. `self.features_overall_average`), and
            (e) the ***Features Selected*** (cf. `self.features_selection`).
        """

        ###
        ### Features F-Measure Overall Average computation.
        ###

        # Temporary variable used to store the overall sum in order to compute the overall average of Features F-Measure.
        overall_sum: float = 0.0
        nb_overall: int = 0

        # For each feature...
        for feature1 in self.list_of_possible_features:
            # For each classe...
            for classe1 in self.list_of_possible_classes:
                # Update the overall sum and count.
                overall_sum += self.features_fmeasure[feature1][classe1]
                nb_overall += 1

        # Compute the overall average of Features F-Measure.
        self.features_overall_average = 0.0 if nb_overall == 0 else overall_sum / nb_overall  # TODO: set to np.nan ?

        ###
        ### Features Selection computation.
        ###

        # Temporary variable used store the selected features.
        self.features_selection = {}

        # Browse features to determine the selected ones.
        for feature2 in self.list_of_possible_features:
            # Set default state of selection.
            self.features_selection[feature2] = False

            # For each feature, browse class to find one for which the Features F-Measure is bigger than the overall average.
            for classe2 in self.list_of_possible_classes:
                # Check that the Feature F-Measure is bigger than the overall average.
                if self.features_fmeasure[feature2][classe2] > self.features_overall_average:
                    # Approve the selection and then break the loop.
                    self.features_selection[feature2] = True
                    break

    # =============================================================================================
    # COMPUTE FEATURES CONSTRAST AND ACTIVATION
    # =============================================================================================

    def _compute_features_contrast_and_activation(
        self,
    ) -> None:
        """
        Compute:
            (g) The ***F-Measure Marginal Averages*** (cf. `self.features_marginal_averages`), and
            (h) The ***Features Contrast*** (cf. `self.features_contrast`).
            (i) the ***Features Activation*** (cf. `self.features_activation`).
        """

        ###
        ### Features F-Measure Marginal computation.
        ###

        # Initialize the marginal average of Features F-Measure.
        self.features_marginal_averages = {}

        # Browse features to compute the averages.
        for feature1 in self.list_of_possible_features:
            # Temporary variable used to store the marginal sum in order to compute the marginal average of Features F-Measure over the current feature.
            sum_marginal: float = 0.0
            nb_marginal: int = 0

            # Update the marginal sum of Features F-Measure over the current feature.
            for classe1 in self.list_of_possible_classes:
                sum_marginal += self.features_fmeasure[feature1][classe1]
                nb_marginal += 1

            # Compute the marginal averages of Features F-Measure over the current feature.
            self.features_marginal_averages[feature1] = (
                0.0 if nb_marginal == 0 else sum_marginal / nb_marginal
            )  # TODO: set to np.nan ?

        ###
        ### Features Contrast computation.
        ###

        # Temporary variable used to store the contrast of a feature for a class.
        self.features_contrast = {
            feature2: {
                classe2: (
                    0.0  # TODO: set to np.nan ?
                    if (self.features_selection[feature2] is False or self.features_marginal_averages[feature2] == 0)
                    else (self.features_fmeasure[feature2][classe2] / self.features_marginal_averages[feature2])
                    ** self.amplification_factor
                )
                for classe2 in self.list_of_possible_classes
            }
            for feature2 in self.list_of_possible_features
        }

        ###
        ### Features Activation computation.
        ###

        # Temporary variable used store the features activation.
        self.features_activation = {
            feature3: {
                classe3: bool(
                    self.features_selection[feature3] is True and self.features_contrast[feature3][classe3] > 1
                )
                for classe3 in self.list_of_possible_classes
            }
            for feature3 in self.list_of_possible_features
        }

    # =============================================================================================
    # GET: MOST ACTIVATED CLASSES FOR A FEATURE
    # =============================================================================================

    def get_most_activated_classes_by_a_feature(
        self,
        feature: str,
        activation_only: bool = True,
        sort_by: Literal["contrast", "fmeasure"] = "contrast",
        max_number: Optional[int] = None,
    ) -> List[str]:
        """
        Get the list of classes for which the requested feature is the most relevant.

        Args:
            feature (str): The feature to analyze.
            sort_by (Literal["contrast", "fmeasure"]): The sort criterion for the list of classes. Defaults to `"contrast"`.
            activation_only (bool): The option to get only activated classes. Defaults to `True`.
            max_number (Optional[int]): The maximum number of classes to return. Defaults to `None`.

        Raises:
            ValueError: if `feature` is not in `self.list_of_possible_features`.
            ValueError: if `sort_by` is not in `{"contrast", "fmeasure"}`.

        Returns:
            List[str]: The list of classes for which the requested feature is the most relevant.
        """

        ###
        ### Check parameters.
        ###

        # Check parameter `feature`.
        if feature not in self.list_of_possible_features:
            raise ValueError(
                "The requested feature `'{0}'` is unknown.".format(
                    feature,
                )
            )

        # Check parameter `sort_by`.
        if sort_by not in {"contrast", "fmeasure"}:
            raise ValueError(
                "The sort option factor `sort_by` has to be in the following values: `{{'contrast', 'fmeasure'}}` (currently: '{0}').".format(
                    sort_by,
                )
            )

        ###
        ### Compute the requested list.
        ###

        # Define list of possible results (classe + contrast/fmeasure).
        list_of_possible_results: List[Tuple[float, str]] = [
            (
                # 0: the metric: contrast or fmeasure.
                (
                    self.features_contrast[feature][classe]
                    if sort_by == "contrast"
                    else self.features_fmeasure[feature][classe]
                ),
                # 1: the classe.
                classe,
            )
            for classe in self.list_of_possible_classes
            if (activation_only is False or self.features_activation[feature][classe] is True)
        ]

        # Return top classes sorted by requested metric.
        return [
            activated_classe
            for _, activated_classe in sorted(
                list_of_possible_results,
                reverse=True,
            )
        ][:max_number]

    # =============================================================================================
    # GET: MOST ACTIVATED FEATURES FOR A CLASSE
    # =============================================================================================

    def get_most_active_features_by_a_classe(
        self,
        classe: str,
        activation_only: bool = True,
        sort_by: Literal["contrast", "fmeasure"] = "contrast",
        max_number: Optional[int] = None,
    ) -> List[str]:
        """
        Get the list of features which are the most relevant for the requested classe.

        Args:
            classe (str): The classe to analyze.
            sort_by (Literal["contrast", "fmeasure"]): The sort criterion for the list of features. Defaults to `"contrast"`.
            activation_only (bool): The option to get only active features. Defaults to `True`.
            max_number (Optional[int]): The maximum number of features to return. Defaults to `None`.

        Raises:
            ValueError: if `classe` is not in `self.list_of_possible_classes`.
            ValueError: if `sort_by` is not in `{"contrast", "fmeasure"}`.

        Returns:
            List[str]: The list of features which are the most relevant for the requested classe.
        """

        ###
        ### Check parameters.
        ###

        # Check parameter `feature`.
        if classe not in self.list_of_possible_classes:
            raise ValueError(
                "The requested classe `'{0}'` is unknown.".format(
                    classe,
                )
            )

        # Check parameter `sort_by`.
        if sort_by not in {"contrast", "fmeasure"}:
            raise ValueError(
                "The sort option factor `sort_by` has to be in the following values: `{{'contrast', 'fmeasure'}}` (currently: '{0}').".format(
                    sort_by,
                )
            )

        ###
        ### Compute the requested list.
        ###

        # Define list of possible results (feature + contrast/fmeasure).
        list_of_possible_results: List[Tuple[float, str]] = [
            (
                # 0: the metric: contrast or fmeasure.
                (
                    self.features_contrast[feature][classe]
                    if sort_by == "contrast"
                    else self.features_fmeasure[feature][classe]
                ),
                # 1: the feature.
                feature,
            )
            for feature in self.list_of_possible_features
            if (activation_only is False or self.features_activation[feature][classe] is True)
        ]

        # Return top features sorted by requested metric.
        return [
            active_feature
            for _, active_feature in sorted(
                list_of_possible_results,
                reverse=True,
            )
        ][:max_number]

    # =============================================================================================
    # COMPARE: WITH AN OTHER FMC
    # =============================================================================================

    def compare(
        self,
        fmc_reference: "FeaturesMaximizationMetric",
        rounded: Optional[int] = None,
    ) -> Tuple[float, float, float]:
        """
        Gives a similarity score in agreement with an reference FMC modelization.
        The similarity score computation is based on common metrics on clustering (homogeneity, completeness, v_measure),
        where each FMC modelization is represented by the Features Activation of their vector features.
        In order to be able to compute these similarity, data classes can be different, but vector features must be the same in both FMC modelization.


        Args:
            fmc_reference (FeaturesMaximizationMetric): Another Features Maximization modelization used as reference for the comparison.
            rounded (Optional[int]): The option to round the result to counter log approximation. Defaults to `None`.

        Raises:
            ValueError: if `list_of_possible_features` are different.

        Returns:
            Tuple[float, float, float]: Computation
        """

        ###
        ### Check parameters.
        ###

        # Check list_of_possible_features equality.
        if self.list_of_possible_features != fmc_reference.list_of_possible_features:
            list_of_in_excess_features: List[str] = [
                feature
                for feature in self.list_of_possible_features
                if feature not in fmc_reference.list_of_possible_features
            ]
            list_of_missing_features: List[str] = [
                feature
                for feature in fmc_reference.list_of_possible_features
                if feature not in self.list_of_possible_features
            ]
            raise ValueError(
                "The list of features `list_of_possible_features` must be the same for both FMC modelization. +: {0}, -: {1}".format(
                    str(list_of_in_excess_features), str(list_of_missing_features)
                )
            )

        ###
        ### Format Features Activation as classification label of features.
        ###

        # Initialize
        list_of_self_features_activations: List[str] = []
        list_of_reference_features_activations: List[str] = []

        # Define default value if feature not activated.
        # NB: we can't set a fixed value in case this value is in the list of possible classes...
        # Example: can't set `""` or `"None"` in case self.list_of_possible_classes==["A", ""] and fmc_reference.list_of_possible_classes==["B", "None"].
        default_label_if_feature_not_activated: str = "NOT_ACTIVATED:{possible_classe}".format(
            possible_classe=self.list_of_possible_classes + fmc_reference.list_of_possible_classes
        )

        # Browse activated features toà compare Features Activation.
        for feature in fmc_reference.list_of_possible_features:
            # Get Features Activation.
            list_of_most_activated_classes_for_feature_in_self: List[
                str
            ] = self.get_most_activated_classes_by_a_feature(
                feature=feature,
            )
            list_of_most_activated_classes_for_feature_in_reference: List[
                str
            ] = fmc_reference.get_most_activated_classes_by_a_feature(
                feature=feature,
            )

            # TODO: Skip if feature is not activated in both modelization.
            if (
                len(list_of_most_activated_classes_for_feature_in_self) != 1
                and len(list_of_most_activated_classes_for_feature_in_reference) != 1
            ):
                continue

            # Format Feature Activation as classification label. Set to `-1` if not activated.
            list_of_self_features_activations.append(
                list_of_most_activated_classes_for_feature_in_self[0]
                if len(list_of_most_activated_classes_for_feature_in_self) == 1
                else default_label_if_feature_not_activated
            )
            list_of_reference_features_activations.append(
                list_of_most_activated_classes_for_feature_in_reference[0]
                if len(list_of_most_activated_classes_for_feature_in_reference) == 1
                else default_label_if_feature_not_activated
            )

        ###
        ### Compute FMC modelizations similarity.
        ###

        # Compute standard metrics for clustering.
        homogeneity: float
        completeness: float
        v_measure: float
        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
            labels_pred=list_of_self_features_activations,
            labels_true=list_of_reference_features_activations,
        )

        # Round the results.
        if rounded is not None:
            homogeneity = round(homogeneity, rounded)
            completeness = round(completeness, rounded)
            v_measure = round(v_measure, rounded)

        # Return values.
        return homogeneity, completeness, v_measure
