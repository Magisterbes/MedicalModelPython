"""Model parameters: loading, storage, and calibration data containers.

Replaces C# Tech/Constants.cs (Parameters class).

Uses TOML for configuration instead of the custom key:value parser.
Supports type-safe access via dataclass fields and automatic data loading.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from pathlib import Path
import logging

import numpy as np
import pandas as pd

from .distribution import Distribution, risks_to_distribution
from .hazard import Hazard, parse_hazard
from .gompertz import GompertzModel
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


def _get_age_group(age: float) -> int:
    """Assign age to a decadal group (0, 10, 20, ..., 110).
    
    Equivalent to C# Tech.GetAgeGroupInt().
    """
    return 10 * int(np.floor(np.round(age) / 10))


def _get_age_group_vector(age: float, n_groups: int = 12) -> np.ndarray:
    """One-hot encode age group for logistic regression.
    
    Equivalent to C# Tech.GetAgeGroup().
    """
    vec = np.zeros(n_groups)
    idx = int(np.floor(np.round(age) / 10))
    if 0 <= idx < n_groups:
        vec[idx] = 1.0
    return vec


@dataclass
class Parameters:
    """Model parameters and calibration data.
    
    Loaded from a TOML configuration file and CSV data files.
    Corresponds to the C# Parameters class in Tech/Constants.cs.
    
    All fields use Python-native types. Parameter arrays are numpy arrays.
    """
    
    # ---- File paths ----
    data_dir: str = "data"
    train_data_filename: str = "data_agg_rus.csv"
    train_staging_data_filename: str = "data_ind.csv"
    
    # ---- Simulation settings ----
    years_to_simulate: int = 30
    init_population: int = 1_000_000
    unreal_life_length: int = 110
    
    # ---- Stage configuration ----
    stage_distribution_length: int = 4
    
    # ---- Hazard specifications (raw strings, parsed lazily) ----
    diagnose_hazard_spec: str = "piecewise, -1@-11.9;0@0.01;40@0.28;50@-0.11;60@-0.1;70@-0.1"
    cancer_death_hazard_spec: str = "exp, 1"
    
    # ---- Gompertz model initial parameters (before fitting) ----
    reduced_gompertz_non_aggressive: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.1, 0.1])
    reduced_gompertz_aggressive: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.1, 0.1])
    
    # ---- Lead time means by stage (exponential distribution means) ----
    lead_time_by_stage_means: List[float] = field(default_factory=lambda: [1.0, 3.0, 4.0, 5.0])
    
    # ---- Natural history ----
    reoccurrence_probability: float = 0.2
    treatment_mortality_adjustment: float = 1.0
    aggressiveness_rate_threshold: float = 0.2
    growth_rate_limits: List[float] = field(default_factory=lambda: [1.5, 4.0])
    
    # ---- Treatment parameters ----
    treatment_efficiency: List[float] = field(default_factory=lambda: [0.7, 0.3, 0.2, 0.2])
    complications: List[float] = field(default_factory=lambda: [0.1])
    age_cure_constants: List[float] = field(default_factory=lambda: [1.0, 0.9, 0.5, 0.3, 0.1])
    # Cure-odds multiplier for an aggressive tumour, per stage I..IV (set by hand).
    aggressiveness_cure_odds_ratio: List[float] = field(default_factory=lambda: [1.5, 1.5, 1.5, 1.5])
    
    # ---- Risk factors (defined but not currently used in simulation core) ----
    factors_rr: Dict[str, float] = field(default_factory=lambda: {"Smoking": 1.1, "Obesity": 1.05, "Activity": 0.95})
    age_irrelevant_factors_rr: Dict[str, float] = field(default_factory=lambda: {"Gene": 1.05})
    factor_age_correction: List[float] = field(default_factory=lambda: [0.1, 2.0])
    
    # ---- Screening parameters ----
    screening_date: int = 10
    start_age: int = 50
    finish_age: int = 60
    frequency: int = 2
    selected_test: int = 70
    participation_rate: float = 0.5
    test_parameters: List[int] = field(default_factory=lambda: [50, 60, 70, 80])
    test_tp: List[float] = field(default_factory=lambda: [0.6, 0.7, 0.9, 0.95])
    test_fp: List[float] = field(default_factory=lambda: [0.05, 0.07, 0.1, 0.15])
    
    # ---- Economic parameters ----
    test_per_person_price: float = 1.0
    screening_price: float = 1.0
    stage_treatment_price: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0, 4.0])
    complications_price: float = 1.0
    reoccurrence_cost: float = 1.0
    
    # ---- Loaded data (populated by init_data()) ----
    # Distributions
    init_age_dist: Optional[Distribution] = None
    aging_dist: Optional[Distribution] = None
    
    # Hazard objects
    diagnose_hazard: Optional[Hazard] = None
    cancer_death_hazard: Optional[Hazard] = None
    
    # Gompertz models
    gompertz_aggressive: Optional[GompertzModel] = None
    gompertz_non_aggressive: Optional[GompertzModel] = None
    
    # Lead time distributions (exponential scales)
    lead_time_distributions: Optional[List[float]] = None
    
    # Stage-by-age regression
    stage_by_age_reg_generator: Optional[Dict[int, np.ndarray]] = None  # age_group -> multinomial probs
    stage_by_age_generator: Optional[Dict[int, np.ndarray]] = None
    progression_regression: Optional[np.ndarray] = None
    
    # Proportion of aggressive by stage
    proportion_of_aggressive: Optional[np.ndarray] = None
    
    # Training data arrays
    train_incidence: Optional[np.ndarray] = None
    train_mortality: Optional[np.ndarray] = None
    
    # Expanded training data for Gompertz fitting
    expanded_data_aggressive: Optional[pd.DataFrame] = None
    expanded_data_non_aggressive: Optional[pd.DataFrame] = None
    
    # Raw data frames
    train_data: Optional[pd.DataFrame] = None
    individual_data: Optional[pd.DataFrame] = None
    
    # Test sensitivity/specificity (set during init)
    test_tp_selected: float = 0.0
    test_fp_selected: float = 0.0
    
    def __post_init__(self):
        """Convert lists to numpy arrays where appropriate."""
        self.reduced_gompertz_non_aggressive = list(self.reduced_gompertz_non_aggressive)
        self.reduced_gompertz_aggressive = list(self.reduced_gompertz_aggressive)
        self.lead_time_by_stage_means = list(self.lead_time_by_stage_means)
        self.treatment_efficiency = list(self.treatment_efficiency)
        self.age_cure_constants = list(self.age_cure_constants)
        self.aggressiveness_cure_odds_ratio = list(self.aggressiveness_cure_odds_ratio)
        self.growth_rate_limits = list(self.growth_rate_limits)
    
    @classmethod
    def from_toml(cls, path: str) -> 'Parameters':
        """Load parameters from a TOML file.
        
        Parameters
        ----------
        path : str
            Path to the .toml configuration file.
        
        Returns
        -------
        Parameters
            Populated parameters object (data not yet loaded).
        """
        try:
            import tomli  # Python 3.11+ can use tomllib
        except ImportError:
            import tomllib as tomli
        
        with open(path, 'rb') as f:
            config = tomli.load(f)
        
        # Flatten TOML sections into a single dict
        flat: Dict[str, Any] = {}
        for section_key, section_value in config.items():
            if isinstance(section_value, dict):
                for key, value in section_value.items():
                    flat[key] = value
            else:
                flat[section_key] = section_value
        
        # Create Parameters with default values, then override
        params = cls()
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        
        for key, value in flat.items():
            if key in known_fields:
                setattr(params, key, value)
        
        return params
    
    @classmethod
    def from_legacy_txt(cls, path: str) -> 'Parameters':
        """Load parameters from legacy C#-format parameters.txt.
        
        Parses the key:value format used in the original C# implementation.
        """
        params = cls()
        
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if ':' not in line:
                continue
            
            key, value_str = line.split(':', 1)
            key = key.strip()
            value_str = value_str.strip()
            
            # Map C# property names to Python field names
            py_key = _csharp_to_python_name(key)
            
            if key == 'FactorsRR' or key == 'AgeIrrelevantFactorsRR':
                # Parse factor definitions: Name@RR,Name@RR,...
                factors = {}
                for item in value_str.split(','):
                    name, rr = item.split('@')
                    factors[name.strip()] = float(rr.strip())
                if key == 'FactorsRR':
                    params.factors_rr = factors
                else:
                    params.age_irrelevant_factors_rr = factors
            
            elif key in ('DiagnoseHazard', 'CancerDeathHazard'):
                setattr(params, f"{_csharp_to_python_name(key)}_spec", value_str)
            
            elif key.endswith('Hazard'):
                setattr(params, f"{_csharp_to_python_name(key)}_spec", value_str)
            
            elif key in ('ReducedGompertzNonAggressive', 'ReducedGompertzAggressive'):
                values = [float(x.strip()) for x in value_str.split(',')]
                setattr(params, _csharp_to_python_name(key), values)
            
            elif hasattr(params, py_key):
                attr = getattr(params, py_key)
                if isinstance(attr, list):
                    values = [float(x.strip()) for x in value_str.split(',')]
                    setattr(params, py_key, values)
                elif isinstance(attr, bool):
                    setattr(params, py_key, value_str.lower() == 'true')
                elif isinstance(attr, int):
                    setattr(params, py_key, int(value_str))
                elif isinstance(attr, float):
                    setattr(params, py_key, float(value_str))
                elif isinstance(attr, str):
                    setattr(params, py_key, value_str)
        
        return params
    
    def init_data(self):
        """Initialize all data-dependent components after loading parameters.
        
        This method must be called before running the simulation.
        It loads CSV data, constructs distributions, trains regressions,
        and sets up hazard objects.
        
        Equivalent to C# Parameters.InitData() + Parameters.InitFrames() +
        Parameters.LoadStagingTrainData() + Parameters.InitLeadTime().
        """
        from pathlib import Path
        
        data_path = Path(self.data_dir)
        logger.info(f"Data directory: {data_path.absolute()}")
        
        # Step 1: Load aggregate data
        self._init_frames(data_path)
        
        # Step 2: Parse hazards
        self.diagnose_hazard = parse_hazard(self.diagnose_hazard_spec, self.unreal_life_length)
        self.cancer_death_hazard = parse_hazard(self.cancer_death_hazard_spec, self.unreal_life_length)
        
        # Step 3: Initialize lead time distributions
        self._init_lead_time()
        
        # Step 4: Load individual staging data and train models
        self._load_staging_train_data(data_path)
        
        # Step 5: Set screening test parameters
        test_idx = self.test_parameters.index(self.selected_test)
        self.test_tp_selected = self.test_tp[test_idx]
        self.test_fp_selected = self.test_fp[test_idx]
        
        logger.info("Parameters initialization complete.")
    
    def _init_frames(self, data_path: Path):
        """Load aggregate demographic and epidemiological data.
        
        Equivalent to C# Parameters.InitFrames().
        """
        train_file = data_path / self.train_data_filename
        logger.info(f"Loading aggregate data: {train_file}")
        
        self.train_data = pd.read_csv(train_file, sep=';')
        
        # Build initial age distribution
        pop_col = self.train_data['population'].values.astype(np.float64)
        self.init_age_dist = Distribution(pop_col, 'pdf')
        
        # Build aging distribution (natural mortality)
        # C#: Aging = new Distribution(rate, DistributionInputType.PDF)
        # where rate = deaths_all / fsum
        deaths_col = self.train_data['deaths all'].values.astype(np.float64)
        # Convert risks to distribution
        # C# multiplies rates by 3 for InitAgeDist, and converts from risks for Aging
        # Here we follow the exact C# logic
        risks = deaths_col / pop_col.clip(min=1)
        # The C# code does: FromRisksToDistribution(doubles.Select(a => a/100000).ToArray())
        # Actually looking at the C# more carefully:
        # In the C# code, the aging distribution is constructed from 
        # deaths_all/population, then converted via FromRisksToDistribution
        # with each risk divided by 100000 first.
        # The input array to Distribution is: risks / 100000 (as PDF)
        # But wait - the C# code actually does:
        #   rate = Tech.DFCtoArray(TrainData.Columns["deaths all"] / fsum)
        #   Aging = new Distribution(rate, DistributionInputType.PDF)
        # So it's deaths_all divided by total population sum, used as PDF directly.
        fsum = float(pop_col.sum())
        aging_pdf = deaths_col / fsum
        self.aging_dist = Distribution(aging_pdf, 'pdf')
        
        # Build training incidence and mortality arrays
        self.train_incidence = (self.train_data['cases'].values.astype(np.float64) /
                                 pop_col.clip(min=1))
        self.train_mortality = (self.train_data['deaths cancer'].values.astype(np.float64) /
                                 pop_col.clip(min=1))
        
        logger.info(f"Loaded {len(self.train_data)} age groups from aggregate data.")
    
    def _init_lead_time(self):
        """Initialize lead time exponential distributions.
        
        Equivalent to C# Parameters.InitLeadTime().
        """
        self.lead_time_distributions = [1.0 / mean for mean in self.lead_time_by_stage_means]
    
    def reinit_with_data_files(self, train_file: str = None, staging_file: str = None):
        """Reinitialize model with alternative data files.
        
        This allows switching data sources without recreating the Simulation object.
        If a filename is provided, it is used; otherwise, the existing filename is kept.
        
        Parameters
        ----------
        train_file : str, optional
            Filename for aggregate demographic data (e.g., 'data_agg_rus.csv').
        staging_file : str, optional
            Filename for individual staging data (e.g., 'data_ind.csv').
        """
        if train_file is not None:
            self.train_data_filename = train_file
        if staging_file is not None:
            self.train_staging_data_filename = staging_file
        
        # Reload everything
        from pathlib import Path
        data_path = Path(self.data_dir)
        self._init_frames(data_path)
        self._init_lead_time()
        self._load_staging_train_data(data_path)
        
        # Update test params
        test_idx = self.test_parameters.index(self.selected_test)
        self.test_tp_selected = self.test_tp[test_idx]
        self.test_fp_selected = self.test_fp[test_idx]
        
        logger.info(f"Reinitialized with train={self.train_data_filename}, staging={self.train_staging_data_filename}")
    
    def _load_staging_train_data(self, data_path: Path):
        """Load individual staging data and train regression + Gompertz models.
        
        Equivalent to C# Parameters.LoadStagingTrainData().
        """
        staging_file = data_path / self.train_staging_data_filename
        logger.info(f"Loading staging data: {staging_file}")
        
        self.individual_data = pd.read_csv(staging_file, sep=';')
        df = self.individual_data
        
        logger.info(f"Loaded {len(df)} individual records.")
        
        # Train stage-by-age regression
        logger.info("Training multinomial logistic regression for stage-by-age...")
        staging_reg = self._get_model(df)
        self._get_stage_by_age_reg_generator(staging_reg, df)
        
        # Train Gompertz models
        logger.info("Training Gompertz models...")
        aggressive_df = df[df['Aggressiveness'] == 1]
        non_aggressive_df = df[df['Aggressiveness'] == 0]
        
        self.gompertz_aggressive = self._get_gompertz_model(aggressive_df, True)
        self.gompertz_non_aggressive = self._get_gompertz_model(non_aggressive_df, False)
        
        # Aggressiveness distribution by stage
        self.proportion_of_aggressive = self._get_aggressiveness_distribution(df)
        
        logger.info("Staging data training complete.")
    
    def _get_model(self, df: pd.DataFrame) -> np.ndarray:
        """Train multinomial logistic regression for stage prediction by age.
        Returns weights in C#-compatible format: (n_features+1, 4) with bias in last row.
        """
        ages = df['Age'].values
        stages = df['Stage'].values.astype(int)
        n, n_classes = len(ages), 4
        
        train_X = np.array([_get_age_group_vector(a) for a in ages])
        
        # Train sklearn multinomial logistic regression (L-BFGS, fast ~0.1s)
        model = LogisticRegression(solver='lbfgs', C=1.0, max_iter=1000, random_state=42)
        model.fit(train_X, stages - 1)
        
        # Extract weights in C# format: (n_features + 1, n_classes), bias in last row
        weights = np.zeros((train_X.shape[1] + 1, n_classes), dtype=np.float64)
        weights[:train_X.shape[1], :] = model.coef_.T
        weights[train_X.shape[1], :] = model.intercept_
        return weights
    
    def _get_stage_by_age_reg_generator(self, reg: np.ndarray, df: pd.DataFrame):
        """Build stage-by-age multinomial generators.
        
        Equivalent to C# Parameters.GetStageByAgeRegGenerator().
        """
        self.stage_by_age_reg_generator = {}
        
        min_gr = _get_age_group(df['Age'].min())
        max_gr = _get_age_group(df['Age'].max())
        nf = reg.shape[0] - 1  # number of features (bias in last row)
        
        for i in range(12):
            age_gr = i * 10
            gr_vec = _get_age_group_vector(age_gr)
            
            # Clamp to data range
            if age_gr <= min_gr:
                gr_vec = _get_age_group_vector(min_gr)
            if age_gr >= max_gr:
                gr_vec = _get_age_group_vector(max_gr)
            
            # Compute softmax probabilities from weight matrix
            scores = gr_vec @ reg[:nf, :] + reg[nf, :]
            exp_s = np.exp(scores - np.max(scores))
            probs = exp_s / exp_s.sum()
            self.stage_by_age_reg_generator[age_gr] = probs
    
    def _get_gompertz_model(self, df: pd.DataFrame, is_aggressive: bool) -> GompertzModel:
        """Train Gompertz model via Nelder-Mead optimization.
        
        This is a placeholder — actual fitting is done by the optimization module.
        Here we return a model with initial parameters.
        
        Equivalent to C# Parameters.GetGompertzModel().
        """
        if is_aggressive:
            params = self.reduced_gompertz_aggressive
        else:
            params = self.reduced_gompertz_non_aggressive
        
        return GompertzModel(
            K=params[0], C=params[1],
            B_pop=params[2], B_std=params[3]
        )
    
    def _get_aggressiveness_distribution(self, df: pd.DataFrame) -> np.ndarray:
        """Compute proportion of aggressive cancers by stage.
        
        Equivalent to C# Parameters.GetAggressivenessDistribution().
        """
        agg_prop = np.zeros(4)
        for stage in range(1, 5):
            stage_df = df[df['Stage'] == stage]
            if len(stage_df) > 0:
                agg_prop[stage - 1] = stage_df['Aggressiveness'].mean()
        
        return agg_prop
    
    def get_age_cure_efficiency(self, age: int) -> float:
        """Get age-dependent cure efficiency multiplier.
        
        Equivalent to C# Cancer.AgeCureEff().
        Age groups: 0-40, 40-50, 50-60, 70-80, 80+
        """
        if age <= 40:
            return self.age_cure_constants[0]
        elif age <= 50:
            return self.age_cure_constants[1]
        elif age <= 60:
            return self.age_cure_constants[2]
        elif age <= 70:
            return self.age_cure_constants[3]
        else:
            return self.age_cure_constants[4]
    
    def has_factor(self, name: str) -> bool:
        """Check if a risk factor is defined."""
        return name in self.factors_rr or name in self.age_irrelevant_factors_rr
    
    def get_factor_rr(self, name: str, age: float) -> float:
        """Get age-adjusted relative risk for a factor.
        
        Equivalent to C# factor mechanism (which was defined but not used).
        RR_age_adjusted = RR ^ correction, where correction is interpolated
        between factor_age_correction[0] (young) and factor_age_correction[1] (old).
        """
        if name in self.factors_rr:
            base_rr = self.factors_rr[name]
            # Interpolate age correction
            correction = self.factor_age_correction[0] + (
                self.factor_age_correction[1] - self.factor_age_correction[0]
            ) * min(age / 80.0, 1.0)
            return base_rr ** correction
        elif name in self.age_irrelevant_factors_rr:
            return self.age_irrelevant_factors_rr[name]
        return 1.0
    
    def sample_stage_by_age(self, age: int) -> int:
        """Sample a clinical stage given age, using the trained regression.
        
        Returns stage (1-indexed: 1, 2, 3, or 4).
        
        Equivalent to C# stageByAgeRegGenerator[ageGroup].Sample() + IndexOf(1).
        """
        age_gr = _get_age_group(age)
        if self.stage_by_age_reg_generator is None:
            return 1  # fallback
        
        # Get probs for the nearest age group
        available_groups = sorted(self.stage_by_age_reg_generator.keys())
        # Find closest
        closest = min(available_groups, key=lambda g: abs(g - age_gr))
        probs = self.stage_by_age_reg_generator[closest]
        
        # Multinomial sample
        from .random import get_random
        rng = get_random()
        u = rng.next_double()
        cumsum = 0.0
        for s in range(len(probs)):
            cumsum += probs[s]
            if u < cumsum:
                return s + 1
        
        return len(probs)  # last stage


def _csharp_to_python_name(csharp_name: str) -> str:
    """Convert C# PascalCase property names to Python snake_case.
    
    Handles common C# naming patterns:
        YearsToSimulate -> years_to_simulate
        InitPopulation -> init_population
    """
    import re
    # Insert underscore before capital letters, then lowercase
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', csharp_name)
    s2 = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1)
    return s2.lower()