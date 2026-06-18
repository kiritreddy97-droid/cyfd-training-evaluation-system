"""
CYFD Statewide Training Evaluation Pipeline
============================================
Automates pre/post assessment analysis, county-level reporting,
and Power BI dataset refresh for 33 New Mexico counties.
"""

import pandas as pd
import numpy as np
import pyodbc
import sqlalchemy
from sqlalchemy import create_engine, text
import logging
import os
import json
from datetime import datetime, timedelta
from typing import Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(f'evaluation_log_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CYFD_Evaluation')

# ─── Configuration ─────────────────────────────────────────────────────────────
NM_COUNTIES = [
    'Bernalillo', 'Catron', 'Chaves', 'Cibola', 'Colfax', 'Curry', 'De Baca',
    'Dona Ana', 'Eddy', 'Grant', 'Guadalupe', 'Harding', 'Hidalgo', 'Lea',
    'Lincoln', 'Los Alamos', 'Luna', 'McKinley', 'Mora', 'Otero', 'Quay',
    'Rio Arriba', 'Roosevelt', 'Sandoval', 'San Juan', 'San Miguel', 'Santa Fe',
    'Sierra', 'Socorro', 'Taos', 'Torrance', 'Union', 'Valencia'
]

TRAINING_PROGRAMS = [
    'Child Welfare Core',
    'Trauma-Informed Care',
    'Safety Assessment & Planning',
    'Family Engagement',
    'Supervisor Leadership',
    'Placement Stability',
    'CQI Foundations',
    'Mandated Reporter Training'
]

PASSING_THRESHOLD = 0.70
KPI_TARGETS = {
    'completion_rate': 0.90,
    'passing_rate': 0.80,
    'knowledge_gain_pct': 0.30,
    'avg_post_score': 0.80,
    'satisfaction_score': 4.0
}

# ─── Database Engine ───────────────────────────────────────────────────────────
def get_engine():
    """Create SQLAlchemy engine for SQL Server connection."""
    server   = os.getenv('SQL_SERVER', 'localhost')
    database = os.getenv('SQL_DATABASE', 'CYFD_Training')
    username = os.getenv('SQL_USER', '')
    password = os.getenv('SQL_PASSWORD', '')
    
    if username and password:
        conn_str = (
            f"mssql+pyodbc://{username}:{password}@{server}/{database}"
            "?driver=ODBC+Driver+17+for+SQL+Server"
        )
    else:
        conn_str = (
            f"mssql+pyodbc://{server}/{database}"
            "?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
        )
    
    return create_engine(conn_str, echo=False, pool_pre_ping=True)

# ─── Data Extraction ───────────────────────────────────────────────────────────
def extract_assessment_data(
    engine,
    start_date: str,
    end_date: str,
    programs: Optional[list] = None
) -> pd.DataFrame:
    """Extract pre/post assessment scores from SQL Server."""
    
    program_filter = ""
    if programs:
        placeholders = ', '.join([f"'{p}'" for p in programs])
        program_filter = f"AND tp.program_name IN ({placeholders})"
    
    query = f"""
    SELECT 
        a.assessment_id,
        a.staff_id,
        s.full_name,
        s.county,
        s.job_title,
        s.hire_date,
        s.supervisor_id,
        tp.program_name,
        tp.program_code,
        tp.training_date,
        tp.trainer_name,
        tp.training_location,
        a.assessment_type,          -- 'PRE' or 'POST'
        a.score_raw,
        a.score_percentage,
        a.passed,
        a.completion_date,
        a.time_to_complete_minutes,
        a.satisfaction_rating,
        a.feedback_text
    FROM Assessments a
    JOIN Staff s ON a.staff_id = s.staff_id
    JOIN TrainingPrograms tp ON a.program_id = tp.program_id
    WHERE tp.training_date BETWEEN '{start_date}' AND '{end_date}'
    {program_filter}
    AND a.is_valid = 1
    ORDER BY s.county, tp.program_name, a.assessment_type, a.completion_date
    """
    
    df = pd.read_sql(query, engine)
    logger.info(f"Extracted {len(df)} assessment records ({start_date} to {end_date})")
    return df

# ─── Pre/Post Analysis ─────────────────────────────────────────────────────────
def compute_prepost_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute matched pre/post assessment pairs and knowledge gain metrics.
    Returns staff-level paired analysis.
    """
    pre  = df[df['assessment_type'] == 'PRE'].copy()
    post = df[df['assessment_type'] == 'POST'].copy()
    
    pre  = pre.rename(columns={'score_percentage': 'pre_score',  'passed': 'pre_passed'})
    post = post.rename(columns={'score_percentage': 'post_score', 'passed': 'post_passed',
                                'satisfaction_rating': 'satisfaction'})
    
    merge_keys = ['staff_id', 'program_name']
    paired = pd.merge(
        pre[merge_keys + ['pre_score', 'pre_passed', 'county', 'job_title', 'full_name']],
        post[merge_keys + ['post_score', 'post_passed', 'satisfaction', 'time_to_complete_minutes']],
        on=merge_keys,
        how='inner'
    )
    
    paired['knowledge_gain']     = paired['post_score'] - paired['pre_score']
    paired['knowledge_gain_pct'] = (paired['knowledge_gain'] / paired['pre_score'].replace(0, 0.01)).round(4)
    paired['improved']           = paired['knowledge_gain'] > 0
    paired['met_passing']        = paired['post_score'] >= PASSING_THRESHOLD
    
    logger.info(f"Paired analysis: {len(paired)} matched pre/post records")
    logger.info(f"Average knowledge gain: {paired['knowledge_gain'].mean():.1%}")
    
    return paired

# ─── County-Level Aggregation ──────────────────────────────────────────────────
def aggregate_by_county(paired: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate evaluation metrics by NM county.
    Produces KPI scorecard for each of the 33 counties.
    """
    post = raw[raw['assessment_type'] == 'POST'].copy()
    
    # Completion stats from POST assessments
    completion_stats = post.groupby('county').agg(
        total_trained    = ('staff_id', 'count'),
        completions      = ('passed',   'count'),
        avg_satisfaction = ('satisfaction_rating', 'mean')
    ).reset_index()
    
    # Enrollment base (everyone who started = has PRE)
    pre = raw[raw['assessment_type'] == 'PRE']
    enrollment = pre.groupby('county')['staff_id'].count().reset_index()
    enrollment.columns = ['county', 'enrolled']
    
    # Pre/Post gains
    gain_stats = paired.groupby('county').agg(
        avg_pre_score       = ('pre_score',          'mean'),
        avg_post_score      = ('post_score',         'mean'),
        avg_knowledge_gain  = ('knowledge_gain',     'mean'),
        pct_improved        = ('improved',           'mean'),
        pct_met_passing     = ('met_passing',        'mean'),
        paired_count        = ('staff_id',           'count')
    ).reset_index()
    
    # Merge everything
    county_df = pd.merge(completion_stats, enrollment, on='county', how='left')
    county_df = pd.merge(county_df, gain_stats, on='county', how='left')
    
    # KPI calculations
    county_df['completion_rate']   = (county_df['completions'] / county_df['enrolled']).round(4)
    county_df['knowledge_gain_pct'] = county_df['avg_knowledge_gain'].round(4)
    county_df['avg_post_score']     = county_df['avg_post_score'].round(4)
    county_df['satisfaction_score'] = county_df['avg_satisfaction'].round(2)
    
    # Flag counties meeting/missing KPI targets
    for kpi, target in KPI_TARGETS.items():
        if kpi in county_df.columns:
            county_df[f'{kpi}_met'] = county_df[kpi] >= target
    
    # Add all 33 counties (fill NaN for counties with no data)
    all_counties = pd.DataFrame({'county': NM_COUNTIES})
    county_df = pd.merge(all_counties, county_df, on='county', how='left')
    county_df['data_available'] = ~county_df['enrolled'].isna()
    
    logger.info(f"County report: {county_df['data_available'].sum()}/33 counties with data")
    return county_df

# ─── Statistical Testing ───────────────────────────────────────────────────────
def run_statistical_analysis(paired: pd.DataFrame) -> dict:
    """
    Run paired t-tests and effect size calculations on pre/post scores.
    Returns statistical significance results by program.
    """
    results = {}
    
    for program in paired['program_name'].unique():
        prog_data = paired[paired['program_name'] == program]
        
        if len(prog_data) < 5:
            continue
        
        pre_scores  = prog_data['pre_score'].values
        post_scores = prog_data['post_score'].values
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(post_scores, pre_scores)
        
        # Cohen's d effect size
        diff = post_scores - pre_scores
        cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0
        
        # Confidence interval
        ci = stats.t.interval(
            0.95,
            df=len(diff) - 1,
            loc=diff.mean(),
            scale=stats.sem(diff)
        )
        
        results[program] = {
            'n': len(prog_data),
            'avg_pre':    round(pre_scores.mean(), 4),
            'avg_post':   round(post_scores.mean(), 4),
            'mean_gain':  round(diff.mean(), 4),
            't_statistic': round(t_stat, 4),
            'p_value':    round(p_value, 6),
            'significant': p_value < 0.05,
            'cohens_d':   round(cohens_d, 4),
            'effect_size': 'large' if abs(cohens_d) >= 0.8 else 'medium' if abs(cohens_d) >= 0.5 else 'small',
            'ci_95_lower': round(ci[0], 4),
            'ci_95_upper': round(ci[1], 4)
        }
    
    sig_count = sum(1 for r in results.values() if r['significant'])
    logger.info(f"Statistical analysis: {sig_count}/{len(results)} programs show significant improvement (p < 0.05)")
    return results

# ─── Report Generator ──────────────────────────────────────────────────────────
def generate_executive_report(
    county_df: pd.DataFrame,
    stats_results: dict,
    report_date: str,
    output_dir: str = 'reports'
) -> dict:
    """Generate executive summary metrics for Power BI dataset push."""
    os.makedirs(output_dir, exist_ok=True)
    
    summary = {
        'report_date': report_date,
        'generated_at': datetime.now().isoformat(),
        'statewide_kpis': {
            'total_counties': 33,
            'counties_with_data': int(county_df['data_available'].sum()),
            'total_enrolled': int(county_df['enrolled'].sum()),
            'total_completed': int(county_df['completions'].sum()),
            'statewide_completion_rate': round(
                county_df['completions'].sum() / county_df['enrolled'].sum(), 4
            ) if county_df['enrolled'].sum() > 0 else 0,
            'statewide_avg_knowledge_gain': round(county_df['avg_knowledge_gain'].mean(), 4),
            'statewide_avg_satisfaction': round(county_df['satisfaction_score'].mean(), 2),
            'counties_meeting_completion_target': int(county_df['completion_rate_met'].sum()),
            'counties_meeting_passing_target': int(county_df['pct_met_passing_met'].sum())
        },
        'program_analysis': stats_results,
        'county_scorecards': county_df.to_dict('records')
    }
    
    # Save JSON for Power BI
    json_path = os.path.join(output_dir, f'cyfd_eval_report_{report_date}.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Save CSV for stakeholders
    county_df.to_csv(
        os.path.join(output_dir, f'county_scorecard_{report_date}.csv'),
        index=False
    )
    
    logger.info(f"Executive report saved: {json_path}")
    logger.info(f"Statewide completion rate: {summary['statewide_kpis']['statewide_completion_rate']:.1%}")
    logger.info(f"Avg knowledge gain: {summary['statewide_kpis']['statewide_avg_knowledge_gain']:.1%}")
    
    return summary

# ─── Main Pipeline ─────────────────────────────────────────────────────────────
def run_evaluation_pipeline(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    programs: Optional[list] = None
) -> dict:
    """
    Full evaluation pipeline:
    1. Extract assessment data from SQL Server
    2. Compute pre/post analysis  
    3. Aggregate by county
    4. Run statistical tests
    5. Generate executive report
    """
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    report_date = datetime.now().strftime('%Y%m%d')
    
    logger.info("=" * 70)
    logger.info("CYFD Training Evaluation Pipeline Starting")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info("=" * 70)
    
    engine = get_engine()
    
    # Step 1: Extract
    raw_df = extract_assessment_data(engine, start_date, end_date, programs)
    
    # Step 2: Pre/Post Analysis
    paired_df = compute_prepost_analysis(raw_df)
    
    # Step 3: County Aggregation
    county_df = aggregate_by_county(paired_df, raw_df)
    
    # Step 4: Statistical Analysis
    stat_results = run_statistical_analysis(paired_df)
    
    # Step 5: Report Generation
    report = generate_executive_report(county_df, stat_results, report_date)
    
    engine.dispose()
    
    logger.info("=" * 70)
    logger.info("Pipeline Complete!")
    logger.info(f"Counties with data: {report['statewide_kpis']['counties_with_data']}/33")
    logger.info(f"Completion rate: {report['statewide_kpis']['statewide_completion_rate']:.1%}")
    logger.info("=" * 70)
    
    return report

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='CYFD Training Evaluation Pipeline')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end',   type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--programs', nargs='+', help='Specific program names to analyze')
    args = parser.parse_args()
    
    run_evaluation_pipeline(
        start_date=args.start,
        end_date=args.end,
        programs=args.programs
    )
