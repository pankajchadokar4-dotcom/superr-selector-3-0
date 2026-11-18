import io, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, brier_score_loss

st.set_page_config(page_title='SupeRR Selector 3.0', page_icon='🏏', layout='wide')
st.title('🏏 SupeRR Selector 3.0')
st.caption('Combined IPL + T20I pressure-performance ranking | contextual impact + personal uplift + consistency')
st.info('The supplied hackathon datasets are bundled with this deployment. Rankings pool IPL and T20I into one batter list and one bowler list; format is retained as a context feature, not as separate ranking lists.')

FEATURES = ['innings','overs_elapsed','wickets_before','score_before','current_rr','required_rr','balls_remaining','batting_order','format_t20i','season_num']

@st.cache_data(show_spinner=False)
def read_csv(blob): return pd.read_csv(io.BytesIO(blob), low_memory=False)

def flag(s):
    if pd.api.types.is_bool_dtype(s): return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin(['true','1','yes','y'])

def prepare(bb, ma, co, playing=None):
    ma = ma.merge(co[['comp_id','comp_name','match_format','season']], on='comp_id', how='left')
    keep = ma.comp_name.astype(str).str.fullmatch(r'IPL\s+\d{4}', case=False) | ma.match_format.astype(str).str.upper().eq('T20I')
    ma = ma[keep].copy()
    ma['match_date'] = pd.to_datetime(ma.match_date, errors='coerce')
    cols = ['match_id','match_date','comp_name','match_format','season','first_inn_revised_overs','second_inn_revised_overs','revised_target']
    d = bb.merge(ma[cols], on='match_id', how='inner')
    d = d[d.innings.isin([1,2])].copy()
    if d.empty: raise ValueError('No IPL/T20I delivery records found.')
    for c in ['runs','extras1_runs','extras_2_runs']:
        d[c] = pd.to_numeric(d[c], errors='coerce').fillna(0)
    for c in ['is_wicket','is_four','is_six']:
        d[c] = flag(d[c])
    d['extras_type'] = d.extras_type.fillna('').astype(str).str.lower().str.strip()
    d['legal_ball'] = ~d.extras_type.isin(['wide','overstep','no ball','noball'])
    d['batter_ball'] = ~d.extras_type.eq('wide')
    d['total_runs'] = d.runs + d.extras1_runs + d.extras_2_runs
    d = d.sort_values(['match_date','match_id','innings','innings_ball_num']).reset_index(drop=True)
    g = d.groupby(['match_id','innings'], sort=False)
    d['score_before'] = g.total_runs.cumsum() - d.total_runs
    d['legal_before'] = g.legal_ball.cumsum() - d.legal_ball.astype(int)
    d['wickets_before'] = g.is_wicket.cumsum() - d.is_wicket.astype(int)
    d['max_balls'] = 120.0
    for inn, field in [(1,'first_inn_revised_overs'),(2,'second_inn_revised_overs')]:
        m = (d.innings.eq(inn)) & d[field].notna()
        d.loc[m,'max_balls'] = 6*pd.to_numeric(d.loc[m,field], errors='coerce').fillna(20)
    d['max_balls'] = d.max_balls.clip(lower=1)
    d['balls_remaining'] = (d.max_balls-d.legal_before).clip(lower=0)
    d['overs_elapsed'] = d.legal_before/6
    d['current_rr'] = np.where(d.legal_before>0, d.score_before*6/d.legal_before, 0)
    first_totals = d[d.innings.eq(1)].groupby('match_id').total_runs.sum()
    d['target'] = d.match_id.map(first_totals).fillna(0)+1
    rev = pd.to_numeric(d.revised_target, errors='coerce')
    d.loc[rev.notna(),'target'] = rev[rev.notna()]
    d['runs_needed'] = (d.target-d.score_before).clip(lower=0)
    d['required_rr'] = np.where((d.innings.eq(2)) & (d.balls_remaining>0), d.runs_needed*6/d.balls_remaining, 0)
    d['format_t20i'] = d.match_format.astype(str).str.upper().eq('T20I').astype(int)
    d['season_num'] = d.match_date.dt.year.fillna(2024).astype(int)
    d['batting_order'] = pd.to_numeric(d.batting_order, errors='coerce').fillna(7).clip(1,12)
    # Pressure ingredients: bounded 0..1.
    progress = (1-d.balls_remaining/d.max_balls).clip(0,1)
    wicket_lost = (d.wickets_before/9).clip(0,1)
    late = ((d.overs_elapsed-10)/9).clip(0,1)
    chase_rrr = ((d.required_rr-6)/8).clip(0,1)
    # First-innings par is data-derived: mean score rate at the same format/innings phase/wicket band.
    d['phase'] = pd.cut(d.overs_elapsed, bins=[-0.01,6,15,30], labels=['powerplay','middle','death'])
    d['wicket_band'] = pd.cut(d.wickets_before, bins=[-1,1,4,9], labels=['0-1','2-4','5-9'])
    par = d[d.innings.eq(1)].groupby(['format_t20i','phase','wicket_band'], observed=True).current_rr.median()
    keys = pd.MultiIndex.from_arrays([d.format_t20i, d.phase, d.wicket_band])
    d['par_rr'] = par.reindex(keys).to_numpy()
    fallback = d[d.innings.eq(1)].groupby('format_t20i').current_rr.median().to_dict()
    d['par_rr'] = [v if pd.notna(v) else fallback.get(int(fmt),8.0) for v,fmt in zip(d.par_rr,d.format_t20i)]
    first_bat_pressure = 100*(.35*((d.par_rr-d.current_rr).clip(lower=0)/np.maximum(d.par_rr,1)).clip(0,1) + .25*wicket_lost + .20*progress + .20*((d.batting_order-1)/10).clip(0,1)*late)
    chase_bat_pressure = 100*(.35*chase_rrr + .25*wicket_lost + .20*progress + .20*((d.batting_order-1)/10).clip(0,1)*late)
    d['bat_pressure'] = np.where(d.innings.eq(2), chase_bat_pressure, first_bat_pressure).clip(0,100)
    # Bowling pressure: defend against a threatening batting position; high RRR and wickets in hand in a chase.
    wickets_in_hand = (1-wicket_lost).clip(0,1)
    first_bowl_pressure = 100*(.35*((d.current_rr-d.par_rr).clip(lower=0)/np.maximum(d.par_rr,1)).clip(0,1) + .25*wickets_in_hand + .20*progress + .20*late)
    chase_bowl_pressure = 100*(.35*chase_rrr + .25*wickets_in_hand + .20*progress + .20*late)
    d['bowl_pressure'] = np.where(d.innings.eq(2), chase_bowl_pressure, first_bowl_pressure).clip(0,100)
    # Responsibility factor is intentionally conservative and transparent; batting order is a proxy, not a specialist-role label.
    d['batter_out'] = (d.is_wicket & d.dismissed_player_id.eq(d.striker_id)).astype(int)
    non_bowler_dismissals = ['run out','runout','retired out','retired hurt','obstructing the field','obstracting the field']
    d['bowler_wicket'] = (d.is_wicket & ~d.dismissal_type.fillna('').astype(str).str.lower().isin(non_bowler_dismissals)).astype(int)
    d['bowler_conceded'] = d.runs + np.where(d.extras_type.isin(['wide','overstep','no ball','noball']), d.extras1_runs+d.extras_2_runs, 0)
    if playing is not None and {'match_id','player_id','is_captain','is_wk'}.issubset(playing.columns):
        p = playing.copy()
        for c in ['is_captain','is_wk']:
            p[c] = flag(p[c])
        # Supporting audit fields; not used to inflate pressure without reliable role definitions.
        p['leadership_flag'] = (p.is_captain | p.is_wk).astype(int)
        d = d.merge(p[['match_id','player_id','leadership_flag']].drop_duplicates(), left_on=['match_id','striker_id'], right_on=['match_id','player_id'], how='left').drop(columns=['player_id'], errors='ignore')
    return d

def chronological_predictions(d):
    match_order = d[['match_id','match_date']].drop_duplicates().sort_values(['match_date','match_id']).match_id.to_numpy()
    if len(match_order)<30: raise ValueError('At least 30 in-scope matches are required.')
    edges = [int(len(match_order)*x) for x in [.35,.57,.79,1.0]]
    out = d.copy()
    for c in ['exp_bat_runs','exp_bowl_runs','p_batter_out','p_bowler_wicket']: out[c]=np.nan
    logs=[]
    for fold in range(3):
        train_ids, test_ids = set(match_order[:edges[fold]]), set(match_order[edges[fold]:edges[fold+1]])
        tr, te = out.match_id.isin(train_ids), out.match_id.isin(test_ids)
        Xtr=out.loc[tr,FEATURES].fillna(0).astype('float32'); Xte=out.loc[te,FEATURES].fillna(0).astype('float32')
        for name,target,kind in [('exp_bat_runs','runs','reg'),('exp_bowl_runs','bowler_conceded','reg'),('p_batter_out','batter_out','cls'),('p_bowler_wicket','bowler_wicket','cls')]:
            y=out.loc[tr,target]
            if kind=='reg':
                model=HistGradientBoostingRegressor(max_iter=55,max_leaf_nodes=15,learning_rate=.08,l2_regularization=3,random_state=42,early_stopping=False).fit(Xtr,y)
                pred=np.clip(model.predict(Xte),0,None); metric=mean_absolute_error(out.loc[te,target],pred); label='MAE'
            else:
                # Handle a rare one-class training fold safely.
                if y.nunique()<2:
                    pred=np.full(len(Xte),float(y.mean())); metric=brier_score_loss(out.loc[te,target],pred)
                else:
                    model=HistGradientBoostingClassifier(max_iter=55,max_leaf_nodes=15,learning_rate=.08,l2_regularization=3,random_state=42,early_stopping=False).fit(Xtr,y)
                    pred=model.predict_proba(Xte)[:,1]; metric=brier_score_loss(out.loc[te,target],pred)
                label='Brier'
            out.loc[te,name]=pred
            logs.append({'fold':fold+1,'target':target,'model':'HistGradientBoosting','metric':label,'value':round(float(metric),4),'train_matches':len(train_ids),'scored_matches':len(test_ids)})
            # Simple training-only baseline for honest model comparison.
            if kind=='reg':
                base_pred=np.full(len(Xte),float(y.mean()))
                base_metric=mean_absolute_error(out.loc[te,target],base_pred)
            else:
                base_pred=np.full(len(Xte),float(y.mean()))
                base_metric=brier_score_loss(out.loc[te,target],base_pred)
            logs.append({'fold':fold+1,'target':target,'model':'Training-mean baseline','metric':label,'value':round(float(base_metric),4),'train_matches':len(train_ids),'scored_matches':len(test_ids)})
    return out,pd.DataFrame(logs)

def percentile(s, higher=True):
    return s.rank(pct=True, ascending=higher)*100

def score_players(d, mapping, threshold=60, min_hp=100, min_matches=10, min_total=250, wicket_value=6):
    x=d[d.exp_bat_runs.notna()].copy()
    x['bat_hp']=x.bat_pressure.ge(threshold); x['bowl_hp']=x.bowl_pressure.ge(threshold)
    x['bat_impact']=x.runs-x.exp_bat_runs+wicket_value*(x.p_batter_out-x.batter_out)
    x['bowl_impact']=x.exp_bowl_runs-x.bowler_conceded+wicket_value*(x.bowler_wicket-x.p_bowler_wicket)
    names=mapping.dropna(subset=['player_id','player_name']).drop_duplicates('player_id').set_index('player_id').player_name.to_dict()
    def role(role, idc, hpcol, ballcol, impactcol, runcol, dismissalcol, expectedcol):
        q=x[x[ballcol]].copy(); hp=q[q[hpcol]].copy(); normal=q[~q[hpcol]].copy()
      if hp.empty:
    return pd.DataFrame(columns=['player_name', 'player_id', 'rank'])
        # Player pressure and normal aggregates.
        h=hp.groupby(idc).agg(hp_balls=(ballcol,'size'),hp_matches=('match_id','nunique'),hp_runs=(runcol,'sum'),hp_outs=(dismissalcol,'sum'),hp_impact=(impactcol,'sum'))
        n=normal.groupby(idc).agg(normal_balls=(ballcol,'size'),normal_runs=(runcol,'sum'),normal_outs=(dismissalcol,'sum'))
        pm=hp.groupby([idc,'match_id'])[impactcol].sum().reset_index()
        cons=pm.groupby(idc).agg(positive_match_pct=(impactcol,lambda s:100*(s>0).mean()),impact_match_sd=(impactcol,'std'))
        a=h.join(n,how='left').join(cons,how='left').fillna({'normal_balls':0,'normal_runs':0,'normal_outs':0,'positive_match_pct':0,'impact_match_sd':0})
        a['total_role_balls']=a.hp_balls+a.normal_balls
        a=a[(a.hp_balls>=min_hp)&(a.hp_matches>=min_matches)&(a.total_role_balls>=min_total)].copy()
      if a.empty:
    a['player_name'] = pd.Series(dtype='object')
    return a
        if role=='batter':
            a['hp_rate']=100*a.hp_runs/a.hp_balls; a['normal_rate']=100*a.normal_runs/a.normal_balls.replace(0,np.nan)
            a['personal_uplift']=a.hp_rate-a.normal_rate
            a['hp_dismissal_pct']=100*a.hp_outs/a.hp_balls
        else:
            # For bowlers, runcol is conceded runs; rates are per 6 legal balls.
            a['hp_rate']=6*a.hp_runs/a.hp_balls; a['normal_rate']=6*a.normal_runs/a.normal_balls.replace(0,np.nan)
            a['personal_uplift']=a.normal_rate-a.hp_rate
            a['hp_dismissal_pct']=100*a.hp_outs/a.hp_balls
        a['impact_per100']=100*a.hp_impact/a.hp_balls
        a['impact_pct']=percentile(a.impact_per100)
        a['uplift_pct']=percentile(a.personal_uplift.fillna(0))
        a['consistency_pct']=percentile(a.positive_match_pct)
        a['raw_rating']=.40*a.impact_pct+.35*a.uplift_pct+.25*a.consistency_pct
        # Confidence combines pressure balls and pressure matches, each capped.
        ball_conf=a.hp_balls/(a.hp_balls+150)
        match_conf=a.hp_matches/(a.hp_matches+10)
        a['reliability']=np.sqrt(ball_conf*match_conf)
        a['rating']=50+(a.raw_rating-50)*a.reliability
        a['player_name']=[names.get(i,str(i)) for i in a.index]
        a=a.sort_values(['rating','hp_matches','hp_balls'],ascending=False).reset_index().rename(columns={idc:'player_id'})
        a.insert(0,'rank',range(1,len(a)+1))
        return a
    bat=role('batter','striker_id','bat_hp','batter_ball','bat_impact','runs','batter_out','exp_bat_runs')
    bowl=role('bowler','bowler_id','bowl_hp','legal_ball','bowl_impact','bowler_conceded','bowler_wicket','exp_bowl_runs')
    return bat,bowl,x

with st.sidebar:
    st.header('3.0 model settings')
    threshold=st.slider('High-pressure threshold',45,80,55)
    min_hp=st.number_input('Minimum pressure balls',20,400,100,step=10)
    min_matches=st.number_input('Minimum pressure matches',2,30,10)
    min_total=st.number_input('Minimum total role-specific balls',100,1000,250,step=25)
    wicket_value=st.slider('Wicket value (run-equivalents)',3,10,6)
    st.caption('Default strict eligibility: 100 pressure balls, 10 pressure matches, 250 total role-specific balls.')

with st.expander('Methodology and assumptions', expanded=False):
    st.markdown('''**Combined selection:** IPL and T20I deliveries are pooled for the final two rankings. Format remains a model feature and is used in diagnostics.\n\n**Proposed rating:** 40% contextual impact + 35% personal uplift + 25% consistency. Components are player percentiles. Reliability shrinks ratings toward 50 using both pressure-ball and pressure-match evidence.\n\n**Pressure:** Separate batting and bowling scores use chase difficulty, wickets, innings progress, phase and transparent responsibility/context proxies. First-innings par run rate is estimated from the supplied data by format, phase and wicket band.\n\n**Limitations:** Batting order is an imperfect responsibility proxy. The 6-run wicket value and component weights are analyst assumptions. First-innings par is a descriptive benchmark, not a pitch-adjusted target. Super overs are excluded.''')

DATA_DIR = Path(__file__).parent

@st.cache_data(show_spinner=False)
def load_bundled_data():
    required = {
        'ball_by_ball': DATA_DIR / 'ball_by_ball.csv.gz',
        'matches': DATA_DIR / 'matches.csv',
        'competitions': DATA_DIR / 'competitions.csv',
        'mapping': DATA_DIR / 'competition_team_player_mapping.csv',
    }
    missing = [str(path.name) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError('Missing bundled dataset files: ' + ', '.join(missing))
    bb = pd.read_csv(required['ball_by_ball'], low_memory=False, compression='gzip')
    ma = pd.read_csv(required['matches'], low_memory=False)
    co = pd.read_csv(required['competitions'], low_memory=False)
    mp = pd.read_csv(required['mapping'], low_memory=False)
    playing_path = DATA_DIR / 'playing_eleven.csv'
    playing = pd.read_csv(playing_path, low_memory=False) if playing_path.exists() else None
    return bb, ma, co, mp, playing

try:
    bb, ma, co, mp, playing = load_bundled_data()
    st.success(f"Datasets ready: {len(bb):,} deliveries | {ma.match_id.nunique():,} match records | IPL + T20I")
    st.caption('Select **Run SupeRR 3.0 analysis** to generate the combined rankings. No manual dataset upload is required.')
    if st.button('🚀 Run SupeRR 3.0 analysis', type='primary'):
        try:
            with st.spinner('Preparing match states, training chronological models and calculating ratings...'):
                d = prepare(bb, ma, co, playing)
                pred, metrics = chronological_predictions(d)
                bat, bowl, scored = score_players(pred, mp, threshold, min_hp, min_matches, min_total, wicket_value)
                st.session_state['supeRR3'] = (pred, metrics, bat, bowl, scored, mp)
            st.success('Analysis completed.')
        except Exception as e:
            st.error(f'Analysis failed: {e}')
            st.exception(e)
except Exception as e:
    st.error(f'Could not load the bundled datasets: {e}')
    st.exception(e)

if 'supeRR3' in st.session_state:
    d,metrics,bat,bowl,scored,mp=st.session_state['supeRR3']
    a,b,c,e=st.columns(4)
    a.metric('Matches in scope',f"{d.match_id.nunique():,}")
    b.metric('Predicted matches',f"{d.loc[d.exp_bat_runs.notna(),'match_id'].nunique():,}")
    c.metric('Eligible batters',len(bat)); e.metric('Eligible bowlers',len(bowl))
    tabs=st.tabs(['🏏 Top 10 batters','🎯 Top 10 bowlers','🔎 Player comparison','📊 Pressure analysis','🧪 Validation'])
    with tabs[0]:
        st.subheader('Combined IPL + T20I batter ranking')
        if len(bat):
            st.dataframe(bat.head(10).round(2),hide_index=True,width="stretch")
            st.plotly_chart(px.bar(bat.head(10),x='player_name',y='rating',title='Top 10 batter ratings'),width="stretch")
            st.download_button('Download Top 10 batters CSV',bat.head(10).to_csv(index=False),'SupeRR_3_0_Top10_Batters.csv','text/csv')
            st.download_button('Download all batter scorecards',bat.to_csv(index=False),'SupeRR_3_0_All_Batters.csv','text/csv')
        else: st.warning('No batters qualify with the current strict eligibility settings.')
    with tabs[1]:
        st.subheader('Combined IPL + T20I bowler ranking')
        if len(bowl):
            st.dataframe(bowl.head(10).round(2),hide_index=True,width="stretch")
            st.plotly_chart(px.bar(bowl.head(10),x='player_name',y='rating',title='Top 10 bowler ratings'),width="stretch")
            st.download_button('Download Top 10 bowlers CSV',bowl.head(10).to_csv(index=False),'SupeRR_3_0_Top10_Bowlers.csv','text/csv')
            st.download_button('Download all bowler scorecards',bowl.to_csv(index=False),'SupeRR_3_0_All_Bowlers.csv','text/csv')
        else: st.warning('No bowlers qualify with the current strict eligibility settings.')
    with tabs[2]:
       bat_names = bat['player_name'].tolist() if 'player_name' in bat.columns else []
bowl_names = bowl['player_name'].tolist() if 'player_name' in bowl.columns else []
pool = sorted(set(bat_names + bowl_names))
        if pool:
            chosen=st.selectbox('Choose a player',pool)
            rows=[]
            for role_name,df in [('Batter',bat),('Bowler',bowl)]:
                r=df[df.player_name.eq(chosen)]
                if len(r): rows.append(r.assign(role=role_name))
            if rows: st.dataframe(pd.concat(rows,ignore_index=True).round(2),hide_index=True,width="stretch")
            else: st.info('This player is not eligible in the selected role.')
    with tabs[3]:
        st.plotly_chart(px.histogram(d,x='bat_pressure',nbins=25,title='Batting pressure distribution'),width="stretch")
        st.plotly_chart(px.histogram(d,x='bowl_pressure',nbins=25,title='Bowling pressure distribution'),width="stretch")
        st.download_button('Download delivery-level pressure data',scored.to_csv(index=False),'SupeRR_3_0_Pressure_Deliveries.csv','text/csv')
    with tabs[4]:
        st.subheader('Chronological out-of-sample model validation')
        st.dataframe(metrics,hide_index=True,width="stretch")
        st.caption('Lower MAE and Brier scores are better. Each model fold is now shown beside a training-mean baseline; ML should only be credited where it improves on that baseline.')
        # Format diagnostic: compare pressure sample and eligibility contributions, not separate final lists.
        st.subheader('Format contribution diagnostic')
        if len(bat):
            st.write('Eligible batter pool format appearances (players may appear in both formats).')
            st.dataframe(d.groupby('match_format').agg(matches=('match_id','nunique'),deliveries=('match_id','size')).reset_index(),hide_index=True)
        st.warning('Diagnostic rankings are not final until threshold/eligibility/weight sensitivity and baseline-model comparisons are completed.')
