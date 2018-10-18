computers = ["m1", "m2", "p1", "p2", "s1", "s2", "bo1", "be1", "sc1", "sc2", "sc3", "e1", "gu1"]  # , "gw1", "je1"
# computers = ["test"]

models = ["var", "fix"]

params_v1 = [
    'z3_factor',
    'cZ0Z1', 'cZ',
    'c_adr',
    'k_g',
    'gamma01', 'gammaZ',
    'f_transp',
    'f_oc', 'k_oc',
    'beta_runoff',
    'dt_50_aged',
    'dt_50_ab',
    'dt_50_ref',
    'epsilon_iso',
    'beta_moisture'
]

params_v2 = [
    'z3_factor',
    'cZ0Z1', 'cZ',
    'c_adr',
    'k_g',
    'gamma01', 'gammaZ',
    'f_transp',
    'f_evap',  # <- new in v2
    'f_oc', 'k_oc',
    'beta_runoff',
    'dt_50_aged',
    'dt_50_ab',
    'dt_50_ref',
    'epsilon_iso',
    'beta_moisture'
]

bounds_v1 = [[0.75, 0.99],  # z3_factor
             [0.01, 1.0], [0.01, 1.0],  # c_lf
             [0.001, 1.0],  # cadr
             [1100.0, 3650.0],  # k_g
             [0.0001, 1.0], [0, 1.0],  # gamma
             [0.0001, 1.0],  # f_transp
             [0.01, 0.05], [7.0, 16180],  # f_oc, k_oc
             [0.01, 1.0],  # beta_runoff
             [140.0, 7000.0],  # age_rate
             [130.0, 230.0],  # dt_50_ab
             [26.0, 37.0],  # dt_50_ref
             [1.0, 3.0],  # epsilon (in absolute, convert to negative!!)
             [0.01, 1.0]]  # beta_moisture

# For models var3+ / fix3+ < var5 / fix5
bounds_v2 = [[0.75, 0.99],  # z3_factor
             [0.01, 1.0], [0.01, 1.0],  # c_lf
             [0.001, 1.0],  # cadr
             [1100.0, 3650.0],  # k_g
             [0.0001, 1.0], [0, 1.0],  # gamma
             [0.0001, 1.0],  # f_transp
             [0.01, 0.05], [7.0, 16180],  # f_oc, k_oc
             [0.01, 1.0],  # beta_runoff
             [140.0, 7000.0],  # age_rate
             [130.0, 230.0],  # dt_50_ab
             [10.0, 40.0],  # dt_50_ref
             [0.5, 5.0],  # epsilon (in absolute, convert to negative!!)
             [0.01, 1.0]]  # beta_moisture

# For models var5+ / fix5+
bounds_v3 = [[0.85, 0.99],  # z3_factor
             [0.2, 1.0], [0.2, 1.0],  # c_lf
             [0.001, 1.0],  # cadr
             [1500.0, 3650.0],  # k_g
             [0.4, 1.0], [0.01, 0.6],  # gamma01, gammaZ
             [0.2, 0.8],  # f_transp
             [0.01, 0.05],  # f_oc,
             [7.0, 16180],  # k_oc
             [0.01, 1.0],  # beta_runoff
             [140.0, 7000.0],  # age_rate
             [130.0, 230.0],  # dt_50_ab
             [10.0, 40.0],  # dt_50_ref
             [0.5, 5.0],  # epsilon (in absolute, convert to negative!!)
             [0.01, 1.0]]  # beta_moisture

# For models var6+ / fix6+
# Based on NSE metric (more stringent),
bounds_v4 = [[0.85, 0.99],  # z3_factor
             [0.2, 1.0], [0.2, 1.0],  # 'cZ0Z1', 'cZ'
             [0.01, 0.6],  # cadr
             [1500.0, 3650.0],  # k_g
             [0.5, 1.0], [0.0, 0.4],  # gamma01, gammaZ
             [0.3, 0.8],  # f_transp
             [0.01, 0.05],  # f_oc,
             [7.0, 10000],  # k_oc <- New
             [0.01, 0.4],  # beta_runoff <- New
             [140.0, 7000.0],  # age_rate
             [130.0, 230.0],  # dt_50_ab
             [10.0, 40.0],  # dt_50_ref
             [0.5, 5.0],  # epsilon (in absolute, convert to negative!!)
             [0.01, 1.0]]  # beta_moisture

# For models var7+ / fix7+ <= 8
# Kd expectations < 200 from best KGE transect models, with outlet constraints
bounds_v5 = [[0.85, 0.99],  # z3_factor
             [0.2, 1.0], [0.2, 1.0],  # 'cZ0Z1', 'cZ'
             [0.01, 0.6],  # cadr
             [1500.0, 3650.0],  # k_g
             [0.5, 1.0], [0.0, 0.4],  # gamma01, gammaZ
             [0.3, 0.8],  # f_transp
             [0.01, 0.05],  # f_oc,
             [0.3, 5000],  # k_oc, max Kd = 250, min due to Boithias2014
             [0.01, 0.4],  # beta_runoff
             [140.0, 7000.0],  # age_rate
             [130.0, 230.0],  # dt_50_ab
             [10.0, 40.0],  # dt_50_ref
             [0.5, 5.0],  # epsilon (in absolute, convert to negative!!)
             [0.01, 1.0]]  # beta_moisture

# For models var9+ / fix9+ <= v16
# Reducing DT50 further, as boundary is not yet best defined.
bounds_v6 = [[0.85, 0.99],  # z3_factor
             [0.2, 1.0], [0.2, 1.0],  # 'cZ0Z1', 'cZ'
             [0.01, 0.6],  # cadr
             [1500.0, 3650.0],  # k_g
             [0.5, 1.0], [0.0, 0.4],  # gamma01, gammaZ
             [0.3, 0.8],  # f_transp
             [0.01, 0.05],  # f_oc,
             [0.3, 5000],  # k_oc, max Kd = 250, min due to Boithias2014
             [0.01, 0.4],  # beta_runoff
             [140.0, 7000.0],  # age_rate
             [130.0, 230.0],  # dt_50_ab
             [1.0, 50.0],  # dt_50_ref
             [0.5, 4.0],  # epsilon (in absolute, convert to negative!!) <- New v9
             [0.01, 1.0]]  # beta_moisture

# For models var17+ / fix17+ <= 23
# Reducing Koc further to 2000, yielding a max Kd of 100.
bounds_v7 = [[0.87, 0.97],  # z3_factor <- New
             [0.6, 1.0], [0.2, 0.6],  # 'cZ0Z1', 'cZ' <- New
             [0.2, 0.4],  # cadr <- New
             [1500.0, 3650.0],  # k_g
             [0.8, 1.0], [0.0, 0.1],  # gamma01, gammaZ <- New
             [0.5, 0.6],  # f_transp <- New
             [0.01, 0.05],  # f_oc,
             [0.3, 2000],  # k_oc, max Kd = 100 <- New
             [0.01, 0.4],  # beta_runoff
             [140.0, 7000.0],  # age_rate
             [130.0, 230.0],  # dt_50_ab
             [1.0, 50.0],  # dt_50_ref
             [0.5, 4.0],  # epsilon (in absolute, convert to negative!!) <- New v9
             [0.01, 1.0]]  # beta_moisture

# For models var24+ / fix24+ < 30
bounds_v8 = [[0.85, 0.97],  # z3_factor <- New
             [0.6, 1.0], [0.2, 0.6],  # 'cZ0Z1', 'cZ'
             [0.2, 0.4],  # cadr
             [1500.0, 3650.0],  # k_g
             [0.2, 0.8],  # gamma01, <- New KEY!!
             [0.0, 0.4],  # gammaZ <- New
             [0.4, 0.6],  # f_transp <- New
             [0.01, 0.05],  # f_oc,
             [0.3, 2000],  # k_oc, max Kd = 100
             [0.01, 0.4],  # beta_runoff
             [140.0, 7000.0],  # age_rate
             [130.0, 230.0],  # dt_50_ab
             [1.0, 50.0],  # dt_50_ref
             [0.5, 4.0],  # epsilon (in absolute, convert to negative!!)
             [0.01, 1.0]]  # beta_moisture

# For models var30 < 32
# Generation 5, volat increase.
bounds_v9 = [[0.85, 0.97],  # z3_factor
             [0.6, 1.0], [0.2, 0.6],  # 'cZ0Z1', 'cZ'
             [0.2, 0.4],  # cadr
             [1500.0, 3650.0],  # k_g
             [0.2, 1.],  # gamma01, <- New!
             [0.0, 0.4],  # gammaZ
             [0.4, 0.6],  # f_transp
             [0.01, 0.05],  # f_oc,
             [0.3, 2000],  # k_oc, max Kd = 100
             [0.01, 0.4],  # beta_runoff
             [140.0, 7000.0],  # age_rate
             [130.0, 230.0],  # dt_50_ab
             [1.0, 50.0],  # dt_50_ref
             [0.5, 4.],  # epsilon (in absolute, convert to negative!!)
             [0.01, 1.0]]

# For models var32+ < 36
# Gen5 (volat increase) and Gen6 (soil temp corr.),
# new age_rate and abiotic deg
bounds_v10 = [[0.85, 0.97],  # z3_factor
              [0.6, 1.0], [0.2, 0.6],  # 'cZ0Z1', 'cZ'
              [0.2, 0.4],  # cadr
              [1500.0, 3650.0],  # k_g
              [0.2, 1.],  # gamma01,
              [0.0, 0.4],  # gammaZ
              [0.4, 0.6],  # f_transp
              [0.01, 0.05],  # f_oc,
              [0.3, 2000],  # k_oc, max Kd = 100
              [0.01, 0.4],  # beta_runoff
              [1.0, 200.0],  # age_rate <- New
              [50.0, 230.0],  # dt_50_ab <- New
              [1.0, 50.0],  # dt_50_ref
              [0.5, 4.],  # epsilon (in absolute, convert to negative!!)
              [0.01, 1.0]]  # beta_moisture

# For models var36+ < 44
bounds_v11 = [[0.85, 0.97],  # z3_factor
              [0.6, 1.0], [0.2, 0.6],  # 'cZ0Z1', 'cZ'
              [0.2, 0.4],  # cadr
              [1500.0, 3650.0],  # k_g
              [0.2, 1.],  # gamma01, <- New!
              [0.0, 0.4],  # gammaZ
              [0.4, 0.6],  # f_transp
              [0.01, 0.05],  # f_oc,
              [0.3, 2000],  # k_oc, max Kd = 100
              [0.01, 0.4],  # beta_runoff
              [1.0, 3000.0],  # age_rate <- New
              [30.0, 230.0],  # dt_50_ab <- New
              [1.0, 50.0],  # dt_50_ref
              [0.5, 4.],  # epsilon (in absolute, convert to negative!!)
              [0.01, 1.0]]  # beta_moisture

# For models var44+
bounds_v12 = [[0.85, 0.97],  # z3_factor
              [0.6, 1.0], [0.2, 0.6],  # 'cZ0Z1', 'cZ'
              [0.2, 0.4],  # cadr
              [1500.0, 3650.0],  # k_g
              [0.2, 1.],  # gamma01,
              [0.0, 0.4],  # gammaZ
              [0.4, 0.6],  # f_transp
              [0.01, 0.05],  # f_oc,
              [0.3, 2000],  # k_oc, max Kd = 100
              [0.01, 0.5],  # beta_runoff
              [1.0, 3000.0],  # age_rate
              [50.0, 250.0],  # dt_50_ab
              [10.0, 50.0],  # dt_50_ref
              [0.5, 4.5],  # epsilon (in absolute, convert to negative!!)
              [0.01, 1.0]]  # beta_moisture

# For models var50+ (Gen 8)
bounds_v13 = [[0.85, 0.99],  # z3_factor
              [0.01, 1.], [0.2, 0.6],  # 'cZ0Z1', 'cZ'
              [0.01, 1.],  # cadr
              [1500.0, 3650.0],  # k_g
              [0.01, 1.],  # gamma01,
              [0.01, 1.],  # gammaZ
              [0.01, 1.],  # f_transp
              [0.01, 0.05],  # f_oc,
              [0.3, 2000],  # k_oc, max Kd = 100
              [0.01, 0.5],  # beta_runoff
              [1.0, 3000.0],  # age_rate
              [50.0, 250.0],  # dt_50_ab
              [10.0, 50.0],  # dt_50_ref
              [0.5, 4.5],  # epsilon (in absolute, convert to negative!!)
              [0.01, 1.0]]  # beta_moisture

# For models var55+ (Gen 9)
# f_evap <- New!
bounds_v14 = [[0.85, 0.99],  # z3_factor
              [0.01, 1.], [0.2, 0.6],  # 'cZ0Z1', 'cZ'
              [0.01, 1.],  # cadr
              [1500.0, 3650.0],  # k_g
              [0.01, 1.],  # gamma01,
              [0.01, 1.],  # gammaZ
              [0.1, 1.],  # f_transp
              [0.1, 0.9],  # f_evap
              [0.01, 0.05],  # f_oc,
              [0.3, 2000],  # k_oc, max Kd = 100
              [0.01, 0.5],  # beta_runoff
              [10.0, 3000.0],  # age_rate
              [65.0, 350.0],  # dt_50_ab
              [5.0, 65.0],  # dt_50_ref
              [0.3, 5.],  # epsilon (in absolute, convert to negative!!)
              [0.01, 1.0]]  # beta_moisture

# For models var61+ (Gen 10)
bounds_v15 = [[0.85, 0.99],  # z3_factor
              [0.01, 1.], [0.2, 0.6],  # 'cZ0Z1', 'cZ'
              [0.01, 1.],  # cadr
              [1500.0, 3650.0],  # k_g
              [0.01, 1.],  # gamma01,
              [0.01, 1.],  # gammaZ
              [0.1, 1.],  # f_transp
              [0.1, 0.9],  # f_evap
              [0.01, 0.05],  # f_oc,
              [0.3, 2000],  # k_oc, max Kd = 100
              [0.01, 0.5],  # beta_runoff
              [10.0, 3000.0],  # age_rate
              [65.0, 350.0],  # dt_50_ab
              [5.0, 65.0],  # dt_50_ref
              [0.3, 5.],  # epsilon (in absolute, convert to negative!!)
              [0.01, 1.0]]  # beta_moisture

bds = [bounds_v1, bounds_v2, bounds_v3, bounds_v4,
       bounds_v5, bounds_v6, bounds_v7, bounds_v8,
       bounds_v9, bounds_v10, bounds_v11, bounds_v12,
       bounds_v13, bounds_v14, bounds_v15]


def get_upper(bounds):
    upper = []
    for e in bounds:
        upper.append(e[1])
    return upper


plots = ['n1', 'n2', 'n3', 'n4', 'n5', 'n7', 'n8',
         'v4', 'v5', 'v7', 'v8', 'v9', 'v10',
         's11', 's12', 's13']

transects = ['nor', 'val', 'sou']

metrics_v1 = ['NSE', 'KGE', 'MAE', 'BIAS']
metrics_v2 = ['KGE']

levels_v1 = ['tra', 'det', 'blk', 'tot']
levels_v2 = ['tra', 'blk', 'tot']

measures = ['CONC', 'd13C', 'Q_out', 'CONC_out', 'LDS_out', 'd13C_out']  # Used in 'thin'
soilMeasures = ['CONC', 'd13C', 'theta']  # Used in 'fat'/free
outMeasures = ['Q_out', 'CONC_out', 'LDS_out', 'd13C_out']  # Used in 'fat'
