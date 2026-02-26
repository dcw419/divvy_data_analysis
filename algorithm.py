import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 核心特征工程：精准区分 Casual 与 Member
# ==========================================
def preprocess_for_algorithm(raw_df):
    print("⏳ [Feature Engineering] 正在进行含『用户分层(Member/Casual)』的数据透视...")
    df = raw_df.copy()
    
    df['date'] = df['started_at'].dt.date
    df['hour'] = df['started_at'].dt.hour
    df['is_member'] = (df['member_casual'] == 'member').astype(int)
    
    # 严格根据 Divvy 真实费率规则逆向推演历史 ARPU
    def calculate_historical_arpu(row):
        t = row['duration_min']
        is_mem = row['is_member']
        bike_type = row['rideable_type']
        
        if is_mem == 0: # Casual
            if bike_type == 'classic_bike': return 1.00 + 0.19 * t
            else: return 1.00 + 0.44 * t 
        else: # Member 
            if bike_type == 'classic_bike':
                return 0.0 if t <= 45 else 0.19 * (t - 45)
            else: 
                if t <= 30: return 0.19 * t
                elif 31 <= t <= 45: return 5.70
                else: return 5.70 + 0.19 * (t - 45)
                
    df['arpu'] = df.apply(calculate_historical_arpu, axis=1)
    
    panel_df = df.groupby(['date', 'hour', 'rideable_type', 'is_member']).agg(
        demand=('ride_id', 'count'),
        avg_price=('arpu', 'mean')
    ).reset_index()
    
    np.random.seed(42)
    panel_df['weather_factor'] = np.random.uniform(-15, 5, len(panel_df))
    
    print(f"✅ [Feature Engineering] 用户分层转换成功！生成 {len(panel_df)} 条数据。")
    return panel_df

# ==========================================
# 2. 训练需求拟合模型
# ==========================================
def train_demand_models(panel_df):
    print("⏳ [Algorithm] 正在训练区分 Member/Casual 的需求拟合模型...")
    data_e = panel_df[panel_df['rideable_type'] == 'electric_bike']
    X_e = data_e[['avg_price', 'weather_factor', 'hour', 'is_member']]
    model_e = xgb.XGBRegressor(n_estimators=100, max_depth=4).fit(X_e, data_e['demand'])
    
    data_c = panel_df[panel_df['rideable_type'] == 'classic_bike']
    X_c = data_c[['avg_price', 'weather_factor', 'hour', 'is_member']]
    model_c = xgb.XGBRegressor(n_estimators=100, max_depth=4).fit(X_c, data_c['demand'])
    
    return model_e, model_c

# ==========================================
# 3. 运筹优化求解器 (四维定价 + 共享供需池 + 结果打印)
# ==========================================
def run_pricing_optimization(raw_df, current_weather, current_hour, params):
    print("\n" + "="*65)
    print("🚀 [Algorithm Engine] 启动: 极寒淡季精细化 (Member vs Casual) 决策")
    print("="*65)
    
    panel_df = preprocess_for_algorithm(raw_df)
    model_e, model_c = train_demand_models(panel_df)
    
    def objective(trial):
        P_e_cas = trial.suggest_float("P_e_cas", 4.0, 15.0) 
        P_e_mem = trial.suggest_float("P_e_mem", 1.0, 6.0)  
        P_c_cas = trial.suggest_float("P_c_cas", 2.0, 8.0)
        P_c_mem = trial.suggest_float("P_c_mem", 0.0, 2.0)
        
        Q_e = trial.suggest_int("Q_e", 0, params['M_e'])
        Q_c = trial.suggest_int("Q_c", 0, params['M_c'])
        
        if Q_e + Q_c < params['Q_min']: return 1e9 # SLA 约束
            
        def predict_d(model, p_cas, p_mem):
            d_cas = max(0, model.predict(pd.DataFrame([[p_cas, current_weather, current_hour, 0]], columns=model.feature_names_in_))[0])
            d_mem = max(0, model.predict(pd.DataFrame([[p_mem, current_weather, current_hour, 1]], columns=model.feature_names_in_))[0])
            return d_cas, d_mem
            
        D_e_cas, D_e_mem = predict_d(model_e, P_e_cas, P_e_mem)
        D_c_cas, D_c_mem = predict_d(model_c, P_c_cas, P_c_mem)
        
        Total_D_e = D_e_cas + D_e_mem + 1e-5 
        Total_D_c = D_c_cas + D_c_mem + 1e-5
        
        Y_e_total = min(Total_D_e, Q_e)
        Y_c_total = min(Total_D_c, Q_c)
        
        Y_e_cas = Y_e_total * (D_e_cas / Total_D_e)
        Y_e_mem = Y_e_total * (D_e_mem / Total_D_e)
        Y_c_cas = Y_c_total * (D_c_cas / Total_D_c)
        Y_c_mem = Y_c_total * (D_c_mem / Total_D_c)
        
        revenue = (P_e_cas * Y_e_cas) + (P_e_mem * Y_e_mem) + (P_c_cas * Y_c_cas) + (P_c_mem * Y_c_mem)
        cost_ops = params['C_e'] * Y_e_total + params['C_c'] * Y_c_total
        cost_dep = params['F_e'] * Q_e + params['F_c'] * Q_c
        profit = revenue - cost_ops - cost_dep
        
        return -profit 

    print("⏳ [Algorithm] 寻找 Casual/Member 双重最优解...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=300) 

    best_strategy = study.best_params
    expected_profit = -study.best_value

    # ==========================================
    # 🚨 新增：将打印报告逻辑内聚到 Algorithm 内部
    # ==========================================
    print("\n" + "="*60)
    print("🎯 今日极寒早高峰：双客群 (Member/Casual) 分层执行指令")
    print("="*60)
    print(f"📈 预期系统早高峰总利润 : ${expected_profit:,.2f}\n")
    
    print("【1. 散客策略 (Casual) - 主攻利润与拉新】")
    print(f"   ⚡ 电单车目标 ARPU   : ${best_strategy['P_e_cas']:.2f} (建议: 推出冬季暖心一口价)")
    print(f"   🚲 经典车目标 ARPU   : ${best_strategy['P_c_cas']:.2f} (建议: 维持 $1+$0.19 基础费率)\n")
    
    print("【2. 会员策略 (Member) - 主攻防流失与复购】")
    print(f"   ⚡ 电单车目标 ARPU   : ${best_strategy['P_e_mem']:.2f} (建议: 触发费率打折或调整封顶上限)")
    print(f"   🚲 经典车目标 ARPU   : ${best_strategy['P_c_mem']:.2f} (建议: 继续保持 45min 内免费)\n")
    
    print("【3. 资产调度方案 (冬眠计划)】")
    print(f"   🚛 建议电车投放 (Q_e) : {best_strategy['Q_e']} 辆")
    print(f"   🚛 建议经典车投放(Q_c): {best_strategy['Q_c']} 辆")
    print("="*60 + "\n")

    return best_strategy, expected_profit
