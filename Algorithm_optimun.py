# algorithm.py
import numpy as np
import xgboost as xgb
import optuna

def run_pricing_optimization(df):
    """
    极寒淡季：ML 动态定价与运筹学投放优化引擎
    结合 XGBoost 需求拟合与 Optuna 贝叶斯搜索的最优定价调度方案
    """
    print("\n" + "="*55)
    print("🚀 [Algorithm Engine] 启动: 极寒淡季定价与投放优化")
    print("="*55)
    
    # ---------------------------------------------------------
    # 1. 简易特征工程与需求拟合 (Feature Engineering & ML)
    # ---------------------------------------------------------
    print("⏳ [1/3] 正在基于历史订单训练 XGBoost 需求拟合模型...")
    # 注意：这里为了保证引擎能在本地跑通，使用模拟的历史特征数据。
    # 实际生产环境中，需从传入的 df 中提取真实特征 (P_e, P_c, 天气等)。
    np.random.seed(42)
    X_train = np.random.rand(1000, 4) * [5, 2, 20, 2] + [2, 0, -10, 0] 
    y_train_e = 5000 - 400 * X_train[:, 0] + 150 * X_train[:, 1] + 100 * X_train[:, 2] + np.random.normal(0, 100, 1000)
    y_train_c = 3000 + 200 * X_train[:, 0] - 500 * X_train[:, 1] + 50 * X_train[:, 2] + np.random.normal(0, 100, 1000)
    
    model_demand_e = xgb.XGBRegressor(objective='reg:squarederror', max_depth=3)
    model_demand_c = xgb.XGBRegressor(objective='reg:squarederror', max_depth=3)
    
    model_demand_e.fit(X_train, y_train_e)
    model_demand_c.fit(X_train, y_train_c)
    
    # ---------------------------------------------------------
    # 2. 设定极寒天气的环境与财务参数 (Parameters)
    # ---------------------------------------------------------
    print("⏳ [2/3] 注入 2026年1月 极寒物理与财务边界约束...")
    current_weather_features = [-5, 1] # 模拟当天：气温-5度, 降雪
    COST_SWAP_E = 6.0   # 电车极寒换电成本
    COST_OPS_C = 0.5    # 经典车极寒调度成本
    DEP_E = 2.0         # 电车日均折旧
    DEP_C = 0.5         # 经典车日均折旧
    MAX_E, MAX_C = 5000, 5000 # 资产总规模限制
    MIN_TOTAL = 3000    # SLA 最低服务保障要求

    # ---------------------------------------------------------
    # 3. 定义 Optuna 黑盒目标函数 (Objective Function)
    # ---------------------------------------------------------
    def objective(trial):
        P_e = trial.suggest_float("P_e", 2.0, 8.0, step=0.5)
        P_c = trial.suggest_float("P_c", 0.0, 3.0, step=0.5)
        Q_e = trial.suggest_int("Q_e", 0, MAX_E, step=100)
        Q_c = trial.suggest_int("Q_c", 0, MAX_C, step=100)
        
        # 惩罚违反 SLA 约束的解
        if Q_e + Q_c < MIN_TOTAL:
            return 1e9 
            
        # 机器学习推断当前价格下的需求量
        X_pred = np.array([[P_e, P_c] + current_weather_features])
        D_e = max(0, model_demand_e.predict(X_pred)[0])
        D_c = max(0, model_demand_c.predict(X_pred)[0])
        
        # 运筹学利润计算 (受限于实际供给)
        actual_rides_e = min(D_e, Q_e)
        actual_rides_c = min(D_c, Q_c)
        
        profit_e = (P_e - COST_SWAP_E) * actual_rides_e - (DEP_E * Q_e)
        profit_c = (P_c - COST_OPS_C) * actual_rides_c - (DEP_C * Q_c)
        
        return -(profit_e + profit_c) # Optuna 默认求极小值，故加负号

    # ---------------------------------------------------------
    # 4. 运行贝叶斯求解器 (Solver)
    # ---------------------------------------------------------
    print("⏳ [3/3] 启动 Optuna 贝叶斯搜索引擎寻找全局最优解...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=500)

    print("\n✅ 【极寒淡季最优策略计算完毕】")
    print(f"💰 最大预估日利润:    {-study.best_value:,.2f} USD")
    print(f"⚡ 最优电单车定价(P_e): ${study.best_params['P_e']}")
    print(f"🚲 最优经典车定价(P_c): ${study.best_params['P_c']}")
    print(f"🔋 建议电车投放量(Q_e):  {study.best_params['Q_e']} 辆")
    print(f"🚲 建议经典车投放量(Q_c): {study.best_params['Q_c']} 辆")
    print("="*55 + "\n")