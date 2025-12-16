from src.ai_modules.nn_rl_training import train_agent

exp_folder_path, model, env = train_agent(
    puzzle_name="hypercuboid_2x2x2x3",
    n_steps=100000,  
    n_envs=20,      
    batch_size=1000,
    success_threshold=0.50,
    reward='binary',
    device="cuda"
)
''''''