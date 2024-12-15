import os
import sys
import torch

# Añadir tanto src como examples al path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'examples'))

# Ahora importar
from complete_pipeline import process_complete

# Usar el proceso
summary = process_complete(
    video_url="https://www.youtube.com/watch?v=FBVi4wLxotU",
    output_dir="examples/output",
    model_players_path="models/players/best.pt",
    model_ball_path="models/ball/best.pt",
    example_path="examples/pre_saved_video/example_hardcourt.mp4",
    device="cuda" if torch.cuda.is_available() else "cpu",
    start_time="00:07:47",
    duration="00:08:01"
)