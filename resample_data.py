import os
import sys
import hydra
import librosa
from tqdm import tqdm
from pathlib import Path
import soundfile as sf

from src.utils.io_utils import ROOT_PATH


def resample_data(data_dir, out_dirs, target_srs):
    for subdir in tqdm(Path(data_dir).iterdir(), desc='preparing data'):
        for fpath in Path(subdir).iterdir():
            audio, original_sr = librosa.load(fpath, sr=None)

            for out_dir, target_sr in zip(out_dirs, target_srs):
                out_path = ROOT_PATH / out_dir / subdir.name
                if not out_path.exists():
                    out_path.mkdir(exist_ok=True, parents=True)
        
                audio_resampled = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
                sf.write(out_path / fpath.name, audio_resampled, target_sr)
        

@hydra.main(version_base=None, config_path="src/configs", config_name="resample")
def main(config):
    resample_data(config.data_dir, config.out_dirs, config.target_srs)



if __name__ == '__main__':
    main()
