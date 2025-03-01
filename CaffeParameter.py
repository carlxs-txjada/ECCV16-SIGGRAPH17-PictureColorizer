# CaffeParameter.py
import argparse
import os

def get_parameters():
    parser = argparse.ArgumentParser(description="Caffe Colorizer Parameters")

    # Parámetros de entrada/salida
    parser.add_argument('-i', '--img_path', type=str, required=True, help='Path to input image')
    parser.add_argument('-o', '--output_dir', type=str, default='Outputs', help='Directory to save results')

    # Opciones del modelo
    parser.add_argument('--use_gpu', action='store_true', help='Enable GPU acceleration if available')
    parser.add_argument('--model', choices=['eccv16', 'siggraph17', 'both'], default='both', help='Choose colorization model')

    args = parser.parse_args()

    # Crear directorio de salida si no existe
    os.makedirs(args.output_dir, exist_ok=True)

    return args
