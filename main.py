from data_handler import DATA
from trainer import Trainer
import argparse


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--niters', type=int, default=5000, help='Number of training iterations')
	parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
	parser.add_argument('--filters', type=int, default=2, help='Number of filters')
	parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate imG')
	parser.add_argument('--data_path', type=str, default='golf.npz',help='Path to data.')
	parser.add_argument('--val_data_path', type=str, default='golf.npz',help='Path to val data.')
	parser.add_argument('--ngpu', type=int, default=2, help='Number of GPUs')
	parser.add_argument('--steps_per_log', type=int, default=10, help='Output Iterations')
	parser.add_argument('--steps_per_checkpoint', type=int, default=100, help='Checkpoint Save Iterations')
	parser.add_argument('--log_dir', type=str, default='log', help='Save Location')
	parser.add_argument('--device', type=str, default='cuda', help='Torch Device Choice')
	parser.add_argument('--load_params', type=bool, default=False, help='Load Parameters form pickle in log dir')
	params = parser.parse_args()
	print(params)
	dataset_train = DATA(path=params.data_path)

	trainer = Trainer(dataset_train, params=params)
	trainer.train()

if __name__ == '__main__':
	main()
