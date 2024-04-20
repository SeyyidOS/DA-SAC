from src.args import get_args
import mlflow


def main():
    args, exp_manager = get_args()

    if args.track:
        mlflow.set_experiment(args.experiment_name)
        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.start_run()

    results = exp_manager.setup_experiment()
    if results is not None:
        model, saved_hyperparams = results
        if args.track:
            mlflow.log_params(saved_hyperparams)

        if model is not None:
            exp_manager.learn(model)
            exp_manager.save_trained_model(model)
    else:
        exp_manager.hyperparameters_optimization()


if __name__ == '__main__':
    main()
