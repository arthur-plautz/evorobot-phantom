from datainterface.baseinterface import BaseInterface

class SpecialistFitEvaluation(BaseInterface):
    def __init__(self, env, seed, name):
        features = [
            'gen',
            'prefit_tp',
            'prefit_fp',
            'prefit_fn',
            'prefit_tn',
            'postfit_tp',
            'postfit_fp',
            'postfit_fn',
            'postfit_tn',
        ]
        super().__init__(env, seed, features, name)
