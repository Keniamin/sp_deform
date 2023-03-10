{
    'normalization': {
        'yield_stress': 0.5,
        'normalized_young_modulus': 2000,
        'reference_rate': 0.001,
    },
    'dunne': {
        'full': {
            'A': 1.028072092847282,
            'B': 0.057970577625601226,
            'alpha': 1.6006430546660633,
            'D': 0.06757199257401778,
            'G': 0.8834128604059808,
            'beta': 0,
            'phi': 0,
            'H_0': 8.074063990197285,
            'H': 4.140255177841624,
        },
        'critical_size_mixin': {
            'reference_critical_rate': 487.45351175184663,
            'mu': 1.767083703517052,
        },
        'refinement_mixin': {
            's_0': 9.061962422094087,
            'theta': 2.2540400672768985,
            'r': 1.573712461538416,
        },
        'refinement_mixin_strong': {
            # Параметры из оригинальной работы, приводящие к полному
            # измельчению при сравнительно низком размере зерна
            's_0': 2.08,
            'theta': 7.10,
            'r': 1.2442,
        },
    },
}
