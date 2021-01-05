def create_quadratic_effect_formula(cov_names):

    quad_terms = "+".join([f"I({x}**2)" for x in cov_names])
    return quad_terms


def create_main_effect_formula(cov_names):

    main_effects = "+".join(cov_names)
    return main_effects


def create_interaction_formula(cov_names):

    main_effects = create_main_effect_formula(cov_names)
    inter_terms = "(" + main_effects + ")"
    inter_terms = inter_terms + ":" + inter_terms

    return inter_terms


def create_formula(
    cov_names,
    main_effects=True,
    quadratic_effects=True,
    interactions=False,
    intercept=True,
):

    model_str = ""

    if main_effects:
        model_str = model_str + create_main_effect_formula(cov_names)

    if quadratic_effects:
        addition = "+" if len(model_str) > 0 else ""
        model_str = model_str + addition + create_quadratic_effect_formula(cov_names)

    if interactions:
        addition = "+" if len(model_str) > 0 else ""
        model_str = model_str + addition + create_interaction_formula(cov_names)

    if not intercept:
        model_str = model_str + "- 1"

    return model_str
