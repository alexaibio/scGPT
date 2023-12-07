from tutorials._predict import plot_perturbation, predict


################### Predict and Plot

# load best model and pert_data
save_dir = None
best_model = None
pert_data = None

# predict
predict(best_model, [["FEV"], ["FEV", "SAMD11"]])

# plot

data_name = "adamson"
split = "simulation"
if data_name == "norman":
    perts_to_plot = ["SAMD1+ZBTB1"]
elif data_name == "adamson":
    perts_to_plot = ["KCTD16+ctrl"]


for p in perts_to_plot:
    plot_perturbation(best_model, pert_data, p, pool_size=300, save_file=f"{save_dir}/{p}.png")