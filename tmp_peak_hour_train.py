from src.models.peak_hour import train_peak_hour_classifier

result = train_peak_hour_classifier()
print(result["model_name"])
print(round(result["report"]["accuracy"], 4))
print(round(result["report"]["1"]["precision"], 4))
print(round(result["report"]["1"]["recall"], 4))
print(result["plot_path"])
print(result["report_path"])
