from nsfw_detector import predict
model = predict.load_model("nsfw.299x299.h5")  # Load once
result = predict.classify(model, "test.jpg")
print(result)
