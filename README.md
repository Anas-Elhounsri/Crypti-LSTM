# Crypti-LSTM 
This is an enhanced version of the [Crypti](https://github.com/1337Farhan/Crypti) project, which now uses a simple artificial neural networks model (Long Short-Term Memory) that takes one input (prices) and one output (predicted prices) all implemented in an automated cloud architecture (AWS).
## 1. Introduction 
The development team consists of me, [@1337Farhan](https://github.com/1337Farhan) and [@mosman4](https://github.com/mosman4/), with me responsible of developing the Crypti-LSTM and AWS architecture.
## 2. How was the cloud was set up? ☁️
Unfortunately I can't include more details on how to implement the architecture in detail but I will show an overview on what you need! We are going to list the necessary services needed for our architecture:
- AWS CLI
- An Amazon EventBridge to trigger the necessary services.
- Lambda functions:
  - One that uses coinGecko API that's triggered by Amazon EventBridge once a month to collect the necessary historical data, as seen in `GeckoApiTools.py`.
  - One that's triggered by Amazon EventBridge once a month to turn on our EC2 as seen in `ec2_on.py`.
  - One that's triggered by Amazon EventBridge once a month to turn off our EC2 as seen in `ec2_off.py`.
- An EC2 instance to host our `lstm.py` script and run it once it's triggered using a cron job.
- An s3 bucket wth two objects where one stores the historical data and the other one stores prediction.
- setup environment variables in your terminal
  - export AWS_ACCESS_KEY_ID="<your_s3_access_key_id>";
  - export AWS_SECRET_ACCESS_KEY="<your_s3_secret_key>";
  - export AWS_DEFAULT_REGION="your_aws_region";
- VPC, subnets and security groups.
## 3. High overview on how the cloud works ☁️
All of the scripts mentioned are well commented to expalin how they work:
- The eventBridge triggers `GeckoApiTools.py` once a month at a specific time, say at the begining of each month at 11:00AM.
- Then it triggers the Lambda function `ec2_on.py`  to turn on the EC2 instance at 11:10AM (it's better to give the instance some time to turn on).
- Now the EC2 will get triggered by a cron job at 11:20AM to run `lstm.py`.
- `lstm.py` will process, train and predict data to put it on the s3 bucket (`lstm.py` is well commented and explains how preparing the data, training and predicting process works)
- After a while, the eventBridge will trigger `ec2_off.py` at 11:30AM to turn off the EC2.
## 4. The cloud architecture
(A picture will be here)
## 5. Weakness in Crypti's LSTM 🐛
This model is better than the linear regression implemented on our old [Crypti](https://github.com/1337Farhan/Crypti) project, however for the sake of simplicity the LSTM only uses one input which is the historical prices, which is not the only factor that affect the prices, we also have market cap, volumes etc. For now we didnt implement it because our LSTM only produces one day worth of prediction and loops 30 times to predict 30 days, and if we use a Multivariate model, we will need to predict also for volume and market cap each time in order to produce new predicted prices, which will be harder to implement but not impossible of course.
## 6. Roadmap 🗺️
This is a long running project that will keep getting improved along development, and so far this is our goals to achieve:
- Implement a Multivariate LSTM. 
- Improve the AWS architecture by:
  - Using SageMaker instead of EC2.
  - Improving the security.
- Use a different neural networks model (Transformers).
## 7. Disclaimer ❗❗❗
This is in no way can be used as a reliable financial indicator, this was only developed for educational and testing purposes and should not be relied for any financial decisions.
