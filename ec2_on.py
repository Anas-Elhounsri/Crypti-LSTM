import boto3
import json
region = 'us-east-1'
instances = ['i-0fec921f36ad08370']
ec2 = boto3.client('ec2', region_name=region)

def lambda_handler(event, context):
    ec2.start_instances(InstanceIds=instances)
    return {
        "statusCode": 200,
        "body" : "Started EC2"
    }