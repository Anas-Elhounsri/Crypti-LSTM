import boto3
import json
region = '<region name>'
instances = '[<Intance id>]'
ec2 = boto3.client('ec2', region_name=region)

def lambda_handler(event, context):
    ec2.start_instances(InstanceIds=instances)
    return {
        "statusCode": 200,
        "body" : "Started EC2"
    }