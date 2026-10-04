provider "aws" {
  region = "ap-south-1"   # Change to your region
}

resource "aws_s3_bucket" "my_bucket" {
  bucket = "gopi-credit-card-fraud-detection"   # Must be globally unique
  acl    = "private"

  tags = {
    Name        = "CreditCardFraudBucket"
    Environment = "Dev"
  }
}

# Enable versioning
resource "aws_s3_bucket_versioning" "my_bucket_versioning" {
  bucket = aws_s3_bucket.my_bucket.id

  versioning_configuration {
    status = "Enabled"
  }
}
