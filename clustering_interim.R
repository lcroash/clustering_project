# Customer Segmentation using The Instacart Online Grocery Shopping Dataset 2017

# The instacart dataset was developed for a kaggle competition to develop a predictive market basket analysis model.
# For the purposes of this project, I will be evaluating various clustering techniques to define a set of customer types.
# It is my hypothesis that a more fractured market basket analysis will provide association rules with more substantial
# support for inter-cluster groups that for the customer base as a whole.


### Loading the necessary libraries:

library(dplyr)
library(tidyr)
library(stringr)
library(cluster)

library(Rclusterpp)
library(clValid)
library(factoextra)
library(fpc)
library(NbClust)
library(clustertend)
library(circlize)
library(RColorBrewer)
library(dendextend)
library(scatterplot3d)


### Begin by loading each of the supplied csv files
setwd("E:/College/Project/data")

aisles = read.csv("aisles.csv", header = T)
departments = read.csv("departments.csv", header = T)
order_prior = read.csv("order_products__prior.csv", header = T)
order_train = read.csv("order_products__train.csv", header = T)
orders = read.csv("orders.csv", header = T)
products = read.csv("products.csv", header = T)

### Aisle data
# aisle consists of 134 entries of aisle id and name.
# aisle could be important to differentiate between customer types:
#   - most frequent aisles
# - does a customer only shop certain aisles?
  
head(aisles, 5)
str(aisles)

### Department data
# departments consists of 21 entries of department id and name
# department could be important to defining customers similarly to aisle:
#   - cross department shopping
# - does the customer shop certain departments first?
head(departments, 5)
str(departments)

### Prior order data
# - order_prior consists 32 millions entries corresponding the orders placed (at sku level) before the users most recent order:
#   - order_id: the id of the order
# - product_id: the id of the product
# - add_to_cart_order: the order that the product was added to the basket
# - reorder: whether the product has previously been ordered by this customer - 1 if ordered previously, 0 otherwise.
# 
# - add_to_cart_order could tell us about product affinity or importance of products to the customer.
# - do they shop staples first, is there a shopping path used by customers.
# - reorder: is the customer likely to try new products or do they consistently purchase the same items, if so, how regularly?

head(order_prior, 5)
str(order_prior)
  
### Training order data
# Similar to order_prior

head(order_train, 5)
str(order_train)

### Orders data
# - orders consists of 3.4M entries:
#   - order_id: the id of the order
# - user_id: the id of the user
# - eval_set: 3 possible values for data type: 
#   - prior: orders prior to the users most recent order
# - train: training data for kaggle competition
# - test: test data held back for kaggle competition
# - order_number: the order sequence number for this customer (1 = 1st, n = nth)
# - order_dow: order day of the week. This is not explicitly defined so we will make an assumption based on EDA.
# - order_hour_of_day: the hour of the day when this order was placed
# - days_since_prior_order: days since their last order
  
summary(orders)
head(orders, 5)
str(orders)  
  
### Product data
# - products consists of 50k product lines:
#   - product_id: the id of the product
# - product_name: descriptive name of the product
# - aisle_id: the aisle id
# - department_id: department id

head(products, 5)
str(products)

# ## Initial Intuition #########
# 
# - Product level information will not be useful to isolate macro level clusters of customers.
# - Training and test sets should be drawn from the complete orders set rather than using the train, test, prior markers in the orders data since we are looking at customer data. Then we split the based on customer id.
# 
# ### Clustering can be performed using a variety of methods:
# RFM: calculate the recency, monetary and frequency characteristics of customers then cluster using KMEANS
# - Department Affinity: calculate the frequency of orders within departments, average transaction values, customer value and department preferences in absolute terms.
# - Develop higher level divisions: fresh food, chilled, ambient, frozen, organic only and use the same metric types as department
# - Classify customers based on when they shop, weekday and time. 
# - Classify customers based on dietary preferences
# 
# ### Build out the products data 
# Tag products with dietary preferences by specifying whether or not a product contain the text:
#   - Organic
# - Gluten Free
# - Vegetarian
# - Dairy Free

products = products%>%
  mutate(is_organic = ifelse(str_detect(str_to_lower(products$product_name),'organic'),1,0))

products = products%>%
  mutate(is_gluten_free = ifelse(str_detect(str_to_lower(products$product_name),'gluten free'), 1,0))

products = products%>%
  mutate(is_vegetarian = ifelse(str_detect(str_to_lower(products$product_name),'vegetarian'), 1,0))

products = products%>%
  mutate(is_dairy_free = ifelse(str_detect(str_to_lower(products$product_name),'dairy free'), 1,0))

str(products)
summary(products)

### Build out the transaction data 

# Concatenate the prior and training order dataframes to have a full set of transaction details
# Before merging the dataframes, check of they have the same column structure

# Do the dataframes have the same column headers?
names(order_prior) == names(order_train)

# Column headers are the same.
# To avoid duplication of transaction data, check for order_id duplication in both dataframes
order_id_prior = order_prior %>%
  select(order_id) %>%
  unique()

order_id_train = order_train %>%
  select(order_id) %>%
  unique()

# If there is any duplication of order_id we will have to perform a dedupe of the data.
intersect(order_id_prior, order_id_train)


# Since there is no order_id duplication, we can safety perform a row-wise concatenation of the 2 dataframes
item_level_orders = rbind(order_prior, order_train)

# Build a complete set of transactions, 
# To the SKU level transaction data, add the product information and the order information
full_orders = item_level_orders %>%
  left_join(products, by = "product_id")

# Add on order data
full_data = full_orders %>%
  left_join(orders, by = "order_id")

dim(full_orders)
summary(full_orders)

### Build out the user data 
# 
# Examining the full_orders dataframe, we see that test orders do not appear in the transaction data,, they only appear in the orders dataframe. <br>
#   As we are concerned with complete transactions only to build out the user data, we can remove the test orders when developing the RFM matrix
complete_orders = orders[orders$eval_set != "test",]

dim(orders)
dim(complete_orders)
dim(orders[orders$eval_set == "test",])

### Cannot develop an RFM matrix due to the absence of spend data.
# We can however, use quantity of items as a proxy for testing under the assumption that larger numbers of items correspond to larger transaction values.

### Working out the "M" part of the RFM matrix
# Get the quantity of products in each order
quantity =  item_level_orders %>%
  group_by(order_id) %>%
  summarise(no_products = n())

# Add the order quantity to the order detail
complete_orders = complete_orders %>%
  left_join(quantity, by = "order_id")

# Sum the total items purchased by each user
quantity_items = complete_orders %>%
  group_by(user_id) %>%
  summarise(total_products = sum(no_products))

# Summary to check for NA but OK!
summary(quantity_items)

### Working out the "R" part of the RFM matrix
# R in RFM terms refers to recency: how recently the customer / user has made a transaction
recent = complete_orders %>%
  group_by(user_id) %>%
  summarise(average_days_between_transactions = mean(days_since_prior_order, na.rm = TRUE))

### Working out the "F" part of the RFM matrix 
# F in RFM terms refers to Frequency: how frequently does the customer / user make transactions
freq_transactions = complete_orders %>%
  group_by(user_id) %>%
  summarise(no_transactions = n())

# Bringing the RFM data together
users.rfm = recent %>%
  left_join(freq_transactions, by = "user_id") %>%
  left_join(quantity_items, by = "user_id")

names(users.rfm) = c("user_id", "recency", "frequency", "quantity" )
head(users.rfm, 5)

# # Working out the scoring rank of users for each of the RFM (RFQ) scores
# RFM is scored on a scale from 1 - 5, where for:
#   - Recency: 1 = not recent, 5 = very recent
# - Frequency: 1 = not frequent, 5 = very frequent
# - Monetary (Quantity in this case): 1 = low quantity, 5 = high quantity
# 
# ### Scoring methodolgy
# First step is to split the range of values under each RFM heading in the number of groups: max_score <br>
#   Then scores are applied by bucketing users into each of their respective score band/quantiles.

max_score = 10

quant_score = seq(1, max_score)
quant_split = seq(0, 1, 1/max_score)
quants_recency = quantile(users.rfm$recency, quant_split)
quants_freq = quantile(users.rfm$frequency, quant_split)
quants_quantity = quantile(users.rfm$quantity, quant_split)

users.rfm$recency_score = as.numeric(cut(users.rfm$recency, quants_recency, quant_score))
users.rfm$freq_score = as.numeric(cut(users.rfm$frequency, quants_freq, quant_score))
users.rfm$quantity_score = as.numeric(cut(users.rfm$quantity, quants_quantity, quant_score))
users.rfm[is.na(users.rfm)] = 1
summary(users.rfm)


### Due the large size of the data files, it may be necessary to remove these from working memory in order to maintain efficient processing<br>
# Sorting by object size and whether or not the objects are relevant from this point onwards, we can remove some larger objects from memory.
# rm(list = ls())

size_of_objects_a = sapply(ls(), function(x) format(object.size(get(x)), unit = 'auto'))
size_of_objects_b = sapply(ls(), function(x) object.size(get(x)))
size_of_objects = data.frame(ls(), size_of_objects_a, size_of_objects_b)
size_of_objects = size_of_objects %>%
  arrange(desc(size_of_objects_b))

#size_of_objects

# head(users)
# head(full_data)
# Can drop: order_prior, full_orders, item_level_orders, orders, quantity, order_id_prior, order_train,
rm(list = c("order_prior", "full_orders", "item_level_orders", "orders", "quantity", "order_id_prior", "order_train"))

### Building a usable Sample dataset

# The dataset is 33M+ observation, so we take a subset to develop our cluster input. <br>
#   1. Start by identifying a subset of customers and selecting their transactions
summary(users.rfm)

# There are 206,209 users, so we can take a sample of 5% to check on clusterability

set.seed(1)
sample_size = 0.05
mask = sample(1:nrow(users.rfm), round(nrow(users.rfm) * sample_size, 0))
sample_users = users.rfm[mask,]$user_id
sample_orders = complete_orders[complete_orders$user_id %in% sample_users,]$order_id
sample_data = full_data[full_data$order_id %in% sample_orders,]

length(sample_users)
length(sample_orders)
dim(sample_data)

head(sample_data)

### Breakdown by category
# Get a total number of products purchased per department per user
categories = sample_data %>%
  select(user_id,department_id) %>%
  left_join(departments, by = "department_id") %>%
  group_by(user_id, department) %>%
  summarise(
    count = n()
  ) 
summary(categories)

### Breakdown by sub-category
# Get a total number of products purchased per sub-category per user
# Breakdown by sub-category
sub.categories = sample_data %>%
  select(department_id, aisle_id) %>%
  arrange(department_id, aisle_id) %>%
  left_join(aisles, by = "aisle_id") %>%
  left_join(departments, by = "department_id") %>%
  select(department_id, department, aisle_id, aisle) %>%
  group_by(department_id, department, aisle_id, aisle) %>%
  summarise(
    count = n()
  ) %>%
  arrange(desc(count))
sum(data.frame(categories)$count)

# Describe customers with special dietary needs
user.diet = sample_data %>%
  group_by(user_id) %>%
  summarise(
    organic.pref = mean(is_organic),
    gluten.pref = mean(is_gluten_free),
    veg.pref = mean(is_vegetarian), 
    dairy.pref = mean(is_dairy_free)
  )

summary(user.diet)

boxplot(user.diet[-1])
hist(user.diet$gluten.pref)

user.diet$gluten.pref = (user.diet$gluten.pref > 0) * 1
user.diet$veg.pref = (user.diet$veg.pref > 0) * 1
user.diet$dairy.pref = (user.diet$dairy.pref > 0) * 1

summary(sample_user)
summary(user.data)
sample_user = users.rfm[users.rfm$user_id %in% sample_users,]

# Join the dietary information to the RFM dataframe
sample_user = sample_user %>%
  left_join(user.diet, by = "user_id")


# "One hot encode" the total customer values per department
ohe.users = spread(categories, key = department, value = count, fill = 0)
ohe.users = ohe.users %>% 
  drop_na()

# separate the user_id column from the user data.
user.ids = ohe.users$user_id
user.data = ohe.users[-1]


boxplot(scale(user.data))
boxplot(scale(sample_user[-1]))

### ASSESSING CLUSTER TENDENCY
# Does the data contain meaningful clusters?


### K-Means Clustering
# Given that the first attempts using hieracrchical clustering did not give any insightful groupings using the transactional data, we can move to K-Means.<br>
#   K-Means being very popular and using numerical data, we should expect customers with similar buying patterns to be grouped closely in n-dimensional space. <br>
#   
#   1st step is to identify a useful value for k - the number of clusters that can be identified by the algorithm.<br>
#   Starting with 1 - 10, we will identify the number of clusters that minimises the within group sum of squares distance.

# Perform principle component analysis to allow a visual inspection of the clustering later.
# Perform PCA
perform.pca = function(df, users.included = FALSE, scale = FALSE, color) {
  if(users.included == TRUE) {
    pca.out = prcomp(df[-1], scale = scale)
  } else {
    pca.out = prcomp(df, scale = scale)
  }
  plot(pca.out$x[, c(1,2)], col = color, main = "K-Means Clustering with Principal Components - PC1 vs PC2")
  plot(pca.out$x[, c(2,3)], col = color, main = "K-Means Clustering with Principal Components - PC2 vs PC3")
  # Build scree plot and cumulative PVE plot
  pve = (100 * pca.out$sdev^2) / sum(pca.out$sdev^2)
  
  plot(pve, type = "o", ylab = "PVE", xlab = "Principal Component", col = color, main = "Scree plot for PCA")
  scatterplot3d(pca.out$x[, c(1,3)], color = color)
}


perform.kmeans = function(df, users.included = FALSE, scale = FALSE, number_samples = 10) {
  
  if(users.included == TRUE) {
    df = df[-1]
  }
  if(scale == TRUE) {
    df = scale(df)
  }
  
  set.seed(1)
  sims = c()
  for (i in 1:number_samples) {
    km.out = kmeans(df, i, nstart = 25)
    sims = c(tots, km.out$tot.withinss)
    
  }
  plot(sims, main = "Within Sum of Squares for K = 1:N")
}

# From the plot we see that the within group sum of squares does not greatly decrease with additional clusters after k = 4. 
# We will run K-Means with 4 to set the standard we can use to compare the correlation based hierarchical clustering to.

perform.kmeans(sample_user, users.included = TRUE, scale = TRUE, number_samples = 10)
set.seed(1)
km.out = kmeans(scale(sample_user[,-1]), 5, nstart = 25)
user.kmeans = km.out$cluster

perform.kmeans(user.data, users.included = FALSE, scale = TRUE, number_samples = 10)
set.seed(1)
km.out = kmeans(scale(user.data), 5, nstart = 25)
data.kmeans = km.out$cluster

# Compare if using transaction and behavioural data return the same clustering.
# For what group of customers?
table(user.kmeans, data.kmeans)
table(user.kmeans)

# Plot kmeans clusters on principal components
perform.pca(sample_user,users.included = TRUE, scale = TRUE, color = user.kmeans )
perform.pca(user.data, scale = TRUE, color = data.kmeans)

### Investigate the clustering of user data based on the different linkage varieties to decide if one method is superior.
# All examples use scaled data, as some departments are inherantly more commonly used than others.

# Complete Linkage
h.clust_comp = hclust(dist(scale(user.data)), method = "complete")
plot(h.clust_comp,main = "Complete Linkage", xlab = "", ylab = "", sub = "", cex = 0.9)
scaled.user = cutree(h.clust_comp, 5)
table(scaled.user)

# Average Linkage
h.clust_avg = hclust(dist(scale(user.data)), method = "average")
plot(h.clust_avg,main = "Average Linkage", xlab = "", ylab = "", sub = "", cex = 0.9)
avg.cut = cutree(h.clust_avg, 5)
table(avg.cut)

#Single Linkage
h.clust_sing = hclust(dist(scale(user.data)), method = "single")
plot(h.clust_sing, main = "Single Linkage", xlab = "", ylab = "", sub = "", cex = 0.9)
sing.cut = cutree(h.clust_sing, 5)
table(sing.cut)

### We can see that none of these clustering model are not very useful to identify a diverse set of users.
# We have other options with hierarchical clustering e.g. correlation based rather distance based clustering.
# This may be more practical in this environment where we want to identify customers with similar behaviour rather than similar traits.

# Cluster based on correlated data and complete linkage
samp_dist = as.dist(1-cor(t(user.data)))
h.clust_cor = hclust(samp_dist, method = "complete")
plot(h.clust_cor, main = "Complete Linkage - Correlation", labels = users, xlab = "", ylab = "", sub = "", cex = 0.9)
samp.cut = cutree(h.clust_cor, 5)
table(samp.cut)/sum(table(samp.cut))
corcomp.user = samp.cut

# We can see that correlation based approach ends up with a much more evenly spread cluster distribution. <br>
#   
#   Looking at a matrix of classification of euclidean vs correlation distance, we see that scaled euclidean distance puts 99.97% of users into cluster 1.<br>
#   Correlation distance has a more favourable distribution among the clusters, but a lower level cut would likely result in a more even distribution.

table(corcomp.user, scaled.user)
table(scaled.user)/sum(table(scaled.user))
table(corcomp.user)/sum(table(corcomp.user))






  
  











