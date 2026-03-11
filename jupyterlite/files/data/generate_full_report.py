# %%
import skrub
from sklearn.ensemble import ExtraTreesClassifier
from skrub import selectors as s

data = skrub.datasets.fetch_credit_fraud()

baskets = skrub.var("baskets", data.baskets)
products = skrub.var("products", data.products)  # add a new variable

X = baskets[["ID"]].skb.mark_as_X()
y = baskets["fraud_flag"].skb.mark_as_y()

# %%

vectorizer = skrub.TableVectorizer(high_cardinality=skrub.StringEncoder())
vectorized_products = products.skb.apply(vectorizer, cols=s.all() - "basket_ID")
# %%
# | echo: true
aggregated_products = vectorized_products.groupby("basket_ID").agg("mean").reset_index()

features = X.merge(aggregated_products, left_on="ID", right_on="basket_ID")
features = features.drop(columns=["ID", "basket_ID"])
# %%

predictions = features.skb.apply(ExtraTreesClassifier(n_jobs=-1), y=y)

# %%
learner = predictions.skb.make_learner()
learner.report(
    environment=predictions.skb.get_data(),
    mode="fit",
    output_dir="dataop_report",
    overwrite=True,
)
# %%
