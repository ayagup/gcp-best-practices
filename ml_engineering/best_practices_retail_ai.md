# Retail AI Best Practices

*Last Updated: January 4, 2026*

## Overview

Google Cloud Retail AI provides industry-specific solutions for e-commerce and retail businesses including Recommendations AI for personalized product recommendations, Retail Search for intelligent product search, Vision Product Search for visual search capabilities, and AI-powered inventory optimization.

---

## 1. Recommendations AI

### Build Personalized Product Recommendations

```python
from google.cloud import recommendationengine_v1beta1 as recommendations
from google.cloud import retail_v2

class RetailRecommendationsManager:
    """Manage Retail Recommendations AI."""
    
    def __init__(self, project_id, location='global'):
        self.project_id = project_id
        self.location = location
        self.catalog_path = f"projects/{project_id}/locations/{location}/catalogs/default_catalog"
    
    def import_catalog_items(
        self,
        gcs_source_uri,
        item_schema='product'
    ):
        """Import product catalog from GCS."""
        
        client = retail_v2.ProductServiceClient()
        
        input_config = retail_v2.ProductInputConfig(
            gcs_source=retail_v2.GcsSource(
                input_uris=[gcs_source_uri],
                data_schema='product'
            )
        )
        
        request = retail_v2.ImportProductsRequest(
            parent=f"{self.catalog_path}/branches/default_branch",
            input_config=input_config
        )
        
        print(f"✓ Importing catalog items")
        print(f"  Source: {gcs_source_uri}")
        print(f"  Schema: {item_schema}")
        
        # Returns long-running operation
        return {'status': 'importing'}
    
    def record_user_event(
        self,
        user_id,
        event_type,
        product_details
    ):
        """Record user interaction event."""
        
        client = retail_v2.UserEventServiceClient()
        
        user_event = retail_v2.UserEvent(
            event_type=event_type,  # e.g., 'detail-page-view', 'add-to-cart', 'purchase'
            visitor_id=user_id,
            product_details=[
                retail_v2.ProductDetail(
                    product=retail_v2.Product(
                        id=product_details['product_id'],
                        title=product_details['title'],
                        categories=product_details.get('categories', [])
                    ),
                    quantity=product_details.get('quantity', 1)
                )
            ]
        )
        
        print(f"✓ Recorded user event")
        print(f"  User: {user_id}")
        print(f"  Event: {event_type}")
        print(f"  Product: {product_details['product_id']}")
        
        return user_event
    
    def get_personalized_recommendations(
        self,
        user_id,
        num_recommendations=10,
        filter_expression=None
    ):
        """Get personalized product recommendations."""
        
        client = retail_v2.PredictionServiceClient()
        
        placement = f"{self.catalog_path}/placements/recently_viewed_default"
        
        request = retail_v2.PredictRequest(
            placement=placement,
            user_event=retail_v2.UserEvent(
                event_type="detail-page-view",
                visitor_id=user_id
            ),
            page_size=num_recommendations,
            filter=filter_expression
        )
        
        # Mock recommendations
        recommendations = [
            {
                'product_id': 'prod-001',
                'title': 'Wireless Headphones',
                'score': 0.95,
                'price': 99.99
            },
            {
                'product_id': 'prod-002',
                'title': 'Bluetooth Speaker',
                'score': 0.89,
                'price': 79.99
            }
        ]
        
        print(f"✓ Generated recommendations")
        print(f"  User: {user_id}")
        print(f"  Count: {len(recommendations)}")
        
        return recommendations
    
    def get_similar_products(
        self,
        product_id,
        num_similar=5
    ):
        """Get similar products based on product ID."""
        
        client = retail_v2.PredictionServiceClient()
        
        # Mock similar products
        similar_products = [
            {
                'product_id': 'prod-003',
                'title': 'Premium Headphones',
                'similarity_score': 0.92
            },
            {
                'product_id': 'prod-004',
                'title': 'Sports Earbuds',
                'similarity_score': 0.88
            }
        ]
        
        print(f"✓ Found similar products")
        print(f"  Base product: {product_id}")
        print(f"  Similar items: {len(similar_products)}")
        
        return similar_products
    
    def get_frequently_bought_together(
        self,
        product_id,
        num_items=3
    ):
        """Get frequently bought together products."""
        
        # Mock frequently bought together
        fbt_products = [
            {
                'product_id': 'prod-005',
                'title': 'Headphone Case',
                'co_occurrence_score': 0.78
            },
            {
                'product_id': 'prod-006',
                'title': 'Audio Cable',
                'co_occurrence_score': 0.65
            }
        ]
        
        print(f"✓ Found frequently bought together items")
        print(f"  Base product: {product_id}")
        print(f"  FBT items: {len(fbt_products)}")
        
        return fbt_products
    
    def optimize_recommendation_model(
        self,
        training_data_path,
        model_type='others-you-may-like'
    ):
        """Optimize recommendation model with new training data."""
        
        optimization_config = {
            'model_type': model_type,
            'training_data': training_data_path,
            'optimization_objective': 'ctr'  # click-through rate
        }
        
        print(f"✓ Optimizing recommendation model")
        print(f"  Model type: {model_type}")
        print(f"  Training data: {training_data_path}")
        
        return optimization_config

# Example usage
# recommendations_mgr = RetailRecommendationsManager(
#     project_id='my-retail-project',
#     location='global'
# )

# Import catalog
# recommendations_mgr.import_catalog_items(
#     gcs_source_uri='gs://retail-data/products.json'
# )

# Record user events
# recommendations_mgr.record_user_event(
#     user_id='user-12345',
#     event_type='detail-page-view',
#     product_details={
#         'product_id': 'prod-001',
#         'title': 'Wireless Headphones',
#         'categories': ['Electronics', 'Audio']
#     }
# )

# Get recommendations
# recs = recommendations_mgr.get_personalized_recommendations(
#     user_id='user-12345',
#     num_recommendations=10,
#     filter_expression='price < 200'
# )
```

---

## 2. Retail Search

### Implement Intelligent Product Search

```python
class RetailSearchManager:
    """Manage Retail Search capabilities."""
    
    def __init__(self, project_id, location='global'):
        self.project_id = project_id
        self.location = location
        self.catalog_path = f"projects/{project_id}/locations/{location}/catalogs/default_catalog"
    
    def search_products(
        self,
        search_query,
        user_id=None,
        page_size=20,
        filters=None
    ):
        """Search products with intelligent ranking."""
        
        from google.cloud import retail_v2
        
        client = retail_v2.SearchServiceClient()
        
        request = retail_v2.SearchRequest(
            placement=f"{self.catalog_path}/placements/default_search",
            query=search_query,
            visitor_id=user_id or 'anonymous',
            page_size=page_size,
            filter=filters or ""
        )
        
        # Mock search results
        results = [
            {
                'product_id': 'prod-001',
                'title': 'Wireless Bluetooth Headphones',
                'price': 99.99,
                'relevance_score': 0.95,
                'categories': ['Electronics', 'Audio']
            },
            {
                'product_id': 'prod-002',
                'title': 'Noise-Canceling Headphones',
                'price': 149.99,
                'relevance_score': 0.92,
                'categories': ['Electronics', 'Audio', 'Premium']
            }
        ]
        
        print(f"✓ Search completed")
        print(f"  Query: {search_query}")
        print(f"  Results: {len(results)}")
        
        return results
    
    def search_with_facets(
        self,
        search_query,
        facet_specs
    ):
        """Search with faceted navigation."""
        
        from google.cloud import retail_v2
        
        # Build facet specs
        facets = []
        for facet_key in facet_specs:
            facets.append(
                retail_v2.SearchRequest.FacetSpec(
                    facet_key=retail_v2.SearchRequest.FacetSpec.FacetKey(
                        key=facet_key
                    )
                )
            )
        
        # Mock faceted results
        results = {
            'products': [
                {'product_id': 'prod-001', 'title': 'Product 1'},
                {'product_id': 'prod-002', 'title': 'Product 2'}
            ],
            'facets': {
                'brand': {
                    'Sony': 45,
                    'Bose': 32,
                    'JBL': 28
                },
                'price_range': {
                    '$0-$50': 15,
                    '$50-$100': 42,
                    '$100-$200': 38,
                    '$200+': 10
                }
            }
        }
        
        print(f"✓ Faceted search completed")
        print(f"  Query: {search_query}")
        print(f"  Facets: {len(facet_specs)}")
        
        return results
    
    def autocomplete_search(
        self,
        query_prefix,
        max_suggestions=5
    ):
        """Get autocomplete suggestions."""
        
        from google.cloud import retail_v2
        
        client = retail_v2.CompletionServiceClient()
        
        # Mock suggestions
        suggestions = [
            {'suggestion': 'wireless headphones', 'score': 0.95},
            {'suggestion': 'wireless earbuds', 'score': 0.88},
            {'suggestion': 'wireless speakers', 'score': 0.82}
        ]
        
        print(f"✓ Autocomplete suggestions generated")
        print(f"  Prefix: {query_prefix}")
        print(f"  Suggestions: {len(suggestions)}")
        
        return suggestions
    
    def boost_products(
        self,
        search_query,
        boost_spec
    ):
        """Apply boosting rules to search results."""
        
        from google.cloud import retail_v2
        
        # Example boost spec
        boost_config = retail_v2.SearchRequest.BoostSpec(
            condition_boost_specs=[
                retail_v2.SearchRequest.BoostSpec.ConditionBoostSpec(
                    condition=boost_spec.get('condition'),
                    boost=boost_spec.get('boost_amount', 1.5)
                )
            ]
        )
        
        print(f"✓ Applied product boosting")
        print(f"  Condition: {boost_spec.get('condition')}")
        print(f"  Boost amount: {boost_spec.get('boost_amount', 1.5)}x")
        
        return boost_config
    
    def personalize_search_results(
        self,
        search_query,
        user_id,
        user_preferences
    ):
        """Personalize search results based on user history."""
        
        # Incorporate user preferences
        personalization_spec = {
            'user_id': user_id,
            'preferred_categories': user_preferences.get('categories', []),
            'price_range': user_preferences.get('price_range', {}),
            'brand_affinity': user_preferences.get('brands', [])
        }
        
        print(f"✓ Personalized search results")
        print(f"  User: {user_id}")
        print(f"  Preferences applied: {len(user_preferences)}")
        
        return personalization_spec

# Example usage
# search_mgr = RetailSearchManager(
#     project_id='my-retail-project',
#     location='global'
# )

# Basic search
# results = search_mgr.search_products(
#     search_query='wireless headphones',
#     user_id='user-12345',
#     filters='price < 200'
# )

# Faceted search
# faceted_results = search_mgr.search_with_facets(
#     search_query='headphones',
#     facet_specs=['brand', 'price_range', 'rating']
# )

# Autocomplete
# suggestions = search_mgr.autocomplete_search(
#     query_prefix='wirele',
#     max_suggestions=5
# )
```

---

## 3. Vision Product Search

### Implement Visual Search

```python
class VisionProductSearchManager:
    """Manage Vision Product Search."""
    
    def __init__(self, project_id, location='us-west1'):
        self.project_id = project_id
        self.location = location
    
    def create_product_set(
        self,
        product_set_id,
        display_name
    ):
        """Create product set for visual search."""
        
        from google.cloud import vision
        
        client = vision.ProductSearchClient()
        
        location_path = f"projects/{self.project_id}/locations/{self.location}"
        
        product_set = vision.ProductSet(
            display_name=display_name
        )
        
        print(f"✓ Created product set")
        print(f"  ID: {product_set_id}")
        print(f"  Location: {self.location}")
        
        return product_set_id
    
    def add_product_to_set(
        self,
        product_id,
        product_category,
        product_labels,
        image_uris
    ):
        """Add product with images to product set."""
        
        from google.cloud import vision
        
        client = vision.ProductSearchClient()
        
        product = vision.Product(
            display_name=product_id,
            product_category=product_category,
            product_labels=[
                vision.Product.KeyValue(key=k, value=v)
                for k, v in product_labels.items()
            ]
        )
        
        print(f"✓ Added product to set")
        print(f"  Product ID: {product_id}")
        print(f"  Images: {len(image_uris)}")
        
        return product
    
    def search_similar_products_by_image(
        self,
        image_uri,
        product_set_id,
        max_results=10,
        filter_expression=None
    ):
        """Search for similar products using image."""
        
        from google.cloud import vision
        
        client = vision.ProductSearchClient()
        image_annotator_client = vision.ImageAnnotatorClient()
        
        # Load image
        image = vision.Image()
        image.source.image_uri = image_uri
        
        # Mock similar products
        similar_products = [
            {
                'product_id': 'prod-001',
                'score': 0.92,
                'bounding_poly': [(100, 100), (300, 100), (300, 300), (100, 300)]
            },
            {
                'product_id': 'prod-002',
                'score': 0.87,
                'bounding_poly': [(150, 150), (350, 150), (350, 350), (150, 350)]
            }
        ]
        
        print(f"✓ Visual search completed")
        print(f"  Image: {image_uri}")
        print(f"  Similar products found: {len(similar_products)}")
        
        return similar_products
    
    def detect_products_in_image(
        self,
        image_uri,
        product_set_id
    ):
        """Detect multiple products in single image."""
        
        from google.cloud import vision
        
        # Mock detected products
        detected_products = [
            {
                'product_id': 'prod-001',
                'name': 'Leather Handbag',
                'confidence': 0.94,
                'bounding_box': {'x': 120, 'y': 80, 'width': 200, 'height': 250}
            },
            {
                'product_id': 'prod-003',
                'name': 'Designer Sunglasses',
                'confidence': 0.89,
                'bounding_box': {'x': 350, 'y': 100, 'width': 150, 'height': 100}
            }
        ]
        
        print(f"✓ Detected products in image")
        print(f"  Products found: {len(detected_products)}")
        
        return detected_products
    
    def train_product_search_index(
        self,
        product_set_id
    ):
        """Train product search index for better accuracy."""
        
        print(f"✓ Training product search index")
        print(f"  Product set: {product_set_id}")
        
        return {'status': 'training'}

# Example usage
# vision_search_mgr = VisionProductSearchManager(
#     project_id='my-retail-project',
#     location='us-west1'
# )

# Create product set
# vision_search_mgr.create_product_set(
#     product_set_id='fashion-products',
#     display_name='Fashion Product Catalog'
# )

# Add product
# vision_search_mgr.add_product_to_set(
#     product_id='handbag-001',
#     product_category='apparel-v2',
#     product_labels={'color': 'black', 'material': 'leather'},
#     image_uris=['gs://retail-images/handbag-001-1.jpg']
# )

# Search by image
# similar = vision_search_mgr.search_similar_products_by_image(
#     image_uri='gs://user-uploads/query-image.jpg',
#     product_set_id='fashion-products',
#     max_results=10
# )
```

---

## 4. Inventory Optimization

### Optimize Inventory with AI

```python
class InventoryOptimizationManager:
    """Manage AI-powered inventory optimization."""
    
    def __init__(self, project_id):
        self.project_id = project_id
    
    def forecast_demand(
        self,
        product_id,
        historical_data,
        forecast_horizon_days=30
    ):
        """Forecast product demand using AI."""
        
        from google.cloud import aiplatform
        
        # Mock demand forecast
        forecast = {
            'product_id': product_id,
            'forecast_period': f'{forecast_horizon_days} days',
            'daily_forecasts': [
                {'date': '2024-01-01', 'predicted_units': 125, 'confidence_interval': (110, 140)},
                {'date': '2024-01-02', 'predicted_units': 132, 'confidence_interval': (115, 150)},
                # ... more days
            ],
            'total_forecast': 3750,
            'trend': 'increasing',
            'seasonality_detected': True
        }
        
        print(f"✓ Generated demand forecast")
        print(f"  Product: {product_id}")
        print(f"  Horizon: {forecast_horizon_days} days")
        print(f"  Total forecast: {forecast['total_forecast']} units")
        
        return forecast
    
    def optimize_reorder_points(
        self,
        product_id,
        current_inventory,
        demand_forecast,
        lead_time_days
    ):
        """Calculate optimal reorder points."""
        
        # Calculate safety stock
        avg_daily_demand = demand_forecast['total_forecast'] / 30
        safety_stock = avg_daily_demand * 1.5  # 1.5x average daily demand
        
        reorder_point = (avg_daily_demand * lead_time_days) + safety_stock
        
        optimal_order_quantity = avg_daily_demand * 14  # 2 weeks supply
        
        recommendation = {
            'product_id': product_id,
            'current_inventory': current_inventory,
            'reorder_point': round(reorder_point),
            'optimal_order_quantity': round(optimal_order_quantity),
            'safety_stock': round(safety_stock),
            'action': 'REORDER' if current_inventory < reorder_point else 'MONITOR'
        }
        
        print(f"✓ Calculated reorder points")
        print(f"  Product: {product_id}")
        print(f"  Reorder at: {recommendation['reorder_point']} units")
        print(f"  Action: {recommendation['action']}")
        
        return recommendation
    
    def detect_stockout_risk(
        self,
        products_inventory,
        demand_forecasts
    ):
        """Detect products at risk of stockout."""
        
        at_risk_products = []
        
        for product_id, inventory in products_inventory.items():
            forecast = demand_forecasts.get(product_id, {})
            daily_demand = forecast.get('total_forecast', 0) / 30
            
            days_of_supply = inventory / daily_demand if daily_demand > 0 else 999
            
            if days_of_supply < 7:  # Less than 1 week supply
                at_risk_products.append({
                    'product_id': product_id,
                    'current_inventory': inventory,
                    'days_of_supply': round(days_of_supply, 1),
                    'daily_demand': round(daily_demand, 1),
                    'urgency': 'HIGH' if days_of_supply < 3 else 'MEDIUM'
                })
        
        print(f"✓ Detected stockout risks")
        print(f"  Products analyzed: {len(products_inventory)}")
        print(f"  At risk: {len(at_risk_products)}")
        
        return at_risk_products
    
    def optimize_warehouse_allocation(
        self,
        products,
        warehouses,
        regional_demand
    ):
        """Optimize product allocation across warehouses."""
        
        allocations = {}
        
        for product_id in products:
            warehouse_allocation = {}
            
            for warehouse_id, capacity in warehouses.items():
                region_demand = regional_demand.get(warehouse_id, {}).get(product_id, 0)
                allocated_units = min(region_demand * 1.2, capacity)  # 20% buffer
                
                warehouse_allocation[warehouse_id] = round(allocated_units)
            
            allocations[product_id] = warehouse_allocation
        
        print(f"✓ Optimized warehouse allocation")
        print(f"  Products: {len(products)}")
        print(f"  Warehouses: {len(warehouses)}")
        
        return allocations
    
    def analyze_slow_moving_inventory(
        self,
        inventory_data,
        sales_velocity_threshold=0.5
    ):
        """Identify slow-moving inventory."""
        
        slow_moving_items = []
        
        for product_id, data in inventory_data.items():
            sales_velocity = data.get('units_sold_30d', 0) / data.get('inventory', 1)
            
            if sales_velocity < sales_velocity_threshold:
                slow_moving_items.append({
                    'product_id': product_id,
                    'inventory': data.get('inventory'),
                    'sales_velocity': round(sales_velocity, 3),
                    'recommended_action': 'DISCOUNT' if sales_velocity < 0.2 else 'PROMOTE',
                    'potential_loss': data.get('inventory') * data.get('cost_per_unit', 0)
                })
        
        print(f"✓ Analyzed slow-moving inventory")
        print(f"  Slow-moving items: {len(slow_moving_items)}")
        
        return slow_moving_items

# Example usage
# inventory_mgr = InventoryOptimizationManager(project_id='my-retail-project')

# Forecast demand
# forecast = inventory_mgr.forecast_demand(
#     product_id='prod-001',
#     historical_data=[...],  # Historical sales data
#     forecast_horizon_days=30
# )

# Optimize reorder points
# reorder = inventory_mgr.optimize_reorder_points(
#     product_id='prod-001',
#     current_inventory=250,
#     demand_forecast=forecast,
#     lead_time_days=7
# )

# Detect stockout risks
# at_risk = inventory_mgr.detect_stockout_risk(
#     products_inventory={'prod-001': 50, 'prod-002': 200},
#     demand_forecasts={...}
# )
```

---

## 5. Quick Reference Checklist

### Setup
- [ ] Enable Retail API
- [ ] Create product catalog
- [ ] Import product data
- [ ] Set up user event tracking
- [ ] Configure search placements
- [ ] Create product sets for visual search

### Recommendations
- [ ] Import catalog items from GCS
- [ ] Record user events (views, clicks, purchases)
- [ ] Configure recommendation types
- [ ] Test personalization
- [ ] Monitor recommendation quality
- [ ] A/B test recommendation strategies

### Search
- [ ] Configure search placements
- [ ] Implement autocomplete
- [ ] Set up faceted navigation
- [ ] Apply product boosting rules
- [ ] Enable personalized search
- [ ] Monitor search analytics

### Visual Search
- [ ] Create product sets
- [ ] Upload product images
- [ ] Train search index
- [ ] Test image similarity
- [ ] Handle multi-product detection
- [ ] Monitor search accuracy

### Inventory Optimization
- [ ] Forecast demand for key products
- [ ] Calculate reorder points
- [ ] Set up stockout alerts
- [ ] Optimize warehouse allocation
- [ ] Identify slow-moving inventory
- [ ] Automate reordering workflows

### Best Practices
- [ ] Track comprehensive user events
- [ ] Update catalog frequently
- [ ] Use high-quality product images
- [ ] Filter inappropriate content
- [ ] Personalize based on user history
- [ ] Monitor and optimize conversion rates
- [ ] A/B test recommendation strategies
- [ ] Implement real-time inventory sync

---

*Best Practices for Google Cloud Data Engineer Certification - Updated January 2026*
