# Looker Studio Best Practices

*Last Updated: December 26, 2025*

## Overview

Looker Studio (formerly Google Data Studio) is a free, cloud-based business intelligence and data visualization tool that transforms data into customizable, informative reports and dashboards. It connects to multiple data sources and enables collaborative analytics with easy sharing and real-time updates.

---

## 1. Data Source Configuration

### Connecting Data Sources

**Best Practices:**
- Use native connectors when available
- Implement data blending carefully
- Cache data appropriately
- Monitor quota usage

```javascript
// Common data source types:
// - BigQuery (recommended for large datasets)
// - Google Sheets
// - Google Analytics 4
// - MySQL, PostgreSQL
// - Cloud Storage (CSV files)
// - Google Ads
// - YouTube Analytics
// - Search Console

// BigQuery Connection Best Practices
/*
1. Use partitioned tables for date-based queries
2. Enable billing project for query costs
3. Use custom queries for complex logic
4. Implement incremental data refresh
5. Set appropriate data freshness settings
*/

// Example: BigQuery Custom Query
SELECT
  date,
  product_category,
  region,
  SUM(revenue) AS total_revenue,
  COUNT(DISTINCT order_id) AS order_count,
  AVG(order_value) AS avg_order_value
FROM
  `project.dataset.orders`
WHERE
  date >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
GROUP BY
  date,
  product_category,
  region
```

### Data Freshness Settings

**Configure Refresh Intervals:**
```
Data Source Settings:
├── Auto-update (recommended for real-time data)
│   ├── Enable data cache: YES
│   ├── Cache duration: 4 hours (adjustable)
│   └── Refresh interval: Based on data source
├── Manual refresh
│   └── Use for infrequently changing data
└── Scheduled refresh
    ├── Daily at specific time
    ├── Hourly updates
    └── Every 15 minutes (for real-time needs)

Best Practices:
- BigQuery: 4-12 hours cache for large datasets
- Google Sheets: 15 minutes - 1 hour
- Google Analytics 4: 4-24 hours
- Static reports: Daily or manual refresh
```

### Data Blending

**Combine Multiple Data Sources:**
```
Data Blending Configuration:

Source 1: Sales Data (BigQuery)
├── Join Key: customer_id
├── Fields: customer_id, order_date, revenue
└── Date Range Dimension: order_date

Source 2: Customer Demographics (Google Sheets)
├── Join Key: customer_id
├── Fields: customer_id, age, region, segment
└── Join Type: LEFT OUTER JOIN

Blending Rules:
1. Use consistent join keys (same data type)
2. Limit to 5 data sources max per blend
3. Pre-aggregate data before blending
4. Avoid blending large tables directly
5. Use BigQuery views for complex joins

Example Blended Data Model:
customer_id (Primary Key from Sales Data)
├── From Sales: revenue, order_count, order_date
├── From Demographics: age, region, segment
└── Calculated Field: revenue_per_customer
```

---

## 2. Report Design

### Layout and Structure

**Best Practices:**
- Use grid system for alignment
- Maintain consistent spacing
- Group related visualizations
- Implement responsive design

```
Report Layout Structure:

┌─────────────────────────────────────────────────────┐
│ HEADER SECTION                                       │
│ ├── Company Logo (left)                             │
│ ├── Report Title (center)                           │
│ └── Date Range Filter (right)                       │
├─────────────────────────────────────────────────────┤
│ FILTER BAR                                           │
│ ├── Region Filter                                   │
│ ├── Product Category Filter                         │
│ └── Customer Segment Filter                         │
├─────────────────────────────────────────────────────┤
│ KEY METRICS (Scorecards)                            │
│ ┌──────────┬──────────┬──────────┬──────────┐      │
│ │ Total    │ Total    │ Avg Order│ Customer │      │
│ │ Revenue  │ Orders   │ Value    │ Count    │      │
│ └──────────┴──────────┴──────────┴──────────┘      │
├─────────────────────────────────────────────────────┤
│ TREND ANALYSIS                                       │
│ ┌─────────────────────────────────────────────┐    │
│ │ Revenue Over Time (Line Chart)               │    │
│ │ - Primary: Revenue                           │    │
│ │ - Secondary: Order Count                     │    │
│ └─────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────┤
│ BREAKDOWN ANALYSIS                                   │
│ ┌──────────────────┬──────────────────────────┐    │
│ │ By Region        │ By Product Category       │    │
│ │ (Bar Chart)      │ (Pie Chart)               │    │
│ └──────────────────┴──────────────────────────┘    │
├─────────────────────────────────────────────────────┤
│ DETAILED DATA                                        │
│ ┌─────────────────────────────────────────────┐    │
│ │ Transaction Table                            │    │
│ │ - Sortable columns                           │    │
│ │ - Conditional formatting                     │    │
│ └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘

Design Guidelines:
- Grid: 12-column layout
- Margins: 20px all sides
- Spacing between sections: 30px
- Chart padding: 10px
- Font hierarchy:
  ├── Title: 24px, Bold
  ├── Section headers: 18px, Bold
  ├── Chart titles: 14px, Medium
  └── Body text: 12px, Regular
```

### Color Schemes

**Consistent Color Usage:**
```
Color Palette Recommendations:

Primary Palette (for main metrics):
├── Blue: #4285F4 (Primary metric)
├── Green: #34A853 (Positive indicators)
├── Red: #EA4335 (Negative indicators)
├── Yellow: #FBBC04 (Warnings)
└── Purple: #9C27B0 (Secondary metric)

Categorical Colors (for dimensions):
├── Category 1: #4285F4
├── Category 2: #34A853
├── Category 3: #FBBC04
├── Category 4: #EA4335
├── Category 5: #9C27B0
└── Category 6+: Auto-generate from base palette

Sequential Colors (for heatmaps):
Light to Dark Blue: #E3F2FD → #0D47A1

Diverging Colors (for variance):
Red to Green: #EA4335 → #FFFFFF → #34A853

Accessibility:
- Ensure sufficient contrast (WCAG AA: 4.5:1)
- Avoid red-green only combinations
- Use patterns/shapes in addition to colors
- Test with color-blind simulators

Theme Configuration:
{
  "primaryColor": "#4285F4",
  "secondaryColor": "#34A853",
  "backgroundColor": "#FFFFFF",
  "textColor": "#202124",
  "gridColor": "#DADCE0",
  "accentColor": "#9C27B0"
}
```

---

## 3. Chart Selection and Configuration

### Chart Type Selection

**Choose Appropriate Visualizations:**
```
Chart Selection Guide:

1. SCORECARD
   Use for: Single KPI values
   Best for: Total Revenue, Count, Average
   Configuration:
   ├── Metric: Single measure
   ├── Comparison: Previous period, target
   ├── Compact/Detailed view
   └── Sparkline: Optional trend

2. TIME SERIES (Line Chart)
   Use for: Trends over time
   Best for: Revenue trends, user growth
   Configuration:
   ├── Date dimension: Date, Week, Month
   ├── Metrics: 1-3 measures
   ├── Smoothing: Optional
   └── Missing data: Interpolate or show gaps

3. BAR CHART
   Use for: Comparing categories
   Best for: Sales by region, products
   Configuration:
   ├── Dimension: Category
   ├── Metric: Measure to compare
   ├── Orientation: Horizontal/Vertical
   ├── Sorting: By metric value
   └── Limit: Top 10-20 items

4. PIE/DONUT CHART
   Use for: Part-to-whole relationships
   Best for: Market share, category breakdown
   Configuration:
   ├── Dimension: Category (max 6-8 slices)
   ├── Metric: Percentage or count
   ├── Labels: Show percentage
   └── Legend: Right or bottom

5. GEO MAP
   Use for: Geographic data
   Best for: Sales by country, store locations
   Configuration:
   ├── Geo dimension: Country, Region, City
   ├── Metric: Color intensity
   ├── Map style: Light, Dark, Satellite
   └── Zoom: Auto or custom bounds

6. TABLE
   Use for: Detailed data
   Best for: Transaction lists, rankings
   Configuration:
   ├── Dimensions: Multiple supported
   ├── Metrics: Multiple measures
   ├── Sorting: Enable user sorting
   ├── Pagination: Rows per page
   └── Conditional formatting: Heatmaps, bars

7. PIVOT TABLE
   Use for: Cross-tabulation
   Best for: Multi-dimensional analysis
   Configuration:
   ├── Row dimensions: 1-3 dimensions
   ├── Column dimensions: 1-2 dimensions
   ├── Metrics: Multiple measures
   └── Subtotals: Show row/column totals

8. SCATTER CHART
   Use for: Correlation analysis
   Best for: Price vs. quantity, age vs. revenue
   Configuration:
   ├── X-axis: Continuous dimension
   ├── Y-axis: Metric
   ├── Bubble size: Optional metric
   └── Color: Optional dimension

9. BULLET CHART
   Use for: Progress to goal
   Best for: KPI tracking, targets
   Configuration:
   ├── Actual value: Current metric
   ├── Target: Goal value
   ├── Ranges: Poor, Fair, Good
   └── Comparison: Previous period
```

### Chart Styling

**Visual Customization:**
```
Chart Style Configuration:

GENERAL SETTINGS
├── Title
│   ├── Text: Clear, descriptive
│   ├── Font: 14-16px, Medium weight
│   └── Alignment: Left, Center
├── Subtitle
│   ├── Text: Additional context
│   └── Font: 12px, Regular
└── Background
    ├── Color: Transparent or white
    └── Border: Optional, subtle

AXIS CONFIGURATION
├── X-Axis (Horizontal)
│   ├── Label: Show dimension name
│   ├── Font size: 10-12px
│   ├── Rotation: 0° or -45° for long labels
│   └── Grid lines: Light gray, dashed
├── Y-Axis (Vertical)
│   ├── Label: Show metric name with units
│   ├── Font size: 10-12px
│   ├── Scale: Auto or custom min/max
│   ├── Format: Number, Currency, Percent
│   └── Grid lines: Show for readability

LEGEND
├── Position: Right, Bottom, or Top
├── Font size: 10px
├── Style: Compact or detailed
└── Interaction: Click to filter

DATA LABELS
├── Show: On bars, lines, or on hover
├── Position: Inside, Outside, Auto
├── Format: Same as metric
└── Threshold: Show only for significant values

COLORS & SERIES
├── Color by: Dimension or metric value
├── Color palette: Consistent with theme
├── Series styling: Line weight, markers
└── Opacity: 100% for solid, 70% for overlap

INTERACTIONS
├── Hover: Show tooltip with details
├── Click: Filter or drill-down
├── Zoom: Enable for time series
└── Pan: For large datasets

Example Time Series Configuration:
{
  "title": "Monthly Revenue Trend",
  "subtitle": "Last 12 months vs. previous year",
  "xAxis": {
    "dimension": "Month",
    "gridLines": true,
    "label": "Month"
  },
  "yAxis": {
    "metric": "Revenue",
    "format": "$#,##0",
    "minValue": 0,
    "gridLines": true
  },
  "series": [
    {
      "name": "This Year",
      "color": "#4285F4",
      "lineWeight": 3,
      "showDataLabels": false
    },
    {
      "name": "Last Year",
      "color": "#DADCE0",
      "lineWeight": 2,
      "style": "dashed"
    }
  ],
  "legend": {
    "position": "bottom"
  }
}
```

---

## 4. Calculated Fields

### Basic Calculations

**Create Custom Metrics:**
```
Calculated Field Examples:

1. BASIC ARITHMETIC
   Name: Profit
   Formula: Revenue - Cost
   Type: Number
   Format: Currency

2. PERCENTAGE CALCULATION
   Name: Profit Margin
   Formula: (Revenue - Cost) / Revenue
   Type: Percent
   Format: 0.00%

3. CONDITIONAL LOGIC
   Name: Customer Tier
   Formula:
   CASE
     WHEN Lifetime_Value >= 10000 THEN "VIP"
     WHEN Lifetime_Value >= 5000 THEN "Premium"
     WHEN Lifetime_Value >= 1000 THEN "Standard"
     ELSE "Basic"
   END
   Type: Text

4. DATE CALCULATIONS
   Name: Days Since Last Order
   Formula: DATE_DIFF(CURRENT_DATE(), Last_Order_Date)
   Type: Number
   Format: 0

5. AGGREGATIONS
   Name: Average Order Value
   Formula: SUM(Revenue) / COUNT(Order_ID)
   Type: Number
   Format: Currency

6. YEAR-OVER-YEAR CHANGE
   Name: YoY Revenue Growth
   Formula:
   (SUM(Revenue) - SUM(Revenue_LY)) / SUM(Revenue_LY)
   Type: Percent
   Format: +0.0%;-0.0%

7. RUNNING TOTAL
   Name: Cumulative Revenue
   Formula: RUNNING_TOTAL(SUM(Revenue))
   Type: Number
   Format: Currency

8. MOVING AVERAGE
   Name: 7-Day Moving Avg
   Formula: AVG(Revenue, 7)
   Type: Number
   Format: Currency

9. RANK
   Name: Product Rank
   Formula: RANK(SUM(Revenue))
   Type: Number
   Format: 0

10. TEXT MANIPULATION
    Name: First Name
    Formula: REGEXP_EXTRACT(Full_Name, "^([^ ]+)")
    Type: Text

11. NULL HANDLING
    Name: Revenue Clean
    Formula: IFNULL(Revenue, 0)
    Type: Number

12. CONCAT
    Name: Full Address
    Formula: CONCAT(Street, ", ", City, ", ", State)
    Type: Text
```

### Advanced Calculated Fields

**Complex Logic:**
```sql
-- 1. NESTED CASE STATEMENTS
Name: Customer Segment
Formula:
CASE
  WHEN CASE
    WHEN Order_Count > 10 THEN "Frequent"
    ELSE "Occasional"
  END = "Frequent" AND Lifetime_Value > 5000 THEN "VIP Frequent"
  WHEN CASE
    WHEN Order_Count > 10 THEN "Frequent"
    ELSE "Occasional"
  END = "Frequent" THEN "Regular Frequent"
  WHEN Lifetime_Value > 5000 THEN "High Value Occasional"
  ELSE "Standard"
END

-- 2. MULTIPLE AGGREGATIONS
Name: Conversion Rate
Formula:
COUNT(CASE WHEN Status = "Completed" THEN Order_ID END) /
COUNT(Order_ID)

-- 3. DATE COMPARISONS
Name: Is Recent Customer
Formula:
CASE
  WHEN DATE_DIFF(CURRENT_DATE(), Last_Order_Date) <= 30 THEN "Active"
  WHEN DATE_DIFF(CURRENT_DATE(), Last_Order_Date) <= 90 THEN "At Risk"
  ELSE "Churned"
END

-- 4. PERCENTILE CALCULATION
Name: Revenue Percentile
Formula:
PERCENTILE(Revenue, 0.95)

-- 5. COHORT ANALYSIS
Name: Cohort Month
Formula:
FORMAT_DATETIME("%Y-%m", CAST(First_Order_Date AS DATETIME))

-- 6. GROWTH RATE
Name: Month-over-Month Growth
Formula:
CASE
  WHEN LAG(Revenue, 1) > 0 THEN
    (Revenue - LAG(Revenue, 1)) / LAG(Revenue, 1)
  ELSE NULL
END

-- 7. WEIGHTED AVERAGE
Name: Weighted Avg Price
Formula:
SUM(Price * Quantity) / SUM(Quantity)

-- 8. DISTINCT COUNT WITH CONDITION
Name: Active Products
Formula:
COUNT_DISTINCT(
  CASE
    WHEN Revenue > 0 THEN Product_ID
    ELSE NULL
  END
)

-- 9. STRING OPERATIONS
Name: Domain from Email
Formula:
REGEXP_EXTRACT(Email, "@(.+)$")

-- 10. COMPLEX DATE LOGIC
Name: Fiscal Quarter
Formula:
CONCAT(
  "FY",
  CAST(
    YEAR(Date) +
    CASE WHEN MONTH(Date) >= 4 THEN 1 ELSE 0 END
    AS TEXT
  ),
  " Q",
  CAST(
    CASE
      WHEN MONTH(Date) IN (4,5,6) THEN 1
      WHEN MONTH(Date) IN (7,8,9) THEN 2
      WHEN MONTH(Date) IN (10,11,12) THEN 3
      ELSE 4
    END
    AS TEXT
  )
)
```

---

## 5. Interactive Controls

### Filter Configuration

**Best Practices:**
- Place filters prominently
- Use appropriate filter types
- Set sensible defaults
- Limit filter combinations

```
Filter Types and Use Cases:

1. DATE RANGE FILTER
   Use for: Time-based filtering
   Configuration:
   ├── Type: Date range picker
   ├── Default: Last 30 days
   ├── Available ranges:
   │   ├── Today, Yesterday
   │   ├── Last 7/30/90 days
   │   ├── This/Last month, quarter, year
   │   └── Custom range
   └── Apply to: All date fields

2. DROP-DOWN LIST
   Use for: Single selection from many options
   Configuration:
   ├── Dimension: Category field
   ├── Default: All, or specific value
   ├── Sort: Alphabetical or by metric
   └── Search: Enable for >20 items

3. CHECKBOX LIST
   Use for: Multiple selections
   Configuration:
   ├── Dimension: Category field
   ├── Default: All selected
   ├── Layout: Vertical or horizontal
   └── Limit: 5-10 visible items

4. SLIDER
   Use for: Numeric range
   Configuration:
   ├── Metric: Numeric field
   ├── Range: Min to max values
   ├── Step: Increment size
   └── Default: Full range

5. INPUT BOX
   Use for: Search/text filter
   Configuration:
   ├── Dimension: Text field
   ├── Match type: Contains, Equals
   ├── Case sensitive: Usually no
   └── Placeholder: Search text

6. FIXED SIZE LIST
   Use for: Few options (2-5)
   Configuration:
   ├── Dimension: Category field
   ├── Layout: Horizontal buttons
   ├── Style: Pills or buttons
   └── Default: First option

Filter Bar Layout:
┌─────────────────────────────────────────────────┐
│ Date Range: [Last 30 days ▼] [Region: All ▼]  │
│ [Product Category: All ▼] [Customer Segment ▼]│
└─────────────────────────────────────────────────┘

Best Practices:
- Group related filters together
- Most important filter on the left
- Date filter always visible
- Limit to 4-6 filters per report
- Use filter dependencies when appropriate
```

### Parameters

**Dynamic Report Behavior:**
```
Parameter Examples:

1. METRIC SELECTOR
   Name: Selected Metric
   Data Type: Text
   Permitted Values:
   ├── Revenue (default)
   ├── Order Count
   ├── Average Order Value
   └── Customer Count
   
   Usage in Calculated Field:
   Name: Dynamic Metric
   Formula:
   CASE
     WHEN Selected_Metric = "Revenue" THEN SUM(Revenue)
     WHEN Selected_Metric = "Order Count" THEN COUNT(Order_ID)
     WHEN Selected_Metric = "Average Order Value" THEN AVG(Order_Value)
     WHEN Selected_Metric = "Customer Count" THEN COUNT_DISTINCT(Customer_ID)
   END

2. DATE GRANULARITY
   Name: Date Grouping
   Data Type: Text
   Permitted Values:
   ├── Day
   ├── Week
   ├── Month (default)
   └── Quarter
   
   Usage:
   Name: Dynamic Date
   Formula:
   CASE
     WHEN Date_Grouping = "Day" THEN Date
     WHEN Date_Grouping = "Week" THEN WEEK(Date)
     WHEN Date_Grouping = "Month" THEN MONTH(Date)
     WHEN Date_Grouping = "Quarter" THEN QUARTER(Date)
   END

3. COMPARISON PERIOD
   Name: Compare To
   Data Type: Text
   Permitted Values:
   ├── Previous Period
   ├── Previous Year (default)
   ├── Budget
   └── No Comparison
   
   Usage: Control visibility of comparison series

4. TOP N SELECTOR
   Name: Top N Items
   Data Type: Number
   Permitted Values:
   ├── 5
   ├── 10 (default)
   ├── 20
   └── 50
   
   Usage: Limit chart data

5. THRESHOLD
   Name: Min Revenue
   Data Type: Number
   Default Value: 1000
   
   Usage in Filter:
   WHERE Revenue >= Min_Revenue
```

---

## 6. Performance Optimization

### Query Optimization

**Best Practices:**
- Pre-aggregate data
- Use extracts for large datasets
- Limit date ranges
- Optimize data sources

```
Performance Optimization Checklist:

1. DATA SOURCE OPTIMIZATION
   ✓ Use BigQuery for large datasets (>1M rows)
   ✓ Create aggregated tables/views
   ✓ Partition BigQuery tables by date
   ✓ Use clustering for common filters
   ✓ Avoid SELECT * in custom queries
   ✓ Pre-calculate complex metrics

   Example Optimized BigQuery View:
   CREATE OR REPLACE VIEW analytics.daily_sales_summary AS
   SELECT
     DATE(order_date) AS date,
     product_category,
     region,
     COUNT(DISTINCT customer_id) AS customer_count,
     COUNT(order_id) AS order_count,
     SUM(revenue) AS total_revenue,
     AVG(revenue) AS avg_order_value
   FROM `project.dataset.orders`
   WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 365 DAY)
   GROUP BY 1, 2, 3;

2. REPORT OPTIMIZATION
   ✓ Limit charts per page (max 10-15)
   ✓ Use date range filters to limit data
   ✓ Enable data caching (4-12 hours)
   ✓ Avoid complex calculated fields in large datasets
   ✓ Use extracts for Google Sheets (>50K rows)
   ✓ Minimize blended data sources

3. CHART OPTIMIZATION
   ✓ Limit table rows (use pagination)
   ✓ Use "Top N" for bar charts
   ✓ Aggregate before visualizing
   ✓ Avoid real-time updates for static data
   ✓ Use scorecard summaries instead of tables

4. CALCULATED FIELD OPTIMIZATION
   ✓ Pre-calculate in data source when possible
   ✓ Avoid nested aggregations
   ✓ Use CASE instead of multiple IFs
   ✓ Cache complex calculations
   ✓ Minimize RegEx operations

5. FILTER OPTIMIZATION
   ✓ Set default filter values
   ✓ Use filter dependencies
   ✓ Limit filter options (<1000 values)
   ✓ Apply filters at data source level
   ✓ Use parameters for UI-only filters

Performance Comparison:
┌──────────────────────────┬──────────┬──────────┐
│ Optimization              │ Before   │ After    │
├──────────────────────────┼──────────┼──────────┤
│ Aggregated BigQuery view │ 45s      │ 3s       │
│ Date range filter        │ 30s      │ 5s       │
│ Limited chart count      │ 20s      │ 8s       │
│ Data caching (4 hours)   │ 15s      │ 2s       │
│ Extract (Google Sheets)  │ 40s      │ 4s       │
└──────────────────────────┴──────────┴──────────┘
```

### Data Extracts

**Use Extracts for Performance:**
```
Data Extract Configuration:

When to Use Extracts:
- Google Sheets with >50,000 rows
- Slow database connections
- Reports with many concurrent users
- Data that changes infrequently
- Complex calculated fields

Extract Settings:
├── Schedule: Daily, Weekly, Monthly
├── Time: Off-peak hours (e.g., 2 AM)
├── Incremental: Yes (append new data)
└── Retention: Keep last 30 days

Creating an Extract:
1. Data Source → Create Extract
2. Configure refresh schedule
3. Select incremental or full refresh
4. Set data range (e.g., last 365 days)
5. Choose fields to include

Example Extract Query:
SELECT
  date,
  customer_id,
  product_category,
  SUM(revenue) as revenue,
  COUNT(order_id) as order_count
FROM orders
WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL 365 DAY)
GROUP BY 1, 2, 3

Best Practices:
✓ Extract only necessary fields
✓ Apply filters at extract level
✓ Schedule during off-peak hours
✓ Monitor extract size and query cost
✓ Use incremental refresh when possible
✗ Don't extract PII without encryption
✗ Don't refresh more than needed
```

---

## 7. Sharing and Collaboration

### Report Sharing

**Best Practices:**
- Set appropriate permissions
- Use view-only links for external users
- Schedule email delivery
- Embed in websites/apps

```
Sharing Options:

1. SHARE WITH SPECIFIC PEOPLE
   Permissions:
   ├── View only
   │   └── Can view report and data
   ├── Edit
   │   └── Can modify report structure
   └── Owner
       └── Full control, can delete

2. GET SHAREABLE LINK
   Options:
   ├── Anyone with link can view
   ├── Anyone in organization can view
   └── Specific domain users can view

3. SCHEDULE EMAIL DELIVERY
   Configuration:
   ├── Recipients: Email addresses
   ├── Frequency: Daily, Weekly, Monthly
   ├── Time: Specific hour
   ├── Format: PDF, Link
   ├── Filters: Apply specific values
   └── Attachment: Include or link only

   Example Schedule:
   Subject: Daily Sales Report - {{date}}
   Recipients: sales-team@company.com
   Frequency: Every weekday at 8:00 AM
   Format: PDF with data
   Filters: Region = {{user_region}}

4. EMBED IN WEBSITE
   Embed Code:
   <iframe
     width="800"
     height="600"
     src="https://datastudio.google.com/embed/reporting/REPORT_ID"
     frameborder="0"
     style="border:0"
     allowfullscreen>
   </iframe>

   Embed Options:
   ├── Filter parameters in URL
   ├── Hide report header/footer
   ├── Auto-refresh interval
   └── Responsive sizing

5. EXPORT OPTIONS
   Formats:
   ├── PDF (best for sharing)
   ├── Google Sheets (for further analysis)
   ├── CSV (single table only)
   └── Print (high quality)
```

### Access Control

**Data-Level Security:**
```
Row-Level Security Implementation:

1. USING DATA CREDENTIALS
   - Viewer uses their own credentials
   - Data filtered by their access rights
   - Best for: BigQuery, Cloud SQL
   - Use when: Different users need different data

   Setup:
   ├── Data Source: Enable "Viewer credentials"
   ├── BigQuery: Use authorized views
   └── Report: Share with users

2. USING OWNER CREDENTIALS
   - All viewers see same data
   - Uses report owner's access
   - Best for: Public reports
   - Use when: Same data for all users

3. USING PARAMETERS FOR FILTERING
   - Pass user info via URL parameters
   - Filter data based on parameters
   - Best for: Embedded reports
   - Use when: Custom user contexts

   Example URL:
   https://datastudio.google.com/reporting/REPORT_ID?params={
     "user_region":"West",
     "user_level":"manager"
   }

4. USING EMAIL-BASED FILTERING
   Calculate Field:
   Name: Can View Data
   Formula:
   CASE
     WHEN Customer_Owner_Email = CURRENT_USER()
     THEN "Yes"
     ELSE "No"
   END
   
   Filter: Can View Data = "Yes"

Best Practices:
✓ Use viewer credentials for sensitive data
✓ Document security model
✓ Test with different user accounts
✓ Audit access regularly
✓ Use groups for permission management
✗ Don't share owner credentials
✗ Don't rely on URL obscurity
```

---

## 8. Templating and Themes

### Custom Themes

**Create Consistent Brand Identity:**
```
Theme Configuration:

COLORS
├── Primary color: #4285F4
├── Secondary color: #34A853
├── Accent color: #EA4335
├── Background: #FFFFFF
└── Text color: #202124

FONTS
├── Headings: Google Sans, 16-24px
├── Body: Roboto, 12-14px
└── Numbers: Roboto Mono, 12-14px

CHART STYLES
├── Bar charts: Rounded corners
├── Line charts: 2px weight
├── Grid lines: Light gray, dashed
└── Data labels: Show on hover

LAYOUT
├── Page margins: 20px
├── Chart padding: 10px
├── Section spacing: 30px
└── Filter bar: Fixed at top

BRANDING
├── Logo: Top left, 40x40px
├── Footer: Company name, page numbers
├── Watermark: Optional, low opacity
└── Color scheme: Match corporate brand

Example Theme JSON:
{
  "themeName": "Corporate Brand",
  "themeColors": [
    {"themeColor": {
      "color": {"rgbColor": {"red": 0.26, "green": 0.52, "blue": 0.96}},
      "colorType": "PRIMARY"
    }},
    {"themeColor": {
      "color": {"rgbColor": {"red": 0.2, "green": 0.66, "blue": 0.33}},
      "colorType": "ACCENT1"
    }}
  ],
  "themeFont": {
    "headingFont": "Google Sans",
    "bodyFont": "Roboto"
  }
}

Applying Theme:
1. File → Theme and layout
2. Select custom theme or create new
3. Customize colors, fonts, chart styles
4. Apply to current report
5. Save as template for reuse
```

### Report Templates

**Reusable Report Structures:**
```
Template Categories:

1. EXECUTIVE DASHBOARD
   Components:
   ├── Key metrics scorecards (4-6)
   ├── Trend analysis (line chart)
   ├── Category breakdown (bar chart)
   ├── Geographic view (map)
   └── Top performers table

2. MARKETING ANALYTICS
   Components:
   ├── Campaign performance scorecards
   ├── Conversion funnel
   ├── Traffic sources (pie chart)
   ├── Time series trends
   └── Channel comparison table

3. SALES REPORT
   Components:
   ├── Revenue metrics
   ├── Sales pipeline visualization
   ├── Performance by rep/region
   ├── Product mix analysis
   └── Forecast vs. actual

4. OPERATIONAL DASHBOARD
   Components:
   ├── Real-time status indicators
   ├── Volume trends
   ├── Error rates
   ├── Resource utilization
   └── Alert summary

5. FINANCIAL REPORT
   Components:
   ├── P&L summary
   ├── Budget vs. actual variance
   ├── Cash flow trends
   ├── Department breakdown
   └── Key financial ratios

Creating Template:
1. Build reference report
2. File → Make a copy
3. Replace data source with template source
4. Remove specific filters/data
5. Document usage instructions
6. Share template with team
```

---

## 9. Advanced Features

### Community Visualizations

**Extend with Custom Charts:**
```
Popular Community Visualizations:

1. SANKEY DIAGRAM
   Use for: Flow analysis
   Example: Customer journey, budget allocation
   Configuration:
   ├── Source dimension
   ├── Target dimension
   ├── Metric for flow width
   └── Color scheme

2. CALENDAR HEATMAP
   Use for: Daily patterns
   Example: Website traffic, sales activity
   Configuration:
   ├── Date dimension (day level)
   ├── Metric for intensity
   └── Color gradient

3. RADAR CHART
   Use for: Multi-metric comparison
   Example: Product features, performance scores
   Configuration:
   ├── Category dimension
   ├── Multiple metrics (3-8)
   └── Color per series

4. WATERFALL CHART
   Use for: Incremental changes
   Example: Revenue bridges, cost breakdown
   Configuration:
   ├── Category dimension
   ├── Value metric
   ├── Starting value
   └── Show connectors

5. CANDLESTICK CHART
   Use for: Financial data
   Example: Stock prices, ranges
   Configuration:
   ├── Date dimension
   ├── Open, High, Low, Close metrics
   └── Color for up/down

Installing Community Visualization:
1. Insert → Community visualization
2. Search visualization library
3. Select desired visualization
4. Add to report
5. Configure data and style

Creating Custom Visualization:
1. Develop using D3.js or similar
2. Follow Looker Studio viz framework
3. Deploy to public URL or packaged
4. Submit to community gallery (optional)
5. Share with team or organization
```

### Blend Data

**Combine Multiple Sources:**
```
Data Blending Best Practices:

Scenario 1: Sales + Customer Demographics
├── Left Data Source: Sales (BigQuery)
│   ├── customer_id (Join Key)
│   ├── order_date
│   ├── revenue
│   └── order_id
├── Right Data Source: Customers (Google Sheets)
│   ├── customer_id (Join Key)
│   ├── age
│   ├── region
│   └── segment
└── Join Type: LEFT OUTER JOIN

Scenario 2: Actual vs. Budget
├── Left: Actual Sales (BigQuery)
│   ├── date (Join Key)
│   ├── category (Join Key)
│   └── actual_revenue
├── Right: Budget (Google Sheets)
│   ├── date (Join Key)
│   ├── category (Join Key)
│   └── budget_revenue
└── Join Type: FULL OUTER JOIN

Scenario 3: Multi-Channel Attribution
├── Source 1: Google Ads
├── Source 2: Facebook Ads
├── Source 3: Email Marketing
├── Join Key: campaign_date + campaign_name
└── Calculated: Total Conversions, ROAS

Blending Limitations:
✗ Max 5 data sources per blend
✗ Performance degrades with large datasets
✗ Limited to LEFT and FULL OUTER joins
✗ No cross-filtering between sources

Alternatives to Blending:
✓ Union data in BigQuery
✓ Create database views
✓ Use BigQuery federated queries
✓ ETL data to single source
```

---

## 10. Common Anti-Patterns

### ❌ Anti-Pattern 1: Too Many Charts
**Problem:** Report loads slowly, overwhelming users
**Solution:** Limit to 10-15 charts, use drill-downs

### ❌ Anti-Pattern 2: No Date Filter
**Problem:** Queries entire dataset, slow performance
**Solution:** Always include date range filter with default

### ❌ Anti-Pattern 3: Using Google Sheets for Large Data
**Problem:** Slow performance, hit row limits
**Solution:** Use BigQuery for >100K rows

### ❌ Anti-Pattern 4: Inconsistent Metrics
**Problem:** Confusion about definitions
**Solution:** Use calculated fields, document definitions

### ❌ Anti-Pattern 5: No Data Refresh Strategy
**Problem:** Stale data or excessive query costs
**Solution:** Set appropriate cache duration

---

## 11. Quick Reference Checklist

### Report Setup
- [ ] Connect to appropriate data sources
- [ ] Configure data freshness settings
- [ ] Create necessary calculated fields
- [ ] Set up date range filter
- [ ] Apply data-level security

### Design
- [ ] Use consistent color scheme
- [ ] Apply company theme/branding
- [ ] Implement proper layout structure
- [ ] Choose appropriate chart types
- [ ] Add clear titles and labels

### Performance
- [ ] Enable data caching
- [ ] Limit date ranges with filters
- [ ] Use aggregated data sources
- [ ] Optimize calculated fields
- [ ] Test load times

### Interactivity
- [ ] Add relevant filters
- [ ] Implement drill-downs
- [ ] Use parameters for flexibility
- [ ] Enable hover tooltips
- [ ] Test user interactions

### Sharing
- [ ] Set appropriate permissions
- [ ] Test with different user accounts
- [ ] Schedule email delivery (if needed)
- [ ] Document report usage
- [ ] Provide data definitions

---

*Best Practices for Google Cloud Data Engineer Certification*
