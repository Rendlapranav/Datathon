# The Product Rescue Mission - 15 Minute Winning Pitch

## The Strategy
**The Hook:** Do not start by talking about AI or models. Start with the *business emergency*. The CEO is losing money, and they don't know why.
**The Dynamic:** 
- **Prathmesh:** The Executive / Storyteller. You frame the business problem, the "Silent Killers", and the final ROI. You hold the attention of the room.
- **Pranav:** The Technical Lead. You prove *how* we solved it. You take them under the hood, show the ML pipeline, and do the live dashboard demo to prove the tech works.

---

## Part 1: The Illusion (0:00 - 4:00)
**Speaker: Prathmesh**
**Screen:** Slide 1 (Title)

**Prathmesh:** 
"Good morning everyone. 
Imagine you are the CEO of a global electronics brand. Your flagship product has a 4.1-star average on Amazon. Management is happy. Marketing is happy. 

But your CFO is panicking. Because despite the 4-star average, sales are dropping, and return rates are climbing. Management is blind to the problem because they are looking at the stars. 

We are here today to show you that the stars are lying. This is the Product Rescue Mission."

*(Transition to Slide 2: The Paradox of a 4.1★ Product)*

**Prathmesh:**
"Our team analyzed over 50,000 Amazon electronics reviews. We found a dangerous paradox. 
Stars are a lagging, biased signal. Customers round up. They give 4 stars to be polite, but their actual text is filled with frustration. 

Look at the scatter plot on the screen. We ran a RoBERTa sentiment analysis on the raw text and plotted it against the star rating. We found a massive 'Star-Gap'. The reality of customer satisfaction is artificially inflated by 0.2 points. 

If you wait for your star rating to drop to 3.5 to fix your product... you've already lost thousands of customers."

*(Transition to Slide 3: Unmasking the Silent Killers)*

**Prathmesh:**
"This led us to discover the most dangerous segment in your user base: The Silent Killers. 

We found 420 customers—representing over 8% of the user base—who left 4 or 5-star reviews, but whose textual language triggered our NLP engine for high anger and disappointment. 

These people churn without filing a support ticket. They leave no trace in your CRM. They represent $173,000 in immediate revenue risk. 

So, how did we find them? And more importantly, what exactly are they angry about? I'll hand it over to our Technical Lead, Pranav, to show you under the hood."

---

## Part 2: The Diagnosis & Demo (4:00 - 10:00)
**Speaker: Pranav**
**Screen:** Slide 4 (What Exactly is Broken?)

**Pranav:**
"Thank you, Prathmesh. 

To rescue this product, we couldn't just count bad words. We needed to map *features* to *emotions*. 
We built an NLP pipeline using BERTopic modeling to cluster 50,000 reviews into specific hardware features. Then, we overlaid an emotion detection model to classify the intensity of those topics into Joy, Sadness, or Anger.

As you can see on the heatmap, the results are definitive. 
We aren't just seeing general 'disappointment'. We are seeing targeted **Anger** specifically clustered around one node: **Battery & Charging.** It appears in 11.4% of all reviews. 

Let me show you exactly how our system tracks this in real-time."

*(Action: Switch from Slides to the Streamlit Dashboard)*

**Pranav:**
"This is the live Product Rescue Dashboard that your product managers would use. 

*(*Click to the Timeline/Sentiment Tab*)*
First, we track the 'Drift'. You can see exactly the month where sentiment sharply dropped, which correlates perfectly with the firmware update pushed in Q3. The star ratings didn't catch this, but our NLP pipeline caught the emotional shift immediately.

*(*Click to the Topics/Aspects Tab*)*
If we drill down into the 'Battery' cluster, the model isolates the exact root cause. It's not the battery life itself—our model extracted that the *charging case fails after 3 to 6 months*, and there is *no low-battery warning*. 

*(*Click to the Segment Filter*)*
If I filter strictly by our 'Silent Killers' segment, you can read the raw quotes. These are your 4-star reviewers saying: 'Sound is great, but it dies without warning. Returning it.' 

We have isolated the mechanical failure. Now, we need to fix it."

*(Action: Switch back to Slides - Slide 5: The CEO Priority Matrix)*

**Pranav:**
"But resources are limited. Engineering can't fix everything at once. 
So, we built the CEO Priority Matrix. We mapped the Frequency of a complaint against its Emotional Intensity. 

1. **Battery & Charging** is high frequency, maximum intensity. It is an urgent hardware fix. 
2. **Connectivity (Bluetooth)** is mid-frequency. 
3. **Sound Quality** is high frequency, but low intensity. This tells us it's a marketing perception issue, not a broken speaker. 

I'll hand it back to Prathmesh to present the Rescue Plan."

---

## Part 3: The Pitch & ROI (10:00 - 14:00)
**Speaker: Prathmesh**
**Screen:** Slide 6 (The 90-Day Rescue Plan)

**Prathmesh:**
"Thank you, Pranav. 
This isn't a research report. It is a shipping plan. Here is how we rescue the product and the revenue in 90 days.

**Days 0 to 30:** We deploy urgent engineering resources to push a firmware update for power management and add a 200-cycle charge test at the QA gate. 
**Days 30 to 60:** We patch the Android Bluetooth connectivity stack. 
**Days 60 to 90:** We spend zero engineering dollars. Instead, we launch a marketing tutorial campaign to reset consumer expectations on sound profiles.

*(Transition to Slide 7: Expected ROI & Impact)*

**Prathmesh:**
"If we execute this plan, the AI models forecast a 52.9% lift in brand sentiment. 

By neutralizing the 'Silent Killers'—the people you didn't even know you were losing—we immediately rescue $173,000 in Annual Recurring Revenue. 

We stopped looking at the stars, and we started listening to the customers. 
This is how you rescue a flagship product. 

Thank you. We'd love to take your questions."

---

## Part 4: Q&A Buffer (14:00 - 15:00)
- **Wait for the judges.**
- **If they ask about models:** Pranav takes the question (talk about RoBERTa, thresholds, UMAP).
- **If they ask about business impact:** Prathmesh takes the question (talk about churn prevention, customer lifetime value).
