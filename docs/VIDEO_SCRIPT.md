# Video demo script (about 3 minutes)

Record the deployed Vercel page. Set speed to **Normal**. Do a dry run first.

| Time | On screen | Say |
|---|---|---|
| 0:00 | Dashboard top | "Once a model is deployed, its data changes and labels arrive late. My project is a monitor that tells harmless changes from harmful ones." |
| 0:20 | Point at the timeline rows | "This is a simulated stream of 40 windows. The top strip is what I injected: clean, a harmless shift, clean, then a real drift. The monitor doesn't know this." |
| 0:40 | Click Play | "Each row is a detector. Red means an alert." |
| 1:00 | Pause at window 12 | "Here the noise features shifted. The feature test and the classifier both alert, because the data really changed. But the model's accuracy has not moved, so this is a false alarm. The combined system stays quiet." |
| 1:30 | Play on, pause at window 25 | "Now the important features start to drift. The combined system alerts one window after drift begins." |
| 1:55 | Point at accuracy chart | "Labels arrive 3 windows late. The solid line is what the team sees. Accuracy only visibly drops around window 33, so the alert comes about 8 windows earlier." |
| 2:15 | Scroll to results table | "Averaged over 15 new random seeds, the combined system catches 87% of impactful windows with no alerts on harmless shifts. Feature tests alone alert on 88% of harmless windows." |
| 2:40 | Open the window-size section | "Smaller windows make detection weaker, so window size matters." |
| 2:50 | Footer | "Limits: synthetic data, one model, one drift shape. Next step is a real dataset. Code and report are on GitHub. Thanks for watching." |
