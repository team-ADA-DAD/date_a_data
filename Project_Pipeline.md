# Project Pipeline
This notebook describes the general processing pipeline of the project. It shows a light-version pipeline of data processing on how to answer the research questions as shown in the methods in README.md step by step.

# Required package
[qwikidata](https://qwikidata.readthedocs.io/en/stable/)
[json](https://docs.python.org/3/library/json.html)
[bz2](https://docs.python.org/3/library/bz2.html)
[pandas](https://pandas.pydata.org/)
[nltk](https://www.nltk.org/)
[gensim](https://radimrehurek.com/gensim/)

# Required data
[wikidata dump](https://dumps.wikimedia.org/wikidatawiki/entities/)
[quotebank](https://zenodo.org/record/4277311#.YY6Nj2DMJm8)


```python
import pandas as pd
import gensim
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer as ps
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
```


```python
QUOTEBANK_POLITICIANS_USA_MAPPED = 'quotebank_politicians_USA_mapped.json.bz2'
sample_data = pd.read_json(QUOTEBANK_POLITICIANS_USA_MAPPED, lines=True, compression='bz2')
```

## Task 1 Network Models

In this part we plot a simple network model to prove the feasiblity of the proposal and present a simple demonstration with only 3 politicians.


```python
QUOTEBANK_POLITICIANS_USA_MAPPED = 'quotebank_politicians_USA_mapped.json.bz2'
df = pd.read_json(QUOTEBANK_POLITICIANS_USA_MAPPED, lines=True, compression='bz2' )
```


```python
trump_mentions = df['qids'].str[0]=='Q22686'
biden_mentions = df['qids'].str[0]=='Q6279'
bernie_mentions = df['qids'].str[0]=='Q359442'
trump_mentioned = df['mentions_qids'].apply(lambda x: 'Q22686' in x)
biden_mentioned = df['mentions_qids'].apply(lambda x: 'Q6279' in x)
bernie_mentioned = df['mentions_qids'].apply(lambda x: 'Q359442' in x)
```


```python
print(f'Trump mentioned Trump {(trump_mentions & trump_mentioned).sum()} times in 2020 US data.')
print(f'Trump mentioned Biden {(trump_mentions & biden_mentioned).sum()} times in 2020 US data.')
print(f'Trump mentioned Bernie {(trump_mentions & bernie_mentioned).sum()} times in 2020 US data.')
print(f'Biden mentioned Trump {(biden_mentions & trump_mentioned).sum()} times in 2020 US data.')
print(f'Biden mentioned Biden {(biden_mentions & biden_mentioned).sum()} times in 2020 US data.')
print(f'Biden mentioned Bernie {(biden_mentions & bernie_mentioned).sum()} times in 2020 US data.')
print(f'Bernie mentioned Trump {(bernie_mentions & trump_mentioned).sum()} times in 2020 US data.')
print(f'Bernie mentioned Biden {(bernie_mentions & biden_mentioned).sum()} times in 2020 US data.')
print(f'Bernie mentioned Bernie {(bernie_mentions & bernie_mentioned).sum()} times in 2020 US data.')
```

    Trump mentioned Trump 1604 times in 2020 US data.
    Trump mentioned Biden 275 times in 2020 US data.
    Trump mentioned Bernie 119 times in 2020 US data.
    Biden mentioned Trump 932 times in 2020 US data.
    Biden mentioned Biden 538 times in 2020 US data.
    Biden mentioned Bernie 94 times in 2020 US data.
    Bernie mentioned Trump 1171 times in 2020 US data.
    Bernie mentioned Biden 402 times in 2020 US data.
    Bernie mentioned Bernie 173 times in 2020 US data.



```python
# Build a dataframe defining all the edges and edge weight
df = pd.DataFrame({ 'from':['Trump', 'Trump', 'Trump', 'Biden', 'Biden', 'Biden','Bernie', 'Bernie', 'Bernie'],
                    'to':['Trump', 'Biden', 'Bernie', 'Trump', 'Biden', 'Bernie', 'Trump', 'Biden', 'Bernie'],
                    'weight':[1604, 275, 119, 932, 538, 94, 1171, 402, 173],
                #     'length':[10, 10, 10, 10, 10, 10, 10, 10, 10]
                })
# metions_matrix = np.array([[1604, 275, 119],
#                            [932, 538, 94],
#                            [1171, 402, 173]
# ])

# Build  graph
G=nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.DiGraph(), edge_attr=True)
pos = nx.spring_layout(G)
# Custom the nodes and draw
nx.draw(G,
        pos=pos,
        with_labels=True,
        node_color=['red', 'skyblue', 'skyblue'],
        node_size=[1604+932+1171, 275+538+402, 119+94+173],
        width=np.sqrt(df['weight'])*0.15,
        connectionstyle='arc3, rad = 0.15')
plt.title("Demonstration of US political network with three politicians")
plt.show()
```


![png](assets/images/output_8_0.png)



```python
G = nx.DiGraph()
node = ['Trump', 'Biden', 'Bernie']
weight_matrix = np.array([[1604, 275, 119],
                           [932, 538, 94],
                           [1171, 402, 173]])

for i in range(3):
    for j in range(3):
        G.add_nodes_from([node[i], node[j]])
        G.add_edge(node[i], node[j], length = 10, weight=weight_matrix[i][j])
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color=['red', 'skyblue', 'skyblue'],node_size=[1604+275+119, 932+538+94, 1171+402+173],)
plt.show()
```


![png](./assets/images/output_9_0.png)


## Task 2 Sentiment Analysis

In this task we plan to do sentiment analysis on the quotation mentions and draw a new network graph showing the sentiment when a politician mentions or is mentioned by another politician. To prove the feasibility of the task, a small scale sentiment analysis is done on US quotation mentions in 2020 with the SentimentIntensityAnalyzer in package nltk, which is pretrained with more than [100 datasets](https://www.nltk.org/nltk_data/).

First we load the US politician mentions in 2020 and create a list out of the mentions.


```python
QUOTEBANK_POLITICIANS_USA_MAPPED = 'quotebank_politicians_USA_mapped.json.bz2'
sample_data = pd.read_json(QUOTEBANK_POLITICIANS_USA_MAPPED, lines=True, compression='bz2')
sentences = []
sentences.extend(sample_data['quotation'].tolist())
```

We print mentions with either a positive or negative emotion with their scores. (sentiment score > 0.5)


```python
for sentence in sentences:
...    sid = SentimentIntensityAnalyzer()
...    print(sentence)
...    ss = sid.polarity_scores(sentence)
...    for k in sorted(ss):
...        print('{0}: {1}, '.format(k, ss[k]), end='')
...    print()
```

    but [ President ] Trump (was) eager to make a symbol of the Army officer sooner after the Senate acquitted him of the impeachment charges approved by House Democrats.
    compound: 0.7783, neg: 0.078, neu: 0.645, pos: 0.277, 
    Even if it's all true, exactly in the worst-case scenario as John Bolton may put it, it doesn't really change the facts much, if at all. For me, I also don't think I'd really learn anything new from him, because we already know that the president was concerned about Biden's role, when Joe Biden was vice president of the United States, and a possible corrupt activity in Ukraine. So, if it's further proof of it, so what? We already know about it.
    compound: 0.6808, neg: 0.0, neu: 0.933, pos: 0.067, 
    I enjoy your analysis and instruction on polling. I hope in the future you will continue with more pithy insight on topics such as polling registered voters versus `likely voters,' polling sample size, and polling sample demographics. In regards to President Trump I believe one to three points should be added to his percentages because of two factors. First, Trump supporters are subject to shaming and public ridicule, even assault, and thus will not announce their support for him. Second, there are many of us who are disgusted with his juvenile narcissistic rants (Twitter) and his impulsive actions (pull out of Syria) and relish the thought of expressing our disgust but ultimately will vote for him for a few key reasons such as judicial appointments and the economy. I would be interested to see if any pollsters are trying to quantify these factors (if in fact they exists).
    compound: -0.1972, neg: 0.08, neu: 0.859, pos: 0.062, 
    If your last name was not Biden, do you think you would have been asked to be on the board of Burisma?
    compound: 0.0, neg: 0.0, neu: 1.0, pos: 0.0, 
    In short, the Clinton administration's policy of facilitating the delivery of arms to the Bosnian Muslims made it the de facto partner of an international network of governments and organizations pursuing their own agenda in Bosnia: the promotion of Islamic revolution in Europe. That network not only involves Iran but Brunei, Malaysia, Pakistan, Saudi Arabia, Sudan (a key ally of Iran), and Turkey, together with front groups supposedly pursuing humanitarian and cultural activities. For example, one such group about which details have come to light is the Third World Relief Agency (TWRA), a Sudan-based, phoney humanitarian organization which has been a major link in the arms pipeline to Bosnia. TWRA is believed to be connected with such fixtures of the Islamic terror network as Sheik Omar Abdel Rahman (the convicted mastermind behind the 1993 World Trade Center bombing) and Osama Bin Laden, a wealthy Saudi émigré believed to bankroll numerous militant groups...
    compound: 0.4215, neg: 0.029, neu: 0.924, pos: 0.047, 
    It's just a lot of baggage that @JoeBiden takes into a campaign, which isn't going to create energy & excitement,
    compound: 0.5402, neg: 0.086, neu: 0.663, pos: 0.251, 
    Joe Biden voted for the war in Iraq. I opposed it. Joe Biden voted for a terrible bankruptcy bill. I strongly opposed it. Joe Biden voted for disastrous trade agreements like NAFTA and PNTR with China. I vigorously opposed them. And Joe Biden has been on the floor of the Senate talking about the need to cut social security.
    compound: -0.6597, neg: 0.187, neu: 0.661, pos: 0.152, 
    Sanders is appealing to them and if that doesn't work for them, then the Trump team hopes they'll come their way.
    compound: 0.4215, neg: 0.0, neu: 0.877, pos: 0.123, 
    To defeat Donald Trump, who will be a very formidable opponent for a number of reasons, we need to have the largest voter turnout in American history. That's just a fact. If it is a low turnout election, Trump will win. And I believe that our campaign is the campaign of energy, is the campaign of excitement, is the campaign that can bring millions of people into the political process who normally do not vote,
    compound: 0.6486, neg: 0.064, neu: 0.805, pos: 0.131, 
    and doesn't want to do anything that [ President ] Donald Trump doesn't agree with and that he wants to quickly acquit Donald Trump. Well that's not what you have a trial for.
    compound: 0.1451, neg: 0.1, neu: 0.782, pos: 0.117, 
    But Tom Perez is the head of the DNC, and I do think that there clearly was not the process in place to make sure all these [ protocols ] were going to be followed.
    compound: 0.7579, neg: 0.0, neu: 0.822, pos: 0.178, 
    California used to be a beautiful State. It's a shame what's happened to it, with all the homeless. And what's happened to California, is the fault of Democrats, in particular Nancy Pelosi and Gavin Newsom. And that the federal government will have to step in if it gets any worse,
    compound: -0.6124, neg: 0.157, neu: 0.775, pos: 0.069, 
    I liked Mitt Romney. Mitt Romney I think would've been a great president in 2012. He was the better person to be president. I thought I was friends with Mitt Romney. I don't recognize this guy anymore. Another one. He has been a huge disappointment.
    compound: 0.8979, neg: 0.065, neu: 0.634, pos: 0.301, 
    I think when the night is over, Joe Biden will be the prohibitive favorite to win the Democratic nomination... I think it is time for us to shut this primary down, it is time for us to cancel the rest of these debates.
    compound: 0.7003, neg: 0.043, neu: 0.812, pos: 0.145, 
    I'm going to let the defense and the prosecution decide which they want to call as a witness or witnesses, but John Bolton is one of those I'd like to hear,
    compound: 0.3716, neg: 0.062, neu: 0.77, pos: 0.167, 
    In a Republican-controlled Senate, I can't think of any reason he wouldn't want folks like Secretary of State Pompeo or national security adviser John Bolton, who were in the room, who were on the email chains, who know what happened, to come to the Senate and clear his name.
    compound: 0.3956, neg: 0.065, neu: 0.838, pos: 0.097, 
    It is absolutely imperative that we defeat Trump, that we have a candidate's agenda and record that can defeat Trump. And not only is our record different, the nature of our campaign is different,
    compound: -0.7184, neg: 0.162, neu: 0.838, pos: 0.0, 
    It's clear the Democratic primary voters have chosen Vice President Joe Biden to be the person who will take on President Trump in the general election. I know that [ Biden ] has a good heart and that he's motivated by his love for our country.
    compound: 0.9136, neg: 0.0, neu: 0.75, pos: 0.25, 
    Its top rated town hall was with Sen. Bernie Sanders (I-VT) in April, drawing 2.5 million viewers, and the news channel also has hosted events with Sen. Amy Klobuchar (D-MN), Sen. Kirsten Gillibrand (D-NY) and Julian Castro, the former secretary of Housing and Urban Development. The network also hosted a town hall with Howard Schultz, who was considering an independent presidential bid but ultimately decided against it.
    compound: 0.1027, neg: 0.0, neu: 0.979, pos: 0.021, 
    Kerry & Murphy illegally violated the Logan Act,
    compound: -0.5267, neg: 0.362, neu: 0.638, pos: 0.0, 
    Make no mistake about it, this campaign will send Donald Trump packing.
    compound: -0.5574, neg: 0.315, neu: 0.685, pos: 0.0, 
    Nobody gets out the vote like Bernie Sanders.
    compound: 0.3612, neg: 0.0, neu: 0.737, pos: 0.263, 
    Recognizing that their states have one integrated regional economy, Rhode Island Governor Gina Raimondo, New York Governor Andrew M. Cuomo, New Jersey Governor Phil Murphy, Connecticut Governor Ned Lamont, Pennsylvania Governor Tom Wolf, and Delaware Governor John Carney today announced the creation of a multi-state council to restore the economy and get people back to work. This announcement builds on the states' ongoing regional approach to combatting the COVID-19 pandemic. The coordinating group -- comprised of one health expert, one economic development expert and the respective Chief of Staff from each state -- will work together to develop a fully integrated regional framework to gradually lift the states' stay at home orders while minimizing the risk of increased spread of the virus. The council will create this framework using every tool available to accomplish the goal of easing social isolation without triggering renewed spread - including testing, contact tracing, treatment and social distancing -- and will rely on the best available scientific, statistical, social and economic information to manage and evaluate those tools.
    compound: 0.926, neg: 0.026, neu: 0.864, pos: 0.11, 
    Trump has become furious about the stock market's slide.
    compound: -0.5719, neg: 0.316, neu: 0.684, pos: 0.0, 
    we made it clear that the whole idea was about rallying the country together to defeat Donald Trump and to win the era for the values that we share.
    compound: 0.8074, neg: 0.078, neu: 0.627, pos: 0.295, 
    What you're seeing now is, in the end, even if he considers you a friend, like Elizabeth Warren, Bernie will come first,
    compound: 0.6908, neg: 0.0, neu: 0.769, pos: 0.231, 
    All I can tell you, whether it's Iraq, whether it's DOMA, whether it's `don't ask, don't tell' -- those were difficult votes. I was there, on the right side of history, and my friend Joe Biden was not.
    compound: 0.1779, neg: 0.063, neu: 0.856, pos: 0.081, 
    America needs a President who can start to heal the wounds of this divided nation. Joe Biden will be that President,
    compound: 0.0, neg: 0.0, neu: 1.0, pos: 0.0, 
    As much as I disagree with Mitt Romney, I respect that he stuck with his principles.
    compound: -0.128, neg: 0.246, neu: 0.588, pos: 0.166, 
    Biden say `Ryan was correct.
    compound: 0.0, neg: 0.0, neu: 1.0, pos: 0.0, 
    He... hurt the family. And so I'm not surprised. But I have to tell you, I'm disappointed. He was friends with [ Biden's wife ] Jill. He was friends with [ sons ] Beau... with Hunter.
    compound: 0.358, neg: 0.186, neu: 0.617, pos: 0.197, 
    I think it's increasingly likely that other Republicans will join those of us who think we should hear from John Bolton. Whether there are other witnesses and documents, that's another matter, but John Bolton's relevance to our decision has become increasingly clear.
    compound: 0.6187, neg: 0.0, neu: 0.863, pos: 0.137, 
    Kevin McCarthy, as you know, left for the hoax. Well, we have to do that; otherwise, it becomes a more serious hoax.
    compound: -0.4005, neg: 0.233, neu: 0.683, pos: 0.084, 
    Mitch McConnell said that he believes that the impeachment trial in the Senate slowed down the federal government's response to the pandemic. What's your response to that?
    compound: 0.0, neg: 0.0, neu: 1.0, pos: 0.0, 
    No, President Trump, Massachusetts is not happy,
    compound: -0.6367, neg: 0.51, neu: 0.49, pos: 0.0, 
    Oneida County is home, and I am proud to have our GOP Committee's endorsement in the race to defeat Anthony Brindisi in November,
    compound: 0.34, neg: 0.109, neu: 0.693, pos: 0.197, 
    President Trump violated federal law when he withheld hundreds of millions of dollars that Congress appropriated on a bipartisan basis to help Ukraine combat Russian aggression,
    compound: -0.6486, neg: 0.252, neu: 0.662, pos: 0.085, 
    So a leftist attacks 3 Trump supporters, including a 15 year old child, in NH this week,
    compound: 0.5654, neg: 0.151, neu: 0.524, pos: 0.325, 
    Susan Collins, I'm curious do you really still think Donald Trump learned his lesson, or if the lesson that Donald Trump learned is that with Senators like you giving him a blank check, he can do whatever the hell he wants. Vindman? That's on you. Sondland? That's on you. Roger Stone, Susan Collins, that's on you, and Lamar Alexander it's on you and it's on every Republican that taught Donald Trump once again, there are no consequences to his actions.
    compound: -0.2406, neg: 0.077, neu: 0.837, pos: 0.086, 
    Thanks to President Trump and Republicans in Congress the state of America is strong.... President Trump has delivered on the promises he made to America and I believe the best is yet to come. I look forward to working together with him and all of my colleagues to maintain our strong economy, military, and border.
    compound: 0.9186, neg: 0.0, neu: 0.79, pos: 0.21, 
    The Democrat establishment is trying to take it away from Bernie Sanders.
    compound: 0.0, neg: 0.0, neu: 1.0, pos: 0.0, 
    The labor movement is pushing for the Democratic nominee to address the needs of workers -- to reverse long trends of declines in union membership, stagnant wages, and growing wealth and income inequality. Mike Bloomberg's record offers little reassurance that he would begin to meaningfully address these issues as president.
    compound: 0.7275, neg: 0.0, neu: 0.869, pos: 0.131, 
    The most important thing the President has done, of all the things he's done are very important, but he's finally outed the media. The media in this town has been corrupt, and it took somebody like Trump to finally bring them out of their shell where now they'll just openly go out,
    compound: 0.6534, neg: 0.0, neu: 0.885, pos: 0.115, 
    The reason why it worked for Barack and me so well is we agreed substantively on every major issue. We disagreed on some tactical ways to approach the issues,
    compound: 0.3697, neg: 0.069, neu: 0.785, pos: 0.146, 
    wrapped up the hearings without calling at least two potential witnesses who could have convincingly corroborated Hill's testimony and, by extension, indicated that the nominee had perjured himself on a sustained basis throughout the hearings. As [ Jane Mayer and Jill Abramson report in Strange Justice ], `Hill's reputation was not foremost among the committee's worries. The Democrats in general, and Biden in particular, appear to have been far more concerned with their own reputations,' and feared a Republican-stoked public backlash if they aired more details of Thomas's sexual proclivities.
    compound: -0.2023, neg: 0.081, neu: 0.856, pos: 0.063, 
    Abusing official power to protect political friends and attack opponents is common in authoritarian regimes like Putin's Russia. Trump and Barr's conduct has no place in our democracy.
    compound: -0.0258, neg: 0.216, neu: 0.571, pos: 0.213, 
    And I don't want to waste a whole lot of time on this, because this is what Donald Trump and maybe some media want. Anybody who knows me knows that it's incomprehensible that I would think that a woman cannot be president of the United States.
    compound: 0.5702, neg: 0.054, neu: 0.834, pos: 0.113, 
    Bondi and the other Trump lawyers spend most of the day savaging the Bidens (as expected)... it become crystal clear to me: Trump is trying to use the trial to do what Ukraine wouldn't -- destroy his political rivals.
    compound: 0.6652, neg: 0.0, neu: 0.872, pos: 0.128, 
    Both Vice President Biden and former Mayor Pete have helped shape our economy. Joe Biden helped save the auto industry, which revitalized the economy of the Midwest, and led the passage and the implementation of the Recovery Act, saving our economy from a depression,
    compound: -0.128, neg: 0.077, neu: 0.856, pos: 0.067, 
    Can a woman beat Donald Trump? Look at the men on this stage. Collectively, they have lost 10 elections. The only people on this stage who have won every single election that they've been in are the women... and the only person on this stage who has beaten an incumbent Republican anytime in the past 30 years is me.
    compound: -0.1027, neg: 0.08, neu: 0.862, pos: 0.058, 
    he has dedicated his life to fighting for people, not for the rich and powerful but for the mom, for the farmer, for the dreamer, for the builder, for the veteran. He can bring our country together and build that coalition of our fired-up Democratic base -- and it is fired up -- as well as independents and moderate Republicans, because we do not want to just eke by a victory. We want to win big, and Joe Biden can do that.
    compound: 0.3553, neg: 0.119, neu: 0.758, pos: 0.124, 
    He's making a big bet on Super Tuesday. He's probably hoping it wouldn't be bad for him necessarily if Bernie Sanders were to win on Monday in Iowa,
    compound: 0.9239, neg: 0.0, neu: 0.633, pos: 0.367, 
    I can't think of a ways (sic) that we make it easier for Donald Trump to get re-elected than to listen to this conversation. This is ridiculous,
    compound: 0.0772, neg: 0.088, neu: 0.813, pos: 0.099, 
    I do not blame Hillary Clinton or Leon Panetta. The Benghazi mission was understaffed. We know that now.
    compound: 0.2584, neg: 0.0, neu: 0.887, pos: 0.113, 
    Latino's For Trump Event
    compound: 0.0, neg: 0.0, neu: 1.0, pos: 0.0, 
    Mike Bloomberg has more wealth than the bottom 125 million Americans. That's wrong and immoral
    compound: -0.3832, neg: 0.282, neu: 0.556, pos: 0.162, 
    Obama is now trying to take credit for the Economic Boom taking place under the Trump Administration.
    compound: 0.3818, neg: 0.0, neu: 0.86, pos: 0.14, 
    Pilsen is a diverse, multi-cultural community and in order to effectively serve the people there, it is of the utmost important that the Cesar Chavez post office reflect that,
    compound: 0.5719, neg: 0.0, neu: 0.847, pos: 0.153, 
    President Trump continues to deliver in the face of non-stop political attacks from House Democrats. We've now got a phase I trade deal with China-in ADDITION to already record jobs numbers, wage increases, and economic growth. Let's keep it going. @realDonaldTrump.
    compound: -0.0772, neg: 0.068, neu: 0.871, pos: 0.061, 
    Since last year's Trump shutdown I have repeatedly pushed my Republican colleagues to provide back pay to federal contract employees, many of whom make up janitorial and support staffs,
    compound: 0.3182, neg: 0.047, neu: 0.864, pos: 0.09, 
    That is what we need to do in order to win against Trump.
    compound: 0.5859, neg: 0.0, neu: 0.759, pos: 0.241, 
    that Mr. Parnas served as a direct channel between President Trump's agent, Mr. Giuliani, and individuals close to President Volodymyr Zelensky.
    compound: 0.0, neg: 0.0, neu: 1.0, pos: 0.0, 
    the amazing thing about President Trump
    compound: 0.5859, neg: 0.0, neu: 0.568, pos: 0.432, 
    The most progressive social and economic path gives us the best chance to catch up and Sen. Bernie Sanders represents the most progressive path,
    compound: 0.7351, neg: 0.0, neu: 0.78, pos: 0.22, 
    We are all thankful that today passed without incident. The teams successfully de-escalated what could have been a volatile situation. This resulted from weeks of planning and extensive cooperation among state, local, and federal partners in Virginia and beyond. Virginia's law enforcement and first responders demonstrated tremendous professionalism. I'm proud of their work. I have spoken with Colonel Settle of the State Police, Colonel Pike of the Capitol Police, and Chief Smith of the Richmond Police Department, as well as leaders of the FBI and the U.S. Attorney's office, and thanked them for keeping Virginia safe. Thousands of people came to Richmond to make their voices heard. Today showed that when people disagree, they can do so peacefully. The issues before us evoke strong emotions, and progress is often difficult. I will continue to listen to the voices of Virginians, and I will continue to do everything in my power to keep our Commonwealth safe.
    compound: 0.978, neg: 0.029, neu: 0.793, pos: 0.178, 
    What the President's counsel said was that no foreign policy was being conducted by a private person here. That is Rudy Giuliani was not conducting U.S. foreign policy. Rudy Giuliani was not conducting policy. That is a remarkable admission,
    compound: 0.34, neg: 0.054, neu: 0.858, pos: 0.088, 
    Wherever we have been in the state, people will come up to us and say, what do you think happened to Susan Collins?
    compound: 0.0, neg: 0.0, neu: 1.0, pos: 0.0, 
    You can trust he'll do what's right for Donald Trump.
    compound: 0.5106, neg: 0.0, neu: 0.732, pos: 0.268, 
    Your duty demands that you convict President Trump.
    compound: 0.0, neg: 0.0, neu: 1.0, pos: 0.0, 
    Again to review, President Trump used his personal agent for Ukraine,
    compound: 0.0, neg: 0.0, neu: 1.0, pos: 0.0, 
    aren't tough enough to stand with President Trump.
    compound: 0.0951, neg: 0.0, neu: 0.836, pos: 0.164, 
    I'm focused on this one. And I'm totally committed to holding Donald Trump accountable for what he has done.
    compound: 0.6115, neg: 0.0, neu: 0.773, pos: 0.227, 
    It should result in consequences. Nancy Pelosi disgraced the House of Representatives. She embarrassed our country. All of the world was watching as she had her petulant, childlike behavior of ripping an official record of the House submitted as an original over the signature of the President of the United States.
    compound: -0.1027, neg: 0.095, neu: 0.816, pos: 0.089, 
    It's an honor to be selected by President Trump to serve alongside him and his team.
    compound: 0.4939, neg: 0.0, neu: 0.824, pos: 0.176, 
    President Reagan said it was morning in America. President Trump said `make America great again. How great is it to have a president who embraces his role as leader of the free world?
    compound: 0.91, neg: 0.0, neu: 0.716, pos: 0.284, 
    President Trump didn't say he'd go after a cultural site,
    compound: 0.0, neg: 0.0, neu: 1.0, pos: 0.0, 
    Schumer and Pelosi staff spoke with Secretary Mnuchin and Treasury staff today. They agreed to continue talks tomorrow.
    compound: 0.4404, neg: 0.0, neu: 0.804, pos: 0.196, 
    The difference between Trump and me: His policies benefit large corporations and the rich; our policies benefit working families,
    compound: 0.8625, neg: 0.0, neu: 0.625, pos: 0.375, 
    And it'll never be otherwise. Because the one thing Trump will never be able to accept about the exalted office he holds is that, unlike his company, it doesn't belong to him,
    compound: 0.3818, neg: 0.0, neu: 0.923, pos: 0.077, 
    Donald Trump is using the coronavirus crisis as an excuse to propose a reduction in payroll contributions. This is a Trojan Horse attack on our Social Security system, which will do nothing to meaningfully address the crisis at hand. Other proposals to stimulate the economy, such as restoring the Making Work Pay Tax Credit or expanding the existing Earned Income Tax Credit, are more targeted and provide more fiscal stimulus. They are fairer in their distribution and place no administrative burdens on employers. The only reason to support the Trump proposal above those others is to undermine Social Security. This is true even if borrowed federal funds are substituted for Social Security's dedicated revenue. Under the guise of stimulating the economy, Trump's plan to reduce Social Security contributions would either undermine Social Security's financing or employ general revenue, both of which would set the stage for future demands to cut Social Security. At base, tax cuts do nothing to meaningfully address coronavirus, or even the resulting market panic. We do want to ensure people have the cash they need while they face massive uncertainties around employment and other costs. We want people to stay home as much as needed without having to worry about paying their rent or other costs. What we need most is a robust public health response, which the Trump Administration is utterly failing to provide. Earlier this week, Speaker Nancy Pelosi and Senate Minority Leader Chuck Schumer released an excellent list of steps we should take to combat coronavirus. Their plan includes paid sick leave, free coronavirus testing, and treatment for all. Our government should enact these measures, not slash payroll contributions. And, for supporters of Social Security, their plan, unlike Trump's, will not undermine this vital program.
    compound: 0.9217, neg: 0.126, neu: 0.699, pos: 0.175, 
    I'm glad Congress agreed to extend this relief to our veterans and I urge President Trump to sign this now.
    compound: 0.8564, neg: 0.0, neu: 0.59, pos: 0.41, 
    Jennifer, Richard, you just told me the story about how if you'd taken that small raise, if you did a little more overtime, you would have lost all of the benefits of Care 4 Kids, you would have lost money doing it, and instead we made it easier for middle class families to afford it so that your beautiful daughter, you'll know, will have a great place, a safe place, while you're hard at work,
    compound: 0.9409, neg: 0.068, neu: 0.706, pos: 0.225, 
    Lowering the cost of healthcare and improving the health of Americans has been a key priority of President Trump during his first three years in office, and tonight, he laid out the results that his leadership has produced... The President is fulfilling his promise to protect what works in our healthcare system and make it better, and more success is still to come.
    compound: 0.9117, neg: 0.028, neu: 0.771, pos: 0.201, 
    President Trump viscerally understands that the toppling of Saddam Hussein made Iran stronger. Soleimani, like Hussein, was an evil man who ordered the killing of Americans. Yet, the question remains, whether his death will lead to more instability in the Middle East or less,
    compound: -0.8625, neg: 0.224, neu: 0.687, pos: 0.09, 
    protect and expand affordable housing, preserve our small businesses and continue to strengthen the Richmond District community that 80,000 residents call home.
    compound: 0.7351, neg: 0.0, neu: 0.725, pos: 0.275, 
    So proud of my friend, Vice President Mike Pence. He is a former governor and understands what states are able to do during a crisis like this,
    compound: 0.6115, neg: 0.12, neu: 0.614, pos: 0.266, 
    States are different and I understand that the governor of Florida, great governor, Ron DeSantis, issued one today and that's good, that's great. But there are some states that are different. There are some states that don't have much of a problem,
    compound: 0.3612, neg: 0.076, neu: 0.773, pos: 0.151, 
    We will win the presidency. And most importantly, we will end the fear that so many people in this country have of a second term for Donald Trump.
    compound: 0.4927, neg: 0.095, neu: 0.714, pos: 0.19, 
    ABSOLUTELY ENOUGH of what we are getting from Donald Trump and his fellow-travelers right now.
    compound: 0.0, neg: 0.0, neu: 1.0, pos: 0.0, 
    Any responsible commander-in-chief would have done the same thing, but President Trump's the one who did it, and I'm glad that the president had the fortitude to say we're going to put America first. We're going to protect America and our allies around the world by doing this. And you know, if somebody wants to criticize him for it, I think they're going to have to look at a mirror and say, whose side are you on if you can question something that actually makes America safer?
    compound: 0.8537, neg: 0.035, neu: 0.832, pos: 0.133, 
    But, it's a hole that's been dug for a long time, but I think the hole by Trump has gotten much deeper and much faster, and there's no easy way out. Overall, it represents the stupidity of the interventionist foreign policy.
    compound: -0.4215, neg: 0.146, neu: 0.769, pos: 0.085, 
    I am again encouraging President [ Donald ] Trump and my congressional colleagues to work together to finally pass wide-sweeping legislation to bring our transportation infrastructure into the 21st century,
    compound: 0.5267, neg: 0.0, neu: 0.884, pos: 0.116, 
    I can prove beyond any doubt that Joe Biden's effort in the Ukraine to root out corruption was undercut, because he let his son sit on the board of the most corrupt company in the Ukraine, and we're not going to give him a pass on that,
    compound: -0.3612, neg: 0.054, neu: 0.946, pos: 0.0, 
    I can't think of a way that would make it easier for Donald Trump to get re-elected than listening to this conversation. It's ridiculous. We're not going to throw out capitalism. We tried. Other countries tried that. It was called communism, and it just didn't work.
    compound: 0.0772, neg: 0.053, neu: 0.888, pos: 0.059, 
    I know there's enormous pressure on him to support Biden,
    compound: 0.128, neg: 0.185, neu: 0.588, pos: 0.227, 
    It has nothing to do with you, Donald Trump. Nothing to do with you. Do your job. Stop personalizing everything.
    compound: -0.296, neg: 0.104, neu: 0.896, pos: 0.0, 
    Mike Pence, who enabled an HIV outbreak in Indiana, will lead US coronavirus response
    compound: 0.0, neg: 0.0, neu: 1.0, pos: 0.0, 
    North Carolina Republicans and President Trump have both reduced the tax burden on their voters by billions of dollars, facilitating increased business investment, new jobs, and growing wages,
    compound: -0.0258, neg: 0.091, neu: 0.789, pos: 0.12, 
    Obviously, a message that some people responded to very well, but for whatever reason this time there was a good number of folks who were looking for a different message. And all of the other parts of the ecosystem you deal with: the media coverage, where we start off these campaigns, the feeling, the anxiety of the Trump era and the cautiousness. All of those things.
    compound: 0.6904, neg: 0.03, neu: 0.843, pos: 0.127, 
    One of us has a history of not only fighting cuts to Social Security but working to expand benefits. And that's why we are the campaign best positioned to defeat Donald Trump.
    compound: 0.8968, neg: 0.089, neu: 0.535, pos: 0.376, 


We can also more strictly constrain the sentiment score to pick out mentions with only strong emotions. (sentiment score > 0.7)

This will result in fewer results, but more accurate in sentiment classification in general.


```python
for sentence in sentences:
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(sentence)
#     print(ss)
    if(ss['pos']> 0.7 or ss['neg']> 0.7):
        print(sentence)
        print(ss)
        #print('{0}: {1}, '.format(k, ss[k]), end='')
#     for k in sorted(ss):
        
#     print()
```

    hopeful & optimistic Trump.
    {'neg': 0.0, 'neu': 0.152, 'pos': 0.848, 'compound': 0.6808}
    Biden supports free college. Bernie supports free college.
    {'neg': 0.0, 'neu': 0.256, 'pos': 0.744, 'compound': 0.891}
    Acquitted! Trump Acquitted.
    {'neg': 0.0, 'neu': 0.189, 'pos': 0.811, 'compound': 0.5093}
    a great win for Bernie Sanders.
    {'neg': 0.0, 'neu': 0.275, 'pos': 0.725, 'compound': 0.836}
    Trump's irresponsible trade wars.
    {'neg': 0.765, 'neu': 0.235, 'pos': 0.0, 'compound': -0.7579}
    Trump is guilty as hell. I don't care.
    {'neg': 0.715, 'neu': 0.285, 'pos': 0.0, 'compound': -0.8758}
    will defeat Trump's hate and greed.
    {'neg': 0.758, 'neu': 0.242, 'pos': 0.0, 'compound': -0.8555}
    Trump kills terrorist, escapes quagmire
    {'neg': 0.701, 'neu': 0.171, 'pos': 0.128, 'compound': -0.8271}


## Task 3 LDA Topic Clustering

In this task we need to interpret what the politicians talk about when they are referring to another politician. We propose to do this with the help of LDA topic clustering. Here is a small example showing the feasibility and the current problems of the method.

We first defined some functions for pre-processing the quotations to extract the tokens dictionary with the help of [reference](https://towardsdatascience.com/nlp-extracting-the-main-topics-from-your-dataset-using-lda-in-minutes-21486f5aa925).


```python
def lemmatize_stemming(text):\
    '''lemmatize stem the text to get key tokens'''
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
# Tokenize and lemmatize
def preprocess(text):
    '''preprocess the quotation list and extract tokens to dictionary'''
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result
```

Then we instantiate the stemmer and use it to extract bag of words in each quotations.


```python
stemmer = ps() #instantilize
processed_docs = []
for quotation in sample_data['quotation'].tolist():
    processed_docs.append(preprocess(quotation))
dictionary = gensim.corpora.Dictionary(processed_docs) #words into
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs] #convert to bag of words
```

We build an LDA clustering model to cluster bag of words into popular topics, with a pre-defined cluster number of 8.


```python
lda_model =  gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = 8, 
                                   id2word = dictionary,                                    
                                   passes = 10,
                                   workers = 2)
```

Here is a visualisation of clustered frequent topics.


```python
for idx, topic in lda_model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic ))
    print("\n")
```

    Topic: 0 
    Words: 0.121*"trump" + 0.060*"donald" + 0.039*"presid" + 0.016*"beat" + 0.015*"american" + 0.014*"defeat" + 0.012*"peopl" + 0.011*"countri" + 0.010*"iran" + 0.009*"go"
    
    
    Topic: 1 
    Words: 0.037*"trump" + 0.023*"presid" + 0.016*"administr" + 0.014*"state" + 0.013*"work" + 0.011*"american" + 0.009*"health" + 0.007*"continu" + 0.007*"need" + 0.006*"protect"
    
    
    Topic: 2 
    Words: 0.055*"obama" + 0.019*"barack" + 0.019*"say" + 0.015*"presid" + 0.013*"year" + 0.010*"administr" + 0.010*"like" + 0.007*"bush" + 0.007*"good" + 0.007*"peopl"
    
    
    Topic: 3 
    Words: 0.062*"biden" + 0.023*"democrat" + 0.019*"sander" + 0.018*"berni" + 0.016*"presid" + 0.015*"go" + 0.014*"think" + 0.012*"peopl" + 0.012*"candid" + 0.012*"campaign"
    
    
    Topic: 4 
    Words: 0.014*"sander" + 0.013*"berni" + 0.010*"senat" + 0.008*"pelosi" + 0.008*"nanci" + 0.008*"puerto" + 0.007*"romney" + 0.007*"rico" + 0.007*"support" + 0.007*"state"
    
    
    Topic: 5 
    Words: 0.033*"trump" + 0.032*"bloomberg" + 0.027*"mike" + 0.016*"money" + 0.015*"spend" + 0.014*"talk" + 0.012*"michael" + 0.011*"million" + 0.011*"social" + 0.009*"peopl"
    
    
    Topic: 6 
    Words: 0.029*"presid" + 0.025*"john" + 0.021*"trump" + 0.020*"wit" + 0.019*"hous" + 0.019*"bolton" + 0.014*"investig" + 0.013*"hear" + 0.013*"ukrain" + 0.013*"say"
    
    
    Topic: 7 
    Words: 0.048*"trump" + 0.029*"presid" + 0.019*"senat" + 0.018*"think" + 0.018*"know" + 0.017*"go" + 0.016*"vote" + 0.015*"impeach" + 0.013*"clinton" + 0.012*"say"
    
    


There is an obvious problem that clusters are centered on politician names. This is understandable because we filtered the data to pick out the quotations containing politician names. For the final project in MS3, we should prune any of the politician names from the dictionary. In this way, we expect the clusters to go into meaningful topics in the end.
