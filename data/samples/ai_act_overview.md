# EU AI Act: Structure, Obligations and Timelines

This handbook summarises Regulation (EU) 2024/1689 laying down harmonised rules
on artificial intelligence. It is an internal paraphrase for engineering teams
and is not legal advice. Article numbers refer to the Regulation as published
in the Official Journal.

## Article 3 - Definitions and the risk-based approach

An AI system is defined as a machine-based system designed to operate with
varying levels of autonomy, that may exhibit adaptiveness after deployment, and
that infers from the input it receives how to generate outputs such as
predictions, content, recommendations or decisions that can influence physical
or virtual environments. The Regulation distinguishes a provider, who develops
an AI system or has it developed and places it on the market under its own name
or trademark, from a deployer, who uses an AI system under its own authority.
Importers and distributors carry separate, lighter obligations. The same
organisation can be both provider and deployer, and a deployer becomes a
provider if it puts its own name on a high-risk system, modifies its intended
purpose, or makes a substantial modification.

## Article 5 - Prohibited AI practices

The following practices are prohibited. Placing on the market or using an AI
system that deploys subliminal techniques beyond a person's consciousness, or
purposefully manipulative or deceptive techniques, with the effect of
materially distorting behaviour in a manner that causes or is likely to cause
significant harm. Exploiting vulnerabilities due to age, disability or a
specific social or economic situation. Social scoring, meaning the evaluation
or classification of natural persons over time based on social behaviour or
personal characteristics, where the resulting detrimental treatment is
unrelated to the context in which the data was generated or is unjustified and
disproportionate. Risk assessment of natural persons to predict criminal
offences based solely on profiling or on personality traits, except where used
to support a human assessment already based on objective facts. Untargeted
scraping of facial images from the internet or CCTV footage to build facial
recognition databases. Inferring emotions in the workplace or in education
institutions, except for medical or safety reasons. Biometric categorisation to
deduce race, political opinions, trade union membership, religious or
philosophical beliefs, sex life or sexual orientation. Real-time remote
biometric identification in publicly accessible spaces for law enforcement
purposes, subject to narrowly drawn exceptions requiring prior judicial or
administrative authorisation.

## Article 6 and Annex III - Classification of high-risk AI systems

An AI system is high-risk in two situations. First, where it is a safety
component of a product, or is itself a product, covered by the Union
harmonisation legislation listed in Annex I and required to undergo third-party
conformity assessment. Second, where it falls within one of the eight areas of
Annex III: biometrics, including remote biometric identification and emotion
recognition; critical infrastructure such as road traffic and the supply of
water, gas, heating and electricity; education and vocational training,
including admission and assessment; employment, worker management and access to
self-employment, including recruitment screening and promotion decisions;
access to essential private and public services, including credit scoring and
the assessment of eligibility for benefits; law enforcement; migration, asylum
and border control; and the administration of justice and democratic processes.
An Annex III system is not high-risk where it performs a narrow procedural
task, improves the result of a previously completed human activity, detects
decision-making patterns without replacing human assessment, or performs a
preparatory task; but a system that profiles natural persons is always
high-risk.

## Articles 9 to 15 - Requirements for high-risk AI systems

A risk management system must be established, implemented, documented and
maintained across the entire lifecycle, with regular systematic review. Data
governance under Article 10 requires training, validation and testing data sets
to be relevant, sufficiently representative, and to the best extent possible
free of errors and complete in view of the intended purpose; examination for
possible biases is mandatory. Technical documentation under Article 11 must be
drawn up before the system is placed on the market and kept up to date, and
Annex IV lists its minimum content. Article 12 requires automatic recording of
events, that is logs, over the lifetime of the system, with a minimum retention
period determined by the intended purpose and in any case not less than six
months unless other law provides otherwise. Article 13 requires the system to
be sufficiently transparent to enable deployers to interpret output and use it
appropriately, accompanied by instructions for use. Article 14 requires that
high-risk systems be designed so they can be effectively overseen by natural
persons during the period in which they are in use, including the ability to
decide not to use the output and to intervene or interrupt operation; for
certain remote biometric identification systems, no action may be taken on the
basis of identification unless separately verified and confirmed by at least
two natural persons with the necessary competence. Article 15 requires an
appropriate level of accuracy, robustness and cybersecurity, with accuracy
metrics declared in the instructions for use.

## Articles 16 to 27 - Obligations of providers and deployers

Providers of high-risk systems must ensure compliance with the requirements of
Articles 9 to 15, have a quality management system in place, keep the technical
documentation and the automatically generated logs for ten years after the
system is placed on the market, undergo the relevant conformity assessment,
draw up an EU declaration of conformity, affix the CE marking, and register the
system in the EU database before placing it on the market. Deployers must use
the system in accordance with the instructions for use, assign human oversight
to natural persons with the necessary competence, training and authority,
ensure that input data is relevant and sufficiently representative in view of
the intended purpose, monitor operation, keep the logs they control for at
least six months, and inform workers' representatives before putting a
high-risk system into service in the workplace. Deployers that are bodies
governed by public law, or private entities providing public services, and
deployers of credit-scoring or life and health insurance pricing systems, must
carry out a fundamental rights impact assessment before first use.

## Article 50 - Transparency obligations for certain AI systems

Providers must ensure that AI systems intended to interact directly with
natural persons are designed so that those persons are informed they are
interacting with an AI system, unless this is obvious from the point of view of
a reasonably well-informed person. Providers of systems generating synthetic
audio, image, video or text content must ensure the outputs are marked in a
machine-readable format and detectable as artificially generated or
manipulated. Deployers of emotion recognition or biometric categorisation
systems must inform the exposed persons. Deployers that generate or manipulate
image, audio or video content constituting a deep fake must disclose that the
content has been artificially generated or manipulated; where the content forms
part of an evidently artistic, creative, satirical or fictional work, the
disclosure obligation is limited to an appropriate manner that does not hamper
the display or enjoyment of the work.

## Articles 51 to 55 - General-purpose AI models

A general-purpose AI model is a model that displays significant generality, is
capable of competently performing a wide range of distinct tasks, and can be
integrated into a variety of downstream systems. Providers of such models must
draw up and keep up to date technical documentation, provide information to
downstream providers that integrate the model, put in place a policy to comply
with Union copyright law including respecting rights reservations expressed
through machine-readable means, and publish a sufficiently detailed summary of
the content used for training according to a template provided by the AI
Office. A general-purpose AI model is classified as presenting systemic risk
where it has high impact capabilities, and a model is presumed to have high
impact capabilities where the cumulative amount of computation used for its
training measured in floating point operations is greater than 10 to the power
of 25. Providers of models with systemic risk must in addition perform model
evaluation including adversarial testing, assess and mitigate systemic risks,
track and report serious incidents to the AI Office without undue delay, and
ensure an adequate level of cybersecurity protection.

## Articles 57 to 63 - Regulatory sandboxes and support for SMEs

Member States must ensure that at least one AI regulatory sandbox is
established at national level and is operational by 2 August 2026, either
individually or jointly with other Member States. A sandbox provides a
controlled environment for development, training, testing and validation before
market placement, under regulatory supervision and for a limited time, with an
exit report that competent authorities take into account in subsequent
conformity assessment. Testing of high-risk systems in real-world conditions
outside sandboxes is permitted under Article 60 subject to a testing plan,
approval by the market surveillance authority, informed consent from subjects,
and a maximum duration of six months, extendable once by a further six months.
SMEs and start-ups are entitled to priority access to sandboxes and to reduced
conformity assessment fees.

## Articles 72 and 73 - Post-market monitoring and incident reporting

Providers of high-risk AI systems must establish and document a post-market
monitoring system proportionate to the nature and risks of the system, based on
a post-market monitoring plan. Providers must report any serious incident to
the market surveillance authorities of the Member States where the incident
occurred. A serious incident means an incident or malfunctioning leading
directly or indirectly to the death of a person or serious harm to health, a
serious and irreversible disruption of the management or operation of critical
infrastructure, infringement of obligations under Union law intended to protect
fundamental rights, or serious harm to property or the environment. The report
must be made immediately after the provider has established a causal link, and
in any event not later than 15 days after becoming aware of the incident. In
the event of a widespread infringement or a serious and irreversible disruption
of critical infrastructure the deadline is 2 days, and in the event of the
death of a person the deadline is 10 days.

## Article 99 - Penalties

Non-compliance with the prohibited practices of Article 5 is subject to
administrative fines of up to 35 000 000 EUR or, if the offender is an
undertaking, up to 7 per cent of its total worldwide annual turnover for the
preceding financial year, whichever is higher. Non-compliance with any other
obligation of the Regulation, including the obligations of providers,
deployers, importers, distributors and notified bodies, is subject to fines of
up to 15 000 000 EUR or up to 3 per cent of total worldwide annual turnover.
The supply of incorrect, incomplete or misleading information to notified
bodies or national competent authorities is subject to fines of up to
7 500 000 EUR or up to 1 per cent of turnover. For SMEs including start-ups,
each fine is capped at the lower of the two figures rather than the higher.

## Article 113 - Entry into application

The Regulation entered into force on 1 August 2024. The prohibitions in Article
5 and the AI literacy obligation in Article 4 apply from 2 February 2025. The
rules on general-purpose AI models, on notified bodies, on governance and on
penalties apply from 2 August 2025. The Regulation applies in general from
2 August 2026, which is the date from which the obligations for high-risk
systems listed in Annex III take effect. For high-risk systems that are safety
components of products covered by Annex I harmonisation legislation, the
obligations apply from 2 August 2027. General-purpose AI models placed on the
market before 2 August 2025 must comply by 2 August 2027.
