from causal_discovery.pipeline.pipeline import CausalDiscoveryPipeline, BatchCasualDiscoveryPipeline

# The pipeline runner is fully generic — it just chains stages sequentially.
# We alias it for clarity in shortest_path context.
ShortestPathPipeline = CausalDiscoveryPipeline
BatchShortestPathPipeline = BatchCasualDiscoveryPipeline
