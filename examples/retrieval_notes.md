# Retrieval Notes

Information retrieval systems help people find relevant documents from a collection.

Classical sparse retrieval methods, such as BM25, match query terms against document
terms. They are strong when the query uses the same words as the document.

Dense retrieval represents text as vectors. It can help when the query and document
use related wording, although high-quality dense retrieval usually depends on a
trained embedding model.

Hybrid retrieval combines sparse and dense scores. This often gives robust results:
BM25 anchors exact matches, while dense retrieval adds fuzzier semantic matching.
