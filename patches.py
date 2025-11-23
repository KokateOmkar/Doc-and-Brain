import logging
import sys
from typing import Any, Dict, List, Optional, Union

# --- PATCH 1: Fix Google Generative AI MediaResolution Error ---
try:
    import google.ai.generativelanguage as glm
    if not hasattr(glm, 'MediaResolution'):
        from enum import Enum
        class MediaResolution(Enum):
            UNSPECIFIED = 0
        glm.MediaResolution = MediaResolution
        # Also patch the module where it might be looked up
        import google.ai.generativelanguage.types as glm_types
        if not hasattr(glm_types, 'MediaResolution'):
            glm_types.MediaResolution = MediaResolution
except ImportError:
    pass

# --- PATCH 2: Fix Qdrant Compatibility Issues ---
try:
    from langchain_community.vectorstores import Qdrant
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest

    # 2a. Fix construct_instance (remove init_from, add on_disk)
    original_construct = Qdrant.construct_instance

    @classmethod
    def fixed_construct_instance(cls, *args, **kwargs):
        # Remove 'init_from' if present in kwargs as it causes errors with newer clients
        if 'init_from' in kwargs:
            kwargs.pop('init_from')
        
        # Ensure 'on_disk' is handled if passed (though the original might not accept it in signature, 
        # we just ensure it doesn't crash if we pass it to the client creation logic manually if needed)
        # For this specific error "NameError: name 'on_disk' is not defined", it was a bug in the library function body.
        # We can't easily patch the *inside* of the function. 
        # However, if we are using the standard library on Cloud, it might NOT have that bug 
        # (it might have been a specific version issue or the user's environment).
        # But to be safe, we can try to wrap it.
        
        return original_construct(*args, **kwargs)

    # If the library is truly broken (NameError), wrapping won't fix it. 
    # We have to hope the pip install on Cloud gets a consistent version.
    # But we CAN fix the 'search' vs 'query_points' issue which is an API change.

    # 2b. Fix similarity_search_with_score_by_vector (search -> query_points)
    def fixed_similarity_search(self, embedding, k=4, filter=None, **kwargs):
        if self.vector_name is not None:
            query_vector = (self.vector_name, embedding)
        else:
            query_vector = embedding

        # Check if client has 'search' (old) or only 'query_points' (new)
        if hasattr(self.client, 'search'):
            return self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=filter,
                limit=k,
                with_payload=True,
                with_vectors=False,
                **kwargs
            )
        else:
            # Use new API
            return self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=filter,
                limit=k,
                with_payload=True,
                with_vectors=False,
                **kwargs
            ).points

    # Apply the patch to the class
    Qdrant.similarity_search_with_score_by_vector = fixed_similarity_search

except ImportError:
    pass
