#!/usr/bin/env python3

from memvid import MemvidRetriever

def debug_memvid_results():
    retriever = MemvidRetriever('data/podcast_batch_20_random.mp4', 'data/podcast_batch_20_random_index.json')
    results = retriever.search('Churchill', top_k=2)
    
    print('Results type:', type(results))
    print('Number of results:', len(results))
    
    if results:
        print('\nFirst result:')
        print('  Type:', type(results[0]))
        print('  Attributes:', [attr for attr in dir(results[0]) if not attr.startswith('_')])
        print('  Content:', results[0])
        
        # Try to access common attributes
        try:
            print('  Text:', getattr(results[0], 'text', 'No text attribute'))
        except:
            print('  Text: Error accessing text')
            
        try:
            print('  Score:', getattr(results[0], 'score', 'No score attribute'))
        except:
            print('  Score: Error accessing score')

if __name__ == '__main__':
    debug_memvid_results() 