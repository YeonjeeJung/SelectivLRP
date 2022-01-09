import matplotlib.pyplot as plt

def rescale_score_by_abs(score, max_score, min_score):
    if max_score > 0 and min_score < 0:
        if max_score >= abs(min_score):
            if score >= 0:
                return 0.5 + 0.5 * (score / max_score)
            else:
                return 0.5 - 0.5 * (abs(score) / max_score)
            
            
        else:
            if score >= 0:
                return 0.5 + 0.5 * (score / abs(min_score))
            else:
                return 0.5 - 0.5 * (score / min_score)
            
            
    elif max_score > 0 and min_score >= 0:
        if max_score == min_score:
            return 1.0
        else:
            return 0.5 + 0.5 * (score / max_score)
            
    elif max_score <= 0 and min_score < 0:
        if max_score == min_score:
            return 0.0
        else:
            return 0.5 - 0.5 * (score / min_score)
        
    elif max_score == 0 and min_score == 0:
        return 0.0
            
def getRGB(c_tuple):
    return "#%02x%02x%02x"%(int(c_tuple[0]*255), int(c_tuple[1]*255), int(c_tuple[2]*255))

def span_word(word, score, colormap):
#     print(colormap(score))
    return "<span style=\"background-color:"+getRGB(colormap(score))+"\">"+word+"</span>"

def html_heatmap(words, scores, cmap_name='bwr'):
    colormap = plt.get_cmap(cmap_name)
    
    assert len(words) == len(scores)
    
    max_s = max(scores)
    min_s = min(scores)
    
    output_text = ""
    
    for idx, w in enumerate(words):
        score = rescale_score_by_abs(scores[idx], max_s, min_s)
        output_text = output_text + span_word(w, score, colormap) + " "
        
    return output_text + "\n"