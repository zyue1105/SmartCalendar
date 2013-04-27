import feedparser
from bs4 import BeautifulSoup

def generate_training_data():
    ''' () -> (list, list)

    Return list of documents, in which each document is a dict {'Contents' : '', 'Title' : ''},
    and list of labels (1 : food, 2 : movie, 3 : seminer, 4 : workshop, 5 : music)
    '''
    d = feedparser.parse(r'./Training_data/test1.txt')
    documents = []
    for i in range(len(d.entries)):
        #print i
        tmp_dict = {}
        tmp_dict['Title'] = d.entries[i].title
        #print d.entries[i].title
        # parse HTML
        html_doc = d.entries[i].description    
        soup = BeautifulSoup(html_doc)
        # find the contents
        summary = soup.find('div', class_= 'summary')
        if summary != None:
            tmp_dict['Content'] = summary.string
        else:
            tmp_dict['Content'] = html_doc
                                
        documents.append(tmp_dict)

    with open('./Training_data/labels1.txt') as f :
        labels_str = f.read().split('\n')
    labels = [int(x) for x in labels_str]

    for i in range(len(documents)):
        if 'food' in documents[i] and label[i] != 1:
            print i

    #print labels
    return (documents, labels)
        

def generate_test_data():
    ''' () -> list

    Return list of documents, in which each document is a dict {'Contents' : '', 'Title' : ''}
    '''
    documents = []
    url_prefix = 'http://calendar.tamu.edu/?&y=2013&m=4&d='
    url_affix = '&format=rss'

    for day in range(31):
        
        url = url_prefix + str(day + 1) + url_affix
        d = feedparser.parse(url)

        for i in range(len(d.entries)):
            tmp_dict = {}
            tmp_dict['Title'] = d.entries[i].title
            tmp_dict['link'] = d.entries[i].link
           
            # parse HTML
            html_doc = d.entries[i].description    
            soup = BeautifulSoup(html_doc)            
            # print soup.prettify()        
            # find the contents    
            tmp_dict['Content'] = soup.find('div', class_= 'summary').string           
            
            # find the time
            #print html_doc
            tmp_dict['start'] = soup.find('abbr', class_ = 'dtstart')['title']
            dtend = soup.find('abbr', class_ = 'dtend')
            if dtend != None:
                tmp_dict['end'] = dtend['title']
            else:
                tmp_dict['end'] = tmp_dict['start']            

            # find the location
            tmp_dict['location'] = soup.find('small', class_ = 'location').string
            
            #print tmp_dict
            documents.append(tmp_dict)

        print 'day', day + 1, ':', len(documents)

    print len(documents)

    return documents



