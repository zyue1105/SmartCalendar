import feedparser
from bs4 import BeautifulSoup
from icalendar import Calendar, Event
from datetime import datetime
import numpy as np

def generate_training_data():
    ''' () -> (list, list)

    Return list of documents, in which each document is a dict {'Contents' : '', 'Title' : ''},
    and list of labels (1 : food, 2 : movie, 3 : seminer, 4 : workshop, 5 : music)
    '''
    d = feedparser.parse(r'./Training_data/test.txt')
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

    with open('./Training_data/labels.txt') as f :
        labels_str = f.read().split('\n')
    labels = [int(x) for x in labels_str]

    for i in range(len(documents)):
        if 'food' in documents[i] and label[i] != 1:
            print i

    #print labels
    return (documents, labels)
        

def generate_test_data():
    ''' () -> list

    Return list of documents, in which each document is a dict
    {'Content' : , 'Title' : , 'link' : , 'start' : , 'end' : , 'location' : }
    '''
    documents = []    
    url_affix = '&format=rss'
    
    for month in range(12):
        url_prefix = 'http://calendar.tamu.edu/?&y=2013&m=' + str(month + 1) + '&d='
        print 'month', month + 1, ':'
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

                # add source to content
                if tmp_dict['link'] != None and tmp_dict['Content'] != None:
                    tmp_dict['Content'] += '\n\nSource\n' + tmp_dict['link']
                
                #print tmp_dict
                if tmp_dict not in documents:
                    documents.append(tmp_dict)

            print 'day', day + 1, ':', len(documents)

    print len(documents)

    return documents

def generate_test_data_icalendar():
    ''' () -> list

    Return list of documents, in which each document is a dict
    {'Content' : , 'Title' : , 'link' : , 'start' : , 'end' : , 'location' : }
    '''
    cal = Calendar.from_ical(open('./Test_data/UTaustin.ics','rb').read())
    cnt = 0
##    total = 100
    documents = []
    
    for component in cal.walk():        
        if component.name == 'VEVENT':            
            tmp_dict = {}
            tmp_dict['Content'] = component.get('description').format()
            tmp_dict['Title'] = component.get('summary').format()
            url = component.get('url')            
            if url != None and '{' not in url:
                tmp_dict['link'] = url.format()
            else:
                tmp_dict['link'] = None
            # convert datetime to dateime64
            tmp_dict['start'] = str(np.datetime64(component.get('dtstart').dt))
            tmp_dict['end'] = str(np.datetime64(component.get('dtend').dt))
            tmp_dict['location'] = component.get('location').format()

            if 'T' not in tmp_dict['start'] or 'T' not in tmp_dict['end']:
                continue
            # add source to content
            if tmp_dict['link'] != None and tmp_dict['Content'] != None:
                tmp_dict['Content'] += '\n\nSource\n' + tmp_dict['link']    

            if tmp_dict not in documents:
                documents.append(tmp_dict)
                cnt += 1
##        if cnt > total:
##            break

    print 'Test data generated'
        
    return documents

