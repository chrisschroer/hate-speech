import csv
import codecs

from snorkel.models import Document
from snorkel.parser import DocPreprocessor

from snorkel.matchers import RegexMatch


class HateSpeechPreprocessor(DocPreprocessor):
    """
    Simple parsing of of example files for the NOHATE project.
    Parses csv files like: (<Dokumentenguid>,<Beitragstitel>,>Beitragsinhalt<,<Publikationsdatum>,<Quellentyp>,<Beinhaltet Link>,<Beinhaltet Bild>,<Beinhaltet Video>,<Beinhaltet nur Text>)
    Into: document name -> <Dokumentenguid>
          document text -> <Beitragsinhalt>
    """

    def parse_file(self, fp, file_name):
        
        with codecs.open(fp, encoding=self.encoding) as file:
            
            csv_reader = csv.reader(file, delimiter=',')
    
            for row in csv_reader:
            
                doc_name = row[0]
                doc_text = f"{row[2]}"
                stable_id = self.get_stable_id(doc_name)
                
                doc = Document(
                    name=doc_name,
                    stable_id=stable_id,
                    meta={'file_name': file_name}
                )
                
                yield doc, doc_text



# Use r.search instead of r.match to find also snippets of dict-strings in sentences
class RegexMatchEachSearch(RegexMatch):
    """Matches regex pattern on **each token**"""
    def _f(self, c):
        tokens = c.get_attrib_tokens(self.attrib)
        return True if tokens and all([self.r.search(t) is not None for t in tokens]) else False