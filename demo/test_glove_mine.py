import logging

from nose.tools import assert_equal, assert_true
import numpy as np
from numpy.testing import assert_allclose

import evaluate2
import glove2
import numpy
import pickle

t_corpus = ("""axirassa/overlord/ExecutionSpecification/executeInstance(int)#java/lang/ProcessBuilder/redirectErrorStream(boolean)
axirassa/util/JavaLibraryPath/addFile(java.lang.String)#java/io/File/getParent()
org/apache/zookeeper/server/NettyServerCnxnFactory/CnxnChannelHandler/messageReceived(org.jboss.netty.channel.ChannelHandlerContext,org.jboss.netty.channel.MessageEvent)#org/slf4j/Logger/debug(java.lang.String)
org/apache/zookeeper/server/persistence/FileTxnLog/append(org.apache.zookeeper.txn.TxnHeader,org.apache.jute.Record)#java/util/zip/Checksum/update(byte[],int,int)
org/tukaani/xz/index/IndexEncoder/encode(java.io.OutputStream)#java/util/zip/CRC32/getValue()
org/xwiki/wiki/script/WikiManagerScriptService/canDeleteWiki(java.lang.String,java.lang.String)#org/xwiki/wiki/descriptor/WikiDescriptor/getOwnerId()
org/neo4j/jmx/impl/KernelBean/DataSourceInfo/registered(org.neo4j.kernel.NeoStoreDataSource)#org/neo4j/kernel/impl/store/StoreId/getRandomId()
org/neo4j/harness/internal/Fixtures/add(java.io.File)#org/neo4j/io/fs/FileUtils/readTextFile(java.io.File,java.nio.charset.Charset)
morfologik/speller/Speller/charsToBytes(java.nio.CharBuffer,java.nio.ByteBuffer)#java/nio/ByteBuffer/allocate(int)
org/apache/maven/toolchain/model/io/xpp3/MavenToolchainsXpp3Reader/parsePersistedToolchains(java.lang.String,org.codehaus.plexus.util.xml.pull.XmlPullParser,boolean,java.lang.String)#java/util/List/add(java.lang.Object)
com/zendaimoney/mqmonitor/command/Command/runPerXMill()#java/net/URI/URI(java.lang.String)""").split("\n")

W = numpy.loadtxt(open("/Users/ljl/Downloads/glove/glove.py-master/MV_L/evaluation/W.csv","rb"),delimiter=",",skiprows=0)
valW = numpy.loadtxt(open("/Users/ljl/Downloads/glove/glove.py-master/MV_L/evaluation/valW.csv","rb"),delimiter=",",skiprows=0)
f2 = open("/Users/ljl/Downloads/glove/glove.py-master/MV_L/evaluation/myvocab.txt","rb")
myvocab = pickle.load(f2)
f2.close()
f2 = open("/Users/ljl/Downloads/glove/glove.py-master/MV_L/evaluation/valvocab.txt","rb")
valvocab = pickle.load(f2)
f2.close()

f = open('//Users/ljl/Downloads/glove/glove.py-master/MV_L/evaluation/train.txt')
train_corpus = f.readlines()


corpus = glove2.build_corpus(train_corpus)
tcorpus = glove2.build_corpus(t_corpus)
tvocab = glove2.build_vocab_mine(tcorpus)

evaluate2.Successrate(W,myvocab,corpus,tvocab,n=5)
# methodid=evaluate2.findcommon(myvocab,valvocab)
# evaluate2.MRRcompare(W, valW, myvocab,valvocab, n=10,m=1)