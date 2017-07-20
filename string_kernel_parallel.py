import numpy as np
from sklearn import svm
from sklearn.datasets import fetch_20newsgroups
from sklearn.svm import libsvm
import sys
from time import time
from functools import wraps
import multiprocessing
import itertools


def myapply(X1el, X2,ob):
    len_X2 = len(X2)
    A = [0] * len_X2
    #print "+++started computation of line", i
    for j in xrange(i, len_X2):
        A[j] = ob._gram_matrix_element_par(X1el, X2[j])
    #print "---ended computation of line", i

    return A

def myapply_asymmetric(X1el, X2,ob):
    len_X2 = len(X2)
    A = [0] * len_X2
    #print "+++started computation of line", i
    for j in xrange(len_X2):
        A[j] = ob._gram_matrix_element_par(X1el, X2[j])
    #print "---ended computation of line", i

    return A

def myapply_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return myapply(*a_b)

def myapply_star_asymmetric(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return myapply_asymmetric(*a_b)



class StringKernel():
    """
    Implementation of string kernel from article:
    H. Lodhi, C. Saunders, J. Shawe-Taylor, N. Cristianini, and C. Watkins.
    Text classification using string kernels. Journal of Machine Learning Research, 2, 2002 .
    svm.SVC is a basic class from scikit-learn for SVM classification (in multiclass case, it uses one-vs-one approach)
    """
    def __init__(self, subseq_length=3, lambda_decay=0.5):
        """
        Constructor
        :param lambda_decay: lambda parameter for the algorithm
        :type  lambda_decay: float
        :param subseq_length: maximal subsequence length
        :type subseq_length: int
        """
        self.lambda_decay = lambda_decay
        self.subseq_length = subseq_length

    def _K(self, n, s, t):
        """
        K_n(s,t) in the original article; recursive function
        :param n: length of subsequence
        :type n: int
        :param s: document #1
        :type s: str
        :param t: document #2
        :type t: str
        :return: float value for similarity between s and t
        """

        if min(len(s), len(t)) < n:
            return 0
        else:
            part_sum = 0
            #for j in range(1, len(t)): Nick corretto bug?? perche da 1 ??
            for j in range(0, len(t)):

                if t[j] == s[-1]:
                    #not t[:j-1] as in the article but t[:j] because of Python slicing rules!!!
                    part_sum += self._K1(n - 1, s[:-1], t[:j])
            result = self._K(n, s[:-1], t) + self.lambda_decay ** 2 * part_sum
            return result


    def _K1(self, n, s, t):
        """
        K'_n(s,t) in the original article; auxiliary intermediate function; recursive function
        :param n: length of subsequence
        :type n: int
        :param s: document #1
        :type s: str
        :param t: document #2
        :type t: str
        :return: intermediate float value
        """
        if n == 0:
            return 1
        elif min(len(s), len(t)) < n:
            return 0
        else:
            result = self._K2(n, s, t) + self.lambda_decay * self._K1(n, s[:-1], t)
            return result


    def _K2(self, n, s, t):
        """
        K''_n(s,t) in the original article; auxiliary intermediate function; recursive function
        :param n: length of subsequence
        :type n: int
        :param s: document #1
        :type s: str
        :param t: document #2
        :type t: str
        :return: intermediate float value
        """
        if n == 0:
            return 1
        elif min(len(s), len(t)) < n:
            return 0
        else:
            if s[-1] == t[-1]:
                return self.lambda_decay * (self._K2(n, s, t[:-1]) +
                                            self.lambda_decay * self._K1(n - 1, s[:-1], t[:-1]))
            else:
                return self.lambda_decay * self._K2(n, s, t[:-1])


    # def _gram_matrix_element(self, s, t, sdkvalue1, sdkvalue2):
    #     """
    #     Helper function
    #     :param s: document #1
    #     :type s: str
    #     :param t: document #2
    #     :type t: str
    #     :param sdkvalue1: K(s,s) from the article
    #     :type sdkvalue1: float
    #     :param sdkvalue2: K(t,t) from the article
    #     :type sdkvalue2: float
    #     :return: value for the (i, j) element from Gram matrix
    #     """
    #     print s, t
    #     print sdkvalue1,sdkvalue2
    #     if s == t:
    #         return 1
    #     else:
    #         try:
    #             return self._K(self.subseq_length, s, t) / \
    #                    (sdkvalue1 * sdkvalue2) ** 0.5
    #         except ZeroDivisionError:
    #             print("Maximal subsequence length is less or equal to documents' minimal length."
    #                   "You should decrease it")
    #             sys.exit(2)

    def _gram_matrix_element(self, s, t, sdkvalue1, sdkvalue2):
        """
        NICK: non normalizzo
        Helper function
        :param s: document #1
        :type s: str
        :param t: document #2
        :type t: str
        :param sdkvalue1: K(s,s) from the article
        :type sdkvalue1: float
        :param sdkvalue2: K(t,t) from the article
        :type sdkvalue2: float
        :return: value for the (i, j) element from Gram matrix
        """
        #print s, t
        # if s == t:
        #     return 1
        # else:
        a=0.0
        for length in xrange(1,self.subseq_length+1):
            if(sdkvalue1[length]* sdkvalue2[length] > 0):
                a+=self._K(length, s, t)/ (sdkvalue1[length] * sdkvalue2[length]) ** 0.5
        #print a /(sdkvalue1 * sdkvalue2) ** 0.5
            #else:
        #print "K", s,t, "=",a

        return a
    def _gram_matrix_element_par(self, s, t):
        """
        NICK: non normalizzo
        Helper function
        :param s: document #1
        :type s: str
        :param t: document #2
        :type t: str
        :param sdkvalue1: K(s,s) from the article
        :type sdkvalue1: float
        :param sdkvalue2: K(t,t) from the article
        :type sdkvalue2: float
        :return: value for the (i, j) element from Gram matrix
        """
        #print s, t
        # if s == t:
        #     return 1
        # else:
        a=0.0

        for length in xrange(1,self.subseq_length+1):
            k1 = self._K(length, s, s)
            k2 = self._K(length, t, t)
            if(k1* k2 > 0):
                a+=self._K(length, s, t)/ (k1 * k2) ** 0.5
        #print a /(sdkvalue1 * sdkvalue2) ** 0.5
            #else:
        #print "K", s,t, "=",a

        return a
    def string_kernel(self, X1, X2,n=4):
        """
        String Kernel computation
        :param X1: list of documents (m rows, 1 column); each row is a single document (string)
        :type X1: list
        :param X2: list of documents (m rows, 1 column); each row is a single document (string)
        :type X2: list
        :return: Gram matrix for the given parameters
        """
        len_X1 = len(X1)
        len_X2 = len(X2)
        # numpy array of Gram matrix
        gram_matrix = np.zeros((len_X1, len_X2), dtype=np.float32)
        sim_docs_kernel_value = {}
        pool = multiprocessing.Pool(processes=n)
        #when lists of documents are identical
        if X1 == X2:


            gram_matrix[:, :]=pool.map(myapply_star,itertools.izip((i  for i in X1),itertools.repeat(X2),itertools.repeat(self)))
                    #gram_matrix[i, j] = self._gram_matrix_element(X1[i], X2[j], sim_docs_kernel_value[1][i],  sim_docs_kernel_value[2][j])
            #using symmetry
            for i in range(len_X1):
              for j in range(i, len_X2):
                    gram_matrix[j, i] = gram_matrix[i, j]


                   # gram_matrix[i, j] = self._gram_matrix_element(X1[i], X2[j], sim_docs_kernel_value[i],      sim_docs_kernel_value[j])
        #using symmetry
                   # gram_matrix[j, i] = gram_matrix[i, j]

        #when lists of documents are neither identical nor of the same length
        else:

            gram_matrix[:, :] = pool.map(myapply_star_asymmetric, itertools.izip((i for i in X1), itertools.repeat(X2), itertools.repeat(self)))
            # gram_matrix[i, j] = self._gram_matrix_element(X1[i], X2[j], sim_docs_kernel_value[1][i],  sim_docs_kernel_value[2][j])
            # using symmetry

        return gram_matrix


# #main di prova
# if __name__ == '__main__':
#     cur_f = __file__.split('/')[-1]
#     if len(sys.argv) != 3:
#         print >> sys.stderr, 'usage: ' + cur_f + ' <maximal subsequence length> <lambda (decay)>'
#         sys.exit(1)
#     else:
#         subseq_length = int(sys.argv[1])
#         lambda_decay = float(sys.argv[2])
#         kernel=StringKernelSVM(subseq_length,lambda_decay)
#     #The dataset is the 20 newsgroups dataset. It will be automatically downloaded, then cached.
#         t_start = time()
#         X_train = ["efewfwerfrfrrwwr", "efe4f43f3rfwerfrfrrwwr","efewfwerfrfrrwwr", "efe4f43f3rfwerfrfrrwwr","efewfwerfrfrrwwr", "efe4f43f3rfwerfrfrrwwr","efewfwerfrfrrwwr", "efe4f43f3rfwerfrfrrwwr"]
#         t_start = time()
#
#         G=kernel.string_kernel(X_train,X_train)
#         print('Gram matrix computed in %.3f seconds' % (time() - t_start))
#         print G
