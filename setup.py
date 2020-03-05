from distutils.core import setup
setup(
  name = 'polyproto',
  packages = ['polyproto'],
  version = '0.1',
  license='MIT',
  description = 'generator using polygons for testing a Keras CNN',
  author = 'kevin',
  author_email = 'derkleinehof@gmail.com',
  url = 'https://github.com/kevinkit/polyproto',   
  download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',    # I explain this later on
  keywords = ['testing', 'keras', 'tensorflow','CNN'],
  install_requires=[            # I get to this in a second
          'keras',
          'opencv-python',
          'tensorflow',
          'numpy'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',      
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      
  ],
)