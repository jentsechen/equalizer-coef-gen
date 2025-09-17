from mdwrite import md2pdf

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str, help='source file')
    parser.add_argument('-output', type=str,default="NONE",help='output file')
    args = parser.parse_args()

    if args.output =='NONE':
        args.output=args.source.replace('.md','.pdf')
    md2pdf(args.source,args.output)