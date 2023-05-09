# to extract the useful data information from the origanl format of the whole data.
# and save such sata to csv format

"""
    statistics about the data (initial data)

        gene :  12340
        transcripts : 33154  this should be the total class 
        total reads : 3315400 ==> (33154 * 100)

        each of the reads from the same transcripts has the same lable because the label is the name of the transcripts
"""

# currently, I have got the whole data with 19813 genes and 57899 transcripts and the .csv file is 16 GB



import os
import csv
# path = "../data.fastq"
# root_path = "../data.fastq/gene/"
# path = "../raw_data_fastq"

# path = ""
root_path = "../source_data/data/gene"

save_path = "../Data_full/"  # 刚开始忘记了加这个斜杠



def fastq_data_process(path_to_fastq):
    """
    process the fastq format data and extract the "AGCT" sequence

    """

    label_str = path_to_fastq.split('/')[-1]

    # every 4 lines is an instance and
    with open(save_path+ "fastq.csv","a",newline="") as file:
        writer = csv.writer(file)
        with open(path_to_fastq + '/pass.fastq') as f:
            content  = f.readlines()
            pointer = 1
            while True:
                if pointer > 399:
                    break

                writer.writerow([content[pointer],label_str])
                pointer += 4

def main():

    cnt = 0
    total_file = 0
    print("Begine to process the souce data to .csv format.")

    for root, dirs, files in os.walk(root_path):
        # print((root, dirs,files))
        # print("debug:",len(dirs))
        if len(files) == 1: # fastq data exist
            path_to_fastq = root 
            fastq_data_process(path_to_fastq)
            total_file +=1  # to record the total number of fastq file

        # cnt += 1
        # if cnt == 100:
        #     break
    
    return total_file


if __name__ == "__main__":

    cnt = main()

    print("Data processing ended!")
    print(f"the total transcripts are : {cnt:>5}")


