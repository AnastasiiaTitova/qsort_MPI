#include <iostream>
#include <fstream>
#include <string>
#include "mpi.h"

using namespace std;

void generate(int size, string name)
{
    ofstream out;
    out.open(name);
    out << size << endl;
    for (auto i = 0; i < size; i++)
        out << rand() % 10000 << " ";
    out.close();
    return;
}

int comparator(const void* p1, const void* p2) 
{
    const int* x = (int*)p1;
    const int* y = (int*)p2;

    if (*x > *y) 	  
        return 1;   
    if (*x < *y) 
        return -1;    
    return 0;
}

int getLen(const string name)
{
    int len;
    ifstream in;

    in.open(name);
    if (!in.is_open())
        throw "Can't open file " + name;
    in >> len;
    in.close();

    return len;
}

void readFile(const string name, int* array)
{
    int len;
    ifstream in;
    in.open(name);
    if (!in.is_open())
        throw "Can't open file " + name;

    in >> len;

    cout << "Reading array" << endl;
    for (auto i = 0; i < len; i++)
        in >> array[i];
    cout << "Array is done" << endl;

    in.close();
}

void MPIQsort(const int size, const int rank, const int len, int* array, int* lengths)
{
    auto iterations = log(size) / log(2);
    int* subArray = new int[len];
    int* buffer = new int[len];
    int* offsets = new int[size] {0}; 

    for (auto iteration = 0; iteration < iterations; iteration++)
    {
        // count offsets for Scatterv
        for (auto i = 1; i < size; i++)
            offsets[i] = offsets[i - 1] + lengths[i - 1];

        int mySize;
        MPI_Scatter(lengths, 1, MPI_INT, &mySize, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(array, lengths, offsets, MPI_INT, subArray, mySize, MPI_INT, 0, MPI_COMM_WORLD);

        // create new communicator
        int color = rank / pow(2, iterations - iteration);
        MPI_Comm MPI_LOCAL_COMMUNICATOR;
        MPI_Comm_split(MPI_COMM_WORLD, color, rank, &MPI_LOCAL_COMMUNICATOR);

        // on 1st iteration num * localSize/[localRanks] are 1 * 8/[0-7]; 2nd - 2 * 4/[0-3]; 3rd - 4 * 2/[0-1]
        int localRank, localSize;
        MPI_Comm_rank(MPI_LOCAL_COMMUNICATOR, &localRank);
        MPI_Comm_size(MPI_LOCAL_COMMUNICATOR, &localSize);

        // let's choose pivot for each communicator and broadcast it
        auto pivot = 0;
        if (localRank == 0 && mySize != 0)
            pivot = subArray[0 + rand() % lengths[rank]];
        MPI_Bcast(&pivot, 1, MPI_INT, 0, MPI_LOCAL_COMMUNICATOR);

        // now do in-place partition
        auto lessSize = 0;
        auto greaterSize = 0;
        auto lessBorder = 0;
        auto greaterBorder = lengths[rank] - 1;
        while (lessBorder <= greaterBorder)
        {
            if (subArray[lessBorder] <= pivot)
            {
                lessBorder++;
                lessSize++;
            }
            else if (subArray[greaterBorder] > pivot)
            {
                greaterBorder--;
                greaterSize++;
            }
            else
            {
                auto tmp = subArray[lessBorder];
                subArray[lessBorder] = subArray[greaterBorder];
                subArray[greaterBorder] = tmp;
            }
        }

        // now check if localRank is from "upper" or "lower" group and send/receive
        // from "lower" - send greater to partner's buffer
        // from "upper" - send less to partner's buffer
        auto rankFromLowerGroup = localRank < localSize / 2;
        auto sendTo = 0;
        auto recFrom = 0;
        auto bufferSize = 0;
        if (rankFromLowerGroup)
        {
            sendTo = recFrom = localRank + localSize / 2;
            MPI_Send(&greaterSize, 1, MPI_INT, sendTo, 0, MPI_LOCAL_COMMUNICATOR);
            MPI_Recv(&bufferSize, 1, MPI_INT, recFrom, 0, MPI_LOCAL_COMMUNICATOR, MPI_STATUS_IGNORE);
            MPI_Send(subArray + lessSize, greaterSize, MPI_INT, sendTo, 0, MPI_LOCAL_COMMUNICATOR);
            MPI_Recv(buffer, bufferSize, MPI_INT, recFrom, 0, MPI_LOCAL_COMMUNICATOR, MPI_STATUS_IGNORE);
        }
        else
        {
            sendTo = recFrom = localRank - localSize / 2;
            MPI_Recv(&bufferSize, 1, MPI_INT, recFrom, 0, MPI_LOCAL_COMMUNICATOR, MPI_STATUS_IGNORE);
            MPI_Send(&lessSize, 1, MPI_INT, sendTo, 0, MPI_LOCAL_COMMUNICATOR);
            MPI_Recv(buffer, bufferSize, MPI_INT, recFrom, 0, MPI_LOCAL_COMMUNICATOR, MPI_STATUS_IGNORE);
            MPI_Send(subArray, lessSize, MPI_INT, sendTo, 0, MPI_LOCAL_COMMUNICATOR);
        }

        // now merge arrays
        if (rankFromLowerGroup)
        {
            memcpy(buffer + bufferSize, subArray, lessSize * sizeof(int));
            bufferSize += lessSize;
        }
        else
        {
            memcpy(buffer + bufferSize, subArray + lessSize, greaterSize * sizeof(int));
            bufferSize += greaterSize;
        }

        // if last iteration - do sequential qsort
        if (iteration == iterations - 1)
            qsort(buffer, bufferSize, sizeof(int), comparator);

        // update lengths
        MPI_Allgather(&bufferSize, 1, MPI_INT, lengths, 1, MPI_INT, MPI_COMM_WORLD);

        // recalculate offsets for Gatherv
        for (auto i = 1; i < size; i++)
            offsets[i] = offsets[i - 1] + lengths[i - 1];
        MPI_Gatherv(buffer, bufferSize, MPI_INT, array, lengths, offsets, MPI_INT, 0, MPI_COMM_WORLD);

        MPI_Comm_free(&MPI_LOCAL_COMMUNICATOR);
    }

    delete[] subArray;
    delete[] buffer;
    delete[] offsets;
}

void printResult(const string name, const int len, const int* array)
{
    ofstream out;
    out.open(name);

    out << len << endl;
    for (auto i = 0; i < len; i++)
        out << array[i] << " ";
    out.close();
}


int main(int argc, char* argv[])
{
	try
    {
        MPI_Init(&argc, &argv);
        int size, rank;

        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (rank == 0 && size % 2 && size != 1)
        {
            cerr << "Required size equals power of 2";
            MPI_Abort(MPI_COMM_WORLD, 0);
            return 0;
        }

        int* array = {};
        long len;
        string name;

        if (rank == 0)
        {
            cout << "Enter input file name" << endl;
            cin >> name;

            //generate(10000000, name);
            len = getLen(name);
            array = new int[len];
            readFile(name, array);

            cout << "Enter output file name" << endl;
            cin >> name;
        }

        MPI_Bcast(&len, 1, MPI_LONG, 0, MPI_COMM_WORLD);
 
        // create lengths array outa qsort to calculate clear MPI time
        int* lengths = new int[size];
        for (auto i = 0; i < size; i++)
            lengths[i] = len / size;
        for (auto i = 0; i < len % size; i++)
            lengths[i]++;

        // qsort
        const auto start = MPI_Wtime();
        if (size == 1)
            qsort(array, len, sizeof(int), comparator);
        else
            MPIQsort(size, rank, len, array, lengths);
        const auto finish = MPI_Wtime();

        if (rank == 0)
        {
            cout << "Number of processes: " << size << " Size: " << len << " Time: " << finish - start << endl;
            printResult(name, len, array);
        }

        delete[] array;
        delete[] lengths;

        MPI_Finalize();
    }
    catch (exception e)
    {
        cout << e.what();
    }

    return 0;
}