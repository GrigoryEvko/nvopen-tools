// Function: sub_3510150
// Address: 0x3510150
//
char *__fastcall sub_3510150(int *a1, int *a2, char *a3, char *a4, int *a5)
{
  int v6; // eax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rsi
  int *v10; // rcx
  int v11; // r10d
  char *result; // rax
  __int64 v13; // r9
  __int64 v14; // rsi
  char *v15; // rcx
  int v16; // edi

  if ( a4 != a3 && a2 != a1 )
  {
    do
    {
      if ( (unsigned int)*a1 < *(_DWORD *)a3 )
      {
        v6 = *(_DWORD *)a3;
        a5 += 4;
        a3 += 16;
        *(a5 - 4) = v6;
        *((_QWORD *)a5 - 1) = *((_QWORD *)a3 - 1);
        if ( a2 == a1 )
          break;
      }
      else
      {
        *a5 = *a1;
        v7 = *((_QWORD *)a1 + 1);
        a1 += 4;
        a5 += 4;
        *((_QWORD *)a5 - 1) = v7;
        if ( a2 == a1 )
          break;
      }
    }
    while ( a4 != a3 );
  }
  v8 = (char *)a2 - (char *)a1;
  v9 = ((char *)a2 - (char *)a1) >> 4;
  if ( v8 <= 0 )
  {
    result = (char *)a5;
  }
  else
  {
    v10 = a5;
    do
    {
      v11 = *a1;
      v10 += 4;
      a1 += 4;
      *(v10 - 4) = v11;
      *((_QWORD *)v10 - 1) = *((_QWORD *)a1 - 1);
      --v9;
    }
    while ( v9 );
    result = (char *)a5 + v8;
  }
  v13 = a4 - a3;
  v14 = v13 >> 4;
  if ( v13 > 0 )
  {
    v15 = result;
    do
    {
      v16 = *(_DWORD *)a3;
      v15 += 16;
      a3 += 16;
      *((_DWORD *)v15 - 4) = v16;
      *((_QWORD *)v15 - 1) = *((_QWORD *)a3 - 1);
      --v14;
    }
    while ( v14 );
    result += v13;
  }
  return result;
}
