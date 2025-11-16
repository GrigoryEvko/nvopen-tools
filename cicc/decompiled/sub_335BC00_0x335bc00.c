// Function: sub_335BC00
// Address: 0x335bc00
//
char *__fastcall sub_335BC00(int *a1, int *a2, int *a3, int *a4, int *a5)
{
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // r9
  __int64 v9; // rsi
  int *v10; // rcx
  int v11; // eax
  char *result; // rax
  __int64 v13; // r10
  __int64 v14; // rsi
  char *v15; // rcx
  int v16; // edi

  if ( a2 != a1 )
  {
    while ( a4 != a3 )
    {
      if ( *a3 < (unsigned int)*a1 )
      {
        *a5 = *a3;
        v6 = *((_QWORD *)a3 + 1);
        a5 += 4;
        a3 += 4;
        *((_QWORD *)a5 - 1) = v6;
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
  v13 = (char *)a4 - (char *)a3;
  v14 = v13 >> 4;
  if ( v13 > 0 )
  {
    v15 = result;
    do
    {
      v16 = *a3;
      v15 += 16;
      a3 += 4;
      *((_DWORD *)v15 - 4) = v16;
      *((_QWORD *)v15 - 1) = *((_QWORD *)a3 - 1);
      --v14;
    }
    while ( v14 );
    result += v13;
  }
  return result;
}
