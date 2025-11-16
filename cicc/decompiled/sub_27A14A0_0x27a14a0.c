// Function: sub_27A14A0
// Address: 0x27a14a0
//
char *__fastcall sub_27A14A0(int *a1, int *a2, int *a3, int *a4, _DWORD *a5)
{
  __int64 v6; // rax
  int v7; // ecx
  int v8; // eax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rsi
  _DWORD *v12; // rcx
  int v13; // r10d
  char *result; // rax
  __int64 v15; // r9
  __int64 v16; // rsi
  char *v17; // rcx
  int v18; // edi

  if ( a2 != a1 )
  {
    while ( a4 != a3 )
    {
      v7 = *a3;
      v8 = *a1;
      if ( *a3 < (unsigned int)*a1 || v7 == v8 && *((_QWORD *)a3 + 1) < *((_QWORD *)a1 + 1) )
      {
        *a5 = v7;
        v9 = *((_QWORD *)a3 + 1);
        a5 += 8;
        a3 += 8;
        *((_QWORD *)a5 - 3) = v9;
        *((_QWORD *)a5 - 2) = *((_QWORD *)a3 - 2);
        *((_QWORD *)a5 - 1) = *((_QWORD *)a3 - 1);
        if ( a2 == a1 )
          break;
      }
      else
      {
        *a5 = v8;
        v6 = *((_QWORD *)a1 + 1);
        a1 += 8;
        a5 += 8;
        *((_QWORD *)a5 - 3) = v6;
        *((_QWORD *)a5 - 2) = *((_QWORD *)a1 - 2);
        *((_QWORD *)a5 - 1) = *((_QWORD *)a1 - 1);
        if ( a2 == a1 )
          break;
      }
    }
  }
  v10 = (char *)a2 - (char *)a1;
  v11 = ((char *)a2 - (char *)a1) >> 5;
  if ( v10 <= 0 )
  {
    result = (char *)a5;
  }
  else
  {
    v12 = a5;
    do
    {
      v13 = *a1;
      v12 += 8;
      a1 += 8;
      *(v12 - 8) = v13;
      *((_QWORD *)v12 - 3) = *((_QWORD *)a1 - 3);
      *((_QWORD *)v12 - 2) = *((_QWORD *)a1 - 2);
      *((_QWORD *)v12 - 1) = *((_QWORD *)a1 - 1);
      --v11;
    }
    while ( v11 );
    result = (char *)a5 + v10;
  }
  v15 = (char *)a4 - (char *)a3;
  v16 = v15 >> 5;
  if ( v15 > 0 )
  {
    v17 = result;
    do
    {
      v18 = *a3;
      v17 += 32;
      a3 += 8;
      *((_DWORD *)v17 - 8) = v18;
      *((_QWORD *)v17 - 3) = *((_QWORD *)a3 - 3);
      *((_QWORD *)v17 - 2) = *((_QWORD *)a3 - 2);
      *((_QWORD *)v17 - 1) = *((_QWORD *)a3 - 1);
      --v16;
    }
    while ( v16 );
    result += v15;
  }
  return result;
}
