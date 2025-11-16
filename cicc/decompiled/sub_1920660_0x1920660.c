// Function: sub_1920660
// Address: 0x1920660
//
char *__fastcall sub_1920660(char *a1, char *a2, char *a3, char *a4, _DWORD *a5)
{
  int v6; // eax
  int v7; // ecx
  int v8; // eax
  int v9; // eax
  __int64 v10; // rax
  unsigned __int64 v11; // rsi
  _DWORD *v12; // rcx
  int v13; // r10d
  char *result; // rax
  __int64 v15; // r9
  unsigned __int64 v16; // rsi
  char *v17; // rcx
  int v18; // edi

  if ( a2 != a1 )
  {
    while ( a4 != a3 )
    {
      v7 = *(_DWORD *)a3;
      v8 = *(_DWORD *)a1;
      if ( *(_DWORD *)a3 < *(_DWORD *)a1 || v7 == v8 && *((_DWORD *)a3 + 1) < *((_DWORD *)a1 + 1) )
      {
        *a5 = v7;
        v9 = *((_DWORD *)a3 + 1);
        a5 += 6;
        a3 += 24;
        *(a5 - 5) = v9;
        *((_QWORD *)a5 - 2) = *((_QWORD *)a3 - 2);
        *((_QWORD *)a5 - 1) = *((_QWORD *)a3 - 1);
        if ( a2 == a1 )
          break;
      }
      else
      {
        *a5 = v8;
        v6 = *((_DWORD *)a1 + 1);
        a1 += 24;
        a5 += 6;
        *(a5 - 5) = v6;
        *((_QWORD *)a5 - 2) = *((_QWORD *)a1 - 2);
        *((_QWORD *)a5 - 1) = *((_QWORD *)a1 - 1);
        if ( a2 == a1 )
          break;
      }
    }
  }
  v10 = a2 - a1;
  v11 = 0xAAAAAAAAAAAAAAABLL * ((a2 - a1) >> 3);
  if ( v10 <= 0 )
  {
    result = (char *)a5;
  }
  else
  {
    v12 = a5;
    do
    {
      v13 = *(_DWORD *)a1;
      v12 += 6;
      a1 += 24;
      *(v12 - 6) = v13;
      *(v12 - 5) = *((_DWORD *)a1 - 5);
      *((_QWORD *)v12 - 2) = *((_QWORD *)a1 - 2);
      *((_QWORD *)v12 - 1) = *((_QWORD *)a1 - 1);
      --v11;
    }
    while ( v11 );
    result = (char *)a5 + v10;
  }
  v15 = a4 - a3;
  v16 = 0xAAAAAAAAAAAAAAABLL * (v15 >> 3);
  if ( v15 > 0 )
  {
    v17 = result;
    do
    {
      v18 = *(_DWORD *)a3;
      v17 += 24;
      a3 += 24;
      *((_DWORD *)v17 - 6) = v18;
      *((_DWORD *)v17 - 5) = *((_DWORD *)a3 - 5);
      *((_QWORD *)v17 - 2) = *((_QWORD *)a3 - 2);
      *((_QWORD *)v17 - 1) = *((_QWORD *)a3 - 1);
      --v16;
    }
    while ( v16 );
    result += v15;
  }
  return result;
}
