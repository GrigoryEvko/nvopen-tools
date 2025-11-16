// Function: sub_30F36B0
// Address: 0x30f36b0
//
__int64 __fastcall sub_30F36B0(__int64 *a1, __int64 *a2, __int64 *a3, __int64 *a4, __int64 a5)
{
  int v6; // eax
  bool v7; // cc
  __int64 v8; // rax
  int v9; // eax
  __int64 v10; // rax
  unsigned __int64 v11; // rsi
  __int64 v12; // rcx
  __int64 v13; // r10
  __int64 result; // rax
  __int64 v15; // r9
  unsigned __int64 v16; // rsi
  __int64 v17; // rcx
  __int64 v18; // rdi
  __int64 v19; // rax

  if ( a4 != a3 && a2 != a1 )
  {
    do
    {
      v6 = *((_DWORD *)a3 + 4);
      v7 = *((_DWORD *)a1 + 4) < v6;
      if ( *((_DWORD *)a1 + 4) == v6 )
        v7 = a1[1] < a3[1];
      if ( v7 )
      {
        v8 = *a3;
        a3 += 3;
        *(_QWORD *)a5 = v8;
        *(_QWORD *)(a5 + 8) = *(a3 - 2);
        v9 = *((_DWORD *)a3 - 2);
      }
      else
      {
        v19 = *a1;
        a1 += 3;
        *(_QWORD *)a5 = v19;
        *(_QWORD *)(a5 + 8) = *(a1 - 2);
        v9 = *((_DWORD *)a1 - 2);
      }
      *(_DWORD *)(a5 + 16) = v9;
      a5 += 24;
    }
    while ( a2 != a1 && a4 != a3 );
  }
  v10 = (char *)a2 - (char *)a1;
  v11 = 0xAAAAAAAAAAAAAAABLL * (a2 - a1);
  if ( v10 <= 0 )
  {
    result = a5;
  }
  else
  {
    v12 = a5;
    do
    {
      v13 = *a1;
      v12 += 24;
      a1 += 3;
      *(_QWORD *)(v12 - 24) = v13;
      *(_QWORD *)(v12 - 16) = *(a1 - 2);
      *(_DWORD *)(v12 - 8) = *((_DWORD *)a1 - 2);
      --v11;
    }
    while ( v11 );
    result = a5 + v10;
  }
  v15 = (char *)a4 - (char *)a3;
  v16 = 0xAAAAAAAAAAAAAAABLL * (v15 >> 3);
  if ( v15 > 0 )
  {
    v17 = result;
    do
    {
      v18 = *a3;
      v17 += 24;
      a3 += 3;
      *(_QWORD *)(v17 - 24) = v18;
      *(_QWORD *)(v17 - 16) = *(a3 - 2);
      *(_DWORD *)(v17 - 8) = *((_DWORD *)a3 - 2);
      --v16;
    }
    while ( v16 );
    result += v15;
  }
  return result;
}
