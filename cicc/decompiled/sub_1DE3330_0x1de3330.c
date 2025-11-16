// Function: sub_1DE3330
// Address: 0x1de3330
//
__int64 __fastcall sub_1DE3330(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  int v6; // eax
  unsigned int v7; // ecx
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rcx
  int v12; // r10d
  __int64 result; // rax
  __int64 v14; // r9
  __int64 v15; // rsi
  __int64 v16; // rcx
  int v17; // edi

  if ( a4 != a3 && a2 != a1 )
  {
    do
    {
      v7 = *((_DWORD *)a1 + 2);
      if ( v7 < *(_DWORD *)(a3 + 8) )
      {
        v6 = *(_DWORD *)(a3 + 8);
        a5 += 16;
        a3 += 16;
        *(_DWORD *)(a5 - 8) = v6;
        *(_QWORD *)(a5 - 16) = *(_QWORD *)(a3 - 16);
        if ( a2 == a1 )
          break;
      }
      else
      {
        *(_DWORD *)(a5 + 8) = v7;
        v8 = *a1;
        a1 += 2;
        a5 += 16;
        *(_QWORD *)(a5 - 16) = v8;
        if ( a2 == a1 )
          break;
      }
    }
    while ( a4 != a3 );
  }
  v9 = (char *)a2 - (char *)a1;
  v10 = ((char *)a2 - (char *)a1) >> 4;
  if ( v9 <= 0 )
  {
    result = a5;
  }
  else
  {
    v11 = a5;
    do
    {
      v12 = *((_DWORD *)a1 + 2);
      v11 += 16;
      a1 += 2;
      *(_DWORD *)(v11 - 8) = v12;
      *(_QWORD *)(v11 - 16) = *(a1 - 2);
      --v10;
    }
    while ( v10 );
    result = a5 + v9;
  }
  v14 = a4 - a3;
  v15 = v14 >> 4;
  if ( v14 > 0 )
  {
    v16 = result;
    do
    {
      v17 = *(_DWORD *)(a3 + 8);
      v16 += 16;
      a3 += 16;
      *(_DWORD *)(v16 - 8) = v17;
      *(_QWORD *)(v16 - 16) = *(_QWORD *)(a3 - 16);
      --v15;
    }
    while ( v15 );
    result += v14;
  }
  return result;
}
