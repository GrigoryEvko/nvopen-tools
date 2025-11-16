// Function: sub_2B09EB0
// Address: 0x2b09eb0
//
__int64 __fastcall sub_2B09EB0(__int64 *a1, __int64 *a2, __int64 *a3, __int64 *a4, __int64 a5)
{
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rcx
  __int64 v11; // r10
  __int64 result; // rax
  __int64 v13; // r9
  __int64 v14; // rsi
  __int64 v15; // rcx
  __int64 v16; // rdi

  if ( a4 != a3 && a2 != a1 )
  {
    do
    {
      if ( *((_DWORD *)a3 + 2) < *((_DWORD *)a1 + 2) )
      {
        v6 = *a3;
        a5 += 16;
        a3 += 2;
        *(_QWORD *)(a5 - 16) = v6;
        *(_DWORD *)(a5 - 8) = *((_DWORD *)a3 - 2);
        if ( a2 == a1 )
          break;
      }
      else
      {
        v7 = *a1;
        a1 += 2;
        a5 += 16;
        *(_QWORD *)(a5 - 16) = v7;
        *(_DWORD *)(a5 - 8) = *((_DWORD *)a1 - 2);
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
    result = a5;
  }
  else
  {
    v10 = a5;
    do
    {
      v11 = *a1;
      v10 += 16;
      a1 += 2;
      *(_QWORD *)(v10 - 16) = v11;
      *(_DWORD *)(v10 - 8) = *((_DWORD *)a1 - 2);
      --v9;
    }
    while ( v9 );
    result = a5 + v8;
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
      a3 += 2;
      *(_QWORD *)(v15 - 16) = v16;
      *(_DWORD *)(v15 - 8) = *((_DWORD *)a3 - 2);
      --v14;
    }
    while ( v14 );
    result += v13;
  }
  return result;
}
