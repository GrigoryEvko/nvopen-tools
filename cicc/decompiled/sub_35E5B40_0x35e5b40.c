// Function: sub_35E5B40
// Address: 0x35e5b40
//
__int64 __fastcall sub_35E5B40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  unsigned __int64 v9; // rsi
  __int64 v10; // rcx
  __int64 v11; // r10
  __int64 result; // rax
  __int64 v13; // r9
  unsigned __int64 v14; // rsi
  __int64 v15; // rcx
  __int64 v16; // rdi

  if ( a2 != a1 )
  {
    while ( a4 != a3 )
    {
      if ( *(_DWORD *)a3 < *(_DWORD *)a1 )
      {
        v6 = *(_QWORD *)(a3 + 16);
        a5 += 24;
        a3 += 24;
        *(_QWORD *)(a5 - 8) = v6;
        *(_DWORD *)(a5 - 16) = *(_DWORD *)(a3 - 16);
        *(_DWORD *)(a5 - 20) = *(_DWORD *)(a3 - 20);
        *(_DWORD *)(a5 - 24) = *(_DWORD *)(a3 - 24);
        if ( a2 == a1 )
          break;
      }
      else
      {
        v7 = *(_QWORD *)(a1 + 16);
        a1 += 24;
        a5 += 24;
        *(_QWORD *)(a5 - 8) = v7;
        *(_DWORD *)(a5 - 16) = *(_DWORD *)(a1 - 16);
        *(_DWORD *)(a5 - 20) = *(_DWORD *)(a1 - 20);
        *(_DWORD *)(a5 - 24) = *(_DWORD *)(a1 - 24);
        if ( a2 == a1 )
          break;
      }
    }
  }
  v8 = a2 - a1;
  v9 = 0xAAAAAAAAAAAAAAABLL * ((a2 - a1) >> 3);
  if ( v8 <= 0 )
  {
    result = a5;
  }
  else
  {
    v10 = a5;
    do
    {
      v11 = *(_QWORD *)(a1 + 16);
      v10 += 24;
      a1 += 24;
      *(_QWORD *)(v10 - 8) = v11;
      *(_DWORD *)(v10 - 16) = *(_DWORD *)(a1 - 16);
      *(_DWORD *)(v10 - 20) = *(_DWORD *)(a1 - 20);
      *(_DWORD *)(v10 - 24) = *(_DWORD *)(a1 - 24);
      --v9;
    }
    while ( v9 );
    result = a5 + v8;
  }
  v13 = a4 - a3;
  v14 = 0xAAAAAAAAAAAAAAABLL * (v13 >> 3);
  if ( v13 > 0 )
  {
    v15 = result;
    do
    {
      v16 = *(_QWORD *)(a3 + 16);
      v15 += 24;
      a3 += 24;
      *(_QWORD *)(v15 - 8) = v16;
      *(_DWORD *)(v15 - 16) = *(_DWORD *)(a3 - 16);
      *(_DWORD *)(v15 - 20) = *(_DWORD *)(a3 - 20);
      *(_DWORD *)(v15 - 24) = *(_DWORD *)(a3 - 24);
      --v14;
    }
    while ( v14 );
    result += v13;
  }
  return result;
}
