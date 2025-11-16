// Function: sub_2B11D20
// Address: 0x2b11d20
//
__int64 __fastcall sub_2B11D20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
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

  if ( a2 != a1 )
  {
    while ( a4 != a3 )
    {
      if ( *(_DWORD *)(a3 + 4) < *(_DWORD *)(a1 + 4) )
      {
        v6 = *(_QWORD *)(a3 + 8);
        a5 += 16;
        a3 += 16;
        *(_QWORD *)(a5 - 8) = v6;
        *(_DWORD *)(a5 - 12) = *(_DWORD *)(a3 - 12);
        *(_DWORD *)(a5 - 16) = *(_DWORD *)(a3 - 16);
        if ( a2 == a1 )
          break;
      }
      else
      {
        v7 = *(_QWORD *)(a1 + 8);
        a1 += 16;
        a5 += 16;
        *(_QWORD *)(a5 - 8) = v7;
        *(_DWORD *)(a5 - 12) = *(_DWORD *)(a1 - 12);
        *(_DWORD *)(a5 - 16) = *(_DWORD *)(a1 - 16);
        if ( a2 == a1 )
          break;
      }
    }
  }
  v8 = a2 - a1;
  v9 = (a2 - a1) >> 4;
  if ( v8 <= 0 )
  {
    result = a5;
  }
  else
  {
    v10 = a5;
    do
    {
      v11 = *(_QWORD *)(a1 + 8);
      v10 += 16;
      a1 += 16;
      *(_QWORD *)(v10 - 8) = v11;
      *(_DWORD *)(v10 - 12) = *(_DWORD *)(a1 - 12);
      *(_DWORD *)(v10 - 16) = *(_DWORD *)(a1 - 16);
      --v9;
    }
    while ( v9 );
    result = a5 + v8;
  }
  v13 = a4 - a3;
  v14 = v13 >> 4;
  if ( v13 > 0 )
  {
    v15 = result;
    do
    {
      v16 = *(_QWORD *)(a3 + 8);
      v15 += 16;
      a3 += 16;
      *(_QWORD *)(v15 - 8) = v16;
      *(_DWORD *)(v15 - 12) = *(_DWORD *)(a3 - 12);
      *(_DWORD *)(v15 - 16) = *(_DWORD *)(a3 - 16);
      --v14;
    }
    while ( v14 );
    result += v13;
  }
  return result;
}
