// Function: sub_B4D9A0
// Address: 0xb4d9a0
//
__int64 __fastcall sub_B4D9A0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rcx
  __int64 v10; // rdi
  __int64 v11; // rsi
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdi
  __int64 v16; // rdi

  v7 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  if ( *(_QWORD *)v7 )
  {
    v8 = *(_QWORD *)(v7 + 8);
    **(_QWORD **)(v7 + 16) = v8;
    if ( v8 )
      *(_QWORD *)(v8 + 16) = *(_QWORD *)(v7 + 16);
  }
  *(_QWORD *)v7 = a2;
  if ( a2 )
  {
    v9 = *(_QWORD *)(a2 + 16);
    *(_QWORD *)(v7 + 8) = v9;
    if ( v9 )
      *(_QWORD *)(v9 + 16) = v7 + 8;
    *(_QWORD *)(v7 + 16) = a2 + 16;
    *(_QWORD *)(a2 + 16) = v7;
  }
  v10 = 8 * a4;
  v11 = v10 >> 3;
  v13 = a1 + 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  if ( v10 > 0 )
  {
    do
    {
      v14 = *a3;
      if ( *(_QWORD *)v13 )
      {
        v15 = *(_QWORD *)(v13 + 8);
        **(_QWORD **)(v13 + 16) = v15;
        if ( v15 )
          *(_QWORD *)(v15 + 16) = *(_QWORD *)(v13 + 16);
      }
      *(_QWORD *)v13 = v14;
      if ( v14 )
      {
        v16 = *(_QWORD *)(v14 + 16);
        *(_QWORD *)(v13 + 8) = v16;
        if ( v16 )
          *(_QWORD *)(v16 + 16) = v13 + 8;
        *(_QWORD *)(v13 + 16) = v14 + 16;
        *(_QWORD *)(v14 + 16) = v13;
      }
      ++a3;
      v13 += 32;
      --v11;
    }
    while ( v11 );
  }
  return sub_BD6B50(a1, a5);
}
