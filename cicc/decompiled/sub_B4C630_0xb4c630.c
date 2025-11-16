// Function: sub_B4C630
// Address: 0xb4c630
//
__int64 __fastcall sub_B4C630(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rax
  __int64 v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rax

  v8 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  v10 = (8 * a4) >> 3;
  if ( 8 * a4 > 0 )
  {
    do
    {
      v11 = *a3;
      if ( *(_QWORD *)v8 )
      {
        v12 = *(_QWORD *)(v8 + 8);
        **(_QWORD **)(v8 + 16) = v12;
        if ( v12 )
          *(_QWORD *)(v12 + 16) = *(_QWORD *)(v8 + 16);
      }
      *(_QWORD *)v8 = v11;
      if ( v11 )
      {
        v13 = *(_QWORD *)(v11 + 16);
        *(_QWORD *)(v8 + 8) = v13;
        if ( v13 )
          *(_QWORD *)(v13 + 16) = v8 + 8;
        *(_QWORD *)(v8 + 16) = v11 + 16;
        *(_QWORD *)(v11 + 16) = v8;
      }
      ++a3;
      v8 += 32;
      --v10;
    }
    while ( v10 );
  }
  if ( *(_QWORD *)(a1 - 32) )
  {
    v14 = *(_QWORD *)(a1 - 24);
    **(_QWORD **)(a1 - 16) = v14;
    if ( v14 )
      *(_QWORD *)(v14 + 16) = *(_QWORD *)(a1 - 16);
  }
  *(_QWORD *)(a1 - 32) = a2;
  if ( a2 )
  {
    v15 = *(_QWORD *)(a2 + 16);
    *(_QWORD *)(a1 - 24) = v15;
    if ( v15 )
      *(_QWORD *)(v15 + 16) = a1 - 24;
    *(_QWORD *)(a1 - 16) = a2 + 16;
    *(_QWORD *)(a2 + 16) = a1 - 32;
  }
  return sub_BD6B50(a1, a5);
}
