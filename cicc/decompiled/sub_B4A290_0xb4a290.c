// Function: sub_B4A290
// Address: 0xb4a290
//
__int64 __fastcall sub_B4A290(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  int v13; // edx
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // r10
  __int64 v18; // r10
  __int64 v19; // rax
  __int64 v20; // rax

  v13 = *(_DWORD *)(a1 + 4);
  *(_QWORD *)(a1 + 80) = a2;
  v14 = a1 - 32LL * (v13 & 0x7FFFFFF);
  v15 = (8 * a5) >> 3;
  if ( 8 * a5 > 0 )
  {
    do
    {
      v16 = *a4;
      if ( *(_QWORD *)v14 )
      {
        v17 = *(_QWORD *)(v14 + 8);
        **(_QWORD **)(v14 + 16) = v17;
        if ( v17 )
          *(_QWORD *)(v17 + 16) = *(_QWORD *)(v14 + 16);
      }
      *(_QWORD *)v14 = v16;
      if ( v16 )
      {
        v18 = *(_QWORD *)(v16 + 16);
        *(_QWORD *)(v14 + 8) = v18;
        if ( v18 )
          *(_QWORD *)(v18 + 16) = v14 + 8;
        *(_QWORD *)(v14 + 16) = v16 + 16;
        *(_QWORD *)(v16 + 16) = v14;
      }
      ++a4;
      v14 += 32;
      --v15;
    }
    while ( v15 );
  }
  if ( *(_QWORD *)(a1 - 32) )
  {
    v19 = *(_QWORD *)(a1 - 24);
    **(_QWORD **)(a1 - 16) = v19;
    if ( v19 )
      *(_QWORD *)(v19 + 16) = *(_QWORD *)(a1 - 16);
  }
  *(_QWORD *)(a1 - 32) = a3;
  if ( a3 )
  {
    v20 = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(a1 - 24) = v20;
    if ( v20 )
      *(_QWORD *)(v20 + 16) = a1 - 24;
    *(_QWORD *)(a1 - 16) = a3 + 16;
    *(_QWORD *)(a3 + 16) = a1 - 32;
  }
  sub_B49680(a1, a7, a8, a5);
  return sub_BD6B50(a1, a6);
}
