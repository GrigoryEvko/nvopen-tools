// Function: sub_B4A9C0
// Address: 0xb4a9c0
//
__int64 __fastcall sub_B4A9C0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8,
        __int64 a9,
        __int64 a10)
{
  int v13; // edx
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // r9
  __int64 v20; // r9
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax

  v13 = *(_DWORD *)(a1 + 4);
  *(_QWORD *)(a1 + 80) = a2;
  v16 = a1 - 32LL * (v13 & 0x7FFFFFF);
  v17 = (8 * a8) >> 3;
  if ( 8 * a8 > 0 )
  {
    do
    {
      v18 = *a7;
      if ( *(_QWORD *)v16 )
      {
        v19 = *(_QWORD *)(v16 + 8);
        **(_QWORD **)(v16 + 16) = v19;
        if ( v19 )
          *(_QWORD *)(v19 + 16) = *(_QWORD *)(v16 + 16);
      }
      *(_QWORD *)v16 = v18;
      if ( v18 )
      {
        v20 = *(_QWORD *)(v18 + 16);
        *(_QWORD *)(v16 + 8) = v20;
        if ( v20 )
          *(_QWORD *)(v20 + 16) = v16 + 8;
        *(_QWORD *)(v16 + 16) = v18 + 16;
        *(_QWORD *)(v18 + 16) = v16;
      }
      ++a7;
      v16 += 32;
      --v17;
    }
    while ( v17 );
  }
  if ( *(_QWORD *)(a1 - 96) )
  {
    v21 = *(_QWORD *)(a1 - 88);
    **(_QWORD **)(a1 - 80) = v21;
    if ( v21 )
      *(_QWORD *)(v21 + 16) = *(_QWORD *)(a1 - 80);
  }
  *(_QWORD *)(a1 - 96) = a4;
  if ( a4 )
  {
    v22 = *(_QWORD *)(a4 + 16);
    *(_QWORD *)(a1 - 88) = v22;
    if ( v22 )
      *(_QWORD *)(v22 + 16) = a1 - 88;
    *(_QWORD *)(a1 - 80) = a4 + 16;
    *(_QWORD *)(a4 + 16) = a1 - 96;
  }
  if ( *(_QWORD *)(a1 - 64) )
  {
    v23 = *(_QWORD *)(a1 - 56);
    **(_QWORD **)(a1 - 48) = v23;
    if ( v23 )
      *(_QWORD *)(v23 + 16) = *(_QWORD *)(a1 - 48);
  }
  *(_QWORD *)(a1 - 64) = a5;
  if ( a5 )
  {
    v24 = *(_QWORD *)(a5 + 16);
    *(_QWORD *)(a1 - 56) = v24;
    if ( v24 )
      *(_QWORD *)(v24 + 16) = a1 - 56;
    *(_QWORD *)(a1 - 48) = a5 + 16;
    *(_QWORD *)(a5 + 16) = a1 - 64;
  }
  if ( *(_QWORD *)(a1 - 32) )
  {
    v25 = *(_QWORD *)(a1 - 24);
    **(_QWORD **)(a1 - 16) = v25;
    if ( v25 )
      *(_QWORD *)(v25 + 16) = *(_QWORD *)(a1 - 16);
  }
  *(_QWORD *)(a1 - 32) = a3;
  if ( a3 )
  {
    v26 = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(a1 - 24) = v26;
    if ( v26 )
      *(_QWORD *)(v26 + 16) = a1 - 24;
    *(_QWORD *)(a1 - 16) = a3 + 16;
    *(_QWORD *)(a3 + 16) = a1 - 32;
  }
  sub_B49680(a1, a9, a10, a8);
  return sub_BD6B50(a1, a6);
}
