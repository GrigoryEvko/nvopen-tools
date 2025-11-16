// Function: sub_B4B130
// Address: 0xb4b130
//
__int64 __fastcall sub_B4B130(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned int a6,
        __int64 *a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11)
{
  int v13; // edx
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rdx
  __int64 v18; // r10
  __int64 v19; // r10
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rsi
  __int64 v24; // r9
  unsigned int v25; // edi
  __int64 v26; // rsi
  __int64 v27; // rax
  __int64 v28; // r9
  __int64 v29; // r9
  __int64 v30; // rax
  __int64 v31; // rax

  v13 = *(_DWORD *)(a1 + 4);
  *(_QWORD *)(a1 + 80) = a2;
  v15 = a1 - 32LL * (v13 & 0x7FFFFFF);
  v16 = (8 * a8) >> 3;
  if ( 8 * a8 > 0 )
  {
    do
    {
      v17 = *a7;
      if ( *(_QWORD *)v15 )
      {
        v18 = *(_QWORD *)(v15 + 8);
        **(_QWORD **)(v15 + 16) = v18;
        if ( v18 )
          *(_QWORD *)(v18 + 16) = *(_QWORD *)(v15 + 16);
      }
      *(_QWORD *)v15 = v17;
      if ( v17 )
      {
        v19 = *(_QWORD *)(v17 + 16);
        *(_QWORD *)(v15 + 8) = v19;
        if ( v19 )
          *(_QWORD *)(v19 + 16) = v15 + 8;
        *(_QWORD *)(v15 + 16) = v17 + 16;
        *(_QWORD *)(v17 + 16) = v15;
      }
      ++a7;
      v15 += 32;
      --v16;
    }
    while ( v16 );
  }
  *(_DWORD *)(a1 + 88) = a6;
  v20 = a1 - 32;
  v21 = a1 - 32 - 32LL * a6 - 32;
  if ( *(_QWORD *)v21 )
  {
    v22 = *(_QWORD *)(v21 + 8);
    **(_QWORD **)(v21 + 16) = v22;
    if ( v22 )
      *(_QWORD *)(v22 + 16) = *(_QWORD *)(v21 + 16);
  }
  *(_QWORD *)v21 = a4;
  if ( a4 )
  {
    v23 = *(_QWORD *)(a4 + 16);
    *(_QWORD *)(v21 + 8) = v23;
    if ( v23 )
      *(_QWORD *)(v23 + 16) = v21 + 8;
    *(_QWORD *)(v21 + 16) = a4 + 16;
    *(_QWORD *)(a4 + 16) = v21;
  }
  v24 = *(unsigned int *)(a1 + 88);
  if ( (_DWORD)v24 )
  {
    v25 = 0;
    do
    {
      v26 = *(_QWORD *)(a5 + 8LL * v25);
      v27 = v20 + 32 * (v25 - v24);
      if ( *(_QWORD *)v27 )
      {
        v28 = *(_QWORD *)(v27 + 8);
        **(_QWORD **)(v27 + 16) = v28;
        if ( v28 )
          *(_QWORD *)(v28 + 16) = *(_QWORD *)(v27 + 16);
      }
      *(_QWORD *)v27 = v26;
      if ( v26 )
      {
        v29 = *(_QWORD *)(v26 + 16);
        *(_QWORD *)(v27 + 8) = v29;
        if ( v29 )
          *(_QWORD *)(v29 + 16) = v27 + 8;
        *(_QWORD *)(v27 + 16) = v26 + 16;
        *(_QWORD *)(v26 + 16) = v27;
      }
      v24 = *(unsigned int *)(a1 + 88);
      ++v25;
    }
    while ( (_DWORD)v24 != v25 );
  }
  if ( *(_QWORD *)(a1 - 32) )
  {
    v30 = *(_QWORD *)(a1 - 24);
    **(_QWORD **)(a1 - 16) = v30;
    if ( v30 )
      *(_QWORD *)(v30 + 16) = *(_QWORD *)(a1 - 16);
  }
  *(_QWORD *)(a1 - 32) = a3;
  if ( a3 )
  {
    v31 = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(a1 - 24) = v31;
    if ( v31 )
      *(_QWORD *)(v31 + 16) = a1 - 24;
    *(_QWORD *)(a1 - 16) = a3 + 16;
    *(_QWORD *)(a3 + 16) = v20;
  }
  sub_B49680(a1, a9, a10, a8);
  return sub_BD6B50(a1, a11);
}
