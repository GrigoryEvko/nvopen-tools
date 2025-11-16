// Function: sub_18476F0
// Address: 0x18476f0
//
__int64 __fastcall sub_18476F0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v11; // r14
  int v13; // ebx
  __int64 v14; // r13
  __int64 v15; // rsi
  __int64 v16; // r15
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // rcx
  _BYTE *v20; // r8
  int v21; // r9d
  double v22; // xmm4_8
  double v23; // xmm5_8
  __int64 v24; // r14
  __int64 v25; // rsi
  __int64 i; // r15
  __int64 v27; // rsi
  char v28; // al
  __int64 v30; // rsi
  __int64 v31; // rdx

  v11 = a3 + 24;
  v13 = 0;
  v14 = *(_QWORD *)(a3 + 32);
  if ( v14 == a3 + 24 )
  {
    v30 = a1 + 40;
    v31 = a1 + 96;
LABEL_19:
    *(_QWORD *)(a1 + 24) = 0x100000002LL;
    *(_QWORD *)(a1 + 8) = v30;
    *(_QWORD *)(a1 + 16) = v30;
    *(_QWORD *)(a1 + 56) = 0;
    *(_QWORD *)(a1 + 64) = v31;
    *(_QWORD *)(a1 + 72) = v31;
    *(_QWORD *)(a1 + 80) = 2;
    *(_DWORD *)(a1 + 88) = 0;
    *(_DWORD *)(a1 + 32) = 0;
    *(_QWORD *)(a1 + 40) = &unk_4F9EE48;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  do
  {
    while ( 1 )
    {
      v15 = v14;
      v14 = *(_QWORD *)(v14 + 8);
      if ( *(_DWORD *)(*(_QWORD *)(v15 - 32) + 8LL) >> 8 )
        break;
      if ( v11 == v14 )
        goto LABEL_6;
    }
    v13 |= sub_18457A0(a4, a5, a6, a7, a8, a9, a10, a11, a2, v15 - 56);
  }
  while ( v11 != v14 );
LABEL_6:
  v16 = *(_QWORD *)(a3 + 32);
  if ( v14 != v16 )
  {
    do
    {
      v17 = v16 - 56;
      if ( !v16 )
        v17 = 0;
      sub_1843DB0(a2, v17);
      v16 = *(_QWORD *)(v16 + 8);
    }
    while ( v14 != v16 );
    v24 = *(_QWORD *)(a3 + 32);
    if ( v14 != v24 )
    {
      do
      {
        v25 = v24;
        v24 = *(_QWORD *)(v24 + 8);
        v13 |= sub_18457F0(a2, (_BYTE *)(v25 - 56), a4, a5, a6, a7, v22, v23, a10, a11, v18, v19, v20, v21);
      }
      while ( v14 != v24 );
      for ( i = *(_QWORD *)(a3 + 32); v14 != i; LOBYTE(v13) = v28 | v13 )
      {
        v27 = i - 56;
        if ( !i )
          v27 = 0;
        v28 = sub_1842CA0(a2, v27);
        i = *(_QWORD *)(i + 8);
      }
    }
  }
  v30 = a1 + 40;
  v31 = a1 + 96;
  if ( !(_BYTE)v13 )
    goto LABEL_19;
  memset((void *)a1, 0, 0x70u);
  *(_QWORD *)(a1 + 8) = v30;
  *(_QWORD *)(a1 + 16) = v30;
  *(_DWORD *)(a1 + 24) = 2;
  *(_QWORD *)(a1 + 64) = v31;
  *(_QWORD *)(a1 + 72) = v31;
  *(_DWORD *)(a1 + 80) = 2;
  return a1;
}
