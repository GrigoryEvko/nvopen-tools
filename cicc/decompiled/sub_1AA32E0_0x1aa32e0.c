// Function: sub_1AA32E0
// Address: 0x1aa32e0
//
__int64 __fastcall sub_1AA32E0(
        unsigned __int64 a1,
        __int64 a2,
        __m128 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  void *v13; // rsi
  __int64 v14; // r15
  __int64 v15; // r9
  unsigned __int64 v16; // r14
  double v17; // xmm4_8
  double v18; // xmm5_8
  __int64 *v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  double v25; // xmm4_8
  double v26; // xmm5_8
  __int64 v28; // [rsp+0h] [rbp-50h]
  __int64 v29; // [rsp+8h] [rbp-48h]
  unsigned __int8 v31; // [rsp+1Fh] [rbp-31h]

  v10 = *(__int64 **)(a1 + 8);
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
LABEL_25:
    BUG();
  v13 = &unk_4F9B6E8;
  while ( *(_UNKNOWN **)v11 != &unk_4F9B6E8 )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_25;
  }
  v31 = 0;
  v28 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_4F9B6E8)
      + 360;
  v14 = *(_QWORD *)(a2 + 32);
  if ( v14 == a2 + 24 )
    return 0;
  do
  {
    v15 = v14 - 56;
    if ( !v14 )
      v15 = 0;
    v16 = v15;
    if ( !sub_15E4F60(v15)
      && v16 + 72 != (*(_QWORD *)(v16 + 72) & 0xFFFFFFFFFFFFFFF8LL)
      && (*(_BYTE *)(v16 + 19) & 0x40) != 0
      && (unsigned __int8)sub_1A94BA0(v16) )
    {
      v19 = *(__int64 **)(a1 + 8);
      v20 = *v19;
      v21 = v19[1];
      if ( v20 == v21 )
LABEL_24:
        BUG();
      while ( *(_UNKNOWN **)v20 != &unk_4F9D3C0 )
      {
        v20 += 16;
        if ( v21 == v20 )
          goto LABEL_24;
      }
      v22 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v20 + 8) + 104LL))(
              *(_QWORD *)(v20 + 8),
              &unk_4F9D3C0);
      v29 = sub_14A4050(v22, v16);
      v23 = sub_161ACC0(*(_QWORD *)(a1 + 8), a1, (__int64)&unk_4F9E06C, v16);
      v24 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v23 + 104LL))(v23, &unk_4F9E06C);
      v13 = (void *)v16;
      v31 |= sub_1AA2F10(a3, a4, a5, a6, v25, v26, a9, a10, a1 + 153, v16, v24 + 160, v29, v28);
    }
    v14 = *(_QWORD *)(v14 + 8);
  }
  while ( a2 + 24 != v14 );
  if ( !v31 )
    return 0;
  else
    sub_1A96B10(a2, (__int64)v13, a3, *(double *)a4.m128_u64, a5, a6, v17, v18, a9, a10);
  return v31;
}
