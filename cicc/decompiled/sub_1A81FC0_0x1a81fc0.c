// Function: sub_1A81FC0
// Address: 0x1a81fc0
//
__int64 __fastcall sub_1A81FC0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 *v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 *v15; // rdx
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 **v20; // r14
  __int64 v21; // rax
  double v22; // xmm4_8
  double v23; // xmm5_8
  __int64 v24; // rax
  __int64 v25; // rcx

  if ( (unsigned __int8)sub_1636880(a1, a2) )
    return 0;
  v11 = *(__int64 **)(a1 + 8);
  v12 = *v11;
  v13 = v11[1];
  if ( v12 == v13 )
    goto LABEL_16;
  while ( *(_UNKNOWN **)v12 != &unk_4F9B6E8 )
  {
    v12 += 16;
    if ( v13 == v12 )
      goto LABEL_16;
  }
  v14 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v12 + 8) + 104LL))(*(_QWORD *)(v12 + 8), &unk_4F9B6E8);
  v15 = *(__int64 **)(a1 + 8);
  v16 = v14 + 360;
  v17 = *v15;
  v18 = v15[1];
  if ( v17 == v18 )
LABEL_16:
    BUG();
  while ( *(_UNKNOWN **)v17 != &unk_4F9D3C0 )
  {
    v17 += 16;
    if ( v18 == v17 )
      goto LABEL_16;
  }
  v19 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v17 + 8) + 104LL))(*(_QWORD *)(v17 + 8), &unk_4F9D3C0);
  v20 = (__int64 **)sub_14A4050(v19, a2);
  v21 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4F9E06C, 1u);
  if ( v21 && (v24 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v21 + 104LL))(v21, &unk_4F9E06C)) != 0 )
    v25 = v24 + 160;
  else
    v25 = 0;
  return sub_1A819A0(a2, v16, v20, v25, a3, a4, a5, a6, v22, v23, a9, a10);
}
