// Function: sub_1A7E8A0
// Address: 0x1a7e8a0
//
__int64 __fastcall sub_1A7E8A0(
        __int64 a1,
        _QWORD *a2,
        __m128 a3,
        __m128i a4,
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
  __int64 *v16; // r13
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 *v20; // rdx
  _QWORD *v21; // r14
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rax
  double v26; // xmm4_8
  double v27; // xmm5_8

  if ( (unsigned __int8)sub_1636880(a1, (__int64)a2) )
    return 0;
  v11 = *(__int64 **)(a1 + 8);
  v12 = *v11;
  v13 = v11[1];
  if ( v12 == v13 )
    goto LABEL_16;
  while ( *(_UNKNOWN **)v12 != &unk_4F99CB0 )
  {
    v12 += 16;
    if ( v13 == v12 )
      goto LABEL_16;
  }
  v14 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v12 + 8) + 104LL))(*(_QWORD *)(v12 + 8), &unk_4F99CB0);
  v15 = *(__int64 **)(a1 + 8);
  v16 = *(__int64 **)(v14 + 160);
  v17 = *v15;
  v18 = v15[1];
  if ( v17 == v18 )
    goto LABEL_16;
  while ( *(_UNKNOWN **)v17 != &unk_4F96DB4 )
  {
    v17 += 16;
    if ( v18 == v17 )
      goto LABEL_16;
  }
  v19 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v17 + 8) + 104LL))(*(_QWORD *)(v17 + 8), &unk_4F96DB4);
  v20 = *(__int64 **)(a1 + 8);
  v21 = *(_QWORD **)(v19 + 160);
  v22 = *v20;
  v23 = v20[1];
  if ( v22 == v23 )
LABEL_16:
    BUG();
  while ( *(_UNKNOWN **)v22 != &unk_4F9D3C0 )
  {
    v22 += 16;
    if ( v23 == v22 )
      goto LABEL_16;
  }
  v24 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v22 + 8) + 104LL))(*(_QWORD *)(v22 + 8), &unk_4F9D3C0);
  v25 = sub_14A4050(v24, (__int64)a2);
  return sub_1A7E2B0(a2, a3, a4, a5, a6, v26, v27, a9, a10, v25, v21, v16);
}
