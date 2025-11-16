// Function: sub_1C95630
// Address: 0x1c95630
//
__int64 __fastcall sub_1C95630(
        __int64 a1,
        __int64 a2,
        unsigned __int8 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 *v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 *v16; // rdx
  _QWORD *v17; // r13
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 *v21; // rdx
  __int64 v22; // r14
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 *v26; // rdx
  __int64 v27; // r9
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rax
  double v31; // xmm4_8
  double v32; // xmm5_8
  __int64 v34; // [rsp+8h] [rbp-38h]

  v12 = *(__int64 **)(a1 + 8);
  v13 = *v12;
  v14 = v12[1];
  if ( v13 == v14 )
LABEL_28:
    BUG();
  while ( *(_UNKNOWN **)v13 != &unk_4F9E06C )
  {
    v13 += 16;
    if ( v14 == v13 )
      goto LABEL_28;
  }
  v15 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v13 + 8) + 104LL))(*(_QWORD *)(v13 + 8), &unk_4F9E06C);
  v16 = *(__int64 **)(a1 + 8);
  v17 = (_QWORD *)(v15 + 160);
  v18 = *v16;
  v19 = v16[1];
  if ( v18 == v19 )
LABEL_25:
    BUG();
  while ( *(_UNKNOWN **)v18 != &unk_4F96DB4 )
  {
    v18 += 16;
    if ( v19 == v18 )
      goto LABEL_25;
  }
  v20 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v18 + 8) + 104LL))(*(_QWORD *)(v18 + 8), &unk_4F96DB4);
  v21 = *(__int64 **)(a1 + 8);
  v22 = *(_QWORD *)(v20 + 160);
  v23 = *v21;
  v24 = v21[1];
  if ( v23 == v24 )
LABEL_26:
    BUG();
  while ( *(_UNKNOWN **)v23 != &unk_4FB9E2C )
  {
    v23 += 16;
    if ( v24 == v23 )
      goto LABEL_26;
  }
  v25 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v23 + 8) + 104LL))(*(_QWORD *)(v23 + 8), &unk_4FB9E2C);
  v26 = *(__int64 **)(a1 + 8);
  v27 = v25 + 156;
  v28 = *v26;
  v29 = v26[1];
  if ( v28 == v29 )
LABEL_27:
    BUG();
  while ( *(_UNKNOWN **)v28 != &unk_505440C )
  {
    v28 += 16;
    if ( v29 == v28 )
      goto LABEL_27;
  }
  v34 = v27;
  v30 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v28 + 8) + 104LL))(*(_QWORD *)(v28 + 8), &unk_505440C);
  return sub_1C95350(
           (__int64 *)(a1 + 160),
           a2,
           a3,
           v17,
           v22,
           v34,
           a4,
           a5,
           a6,
           a7,
           v31,
           v32,
           a10,
           a11,
           *(_QWORD *)(v30 + 160));
}
