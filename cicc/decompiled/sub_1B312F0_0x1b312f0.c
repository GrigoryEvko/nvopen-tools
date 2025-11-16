// Function: sub_1B312F0
// Address: 0x1b312f0
//
__int64 __fastcall sub_1B312F0(
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
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 *v14; // rdx
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // r14
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r13
  __int64 v23; // rdi
  __int64 v24; // rdx
  __int64 v25; // rax
  double v26; // xmm4_8
  double v27; // xmm5_8
  __int64 v29; // rax
  double v30; // xmm4_8
  double v31; // xmm5_8

  v10 = *(__int64 **)(a1 + 8);
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
LABEL_20:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_4F9E06C )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_20;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_4F9E06C);
  v14 = *(__int64 **)(a1 + 8);
  v15 = v13 + 160;
  v16 = *v14;
  v17 = v14[1];
  if ( v16 == v17 )
LABEL_21:
    BUG();
  while ( *(_UNKNOWN **)v16 != &unk_4F9D764 )
  {
    v16 += 16;
    if ( v17 == v16 )
      goto LABEL_21;
  }
  v18 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v16 + 8) + 104LL))(*(_QWORD *)(v16 + 8), &unk_4F9D764);
  v19 = sub_14CF090(v18, a2);
  v20 = sub_22077B0(3488);
  v22 = v20;
  if ( v20 )
  {
    v23 = v20;
    sub_1B31180(v20, a2, v15, v19);
    v25 = sub_16BA580(v23, a2, v24);
    sub_1B2BDA0(v22, v25);
    if ( byte_4FB6CE0 )
      nullsub_655();
    sub_1B2A5B0(v22, a2, a3, a4, a5, a6, v26, v27, a9, a10);
    sub_1B2B8B0(v22);
    j_j___libc_free_0(v22, 3488);
  }
  else
  {
    v29 = sub_16BA580(3488, a2, v21);
    sub_1B2BDA0(0, v29);
    if ( byte_4FB6CE0 )
      nullsub_655();
    sub_1B2A5B0(0, a2, a3, a4, a5, a6, v30, v31, a9, a10);
  }
  return 0;
}
