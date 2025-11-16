// Function: sub_1903990
// Address: 0x1903990
//
__int64 __fastcall sub_1903990(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        __m128i a4,
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
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 *v20; // rdx
  __int64 v21; // r12
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 *v25; // rdx
  __int64 v26; // r15
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rax
  double v31; // xmm4_8
  double v32; // xmm5_8
  unsigned int v33; // r12d
  __int64 v35; // [rsp+8h] [rbp-298h]
  __m128i v36[41]; // [rsp+10h] [rbp-290h] BYREF

  v10 = *(__int64 **)(a1 + 8);
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
LABEL_28:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_4F9B6E8 )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_28;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_4F9B6E8);
  v14 = *(__int64 **)(a1 + 8);
  v15 = v13 + 360;
  v16 = *v14;
  v17 = v14[1];
  if ( v16 == v17 )
LABEL_25:
    BUG();
  while ( *(_UNKNOWN **)v16 != &unk_4F9D3C0 )
  {
    v16 += 16;
    if ( v17 == v16 )
      goto LABEL_25;
  }
  v18 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v16 + 8) + 104LL))(*(_QWORD *)(v16 + 8), &unk_4F9D3C0);
  v19 = sub_14A4050(v18, a2);
  v20 = *(__int64 **)(a1 + 8);
  v21 = v19;
  v22 = *v20;
  v23 = v20[1];
  if ( v22 == v23 )
LABEL_26:
    BUG();
  while ( *(_UNKNOWN **)v22 != &unk_4F9E06C )
  {
    v22 += 16;
    if ( v23 == v22 )
      goto LABEL_26;
  }
  v24 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v22 + 8) + 104LL))(*(_QWORD *)(v22 + 8), &unk_4F9E06C);
  v25 = *(__int64 **)(a1 + 8);
  v26 = v24 + 160;
  v27 = *v25;
  v28 = v25[1];
  if ( v27 == v28 )
LABEL_27:
    BUG();
  while ( *(_UNKNOWN **)v27 != &unk_4F9D764 )
  {
    v27 += 16;
    if ( v28 == v27 )
      goto LABEL_27;
  }
  v29 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v27 + 8) + 104LL))(*(_QWORD *)(v27 + 8), &unk_4F9D764);
  v35 = sub_14CF090(v29, a2);
  v30 = sub_1632FA0(*(_QWORD *)(a2 + 40));
  sub_18FBF50((__int64)v36, v30, v15, v21, v26, v35, 0);
  v33 = sub_1900BB0(v36, a3, a4, a5, a6, v31, v32, a9, a10);
  sub_18FC980((__int64)v36);
  return v33;
}
