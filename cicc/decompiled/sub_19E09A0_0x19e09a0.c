// Function: sub_19E09A0
// Address: 0x19e09a0
//
_BOOL8 __fastcall sub_19E09A0(
        __int64 a1,
        __int64 a2,
        __m128i a3,
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
  __int64 v15; // rax
  __int64 *v16; // rdx
  __int64 v17; // r14
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 *v21; // rdx
  __int64 v22; // r13
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 *v26; // rdx
  __int64 v27; // rbx
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 *v31; // rdx
  __int64 v32; // r9
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rax
  double v37; // xmm4_8
  double v38; // xmm5_8
  __int64 v39; // [rsp+8h] [rbp-38h]

  if ( (unsigned __int8)sub_1636880(a1, a2) )
    return 0;
  v11 = *(__int64 **)(a1 + 8);
  v12 = *v11;
  v13 = v11[1];
  if ( v12 == v13 )
LABEL_33:
    BUG();
  while ( *(_UNKNOWN **)v12 != &unk_4F9D764 )
  {
    v12 += 16;
    if ( v13 == v12 )
      goto LABEL_33;
  }
  v14 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v12 + 8) + 104LL))(*(_QWORD *)(v12 + 8), &unk_4F9D764);
  v15 = sub_14CF090(v14, a2);
  v16 = *(__int64 **)(a1 + 8);
  v17 = v15;
  v18 = *v16;
  v19 = v16[1];
  if ( v18 == v19 )
LABEL_34:
    BUG();
  while ( *(_UNKNOWN **)v18 != &unk_4F9E06C )
  {
    v18 += 16;
    if ( v19 == v18 )
      goto LABEL_34;
  }
  v20 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v18 + 8) + 104LL))(*(_QWORD *)(v18 + 8), &unk_4F9E06C);
  v21 = *(__int64 **)(a1 + 8);
  v22 = v20 + 160;
  v23 = *v21;
  v24 = v21[1];
  if ( v23 == v24 )
LABEL_35:
    BUG();
  while ( *(_UNKNOWN **)v23 != &unk_4F9A488 )
  {
    v23 += 16;
    if ( v24 == v23 )
      goto LABEL_35;
  }
  v25 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v23 + 8) + 104LL))(*(_QWORD *)(v23 + 8), &unk_4F9A488);
  v26 = *(__int64 **)(a1 + 8);
  v27 = *(_QWORD *)(v25 + 160);
  v28 = *v26;
  v29 = v26[1];
  if ( v28 == v29 )
LABEL_36:
    BUG();
  while ( *(_UNKNOWN **)v28 != &unk_4F9B6E8 )
  {
    v28 += 16;
    if ( v29 == v28 )
      goto LABEL_36;
  }
  v30 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v28 + 8) + 104LL))(*(_QWORD *)(v28 + 8), &unk_4F9B6E8);
  v31 = *(__int64 **)(a1 + 8);
  v32 = v30 + 360;
  v33 = *v31;
  v34 = v31[1];
  if ( v33 == v34 )
LABEL_37:
    BUG();
  while ( *(_UNKNOWN **)v33 != &unk_4F9D3C0 )
  {
    v33 += 16;
    if ( v34 == v33 )
      goto LABEL_37;
  }
  v39 = v32;
  v35 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v33 + 8) + 104LL))(*(_QWORD *)(v33 + 8), &unk_4F9D3C0);
  v36 = sub_14A4050(v35, a2);
  return sub_19E0940((_QWORD *)(a1 + 160), a2, v17, v22, v27, v39, a3, a4, a5, a6, v37, v38, a9, a10, v36);
}
