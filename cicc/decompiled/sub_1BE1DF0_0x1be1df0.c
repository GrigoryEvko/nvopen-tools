// Function: sub_1BE1DF0
// Address: 0x1be1df0
//
__int64 __fastcall sub_1BE1DF0(
        __int64 a1,
        __int64 a2,
        __m128i a3,
        __m128i a4,
        __m128i a5,
        __m128i a6,
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
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // r13
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rbx
  __int64 *v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 *v27; // rdx
  __int64 v28; // r15
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 *v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 *v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 *v41; // rdx
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 *v45; // rdx
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rax
  double v49; // xmm4_8
  double v50; // xmm5_8
  __int64 v51; // [rsp+8h] [rbp-58h]
  __int64 v52; // [rsp+10h] [rbp-50h]
  __int64 v53; // [rsp+18h] [rbp-48h]
  __int64 v54; // [rsp+20h] [rbp-40h]
  __int64 v55; // [rsp+28h] [rbp-38h]

  if ( (unsigned __int8)sub_1636880(a1, a2) )
    return 0;
  v11 = *(__int64 **)(a1 + 8);
  v12 = *v11;
  v13 = v11[1];
  if ( v12 == v13 )
LABEL_61:
    BUG();
  while ( *(_UNKNOWN **)v12 != &unk_4F9A488 )
  {
    v12 += 16;
    if ( v13 == v12 )
      goto LABEL_61;
  }
  v14 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v12 + 8) + 104LL))(*(_QWORD *)(v12 + 8), &unk_4F9A488);
  v15 = *(__int64 **)(a1 + 8);
  v51 = *(_QWORD *)(v14 + 160);
  v16 = *v15;
  v17 = v15[1];
  if ( v16 == v17 )
LABEL_55:
    BUG();
  while ( *(_UNKNOWN **)v16 != &unk_4F9D3C0 )
  {
    v16 += 16;
    if ( v17 == v16 )
      goto LABEL_55;
  }
  v18 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v16 + 8) + 104LL))(*(_QWORD *)(v16 + 8), &unk_4F9D3C0);
  v19 = sub_14A4050(v18, a2);
  v20 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4F9B6E8, 1u);
  if ( v20 && (v21 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v20 + 104LL))(v20, &unk_4F9B6E8)) != 0 )
    v22 = v21 + 360;
  else
    v22 = 0;
  v23 = *(__int64 **)(a1 + 8);
  v24 = *v23;
  v25 = v23[1];
  if ( v24 == v25 )
LABEL_54:
    BUG();
  while ( *(_UNKNOWN **)v24 != &unk_4F96DB4 )
  {
    v24 += 16;
    if ( v25 == v24 )
      goto LABEL_54;
  }
  v26 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v24 + 8) + 104LL))(*(_QWORD *)(v24 + 8), &unk_4F96DB4);
  v27 = *(__int64 **)(a1 + 8);
  v28 = *(_QWORD *)(v26 + 160);
  v29 = *v27;
  v30 = v27[1];
  if ( v29 == v30 )
LABEL_58:
    BUG();
  while ( *(_UNKNOWN **)v29 != &unk_4F9920C )
  {
    v29 += 16;
    if ( v30 == v29 )
      goto LABEL_58;
  }
  v31 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v29 + 8) + 104LL))(*(_QWORD *)(v29 + 8), &unk_4F9920C);
  v32 = *(__int64 **)(a1 + 8);
  v55 = v31 + 160;
  v33 = *v32;
  v34 = v32[1];
  if ( v33 == v34 )
LABEL_56:
    BUG();
  while ( *(_UNKNOWN **)v33 != &unk_4F9E06C )
  {
    v33 += 16;
    if ( v34 == v33 )
      goto LABEL_56;
  }
  v35 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v33 + 8) + 104LL))(*(_QWORD *)(v33 + 8), &unk_4F9E06C);
  v36 = *(__int64 **)(a1 + 8);
  v54 = v35 + 160;
  v37 = *v36;
  v38 = v36[1];
  if ( v37 == v38 )
LABEL_57:
    BUG();
  while ( *(_UNKNOWN **)v37 != &unk_4F9D764 )
  {
    v37 += 16;
    if ( v38 == v37 )
      goto LABEL_57;
  }
  v39 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v37 + 8) + 104LL))(*(_QWORD *)(v37 + 8), &unk_4F9D764);
  v40 = sub_14CF090(v39, a2);
  v41 = *(__int64 **)(a1 + 8);
  v53 = v40;
  v42 = *v41;
  v43 = v41[1];
  if ( v42 == v43 )
LABEL_59:
    BUG();
  while ( *(_UNKNOWN **)v42 != &unk_4F98D2C )
  {
    v42 += 16;
    if ( v43 == v42 )
      goto LABEL_59;
  }
  v44 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v42 + 8) + 104LL))(*(_QWORD *)(v42 + 8), &unk_4F98D2C);
  v45 = *(__int64 **)(a1 + 8);
  v52 = v44 + 160;
  v46 = *v45;
  v47 = v45[1];
  if ( v46 == v47 )
LABEL_60:
    BUG();
  while ( *(_UNKNOWN **)v46 != &unk_4F99CB0 )
  {
    v46 += 16;
    if ( v47 == v46 )
      goto LABEL_60;
  }
  v48 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v46 + 8) + 104LL))(*(_QWORD *)(v46 + 8), &unk_4F99CB0);
  return sub_1BE1020(
           a1 + 160,
           a2,
           v51,
           v19,
           v22,
           v28,
           a3,
           a4,
           a5,
           a6,
           v49,
           v50,
           a9,
           a10,
           v55,
           v54,
           v53,
           v52,
           *(_QWORD *)(v48 + 160));
}
