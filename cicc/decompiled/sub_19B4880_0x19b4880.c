// Function: sub_19B4880
// Address: 0x19b4880
//
__int64 __fastcall sub_19B4880(
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
  __int64 *v11; // rcx
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 *v15; // rcx
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 *v20; // rcx
  __int64 v21; // rbx
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // rax
  __int64 *v25; // rcx
  __int64 v26; // rax
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 *v29; // rcx
  __int64 v30; // rax
  __int64 v31; // rcx
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 *v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rax
  double v39; // xmm4_8
  double v40; // xmm5_8
  __int64 v41; // [rsp+8h] [rbp-48h]
  __int64 v42; // [rsp+10h] [rbp-40h]
  char *v43; // [rsp+18h] [rbp-38h]

  if ( (unsigned __int8)sub_1404700(a1, a2) )
    return 0;
  v11 = *(__int64 **)(a1 + 8);
  v12 = *v11;
  v13 = v11[1];
  if ( v12 == v13 )
LABEL_39:
    BUG();
  while ( *(_UNKNOWN **)v12 != &unk_4F98F4C )
  {
    v12 += 16;
    if ( v13 == v12 )
      goto LABEL_39;
  }
  v14 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v12 + 8) + 104LL))(*(_QWORD *)(v12 + 8), &unk_4F98F4C);
  v15 = *(__int64 **)(a1 + 8);
  v16 = *(_QWORD *)(v14 + 160);
  v17 = *v15;
  v18 = v15[1];
  if ( v17 == v18 )
LABEL_40:
    BUG();
  while ( *(_UNKNOWN **)v17 != &unk_4F9A488 )
  {
    v17 += 16;
    if ( v18 == v17 )
      goto LABEL_40;
  }
  v19 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v17 + 8) + 104LL))(*(_QWORD *)(v17 + 8), &unk_4F9A488);
  v20 = *(__int64 **)(a1 + 8);
  v21 = *(_QWORD *)(v19 + 160);
  v22 = *v20;
  v23 = v20[1];
  if ( v22 == v23 )
LABEL_41:
    BUG();
  while ( *(_UNKNOWN **)v22 != &unk_4F9E06C )
  {
    v22 += 16;
    if ( v23 == v22 )
      goto LABEL_41;
  }
  v24 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v22 + 8) + 104LL))(*(_QWORD *)(v22 + 8), &unk_4F9E06C);
  v25 = *(__int64 **)(a1 + 8);
  v43 = (char *)(v24 + 160);
  v26 = *v25;
  v27 = v25[1];
  if ( v26 == v27 )
LABEL_42:
    BUG();
  while ( *(_UNKNOWN **)v26 != &unk_4F9920C )
  {
    v26 += 16;
    if ( v27 == v26 )
      goto LABEL_42;
  }
  v28 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v26 + 8) + 104LL))(*(_QWORD *)(v26 + 8), &unk_4F9920C);
  v29 = *(__int64 **)(a1 + 8);
  v42 = v28 + 160;
  v30 = *v29;
  v31 = v29[1];
  if ( v30 == v31 )
LABEL_43:
    BUG();
  while ( *(_UNKNOWN **)v30 != &unk_4F9D3C0 )
  {
    v30 += 16;
    if ( v31 == v30 )
      goto LABEL_43;
  }
  v32 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v30 + 8) + 104LL))(*(_QWORD *)(v30 + 8), &unk_4F9D3C0);
  v33 = sub_14A4050(v32, *(_QWORD *)(**(_QWORD **)(a2 + 32) + 56LL));
  v34 = *(__int64 **)(a1 + 8);
  v35 = v33;
  v36 = *v34;
  v37 = v34[1];
  if ( v36 == v37 )
LABEL_44:
    BUG();
  while ( *(_UNKNOWN **)v36 != &unk_4FB9E2C )
  {
    v36 += 16;
    if ( v37 == v36 )
      goto LABEL_44;
  }
  v41 = v35;
  v38 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v36 + 8) + 104LL))(*(_QWORD *)(v36 + 8), &unk_4FB9E2C);
  return sub_19B15B0(v38 + 156, a1 + 168, a2, v16, v21, v43, a3, a4, a5, a6, v39, v40, a9, a10, v42, v41);
}
