// Function: sub_35B6490
// Address: 0x35b6490
//
__int64 __fastcall sub_35B6490(
        _QWORD *a1,
        __int64 a2,
        __m128i a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10)
{
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v14; // rax
  __int64 *v15; // rdx
  __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 *v20; // rdx
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 *v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 *v29; // rdx
  __int64 v30; // r15
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 *v34; // rcx
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rcx
  __int64 v38; // rax
  __int64 *v39; // rdx
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rax
  __int64 *v43; // rdx
  __int64 v44; // r15
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // rsi
  double v51; // xmm4_8
  double v52; // xmm5_8
  __int64 v53; // rax
  __int64 v54; // rdx
  _QWORD *v55; // rsi
  _QWORD *v56; // rax
  __int64 v57; // rdi
  __int64 v58; // rdi
  __int64 v60; // [rsp+0h] [rbp-A0h]
  __int64 v61; // [rsp+8h] [rbp-98h]
  __int64 v62[4]; // [rsp+10h] [rbp-90h] BYREF
  _QWORD v63[14]; // [rsp+30h] [rbp-70h] BYREF

  v10 = (__int64 *)a1[1];
  a1[121] = a2;
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
LABEL_60:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_501EC08 )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_60;
  }
  v14 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_501EC08);
  v15 = (__int64 *)a1[1];
  v16 = v14 + 200;
  v17 = *v15;
  v18 = v15[1];
  if ( v17 == v18 )
LABEL_53:
    BUG();
  while ( *(_UNKNOWN **)v17 != &unk_501EB0C )
  {
    v17 += 16;
    if ( v18 == v17 )
      goto LABEL_53;
  }
  v19 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v17 + 8) + 104LL))(*(_QWORD *)(v17 + 8), &unk_501EB0C);
  v20 = (__int64 *)a1[1];
  v21 = v19 + 200;
  v22 = *v20;
  v23 = v20[1];
  if ( v22 == v23 )
LABEL_54:
    BUG();
  while ( *(_UNKNOWN **)v22 != &unk_501FE44 )
  {
    v22 += 16;
    if ( v23 == v22 )
      goto LABEL_54;
  }
  v24 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v22 + 8) + 104LL))(*(_QWORD *)(v22 + 8), &unk_501FE44);
  v25 = (__int64 *)a1[1];
  v61 = v24 + 200;
  v26 = *v25;
  v27 = v25[1];
  if ( v26 == v27 )
LABEL_55:
    BUG();
  while ( *(_UNKNOWN **)v26 != &unk_501EAFC )
  {
    v26 += 16;
    if ( v27 == v26 )
      goto LABEL_55;
  }
  v28 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v26 + 8) + 104LL))(*(_QWORD *)(v26 + 8), &unk_501EAFC);
  v29 = (__int64 *)a1[1];
  v30 = v28 + 200;
  v31 = *v29;
  v32 = v29[1];
  if ( v31 == v32 )
LABEL_56:
    BUG();
  while ( *(_UNKNOWN **)v31 != &unk_501EACC )
  {
    v31 += 16;
    if ( v32 == v31 )
      goto LABEL_56;
  }
  v33 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v31 + 8) + 104LL))(*(_QWORD *)(v31 + 8), &unk_501EACC);
  v34 = (__int64 *)a1[1];
  v35 = v33 + 200;
  v36 = *v34;
  v37 = v34[1];
  if ( v36 == v37 )
LABEL_57:
    BUG();
  while ( *(_UNKNOWN **)v36 != &unk_502A66C )
  {
    v36 += 16;
    if ( v37 == v36 )
      goto LABEL_57;
  }
  v60 = v35;
  v38 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v36 + 8) + 104LL))(*(_QWORD *)(v36 + 8), &unk_502A66C);
  sub_35B4B20((__int64)(a1 + 25), v38 + 200, v60, v30);
  v39 = (__int64 *)a1[1];
  v40 = *v39;
  v41 = v39[1];
  if ( v40 == v41 )
LABEL_58:
    BUG();
  while ( *(_UNKNOWN **)v40 != &unk_4F87C64 )
  {
    v40 += 16;
    if ( v41 == v40 )
      goto LABEL_58;
  }
  v42 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v40 + 8) + 104LL))(*(_QWORD *)(v40 + 8), &unk_4F87C64);
  v43 = (__int64 *)a1[1];
  v44 = *(_QWORD *)(v42 + 176);
  v45 = *v43;
  v46 = v43[1];
  if ( v45 == v46 )
LABEL_59:
    BUG();
  while ( *(_UNKNOWN **)v45 != &unk_50208AC )
  {
    v45 += 16;
    if ( v46 == v45 )
      goto LABEL_59;
  }
  v47 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v45 + 8) + 104LL))(*(_QWORD *)(v45 + 8), &unk_50208AC);
  v63[5] = v44;
  v48 = a1[28];
  v63[6] = v16;
  v49 = a1[29];
  v50 = a1[121];
  v63[4] = v47 + 200;
  v63[0] = &unk_4A2AFC8;
  v63[1] = v50;
  v63[2] = v49;
  v63[3] = v48;
  sub_34C97A0(v63, a3, a4, a5, a6, v51, v52, a9, a10);
  v53 = a1[29];
  v54 = a1[28];
  v55 = (_QWORD *)a1[121];
  v62[1] = v21;
  v62[0] = v53;
  v62[3] = v16;
  v62[2] = v61;
  v56 = sub_34F6050(v62, v55, v54, (__int64)v63);
  v57 = a1[122];
  a1[122] = v56;
  if ( v57 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v57 + 16LL))(v57);
  sub_35B5380(a1 + 25);
  sub_35B45F0((__int64)(a1 + 25), (__int64)v55);
  v58 = a1[122];
  a1[122] = 0;
  if ( v58 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v58 + 16LL))(v58);
  return 1;
}
