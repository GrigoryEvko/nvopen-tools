// Function: sub_1972830
// Address: 0x1972830
//
__int64 __fastcall sub_1972830(
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
  __int64 *v10; // rcx
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 *v14; // rcx
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rax
  __int64 *v19; // rcx
  __int64 v20; // rbx
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rax
  __int64 *v24; // rcx
  __int64 v25; // rax
  __int64 v26; // rcx
  __int64 v27; // rax
  __int64 *v28; // rcx
  __int64 v29; // r14
  __int64 v30; // rax
  __int64 v31; // rcx
  __int64 v32; // rax
  __int64 *v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // r13
  __int64 v38; // rax
  unsigned int v39; // r13d
  double v40; // xmm4_8
  double v41; // xmm5_8
  __int64 v42; // rbx
  __int64 v43; // r12
  unsigned __int64 v44; // rdi
  __int64 v45; // rbx
  __int64 v46; // r12
  unsigned __int64 v47; // rdi
  __int64 v49; // [rsp+0h] [rbp-150h]
  __int64 v50; // [rsp+8h] [rbp-148h]
  __int64 v51[10]; // [rsp+10h] [rbp-140h] BYREF
  __int64 v52; // [rsp+60h] [rbp-F0h]
  __int64 v53; // [rsp+68h] [rbp-E8h]
  int v54; // [rsp+70h] [rbp-E0h]
  __int64 v55; // [rsp+78h] [rbp-D8h]
  __int64 v56; // [rsp+80h] [rbp-D0h]
  __int64 v57; // [rsp+88h] [rbp-C8h]
  __int64 v58; // [rsp+90h] [rbp-C0h]
  __int64 v59; // [rsp+98h] [rbp-B8h]
  __int64 v60; // [rsp+A0h] [rbp-B0h]
  int v61; // [rsp+A8h] [rbp-A8h]
  __int64 v62; // [rsp+B0h] [rbp-A0h]
  __int64 v63; // [rsp+B8h] [rbp-98h]
  __int64 v64; // [rsp+C0h] [rbp-90h]
  _BYTE *v65; // [rsp+C8h] [rbp-88h]
  __int64 v66; // [rsp+D0h] [rbp-80h]
  _BYTE v67[120]; // [rsp+D8h] [rbp-78h] BYREF

  v10 = *(__int64 **)(a1 + 8);
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
LABEL_60:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_4F96DB4 )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_60;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_4F96DB4);
  v14 = *(__int64 **)(a1 + 8);
  v15 = *(_QWORD *)(v13 + 160);
  v16 = *v14;
  v17 = v14[1];
  if ( v16 == v17 )
LABEL_55:
    BUG();
  while ( *(_UNKNOWN **)v16 != &unk_4F9E06C )
  {
    v16 += 16;
    if ( v17 == v16 )
      goto LABEL_55;
  }
  v18 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v16 + 8) + 104LL))(*(_QWORD *)(v16 + 8), &unk_4F9E06C);
  v19 = *(__int64 **)(a1 + 8);
  v20 = v18 + 160;
  v21 = *v19;
  v22 = v19[1];
  if ( v21 == v22 )
LABEL_56:
    BUG();
  while ( *(_UNKNOWN **)v21 != &unk_4F9920C )
  {
    v21 += 16;
    if ( v22 == v21 )
      goto LABEL_56;
  }
  v23 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v21 + 8) + 104LL))(*(_QWORD *)(v21 + 8), &unk_4F9920C);
  v24 = *(__int64 **)(a1 + 8);
  v50 = v23 + 160;
  v25 = *v24;
  v26 = v24[1];
  if ( v25 == v26 )
LABEL_57:
    BUG();
  while ( *(_UNKNOWN **)v25 != &unk_4F9A488 )
  {
    v25 += 16;
    if ( v26 == v25 )
      goto LABEL_57;
  }
  v27 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v25 + 8) + 104LL))(*(_QWORD *)(v25 + 8), &unk_4F9A488);
  v28 = *(__int64 **)(a1 + 8);
  v29 = *(_QWORD *)(v27 + 160);
  v30 = *v28;
  v31 = v28[1];
  if ( v30 == v31 )
LABEL_58:
    BUG();
  while ( *(_UNKNOWN **)v30 != &unk_4F9B6E8 )
  {
    v30 += 16;
    if ( v31 == v30 )
      goto LABEL_58;
  }
  v32 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v30 + 8) + 104LL))(*(_QWORD *)(v30 + 8), &unk_4F9B6E8);
  v33 = *(__int64 **)(a1 + 8);
  v49 = v32 + 360;
  v34 = *v33;
  v35 = v33[1];
  if ( v34 == v35 )
LABEL_59:
    BUG();
  while ( *(_UNKNOWN **)v34 != &unk_4F9D3C0 )
  {
    v34 += 16;
    if ( v35 == v34 )
      goto LABEL_59;
  }
  v36 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v34 + 8) + 104LL))(*(_QWORD *)(v34 + 8), &unk_4F9D3C0);
  v37 = sub_14A4050(v36, *(_QWORD *)(**(_QWORD **)(a2 + 32) + 56LL));
  v38 = sub_157EB90(**(_QWORD **)(a2 + 32));
  v51[2] = v20;
  v51[7] = sub_1632FA0(v38);
  v51[3] = v50;
  v51[4] = v29;
  v51[6] = v37;
  v39 = 0;
  v51[1] = v15;
  v51[5] = v49;
  v51[9] = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v65 = v67;
  v66 = 0x800000000LL;
  v51[0] = a2;
  if ( sub_13FC520(a2) )
    v39 = sub_1972280(v51, a2, a3, a4, a5, a6, v40, v41, a9, a10);
  if ( v65 != v67 )
    _libc_free((unsigned __int64)v65);
  v42 = v63;
  v43 = v62;
  if ( v63 != v62 )
  {
    do
    {
      v44 = *(_QWORD *)(v43 + 8);
      if ( v44 != v43 + 24 )
        _libc_free(v44);
      v43 += 88;
    }
    while ( v42 != v43 );
    v43 = v62;
  }
  if ( v43 )
    j_j___libc_free_0(v43, v64 - v43);
  j___libc_free_0(v59);
  v45 = v56;
  v46 = v55;
  if ( v56 != v55 )
  {
    do
    {
      v47 = *(_QWORD *)(v46 + 8);
      if ( v47 != v46 + 24 )
        _libc_free(v47);
      v46 += 88;
    }
    while ( v45 != v46 );
    v46 = v55;
  }
  if ( v46 )
    j_j___libc_free_0(v46, v57 - v46);
  j___libc_free_0(v52);
  return v39;
}
