// Function: sub_1C6D0C0
// Address: 0x1c6d0c0
//
_BOOL8 __fastcall sub_1C6D0C0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        __m128i a4,
        __m128 a5,
        __m128 a6,
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
  __int64 v15; // rbx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 *v19; // rdx
  __int64 v20; // r13
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // r15
  double v24; // xmm4_8
  double v25; // xmm5_8
  __int64 v26; // rcx
  __int64 v27; // rdx
  _BOOL4 v28; // eax
  __int64 v29; // rbx
  _BOOL4 v30; // r12d
  __int64 v31; // rdi
  __int64 v32; // rbx
  __int64 v33; // r13
  __int64 v34; // rdi
  __int64 v35; // rbx
  __int64 v36; // rdi
  _QWORD *v37; // rbx
  _QWORD *v38; // r13
  unsigned __int64 v39; // rdi
  __int64 *v41; // rdx
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // [rsp+8h] [rbp-1A8h] BYREF
  __int64 v45; // [rsp+10h] [rbp-1A0h] BYREF
  _QWORD *v46; // [rsp+18h] [rbp-198h]
  __int64 v47; // [rsp+20h] [rbp-190h]
  __int64 v48; // [rsp+28h] [rbp-188h]
  __int64 v49; // [rsp+30h] [rbp-180h]
  __int64 v50; // [rsp+38h] [rbp-178h]
  __int64 v51; // [rsp+40h] [rbp-170h]
  __int64 v52; // [rsp+48h] [rbp-168h]
  __int64 v53; // [rsp+58h] [rbp-158h] BYREF
  __int64 v54; // [rsp+60h] [rbp-150h]
  __int64 *v55; // [rsp+68h] [rbp-148h]
  __int64 *v56; // [rsp+70h] [rbp-140h]
  __int64 v57; // [rsp+78h] [rbp-138h]
  __int64 v58; // [rsp+80h] [rbp-130h]
  __int64 v59; // [rsp+88h] [rbp-128h]
  __int64 v60; // [rsp+90h] [rbp-120h]
  int v61; // [rsp+98h] [rbp-118h]
  __int64 v62; // [rsp+A0h] [rbp-110h]
  __int64 v63; // [rsp+A8h] [rbp-108h]
  __int64 v64; // [rsp+B0h] [rbp-100h]
  int v65; // [rsp+B8h] [rbp-F8h]
  __int64 v66; // [rsp+C0h] [rbp-F0h]
  __int64 v67; // [rsp+C8h] [rbp-E8h]
  __int64 v68; // [rsp+D0h] [rbp-E0h]
  __int64 v69; // [rsp+D8h] [rbp-D8h]
  __int64 v70; // [rsp+E0h] [rbp-D0h]
  __int64 (__fastcall *v71)(__int64 *, unsigned __int64); // [rsp+E8h] [rbp-C8h]
  __int64 *v72; // [rsp+F0h] [rbp-C0h]
  __int64 v73; // [rsp+F8h] [rbp-B8h]
  __int64 v74; // [rsp+100h] [rbp-B0h]
  __int64 v75; // [rsp+108h] [rbp-A8h]
  __int64 v76; // [rsp+110h] [rbp-A0h]
  __int64 v77; // [rsp+120h] [rbp-90h] BYREF
  __int64 v78; // [rsp+128h] [rbp-88h]
  __int64 *v79; // [rsp+130h] [rbp-80h]
  __int64 *v80; // [rsp+138h] [rbp-78h]
  __int64 v81; // [rsp+140h] [rbp-70h]
  int v82; // [rsp+150h] [rbp-60h] BYREF
  __int64 v83; // [rsp+158h] [rbp-58h]
  int *v84; // [rsp+160h] [rbp-50h]
  int *v85; // [rsp+168h] [rbp-48h]
  __int64 v86; // [rsp+170h] [rbp-40h]

  v10 = *(__int64 **)(a1 + 8);
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
LABEL_43:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_4F9920C )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_43;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_4F9920C);
  v14 = *(__int64 **)(a1 + 8);
  v15 = v13 + 160;
  v16 = *v14;
  v17 = v14[1];
  if ( v16 == v17 )
LABEL_40:
    BUG();
  while ( *(_UNKNOWN **)v16 != &unk_4F9E06C )
  {
    v16 += 16;
    if ( v17 == v16 )
      goto LABEL_40;
  }
  v18 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v16 + 8) + 104LL))(*(_QWORD *)(v16 + 8), &unk_4F9E06C);
  v19 = *(__int64 **)(a1 + 8);
  v20 = v18 + 160;
  v21 = *v19;
  v22 = v19[1];
  if ( v21 == v22 )
LABEL_41:
    BUG();
  while ( *(_UNKNOWN **)v21 != &unk_4FB9E2C )
  {
    v21 += 16;
    if ( v22 == v21 )
      goto LABEL_41;
  }
  v23 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v21 + 8) + 104LL))(*(_QWORD *)(v21 + 8), &unk_4FB9E2C)
      + 156;
  if ( dword_4FBD2C0 | dword_4FBD1E0 )
  {
    v41 = *(__int64 **)(a1 + 8);
    v42 = *v41;
    v43 = v41[1];
    if ( v42 == v43 )
LABEL_42:
      BUG();
    while ( *(_UNKNOWN **)v42 != &unk_4F9A488 )
    {
      v42 += 16;
      if ( v43 == v42 )
        goto LABEL_42;
    }
    v26 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v42 + 8) + 104LL))(
                        *(_QWORD *)(v42 + 8),
                        &unk_4F9A488)
                    + 160);
  }
  else
  {
    v26 = 0;
  }
  v27 = *(_QWORD *)(a1 + 160);
  v67 = v26;
  v55 = &v53;
  v56 = &v53;
  v71 = sub_1C51230;
  v72 = &v44;
  v66 = v27;
  v68 = v15;
  v44 = a1;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  v69 = v20;
  v70 = v23;
  v73 = 0;
  v74 = 0;
  v75 = 0;
  v79 = &v77;
  v80 = &v77;
  v76 = 0;
  v77 = 0;
  v78 = 0;
  v81 = 0;
  v82 = 0;
  v83 = 0;
  v84 = &v82;
  v85 = &v82;
  v86 = 0;
  v28 = sub_1C6A6C0((__int64)&v45, a2, a3, a4, a5, a6, v24, v25, a9, a10);
  v29 = v83;
  v30 = v28;
  while ( v29 )
  {
    sub_1C51850(*(_QWORD *)(v29 + 24));
    v31 = v29;
    v29 = *(_QWORD *)(v29 + 16);
    j_j___libc_free_0(v31, 56);
  }
  v32 = v78;
  while ( v32 )
  {
    v33 = v32;
    sub_1C51BF0(*(_QWORD **)(v32 + 24));
    v34 = *(_QWORD *)(v32 + 48);
    v32 = *(_QWORD *)(v32 + 16);
    j___libc_free_0(v34);
    j_j___libc_free_0(v33, 72);
  }
  j___libc_free_0(v74);
  j___libc_free_0(v63);
  j___libc_free_0(v59);
  v35 = v54;
  while ( v35 )
  {
    sub_1C51A20(*(_QWORD *)(v35 + 24));
    v36 = v35;
    v35 = *(_QWORD *)(v35 + 16);
    j_j___libc_free_0(v36, 48);
  }
  j___libc_free_0(v50);
  if ( (_DWORD)v48 )
  {
    v37 = v46;
    v38 = &v46[4 * (unsigned int)v48];
    do
    {
      if ( *v37 != -16 && *v37 != -8 )
      {
        v39 = v37[1];
        if ( (_QWORD *)v39 != v37 + 3 )
          _libc_free(v39);
      }
      v37 += 4;
    }
    while ( v38 != v37 );
  }
  j___libc_free_0(v46);
  return v30;
}
