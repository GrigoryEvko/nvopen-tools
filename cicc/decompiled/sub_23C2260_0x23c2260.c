// Function: sub_23C2260
// Address: 0x23c2260
//
__int64 __fastcall sub_23C2260(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 *v8; // r12
  __int64 v9; // r15
  void *(*v10)(); // rax
  void *v11; // rax
  __int64 v12; // r12
  __int64 result; // rax
  unsigned int v14; // esi
  __int64 v15; // rdx
  int v16; // r10d
  __int64 *v17; // rdi
  unsigned int v18; // eax
  __int64 *v19; // r13
  void *v20; // r9
  __int64 *v21; // r13
  unsigned int v22; // esi
  __int64 v23; // rdi
  int v24; // r10d
  __int64 *v25; // r9
  unsigned int v26; // edx
  __int64 *v27; // rax
  void *v28; // rcx
  __int64 *v29; // r13
  __int64 v30; // r13
  unsigned int v31; // esi
  __int64 v32; // r8
  int v33; // r14d
  __int64 *v34; // r11
  unsigned int v35; // edx
  __int64 *v36; // rax
  void *v37; // rdi
  __int64 *v38; // r13
  _QWORD *v39; // rax
  __int64 v40; // rdi
  _QWORD *v41; // rax
  __int64 v42; // rdi
  _QWORD *v43; // rax
  __int64 v44; // rdi
  int v45; // eax
  int v46; // edx
  __int64 *v47; // r13
  __int64 v48; // rax
  int v49; // eax
  int v50; // edx
  __int64 v51; // rdx
  int v52; // eax
  int v53; // edx
  __int64 v54; // rdx
  __int64 *v56; // [rsp+18h] [rbp-68h]
  __int64 v57; // [rsp+20h] [rbp-60h] BYREF
  void *v58; // [rsp+28h] [rbp-58h] BYREF
  __int64 *v59; // [rsp+30h] [rbp-50h] BYREF
  int v60; // [rsp+38h] [rbp-48h]
  char v61; // [rsp+40h] [rbp-40h] BYREF

  v4 = *a4;
  *a4 = 0;
  v5 = *(_QWORD *)(a1 + 8);
  v57 = v4;
  sub_23B2720(&v59, &v57);
  v6 = sub_23B66B0((__int64 *)&v59, 1);
  v7 = *(_QWORD *)(sub_BC0510(v5, &unk_4F82418, v6) + 8);
  sub_23B42E0((__int64 *)&v59);
  if ( *(_BYTE *)(a1 + 16) )
    goto LABEL_2;
  v58 = &unk_4FDE338;
  v14 = *(_DWORD *)(v7 + 24);
  if ( !v14 )
  {
    v59 = 0;
    ++*(_QWORD *)v7;
LABEL_88:
    sub_2275E10(v7, 2 * v14);
LABEL_89:
    sub_23519D0(v7, (__int64 *)&v58, &v59);
    v46 = *(_DWORD *)(v7 + 16) + 1;
    goto LABEL_52;
  }
  v15 = *(_QWORD *)(v7 + 8);
  v16 = 1;
  v17 = 0;
  v18 = (v14 - 1) & (((unsigned int)&unk_4FDE338 >> 9) ^ ((unsigned int)&unk_4FDE338 >> 4));
  v19 = (__int64 *)(v15 + 16LL * v18);
  v20 = (void *)*v19;
  if ( (_UNKNOWN *)*v19 == &unk_4FDE338 )
  {
LABEL_19:
    v21 = v19 + 1;
    goto LABEL_20;
  }
  while ( v20 != (void *)-4096LL )
  {
    if ( !v17 && v20 == (void *)-8192LL )
      v17 = v19;
    v18 = (v14 - 1) & (v16 + v18);
    v19 = (__int64 *)(v15 + 16LL * v18);
    v20 = (void *)*v19;
    if ( (_UNKNOWN *)*v19 == &unk_4FDE338 )
      goto LABEL_19;
    ++v16;
  }
  if ( !v17 )
    v17 = v19;
  v59 = v17;
  v45 = *(_DWORD *)(v7 + 16);
  ++*(_QWORD *)v7;
  v46 = v45 + 1;
  if ( 4 * (v45 + 1) >= 3 * v14 )
    goto LABEL_88;
  if ( v14 - *(_DWORD *)(v7 + 20) - v46 <= v14 >> 3 )
  {
    sub_2275E10(v7, v14);
    goto LABEL_89;
  }
LABEL_52:
  *(_DWORD *)(v7 + 16) = v46;
  v47 = v59;
  if ( *v59 != -4096 )
    --*(_DWORD *)(v7 + 20);
  v48 = (__int64)v58;
  v21 = v47 + 1;
  *v21 = 0;
  *(v21 - 1) = v48;
LABEL_20:
  if ( !*v21 )
  {
    v43 = (_QWORD *)sub_22077B0(0x10u);
    if ( v43 )
      *v43 = &unk_4A16110;
    v44 = *v21;
    *v21 = (__int64)v43;
    if ( v44 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v44 + 8LL))(v44);
  }
  v58 = &unk_4FDE330;
  v22 = *(_DWORD *)(v7 + 24);
  if ( !v22 )
  {
    v59 = 0;
    ++*(_QWORD *)v7;
LABEL_85:
    v22 *= 2;
    goto LABEL_86;
  }
  v23 = *(_QWORD *)(v7 + 8);
  v24 = 1;
  v25 = 0;
  v26 = (v22 - 1) & (((unsigned int)&unk_4FDE330 >> 9) ^ ((unsigned int)&unk_4FDE330 >> 4));
  v27 = (__int64 *)(v23 + 16LL * v26);
  v28 = (void *)*v27;
  if ( (_UNKNOWN *)*v27 == &unk_4FDE330 )
    goto LABEL_23;
  while ( v28 != (void *)-4096LL )
  {
    if ( !v25 && v28 == (void *)-8192LL )
      v25 = v27;
    v26 = (v22 - 1) & (v24 + v26);
    v27 = (__int64 *)(v23 + 16LL * v26);
    v28 = (void *)*v27;
    if ( (_UNKNOWN *)*v27 == &unk_4FDE330 )
      goto LABEL_23;
    ++v24;
  }
  if ( !v25 )
    v25 = v27;
  v59 = v25;
  v52 = *(_DWORD *)(v7 + 16);
  ++*(_QWORD *)v7;
  v53 = v52 + 1;
  if ( 4 * (v52 + 1) >= 3 * v22 )
    goto LABEL_85;
  if ( v22 - *(_DWORD *)(v7 + 20) - v53 <= v22 >> 3 )
  {
LABEL_86:
    sub_2275E10(v7, v22);
    sub_23519D0(v7, (__int64 *)&v58, &v59);
    v53 = *(_DWORD *)(v7 + 16) + 1;
  }
  *(_DWORD *)(v7 + 16) = v53;
  v27 = v59;
  if ( *v59 != -4096 )
    --*(_DWORD *)(v7 + 20);
  v54 = (__int64)v58;
  v27[1] = 0;
  *v27 = v54;
LABEL_23:
  v29 = v27 + 1;
  if ( !v27[1] )
  {
    v39 = (_QWORD *)sub_22077B0(0x10u);
    if ( v39 )
      *v39 = &unk_4A16140;
    v40 = *v29;
    *v29 = (__int64)v39;
    if ( v40 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v40 + 8LL))(v40);
  }
  v30 = *(_QWORD *)(a1 + 8);
  v58 = &unk_4FDE328;
  v31 = *(_DWORD *)(v30 + 24);
  if ( !v31 )
  {
    v59 = 0;
    ++*(_QWORD *)v30;
    goto LABEL_69;
  }
  v32 = *(_QWORD *)(v30 + 8);
  v33 = 1;
  v34 = 0;
  v35 = (v31 - 1) & (((unsigned int)&unk_4FDE328 >> 9) ^ ((unsigned int)&unk_4FDE328 >> 4));
  v36 = (__int64 *)(v32 + 16LL * v35);
  v37 = (void *)*v36;
  if ( (_UNKNOWN *)*v36 != &unk_4FDE328 )
  {
    while ( v37 != (void *)-4096LL )
    {
      if ( v37 == (void *)-8192LL && !v34 )
        v34 = v36;
      v35 = (v31 - 1) & (v33 + v35);
      v36 = (__int64 *)(v32 + 16LL * v35);
      v37 = (void *)*v36;
      if ( (_UNKNOWN *)*v36 == &unk_4FDE328 )
        goto LABEL_26;
      ++v33;
    }
    if ( !v34 )
      v34 = v36;
    v59 = v34;
    v49 = *(_DWORD *)(v30 + 16);
    ++*(_QWORD *)v30;
    v50 = v49 + 1;
    if ( 4 * (v49 + 1) < 3 * v31 )
    {
      if ( v31 - *(_DWORD *)(v30 + 20) - v50 > v31 >> 3 )
      {
LABEL_65:
        *(_DWORD *)(v30 + 16) = v50;
        v36 = v59;
        if ( *v59 != -4096 )
          --*(_DWORD *)(v30 + 20);
        v51 = (__int64)v58;
        v36[1] = 0;
        *v36 = v51;
        goto LABEL_26;
      }
LABEL_70:
      sub_23622E0(v30, v31);
      sub_2351850(v30, (__int64 *)&v58, &v59);
      v50 = *(_DWORD *)(v30 + 16) + 1;
      goto LABEL_65;
    }
LABEL_69:
    v31 *= 2;
    goto LABEL_70;
  }
LABEL_26:
  v38 = v36 + 1;
  if ( !v36[1] )
  {
    v41 = (_QWORD *)sub_22077B0(0x10u);
    if ( v41 )
      *v41 = &unk_4A16170;
    v42 = *v38;
    *v38 = (__int64)v41;
    if ( v42 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v42 + 8LL))(v42);
  }
  *(_BYTE *)(a1 + 16) = 1;
LABEL_2:
  sub_23B2720(&v58, &v57);
  sub_23B4300((__int64)&v59, (__int64 *)&v58);
  sub_23B42E0((__int64 *)&v58);
  v56 = &v59[v60];
  if ( v59 != v56 )
  {
    v8 = v59;
    do
    {
      v9 = *v8++;
      sub_BC1CD0(v7, &unk_4FDE338, v9);
      sub_BC1CD0(v7, &unk_4FDE330, v9);
    }
    while ( v56 != v8 );
    v56 = v59;
  }
  if ( v56 != (__int64 *)&v61 )
    _libc_free((unsigned __int64)v56);
  sub_23B2720(&v59, &v57);
  if ( v59
    && ((v10 = *(void *(**)())(*v59 + 24), v10 != sub_23AE340) ? (v11 = v10()) : (v11 = &unk_4CDFBF8),
        v11 == &unk_4C5D162) )
  {
    v12 = v59[1];
    result = sub_23B42E0((__int64 *)&v59);
    if ( v12 )
      result = sub_BC0510(*(_QWORD *)(a1 + 8), &unk_4FDE328, v12);
  }
  else
  {
    result = sub_23B42E0((__int64 *)&v59);
  }
  if ( v57 )
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v57 + 8LL))(v57);
  return result;
}
