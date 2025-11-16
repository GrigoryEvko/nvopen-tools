// Function: sub_1BEFE70
// Address: 0x1befe70
//
void __fastcall sub_1BEFE70(__int64 a1)
{
  __int64 v1; // rdx
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rax
  __int64 *v12; // rdi
  __int64 v13; // rdx
  _QWORD *v14; // rsi
  _QWORD *v15; // r8
  unsigned __int64 v16; // r15
  __int64 v17; // rax
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdx
  _QWORD *v20; // rax
  char v21; // cl
  unsigned __int64 v22; // rcx
  unsigned __int64 v23; // r8
  unsigned __int64 v24; // r15
  __int64 v25; // rax
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // rdx
  unsigned __int64 v28; // rax
  char v29; // si
  unsigned __int64 v30; // rax
  unsigned __int64 v31; // rdx
  unsigned __int64 v32; // rsi
  char v33; // al
  char v34; // r8
  bool v35; // al
  __int64 v36; // rax
  unsigned __int64 v37; // rcx
  unsigned __int64 v38; // r8
  unsigned __int64 v39; // r12
  __int64 v40; // rax
  __int64 v41; // rdi
  __int64 v42; // rdx
  unsigned __int64 v43; // rax
  char v44; // si
  __int64 v45; // rcx
  __int64 v46; // r8
  unsigned __int64 v47; // r12
  __int64 v48; // rax
  unsigned __int64 v49; // rdi
  unsigned __int64 v50; // rdx
  __int64 v51; // rax
  char v52; // si
  unsigned __int64 v53; // rax
  __int64 v54; // rdx
  unsigned __int64 v55; // rsi
  char v56; // al
  char v57; // r8
  bool v58; // al
  _QWORD v60[2]; // [rsp+30h] [rbp-430h] BYREF
  unsigned __int64 v61; // [rsp+40h] [rbp-420h]
  _BYTE v62[64]; // [rsp+58h] [rbp-408h] BYREF
  __int64 v63; // [rsp+98h] [rbp-3C8h]
  __int64 v64; // [rsp+A0h] [rbp-3C0h]
  unsigned __int64 v65; // [rsp+A8h] [rbp-3B8h]
  _QWORD v66[16]; // [rsp+B0h] [rbp-3B0h] BYREF
  __int64 v67; // [rsp+130h] [rbp-330h] BYREF
  _QWORD *v68; // [rsp+138h] [rbp-328h]
  _QWORD *v69; // [rsp+140h] [rbp-320h]
  __int64 v70; // [rsp+148h] [rbp-318h]
  int v71; // [rsp+150h] [rbp-310h]
  _QWORD v72[8]; // [rsp+158h] [rbp-308h] BYREF
  unsigned __int64 v73; // [rsp+198h] [rbp-2C8h] BYREF
  unsigned __int64 v74; // [rsp+1A0h] [rbp-2C0h]
  unsigned __int64 v75; // [rsp+1A8h] [rbp-2B8h]
  __int64 v76; // [rsp+1B0h] [rbp-2B0h] BYREF
  _QWORD *v77; // [rsp+1B8h] [rbp-2A8h]
  _QWORD *v78; // [rsp+1C0h] [rbp-2A0h]
  __int64 v79; // [rsp+1C8h] [rbp-298h]
  int v80; // [rsp+1D0h] [rbp-290h]
  _QWORD v81[8]; // [rsp+1D8h] [rbp-288h] BYREF
  unsigned __int64 v82; // [rsp+218h] [rbp-248h] BYREF
  unsigned __int64 i; // [rsp+220h] [rbp-240h]
  unsigned __int64 v84; // [rsp+228h] [rbp-238h]
  __int64 v85; // [rsp+230h] [rbp-230h] BYREF
  __int64 v86; // [rsp+238h] [rbp-228h]
  unsigned __int64 v87; // [rsp+240h] [rbp-220h]
  char v88[64]; // [rsp+258h] [rbp-208h] BYREF
  unsigned __int64 v89; // [rsp+298h] [rbp-1C8h]
  unsigned __int64 v90; // [rsp+2A0h] [rbp-1C0h]
  unsigned __int64 v91; // [rsp+2A8h] [rbp-1B8h]
  char v92[8]; // [rsp+2B0h] [rbp-1B0h] BYREF
  __int64 v93; // [rsp+2B8h] [rbp-1A8h]
  unsigned __int64 v94; // [rsp+2C0h] [rbp-1A0h]
  __int64 v95; // [rsp+318h] [rbp-148h]
  __int64 v96; // [rsp+320h] [rbp-140h]
  __int64 v97; // [rsp+328h] [rbp-138h]
  _QWORD v98[16]; // [rsp+330h] [rbp-130h] BYREF
  _QWORD v99[2]; // [rsp+3B0h] [rbp-B0h] BYREF
  unsigned __int64 v100; // [rsp+3C0h] [rbp-A0h]
  char v101[64]; // [rsp+3D8h] [rbp-88h] BYREF
  unsigned __int64 v102; // [rsp+418h] [rbp-48h]
  unsigned __int64 v103; // [rsp+420h] [rbp-40h]
  unsigned __int64 v104; // [rsp+428h] [rbp-38h]

  v1 = *(_QWORD *)(a1 + 112);
  memset(v66, 0, sizeof(v66));
  v66[1] = &v66[5];
  v66[2] = &v66[5];
  v70 = 0x100000008LL;
  v72[0] = v1;
  v98[0] = v1;
  LODWORD(v66[3]) = 8;
  v68 = v72;
  v69 = v72;
  v73 = 0;
  v74 = 0;
  v75 = 0;
  v71 = 0;
  v67 = 1;
  LOBYTE(v98[2]) = 0;
  sub_1BE49B0(&v73, (__int64)v98);
  sub_16CCEE0(&v76, (__int64)v81, 8, (__int64)v66);
  v2 = v66[13];
  memset(&v66[13], 0, 24);
  v82 = v2;
  i = v66[14];
  v84 = v66[15];
  sub_16CCEE0(&v85, (__int64)v88, 8, (__int64)&v67);
  v3 = v73;
  v73 = 0;
  v89 = v3;
  v4 = v74;
  v74 = 0;
  v90 = v4;
  v5 = v75;
  v75 = 0;
  v91 = v5;
  sub_16CCEE0(v98, (__int64)&v98[5], 8, (__int64)&v85);
  v6 = v89;
  v89 = 0;
  v98[13] = v6;
  v7 = v90;
  v90 = 0;
  v98[14] = v7;
  v8 = v91;
  v91 = 0;
  v98[15] = v8;
  sub_16CCEE0(v99, (__int64)v101, 8, (__int64)&v76);
  v9 = v82;
  v82 = 0;
  v102 = v9;
  v10 = i;
  i = 0;
  v103 = v10;
  v11 = v84;
  v84 = 0;
  v104 = v11;
  if ( v89 )
    j_j___libc_free_0(v89, v91 - v89);
  if ( v87 != v86 )
    _libc_free(v87);
  if ( v82 )
    j_j___libc_free_0(v82, v84 - v82);
  if ( v78 != v77 )
    _libc_free((unsigned __int64)v78);
  if ( v73 )
    j_j___libc_free_0(v73, v75 - v73);
  if ( v69 != v68 )
    _libc_free((unsigned __int64)v69);
  if ( v66[13] )
    j_j___libc_free_0(v66[13], v66[15] - v66[13]);
  if ( v66[2] != v66[1] )
    _libc_free(v66[2]);
  v12 = &v67;
  sub_16CCCB0(&v67, (__int64)v72, (__int64)v98);
  v14 = (_QWORD *)v98[14];
  v15 = (_QWORD *)v98[13];
  v73 = 0;
  v74 = 0;
  v75 = 0;
  v16 = v98[14] - v98[13];
  if ( v98[14] == v98[13] )
  {
    v18 = 0;
  }
  else
  {
    if ( v16 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_130;
    v17 = sub_22077B0(v98[14] - v98[13]);
    v14 = (_QWORD *)v98[14];
    v15 = (_QWORD *)v98[13];
    v18 = v17;
  }
  v73 = v18;
  v74 = v18;
  v75 = v18 + v16;
  if ( v15 != v14 )
  {
    v19 = v18;
    v20 = v15;
    do
    {
      if ( v19 )
      {
        *(_QWORD *)v19 = *v20;
        v21 = *((_BYTE *)v20 + 16);
        *(_BYTE *)(v19 + 16) = v21;
        if ( v21 )
          *(_QWORD *)(v19 + 8) = v20[1];
      }
      v20 += 3;
      v19 += 24LL;
    }
    while ( v20 != v14 );
    v18 += 8 * ((unsigned __int64)((char *)(v20 - 3) - (char *)v15) >> 3) + 24;
  }
  v14 = v81;
  v74 = v18;
  v12 = &v76;
  sub_16CCCB0(&v76, (__int64)v81, (__int64)v99);
  v22 = v103;
  v23 = v102;
  v82 = 0;
  i = 0;
  v84 = 0;
  v24 = v103 - v102;
  if ( v103 == v102 )
  {
    v26 = 0;
  }
  else
  {
    if ( v24 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_130;
    v25 = sub_22077B0(v103 - v102);
    v22 = v103;
    v23 = v102;
    v26 = v25;
  }
  v82 = v26;
  i = v26;
  v84 = v26 + v24;
  if ( v22 == v23 )
  {
    v30 = v26;
  }
  else
  {
    v27 = v26;
    v28 = v23;
    do
    {
      if ( v27 )
      {
        *(_QWORD *)v27 = *(_QWORD *)v28;
        v29 = *(_BYTE *)(v28 + 16);
        *(_BYTE *)(v27 + 16) = v29;
        if ( v29 )
          *(_QWORD *)(v27 + 8) = *(_QWORD *)(v28 + 8);
      }
      v28 += 24LL;
      v27 += 24LL;
    }
    while ( v22 != v28 );
    v30 = v26 + 8 * ((v22 - 24 - v23) >> 3) + 24;
  }
  for ( i = v30; ; v30 = i )
  {
    v31 = v73;
    if ( v74 - v73 != v30 - v26 )
      goto LABEL_38;
    if ( v73 == v74 )
      break;
    v32 = v26;
    while ( *(_QWORD *)v31 == *(_QWORD *)v32 )
    {
      v33 = *(_BYTE *)(v31 + 16);
      v34 = *(_BYTE *)(v32 + 16);
      if ( v33 && v34 )
        v35 = *(_QWORD *)(v31 + 8) == *(_QWORD *)(v32 + 8);
      else
        v35 = v34 == v33;
      if ( !v35 )
        break;
      v31 += 24LL;
      v32 += 24LL;
      if ( v74 == v31 )
        goto LABEL_48;
    }
LABEL_38:
    sub_1BEFD50((__int64)&v67);
    v26 = v82;
  }
LABEL_48:
  if ( v26 )
    j_j___libc_free_0(v26, v84 - v26);
  if ( v78 != v77 )
    _libc_free((unsigned __int64)v78);
  if ( v73 )
    j_j___libc_free_0(v73, v75 - v73);
  if ( v69 != v68 )
    _libc_free((unsigned __int64)v69);
  if ( v102 )
    j_j___libc_free_0(v102, v104 - v102);
  if ( v100 != v99[1] )
    _libc_free(v100);
  if ( v98[13] )
    j_j___libc_free_0(v98[13], v98[15] - v98[13]);
  if ( v98[2] != v98[1] )
    _libc_free(v98[2]);
  memset(v98, 0, sizeof(v98));
  LODWORD(v98[3]) = 8;
  v98[1] = &v98[5];
  v98[2] = &v98[5];
  v77 = v81;
  v36 = *(_QWORD *)(a1 + 112);
  v78 = v81;
  v82 = 0;
  v81[0] = v36;
  v85 = v36;
  i = 0;
  v84 = 0;
  v79 = 0x100000008LL;
  v80 = 0;
  v76 = 1;
  LOBYTE(v87) = 0;
  sub_1BE49B0(&v82, (__int64)&v85);
  sub_1BEFB70(&v85, &v76, v98);
  if ( v82 )
    j_j___libc_free_0(v82, v84 - v82);
  if ( v78 != v77 )
    _libc_free((unsigned __int64)v78);
  if ( v98[13] )
    j_j___libc_free_0(v98[13], v98[15] - v98[13]);
  if ( v98[2] != v98[1] )
    _libc_free(v98[2]);
  v14 = v62;
  v12 = v60;
  sub_16CCCB0(v60, (__int64)v62, (__int64)&v85);
  v37 = v90;
  v38 = v89;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  v39 = v90 - v89;
  if ( v90 != v89 )
  {
    if ( v39 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v40 = sub_22077B0(v90 - v89);
      v37 = v90;
      v38 = v89;
      v41 = v40;
      goto LABEL_75;
    }
LABEL_130:
    sub_4261EA(v12, v14, v13);
  }
  v41 = 0;
LABEL_75:
  v63 = v41;
  v64 = v41;
  v65 = v41 + v39;
  if ( v37 != v38 )
  {
    v42 = v41;
    v43 = v38;
    do
    {
      if ( v42 )
      {
        *(_QWORD *)v42 = *(_QWORD *)v43;
        v44 = *(_BYTE *)(v43 + 16);
        *(_BYTE *)(v42 + 16) = v44;
        if ( v44 )
          *(_QWORD *)(v42 + 8) = *(_QWORD *)(v43 + 8);
      }
      v43 += 24LL;
      v42 += 24;
    }
    while ( v37 != v43 );
    v41 += 8 * ((v37 - 24 - v38) >> 3) + 24;
  }
  v64 = v41;
  v14 = &v66[5];
  v12 = v66;
  sub_16CCCB0(v66, (__int64)&v66[5], (__int64)v92);
  v45 = v96;
  v46 = v95;
  memset(&v66[13], 0, 24);
  v47 = v96 - v95;
  if ( v96 == v95 )
  {
    v49 = 0;
  }
  else
  {
    if ( v47 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_130;
    v48 = sub_22077B0(v96 - v95);
    v45 = v96;
    v46 = v95;
    v49 = v48;
  }
  v66[13] = v49;
  v66[14] = v49;
  v66[15] = v49 + v47;
  if ( v46 == v45 )
  {
    v53 = v49;
  }
  else
  {
    v50 = v49;
    v51 = v46;
    do
    {
      if ( v50 )
      {
        *(_QWORD *)v50 = *(_QWORD *)v51;
        v52 = *(_BYTE *)(v51 + 16);
        *(_BYTE *)(v50 + 16) = v52;
        if ( v52 )
          *(_QWORD *)(v50 + 8) = *(_QWORD *)(v51 + 8);
      }
      v51 += 24;
      v50 += 24LL;
    }
    while ( v45 != v51 );
    v53 = v49 + 8 * ((unsigned __int64)(v45 - 24 - v46) >> 3) + 24;
  }
  v66[14] = v53;
  while ( 2 )
  {
    v54 = v63;
    if ( v64 - v63 != v53 - v49 )
    {
LABEL_94:
      if ( *(_BYTE *)(*(_QWORD *)(v64 - 24) + 8LL) == 1 )
        sub_1BEFE70();
      sub_1BEFD50((__int64)v60);
      v49 = v66[13];
      v53 = v66[14];
      continue;
    }
    break;
  }
  if ( v63 != v64 )
  {
    v55 = v49;
    while ( *(_QWORD *)v54 == *(_QWORD *)v55 )
    {
      v56 = *(_BYTE *)(v54 + 16);
      v57 = *(_BYTE *)(v55 + 16);
      if ( v56 && v57 )
        v58 = *(_QWORD *)(v54 + 8) == *(_QWORD *)(v55 + 8);
      else
        v58 = v57 == v56;
      if ( !v58 )
        break;
      v54 += 24;
      v55 += 24LL;
      if ( v64 == v54 )
        goto LABEL_108;
    }
    goto LABEL_94;
  }
LABEL_108:
  if ( v49 )
    j_j___libc_free_0(v49, v66[15] - v49);
  if ( v66[2] != v66[1] )
    _libc_free(v66[2]);
  if ( v63 )
    j_j___libc_free_0(v63, v65 - v63);
  if ( v61 != v60[1] )
    _libc_free(v61);
  if ( v95 )
    j_j___libc_free_0(v95, v97 - v95);
  if ( v94 != v93 )
    _libc_free(v94);
  if ( v89 )
    j_j___libc_free_0(v89, v91 - v89);
  if ( v87 != v86 )
    _libc_free(v87);
}
