// Function: sub_398FCD0
// Address: 0x398fcd0
//
void __fastcall sub_398FCD0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // rbx
  __int64 v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // rbx
  __int64 v13; // rdi
  __int64 v14; // rdx
  const void *v15; // r14
  __int64 v16; // rax
  size_t v17; // rdx
  size_t v18; // r8
  const void *v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rdx
  int v22; // eax
  int v23; // ecx
  __int64 v24; // rsi
  unsigned int v25; // edx
  __int64 *v26; // rax
  __int64 v27; // rdi
  __int64 v28; // rsi
  __int64 v29; // rdx
  __int64 v30; // rdi
  _BYTE *v31; // rax
  __int64 v32; // rdx
  __int64 v33; // r14
  __int64 v34; // r11
  _BYTE *v35; // rax
  __int64 v36; // rdx
  _BYTE *v37; // r8
  size_t v38; // rbx
  __int64 v39; // rax
  const void *v40; // rdi
  _BYTE *v41; // rax
  unsigned __int64 v42; // r14
  _BYTE *v43; // rax
  _BYTE *v44; // r8
  unsigned __int64 v45; // rax
  char *v46; // rax
  _BYTE *v47; // r8
  __int64 v48; // r11
  size_t v49; // rdx
  char *v50; // rcx
  _BYTE *v51; // rax
  unsigned __int64 v52; // rsi
  _BYTE *v53; // r8
  unsigned __int64 v54; // rcx
  __int64 v55; // r11
  unsigned __int64 v56; // rax
  size_t v57; // rax
  __int64 v58; // rbx
  _BYTE *v59; // rax
  _BYTE *v60; // r8
  __int64 v61; // r11
  size_t v62; // rdx
  unsigned __int64 v63; // r14
  _BYTE *v64; // rax
  unsigned __int64 v65; // r11
  unsigned __int64 v66; // rax
  __int64 v67; // rdx
  __int64 v68; // rdx
  __int64 v69; // r8
  _BYTE *v70; // rax
  __int64 v71; // rdx
  _BYTE *v72; // r15
  size_t v73; // r14
  _BYTE *v74; // rax
  __int64 v75; // r8
  size_t v76; // rdx
  unsigned __int64 v77; // rbx
  _BYTE *v78; // rax
  unsigned __int64 v79; // rax
  _BYTE *v80; // rax
  _BYTE *v81; // rax
  const void *v82; // rdi
  _BYTE *v83; // rax
  _BYTE *v84; // rax
  _BYTE *v85; // r8
  unsigned __int64 v86; // rax
  char *v87; // rax
  int v88; // eax
  int v89; // r9d
  _BYTE *v90; // [rsp-68h] [rbp-68h]
  __int64 v91; // [rsp-68h] [rbp-68h]
  _BYTE *v92; // [rsp-68h] [rbp-68h]
  _BYTE *v93; // [rsp-60h] [rbp-60h]
  _BYTE *v94; // [rsp-60h] [rbp-60h]
  _BYTE *v95; // [rsp-60h] [rbp-60h]
  unsigned __int64 v96; // [rsp-60h] [rbp-60h]
  _BYTE *v97; // [rsp-60h] [rbp-60h]
  _BYTE *v98; // [rsp-60h] [rbp-60h]
  __int64 v99; // [rsp-60h] [rbp-60h]
  _BYTE *v100; // [rsp-58h] [rbp-58h]
  _BYTE *v101; // [rsp-58h] [rbp-58h]
  __int64 v102; // [rsp-58h] [rbp-58h]
  __int64 v103; // [rsp-58h] [rbp-58h]
  __int64 v104; // [rsp-58h] [rbp-58h]
  size_t v105; // [rsp-50h] [rbp-50h]
  __int64 v106; // [rsp-50h] [rbp-50h]
  _BYTE *v107; // [rsp-50h] [rbp-50h]
  _BYTE *v108; // [rsp-50h] [rbp-50h]
  _BYTE *v109; // [rsp-50h] [rbp-50h]
  __int64 v110; // [rsp-50h] [rbp-50h]
  __int64 v111; // [rsp-50h] [rbp-50h]
  _BYTE *v112; // [rsp-50h] [rbp-50h]
  __int64 v113; // [rsp-50h] [rbp-50h]
  __int64 v114[9]; // [rsp-48h] [rbp-48h] BYREF

  if ( (*(_BYTE *)(a2 + 40) & 8) == 0 )
    return;
  v6 = *(unsigned int *)(a2 + 8);
  v7 = *(_QWORD *)(a2 + 8 * (2 - v6));
  if ( v7 )
  {
    sub_161E970(v7);
    if ( v8 )
    {
      v9 = 2LL - *(unsigned int *)(a2 + 8);
      v10 = *(_QWORD *)(a2 + 8 * v9);
      if ( v10 )
        v10 = sub_161E970(*(_QWORD *)(a2 + 8 * v9));
      else
        v11 = 0;
      sub_398FC80(a1, v10, v11, a3);
    }
    v12 = *(unsigned int *)(a2 + 8);
    v13 = *(_QWORD *)(a2 + 8 * (3 - v12));
    if ( !v13 )
      goto LABEL_20;
  }
  else
  {
    v13 = *(_QWORD *)(a2 + 8 * (3 - v6));
    if ( !v13 )
      return;
  }
  sub_161E970(v13);
  v12 = *(unsigned int *)(a2 + 8);
  if ( !v14 )
    goto LABEL_20;
  v15 = *(const void **)(a2 + 8 * (3 - v12));
  if ( !v15 )
  {
    v19 = *(const void **)(a2 + 8 * (2 - v12));
    if ( !v19 )
      goto LABEL_20;
    v18 = 0;
    goto LABEL_11;
  }
  v16 = sub_161E970(*(_QWORD *)(a2 + 8 * (3 - v12)));
  v12 = *(unsigned int *)(a2 + 8);
  v15 = (const void *)v16;
  v18 = v17;
  v19 = *(const void **)(a2 + 8 * (2 - v12));
  if ( v19 )
  {
LABEL_11:
    v105 = v18;
    v20 = sub_161E970((__int64)v19);
    v12 = *(unsigned int *)(a2 + 8);
    v18 = v105;
    v19 = (const void *)v20;
    goto LABEL_12;
  }
  v21 = 0;
LABEL_12:
  if ( v18 == v21 && (!v18 || !memcmp(v19, v15, v18)) )
    goto LABEL_20;
  if ( *(_BYTE *)(a1 + 4498) )
    goto LABEL_17;
  v22 = *(_DWORD *)(a1 + 4360);
  if ( !v22 )
    goto LABEL_20;
  v23 = v22 - 1;
  v24 = *(_QWORD *)(a1 + 4344);
  v25 = (v22 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v26 = (__int64 *)(v24 + 16LL * v25);
  v27 = *v26;
  if ( a2 == *v26 )
  {
LABEL_16:
    if ( !v26[1] )
      goto LABEL_20;
LABEL_17:
    v28 = *(_QWORD *)(a2 + 8 * (3 - v12));
    if ( v28 )
      v28 = sub_161E970(*(_QWORD *)(a2 + 8 * (3 - v12)));
    else
      v29 = 0;
    sub_398FC80(a1, v28, v29, a3);
    v12 = *(unsigned int *)(a2 + 8);
    goto LABEL_20;
  }
  v88 = 1;
  while ( v27 != -8 )
  {
    v89 = v88 + 1;
    v25 = v23 & (v88 + v25);
    v26 = (__int64 *)(v24 + 16LL * v25);
    v27 = *v26;
    if ( a2 == *v26 )
      goto LABEL_16;
    v88 = v89;
  }
LABEL_20:
  v30 = *(_QWORD *)(a2 + 8 * (2 - v12));
  if ( !v30 )
    return;
  v31 = (_BYTE *)sub_161E970(v30);
  if ( !v32 || ((*v31 - 43) & 0xFD) != 0 )
    return;
  v33 = 2LL - *(unsigned int *)(a2 + 8);
  v34 = *(_QWORD *)(a2 + 8 * v33);
  if ( !v34
    || (v35 = (_BYTE *)sub_161E970(*(_QWORD *)(a2 + 8 * v33)),
        v106 = v36,
        v37 = v35,
        v38 = v36,
        v34 = (__int64)v35,
        v114[0] = (__int64)v35,
        (v114[1] = v36) == 0) )
  {
    v67 = 0;
LABEL_61:
    sub_398FCA0(a1, v34, v67);
    goto LABEL_62;
  }
  if ( ((*v35 - 43) & 0xFD) != 0
    || (v93 = v35, v100 = v35, v39 = sub_16D20C0(v114, ") ", 2u, 0), v34 = (__int64)v100, v37 = v93, v39 == -1) )
  {
    if ( (v38 & 0x8000000000000000LL) != 0LL )
    {
      v104 = v34;
      v112 = v37;
      v80 = memchr(v37, 32, 0x7FFFFFFFFFFFFFFFuLL);
      v60 = v112;
      v61 = v104;
      v62 = 0x7FFFFFFFFFFFFFFFLL;
      v63 = v80 - v112;
      if ( !v80 )
        v63 = -1;
    }
    else
    {
      v102 = v34;
      v108 = v37;
      v59 = memchr(v37, 32, v38);
      v60 = v108;
      v61 = v102;
      v62 = v38;
      v63 = v59 - v108;
      if ( !v59 )
        v63 = -1;
    }
    v103 = v61;
    v109 = v60;
    v64 = memchr(v60, 91, v62);
    v34 = v103;
    if ( v64 )
    {
      v65 = v64 - v109 + 1;
      v66 = v63;
      if ( v65 > v38 )
        v65 = v38;
      if ( v63 < v65 )
        v66 = v65;
      if ( v66 <= v38 )
        v38 = v66;
      v67 = v38 - v65;
      v34 = (__int64)&v109[v65];
    }
    else
    {
      if ( v63 <= v38 )
        v38 = v63;
      v67 = v38;
    }
    goto LABEL_61;
  }
  if ( (v38 & 0x8000000000000000LL) != 0LL )
  {
    v82 = v93;
    v97 = v100;
    v101 = v37;
    v83 = memchr(v82, 40, 0x7FFFFFFFFFFFFFFFuLL);
    v42 = v83 - v101;
    if ( !v83 )
      v42 = -1;
    v84 = memchr(v101, 91, 0x7FFFFFFFFFFFFFFFuLL);
    v85 = v101;
    if ( v84 )
    {
      v86 = v84 - v101 + 1;
      if ( v86 > v38 )
        v86 = v38;
      if ( v42 < v86 )
        v42 = v86;
      v101 += v86;
      if ( v42 > v38 )
        v42 = v38;
      v42 -= v86;
    }
    else if ( v42 > v38 )
    {
      v42 = v38;
    }
    v92 = v97;
    v98 = v85;
    v87 = (char *)memchr(v85, 32, 0x7FFFFFFFFFFFFFFFuLL);
    v47 = v98;
    v48 = (__int64)v92;
    v49 = 0x7FFFFFFFFFFFFFFFLL;
    v50 = (char *)(v87 - v98);
    if ( v87 )
    {
LABEL_39:
      v91 = v48;
      v96 = (unsigned __int64)v50;
      v107 = v47;
      v51 = memchr(v47, 91, v49);
      v52 = 0;
      v53 = v107;
      v54 = v96;
      v55 = v91;
      if ( !v51 )
        goto LABEL_44;
      goto LABEL_40;
    }
  }
  else
  {
    v40 = v93;
    v94 = v100;
    v101 = v37;
    v41 = memchr(v40, 40, v38);
    v42 = v41 - v101;
    if ( !v41 )
      v42 = -1;
    v43 = memchr(v101, 91, v38);
    v44 = v101;
    if ( v43 )
    {
      v45 = v43 - v101 + 1;
      if ( v45 > v38 )
        v45 = v38;
      if ( v42 < v45 )
        v42 = v45;
      v101 += v45;
      if ( v42 > v38 )
        v42 = v38;
      v42 -= v45;
    }
    else if ( v42 > v38 )
    {
      v42 = v38;
    }
    v90 = v94;
    v95 = v44;
    v46 = (char *)memchr(v44, 32, v38);
    v47 = v95;
    v48 = (__int64)v90;
    v49 = v38;
    v50 = (char *)(v46 - v95);
    if ( v46 )
      goto LABEL_39;
  }
  v99 = (__int64)v47;
  v51 = memchr(v47, 91, v49);
  v53 = (_BYTE *)v99;
  v54 = -1;
  if ( v51 )
  {
LABEL_40:
    v56 = v51 - v53 + 1;
    if ( v56 > v38 )
      v56 = v38;
    v52 = v56;
    v55 = (__int64)&v53[v56];
    if ( v54 < v56 )
      v54 = v56;
LABEL_44:
    v57 = v38;
    if ( v54 <= v38 )
      v57 = v54;
    v58 = v55;
    v106 = v57 - v52;
    sub_398FCA0(a1, (__int64)v101, v42);
    if ( !v106 )
      goto LABEL_62;
    goto LABEL_47;
  }
  v58 = v99;
  sub_398FCA0(a1, (__int64)v101, v42);
LABEL_47:
  sub_398FCA0(a1, v58, v106);
LABEL_62:
  v68 = *(unsigned int *)(a2 + 8);
  v69 = *(_QWORD *)(a2 + 8 * (2 - v68));
  if ( v69 )
  {
    v70 = (_BYTE *)sub_161E970(*(_QWORD *)(a2 + 8 * (2 - v68)));
    v72 = v70;
    v73 = v71;
    v69 = (__int64)v70;
    if ( v71 )
    {
      if ( v71 < 0 )
      {
        v113 = (__int64)v70;
        v81 = memchr(v70, 93, 0x7FFFFFFFFFFFFFFFuLL);
        v75 = v113;
        v76 = 0x7FFFFFFFFFFFFFFFLL;
        v77 = v81 - v72;
        if ( !v81 )
          v77 = -1;
      }
      else
      {
        v110 = (__int64)v70;
        v74 = memchr(v70, 93, v71);
        v75 = v110;
        v76 = v73;
        v77 = v74 - v72;
        if ( !v74 )
          v77 = -1;
      }
      v111 = v75;
      v78 = memchr(v72, 32, v76);
      v69 = v111;
      if ( v78 )
      {
        v79 = v78 - v72 + 1;
        if ( v79 > v73 )
          v79 = v73;
        v69 = (__int64)&v72[v79];
        if ( v77 < v79 )
          v77 = v79;
        if ( v77 > v73 )
          v77 = v73;
        v71 = v77 - v79;
      }
      else
      {
        if ( v73 <= v77 )
          v77 = v73;
        v71 = v77;
      }
    }
  }
  else
  {
    v71 = 0;
  }
  sub_398FC80(a1, v69, v71, a3);
}
