// Function: sub_1C65B90
// Address: 0x1c65b90
//
__int64 __fastcall sub_1C65B90(__int64 a1, __int64 *a2, __int64 a3, unsigned int a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v9; // rbx
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 i; // rbx
  __int64 v13; // r12
  __int64 v14; // r13
  __int64 v15; // rax
  __int64 v16; // rax
  _QWORD *v17; // rdx
  __int64 v18; // r11
  char *v19; // r13
  _QWORD *v20; // rdi
  __int64 v21; // rsi
  __int64 v22; // rcx
  unsigned int v23; // esi
  __int64 v24; // r9
  unsigned int v25; // ecx
  char **v26; // rdx
  char *v27; // r8
  int v28; // esi
  __int64 v29; // rax
  __int64 v30; // r8
  int v31; // edx
  __int64 v32; // r9
  int v33; // edx
  __int64 v34; // rdi
  unsigned int v35; // ecx
  __int64 *v36; // rax
  __int64 v37; // r8
  _DWORD *v38; // r15
  unsigned int v39; // r11d
  unsigned __int64 v40; // r12
  size_t v41; // r10
  _DWORD *v42; // rbx
  unsigned __int64 *v43; // r14
  unsigned __int64 v44; // r13
  unsigned int v45; // r15d
  unsigned int v46; // edx
  __int64 **v47; // rdx
  unsigned int *v48; // rsi
  char v49; // al
  int *v50; // rax
  int *v51; // rdx
  unsigned int v52; // eax
  int *v53; // r8
  unsigned int v54; // r9d
  int *v55; // rax
  __int64 v56; // rax
  int *v57; // rax
  __int64 v58; // rdx
  _BOOL8 v59; // rdi
  _QWORD *v61; // rax
  __int64 v62; // r13
  __int64 v63; // r13
  char v64; // al
  char v65; // r8
  __int64 v66; // rax
  unsigned int v67; // esi
  char *v68; // rdx
  __int64 v69; // r15
  unsigned int v70; // r9d
  char **v71; // rax
  char *v72; // rdi
  __int64 v73; // rdi
  _BYTE *v74; // rsi
  signed __int64 v75; // rdx
  __int64 v76; // rax
  __int64 v77; // rcx
  bool v78; // cf
  unsigned __int64 v79; // rax
  __int64 v80; // r12
  __int64 v81; // r12
  __int64 v82; // rax
  char *v83; // rcx
  __int64 v84; // r12
  char *v85; // rax
  int v86; // eax
  char **v87; // rdi
  int v88; // eax
  int v89; // edx
  bool v90; // al
  _QWORD *v91; // rax
  __int64 *v92; // rax
  size_t v93; // r10
  unsigned int v94; // r11d
  __int64 v95; // rdi
  _BYTE *v96; // rsi
  __int64 v97; // rax
  int v98; // r10d
  char **v99; // rcx
  int v100; // eax
  int v101; // r8d
  unsigned int v102; // [rsp+0h] [rbp-150h]
  unsigned int v103; // [rsp+4h] [rbp-14Ch]
  int *v104; // [rsp+8h] [rbp-148h]
  size_t v105; // [rsp+8h] [rbp-148h]
  unsigned int v106; // [rsp+10h] [rbp-140h]
  size_t v107; // [rsp+10h] [rbp-140h]
  int *v108; // [rsp+10h] [rbp-140h]
  __int64 v110; // [rsp+20h] [rbp-130h]
  unsigned int v112; // [rsp+2Ch] [rbp-124h]
  __int64 v115; // [rsp+50h] [rbp-100h]
  _QWORD *v116; // [rsp+58h] [rbp-F8h]
  unsigned int v117; // [rsp+60h] [rbp-F0h]
  size_t v118; // [rsp+60h] [rbp-F0h]
  int *v119; // [rsp+60h] [rbp-F0h]
  unsigned int v120; // [rsp+60h] [rbp-F0h]
  unsigned int v121; // [rsp+60h] [rbp-F0h]
  unsigned int v122; // [rsp+60h] [rbp-F0h]
  unsigned int v123; // [rsp+60h] [rbp-F0h]
  size_t n; // [rsp+68h] [rbp-E8h]
  char *na; // [rsp+68h] [rbp-E8h]
  size_t nb; // [rsp+68h] [rbp-E8h]
  size_t nc; // [rsp+68h] [rbp-E8h]
  size_t nd; // [rsp+68h] [rbp-E8h]
  size_t ne; // [rsp+68h] [rbp-E8h]
  size_t *v130; // [rsp+78h] [rbp-D8h]
  __int64 v131; // [rsp+78h] [rbp-D8h]
  __int64 v132; // [rsp+78h] [rbp-D8h]
  _DWORD *v133; // [rsp+80h] [rbp-D0h]
  int v134; // [rsp+80h] [rbp-D0h]
  __int64 v135; // [rsp+80h] [rbp-D0h]
  void *src; // [rsp+88h] [rbp-C8h]
  __int64 v137; // [rsp+90h] [rbp-C0h]
  __int64 v138; // [rsp+90h] [rbp-C0h]
  __int64 v140; // [rsp+A0h] [rbp-B0h]
  __int64 v141; // [rsp+A0h] [rbp-B0h]
  char *v142; // [rsp+A8h] [rbp-A8h]
  __int64 v143; // [rsp+A8h] [rbp-A8h]
  __int64 v144; // [rsp+A8h] [rbp-A8h]
  char *v145; // [rsp+A8h] [rbp-A8h]
  __int64 v146; // [rsp+B0h] [rbp-A0h]
  unsigned __int64 v147; // [rsp+B0h] [rbp-A0h]
  _QWORD *v148; // [rsp+B8h] [rbp-98h]
  unsigned __int64 v149; // [rsp+B8h] [rbp-98h]
  __int64 v150; // [rsp+C0h] [rbp-90h] BYREF
  __int64 v151; // [rsp+C8h] [rbp-88h] BYREF
  char *v152; // [rsp+D0h] [rbp-80h] BYREF
  _QWORD *v153; // [rsp+D8h] [rbp-78h] BYREF
  char *v154; // [rsp+E0h] [rbp-70h] BYREF
  char **v155; // [rsp+E8h] [rbp-68h] BYREF
  __int64 v156; // [rsp+F0h] [rbp-60h] BYREF
  int v157; // [rsp+F8h] [rbp-58h] BYREF
  int *v158; // [rsp+100h] [rbp-50h]
  int *v159; // [rsp+108h] [rbp-48h]
  int *v160; // [rsp+110h] [rbp-40h]
  __int64 v161; // [rsp+118h] [rbp-38h]

  v6 = a2[1] - *a2;
  v112 = dword_4FBC920;
  v159 = &v157;
  v160 = &v157;
  v7 = v6 >> 3;
  v157 = 0;
  v158 = 0;
  v161 = 0;
  if ( (unsigned __int64)v6 > 0x1FFFFFFFFFFFFFF8LL )
    sub_4262D8((__int64)"vector::reserve");
  if ( !v7 )
    return sub_1C514B0((__int64)v158);
  v9 = 32 * v7;
  v10 = sub_22077B0(32 * v7);
  v146 = v9 + v10;
  v11 = *a2;
  v140 = (a2[1] - *a2) >> 3;
  if ( !v140 )
  {
    v110 = v9;
    src = (void *)v10;
    goto LABEL_68;
  }
  src = (void *)v10;
  v137 = a1 + 32;
  v148 = (_QWORD *)(a1 + 72);
  for ( i = 0; i != v140; ++i )
  {
    v13 = *(_QWORD *)(v11 + 8 * i);
    v14 = *(_QWORD *)(a1 + 184);
    v154 = *(char **)v13;
    v15 = sub_1456040((__int64)v154);
    v16 = sub_1456C90(v14, v15);
    v17 = *(_QWORD **)(a1 + 80);
    v18 = v16;
    v19 = v154;
    if ( v17 )
    {
      v20 = v148;
      do
      {
        while ( 1 )
        {
          v21 = v17[2];
          v22 = v17[3];
          if ( v17[4] >= (unsigned __int64)v154 )
            break;
          v17 = (_QWORD *)v17[3];
          if ( !v22 )
            goto LABEL_10;
        }
        v20 = v17;
        v17 = (_QWORD *)v17[2];
      }
      while ( v21 );
LABEL_10:
      if ( v20 != v148 && v20[4] <= (unsigned __int64)v154 )
        v19 = (char *)v20[5];
    }
    v23 = *(_DWORD *)(a1 + 56);
    if ( v23 )
    {
      v24 = *(_QWORD *)(a1 + 40);
      v142 = v154;
      v25 = (v23 - 1) & (((unsigned int)v154 >> 9) ^ ((unsigned int)v154 >> 4));
      v26 = (char **)(v24 + 16LL * v25);
      v27 = *v26;
      if ( v154 == *v26 )
      {
        v28 = *((_DWORD *)v26 + 2);
        goto LABEL_16;
      }
      v134 = 1;
      v87 = 0;
      while ( v27 != (char *)-8LL )
      {
        if ( v27 != (char *)-16LL || v87 )
          v26 = v87;
        v25 = (v23 - 1) & (v134 + v25);
        v132 = v24 + 16LL * v25;
        v27 = *(char **)v132;
        if ( v154 == *(char **)v132 )
        {
          v28 = *(_DWORD *)(v132 + 8);
          goto LABEL_16;
        }
        ++v134;
        v87 = v26;
        v26 = (char **)(v24 + 16LL * v25);
      }
      v88 = *(_DWORD *)(a1 + 48);
      if ( !v87 )
        v87 = v26;
      ++*(_QWORD *)(a1 + 32);
      v89 = v88 + 1;
      if ( 4 * (v88 + 1) < 3 * v23 )
      {
        if ( v23 - *(_DWORD *)(a1 + 52) - v89 > v23 >> 3 )
          goto LABEL_120;
        v135 = v18;
        goto LABEL_131;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 32);
    }
    v135 = v18;
    v23 *= 2;
LABEL_131:
    sub_1468630(v137, v23);
    sub_145FB10(v137, (__int64 *)&v154, &v155);
    v87 = v155;
    v18 = v135;
    v142 = v154;
    v89 = *(_DWORD *)(a1 + 48) + 1;
LABEL_120:
    *(_DWORD *)(a1 + 48) = v89;
    if ( *v87 != (char *)-8LL )
      --*(_DWORD *)(a1 + 52);
    *((_DWORD *)v87 + 2) = 0;
    v28 = 0;
    *v87 = v142;
LABEL_16:
    v29 = *(_QWORD *)(a1 + 192);
    v30 = 0;
    v31 = *(_DWORD *)(v29 + 24);
    if ( v31 )
    {
      v32 = *(_QWORD *)(v29 + 8);
      v33 = v31 - 1;
      v34 = *(_QWORD *)(*(_QWORD *)(***(_QWORD ***)(v13 + 8) + 16LL) + 40LL);
      v35 = v33 & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
      v36 = (__int64 *)(v32 + 16LL * v35);
      v37 = *v36;
      if ( v34 == *v36 )
      {
LABEL_18:
        v30 = v36[1];
      }
      else
      {
        v86 = 1;
        while ( v37 != -8 )
        {
          v98 = v86 + 1;
          v35 = v33 & (v86 + v35);
          v36 = (__int64 *)(v32 + 16LL * v35);
          v37 = *v36;
          if ( v34 == *v36 )
            goto LABEL_18;
          v86 = v98;
        }
        v30 = 0;
      }
    }
    if ( v146 == v10 )
    {
      v75 = v146 - (_QWORD)src;
      v76 = (v146 - (__int64)src) >> 5;
      if ( v76 == 0x3FFFFFFFFFFFFFFLL )
        sub_4262D8((__int64)"vector::_M_realloc_insert");
      v77 = 1;
      if ( v76 )
        v77 = (v146 - (__int64)src) >> 5;
      v78 = __CFADD__(v77, v76);
      v79 = v77 + v76;
      if ( v78 )
      {
        v81 = 0x7FFFFFFFFFFFFFE0LL;
      }
      else
      {
        if ( !v79 )
        {
          v84 = 0;
          v83 = 0;
LABEL_104:
          v85 = &v83[v75];
          if ( &v83[v75] )
          {
            *(_QWORD *)v85 = v18;
            *((_QWORD *)v85 + 1) = v19;
            *((_QWORD *)v85 + 2) = v30;
            *((_DWORD *)v85 + 6) = v28;
          }
          v10 = (__int64)&v83[v75 + 32];
          if ( v75 > 0 )
          {
            v83 = (char *)memmove(v83, src, v75);
          }
          else if ( !src )
          {
LABEL_108:
            v146 = v84;
            src = v83;
            goto LABEL_23;
          }
          v145 = v83;
          j_j___libc_free_0(src, v146 - (_QWORD)src);
          v83 = v145;
          goto LABEL_108;
        }
        v80 = 0x3FFFFFFFFFFFFFFLL;
        if ( v79 <= 0x3FFFFFFFFFFFFFFLL )
          v80 = v79;
        v81 = 32 * v80;
      }
      v131 = v30;
      v144 = v18;
      v82 = sub_22077B0(v81);
      v18 = v144;
      v30 = v131;
      v75 = v146 - (_QWORD)src;
      v83 = (char *)v82;
      v84 = v82 + v81;
      goto LABEL_104;
    }
    if ( v10 )
    {
      *(_QWORD *)v10 = v18;
      *(_QWORD *)(v10 + 8) = v19;
      *(_QWORD *)(v10 + 16) = v30;
      *(_DWORD *)(v10 + 24) = v28;
    }
    v10 += 32;
LABEL_23:
    v11 = *a2;
  }
  v110 = v146 - (_QWORD)src;
  v149 = (a2[1] - v11) >> 3;
  if ( !v149 )
    goto LABEL_68;
  v116 = (_QWORD *)a1;
  v38 = src;
  v147 = 0;
  while ( 2 )
  {
    v39 = v38[6];
    v40 = ++v147;
    if ( v39 <= 0x3E8 )
    {
      if ( v149 <= v147 )
        goto LABEL_68;
      v133 = v38;
      v115 = 0;
      v141 = *(_QWORD *)v38;
      v143 = *((_QWORD *)v38 + 1);
      v138 = *((_QWORD *)v38 + 2);
      v130 = *(size_t **)(v11 + 8 * v147 - 8);
      v41 = *v130;
      v42 = v38;
LABEL_31:
      v43 = *(unsigned __int64 **)(v11 + 8 * v40);
      v44 = *v43;
      if ( v41 == *v43 || v143 == *((_QWORD *)v42 + 5) && (*((_QWORD *)v42 + 6) != v138 || !v138) )
        goto LABEL_90;
      if ( v141 != *((_QWORD *)v42 + 4) )
        goto LABEL_90;
      v45 = v42[14];
      if ( v45 <= 3 && v39 <= 3 || v45 > 0x3E8 )
        goto LABEL_90;
      v46 = v45 - v39;
      if ( v45 < v39 )
        v46 = v39 - v45;
      if ( v45 > v39 )
        v45 = v39;
      if ( v46 > v45 )
        goto LABEL_90;
      v47 = (__int64 **)v43[1];
      v117 = v39;
      n = v41;
      v48 = (unsigned int *)v130[1];
      v150 = 0;
      v151 = 0;
      v49 = sub_1C620D0(v116, v48, v47, &v150, (unsigned __int64 *)&v151, a3, 0);
      v41 = n;
      v39 = v117;
      if ( !v49 )
        goto LABEL_89;
      v50 = v158;
      if ( !v158 )
        goto LABEL_179;
      v51 = &v157;
      do
      {
        while ( v44 <= *((_QWORD *)v50 + 4) && (v44 != *((_QWORD *)v50 + 4) || n <= *((_QWORD *)v50 + 5)) )
        {
          v51 = v50;
          v50 = (int *)*((_QWORD *)v50 + 2);
          if ( !v50 )
            goto LABEL_49;
        }
        v50 = (int *)*((_QWORD *)v50 + 3);
      }
      while ( v50 );
LABEL_49:
      if ( v51 == &v157 || v44 < *((_QWORD *)v51 + 4) || v44 == *((_QWORD *)v51 + 4) && n < *((_QWORD *)v51 + 5) )
      {
LABEL_179:
        if ( !v115 )
        {
          v97 = sub_1480620(v116[23], n, 0);
          v39 = v117;
          v41 = n;
          v115 = v97;
        }
        v106 = v39;
        v118 = v41;
        v152 = (char *)sub_13A5B00(v116[23], v44, v115, 0, 0);
        v52 = sub_1CCB2B0(v116[23], v152);
        v53 = &v157;
        v41 = v118;
        v54 = v52;
        v39 = v106;
        na = v152;
        v55 = v158;
        if ( !v158 )
          goto LABEL_62;
        do
        {
          while ( v44 <= *((_QWORD *)v55 + 4) && (v44 != *((_QWORD *)v55 + 4) || v118 <= *((_QWORD *)v55 + 5)) )
          {
            v53 = v55;
            v55 = (int *)*((_QWORD *)v55 + 2);
            if ( !v55 )
              goto LABEL_60;
          }
          v55 = (int *)*((_QWORD *)v55 + 3);
        }
        while ( v55 );
LABEL_60:
        if ( v53 == &v157 || v44 < *((_QWORD *)v53 + 4) || v44 == *((_QWORD *)v53 + 4) && v118 < *((_QWORD *)v53 + 5) )
        {
LABEL_62:
          v102 = v106;
          v103 = v54;
          v104 = v53;
          v56 = sub_22077B0(64);
          *(_QWORD *)(v56 + 32) = v44;
          *(_QWORD *)(v56 + 40) = v118;
          *(_QWORD *)(v56 + 48) = 0;
          *(_DWORD *)(v56 + 56) = 0;
          v107 = v118;
          v119 = (int *)v56;
          v57 = (int *)sub_1C56FB0(&v156, v104, (unsigned __int64 *)(v56 + 32));
          if ( v58 )
          {
            v59 = 1;
            if ( !v57 && &v157 != (int *)v58 && v44 >= *(_QWORD *)(v58 + 32) )
            {
              v59 = 0;
              if ( v44 == *(_QWORD *)(v58 + 32) )
                v59 = v107 < *(_QWORD *)(v58 + 40);
            }
            sub_220F040(v59, v119, v58, &v157);
            ++v161;
            v53 = v119;
            v54 = v103;
            v41 = v107;
            v39 = v102;
          }
          else
          {
            v105 = v107;
            v108 = v57;
            j_j___libc_free_0(v119, 64);
            v39 = v102;
            v41 = v105;
            v54 = v103;
            v53 = v108;
          }
        }
        v53[14] = v54;
        *((_QWORD *)v53 + 6) = na;
      }
      else
      {
        v54 = v51[14];
        v152 = (char *)*((_QWORD *)v51 + 6);
      }
      if ( v112 <= a4 && v54 > 8 )
        goto LABEL_89;
      if ( v45 <= v54 )
        goto LABEL_89;
      if ( *(_BYTE *)(v151 + 16) == 18 && v151 != *(_QWORD *)(*(_QWORD *)(**(_QWORD **)v43[1] + 16LL) + 40LL) )
      {
        v121 = v39;
        nc = v41;
        v90 = sub_146D920(v116[23], (__int64)v152, v151);
        v41 = nc;
        v39 = v121;
        if ( !v90 )
          goto LABEL_89;
      }
      v120 = v39;
      nb = v41;
      v61 = (_QWORD *)sub_22077B0(16);
      if ( v61 )
      {
        *v61 = v130[1];
        v61[1] = v43[1];
      }
      v153 = v61;
      v62 = *(unsigned int *)(a5 + 24);
      v154 = v152;
      v63 = *(_QWORD *)(a5 + 8) + 16 * v62;
      v64 = sub_1C506F0(a5, (__int64 *)&v154, &v155);
      v41 = nb;
      v39 = v120;
      v65 = v64;
      v66 = (__int64)v155;
      if ( !v65 )
        v66 = *(_QWORD *)(a5 + 8) + 16LL * *(unsigned int *)(a5 + 24);
      if ( v63 == v66 )
      {
        v91 = (_QWORD *)sub_22077B0(24);
        if ( v91 )
        {
          *v91 = 0;
          v91[1] = 0;
          v91[2] = 0;
        }
        sub_1C53600(a5, (__int64 *)&v152)[1] = (__int64)v91;
        v92 = sub_1C53600(a5, (__int64 *)&v152);
        v93 = nb;
        v94 = v120;
        v95 = v92[1];
        v96 = *(_BYTE **)(v95 + 8);
        if ( v96 == *(_BYTE **)(v95 + 16) )
        {
          sub_1C50D80(v95, v96, &v153);
          v94 = v120;
          v93 = nb;
        }
        else
        {
          if ( v96 )
          {
            *(_QWORD *)v96 = v153;
            v96 = *(_BYTE **)(v95 + 8);
          }
          *(_QWORD *)(v95 + 8) = v96 + 8;
        }
        v122 = v94;
        nd = v93;
        sub_1C55CE0(a6, &v152);
        v41 = nd;
        v39 = v122;
        v11 = *a2;
        goto LABEL_90;
      }
      v67 = *(_DWORD *)(a5 + 24);
      if ( v67 )
      {
        v68 = v152;
        v69 = *(_QWORD *)(a5 + 8);
        v70 = (v67 - 1) & (((unsigned int)v152 >> 9) ^ ((unsigned int)v152 >> 4));
        v71 = (char **)(v69 + 16LL * v70);
        v72 = *v71;
        if ( v152 == *v71 )
        {
LABEL_84:
          v73 = (__int64)v71[1];
          goto LABEL_85;
        }
        v101 = 1;
        v99 = 0;
        while ( v72 != (char *)-8LL )
        {
          if ( !v99 && v72 == (char *)-16LL )
            v99 = v71;
          v70 = (v67 - 1) & (v101 + v70);
          v71 = (char **)(v69 + 16LL * v70);
          v72 = *v71;
          if ( v152 == *v71 )
            goto LABEL_84;
          ++v101;
        }
        if ( !v99 )
          v99 = v71;
        ++*(_QWORD *)a5;
        v100 = *(_DWORD *)(a5 + 16) + 1;
        if ( 4 * v100 < 3 * v67 )
        {
          if ( v67 - *(_DWORD *)(a5 + 20) - v100 <= v67 >> 3 )
          {
            sub_1C53450(a5, v67);
            sub_1C506F0(a5, (__int64 *)&v152, &v155);
            v99 = v155;
            v68 = v152;
            v39 = v120;
            v41 = nb;
            v100 = *(_DWORD *)(a5 + 16) + 1;
          }
          goto LABEL_153;
        }
      }
      else
      {
        ++*(_QWORD *)a5;
      }
      sub_1C53450(a5, 2 * v67);
      sub_1C506F0(a5, (__int64 *)&v152, &v155);
      v99 = v155;
      v68 = v152;
      v41 = nb;
      v39 = v120;
      v100 = *(_DWORD *)(a5 + 16) + 1;
LABEL_153:
      *(_DWORD *)(a5 + 16) = v100;
      if ( *v99 != (char *)-8LL )
        --*(_DWORD *)(a5 + 20);
      *v99 = v68;
      v73 = 0;
      v99[1] = 0;
LABEL_85:
      v74 = *(_BYTE **)(v73 + 8);
      if ( v74 == *(_BYTE **)(v73 + 16) )
      {
        v123 = v39;
        ne = v41;
        sub_1C50D80(v73, v74, &v153);
        v39 = v123;
        v41 = ne;
      }
      else
      {
        if ( v74 )
        {
          *(_QWORD *)v74 = v153;
          v74 = *(_BYTE **)(v73 + 8);
        }
        *(_QWORD *)(v73 + 8) = v74 + 8;
      }
LABEL_89:
      v11 = *a2;
LABEL_90:
      ++v40;
      v42 += 8;
      if ( v40 >= v149 )
      {
        v38 = v133;
LABEL_27:
        v38 += 8;
        continue;
      }
      goto LABEL_31;
    }
    break;
  }
  if ( v149 > v147 )
    goto LABEL_27;
LABEL_68:
  if ( src )
    j_j___libc_free_0(src, v110);
  return sub_1C514B0((__int64)v158);
}
