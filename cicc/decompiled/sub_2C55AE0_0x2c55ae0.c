// Function: sub_2C55AE0
// Address: 0x2c55ae0
//
__int64 __fastcall sub_2C55AE0(__int64 a1, _BYTE *a2)
{
  __int64 v2; // r12
  __int64 v3; // rax
  unsigned int v4; // r10d
  unsigned __int8 v6; // al
  _BYTE *v7; // rsi
  char v9; // bl
  _DWORD *v10; // rdi
  int *v11; // r8
  bool v12; // r9
  __int64 v13; // rax
  __int64 v14; // rdx
  char v15; // r11
  __int64 v16; // rdx
  char v17; // r10
  __int64 v18; // rax
  int v19; // r13d
  int *v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // rax
  int *v23; // rbx
  int *v24; // rax
  __int64 v25; // rdx
  unsigned __int64 v26; // rcx
  char v27; // r14
  int v28; // r12d
  int *v29; // r13
  __int64 v30; // r9
  __int64 v31; // r15
  __int64 v32; // rdx
  __int64 v33; // rdx
  signed int v34; // r13d
  __int64 *v35; // rdi
  __int64 v36; // r9
  __int64 v37; // rax
  int v38; // edx
  int v39; // ebx
  __int64 v40; // r13
  __int64 v41; // rax
  int v42; // edx
  int v43; // r8d
  int v44; // edx
  char v45; // r10
  bool v46; // of
  unsigned __int64 v47; // rax
  __int64 *v48; // rdi
  __int64 v49; // r9
  __int64 v50; // rax
  int v51; // edx
  bool v52; // zf
  int v53; // edx
  __int64 *v54; // rdi
  __int64 v55; // r9
  __int64 v56; // rax
  int v57; // edx
  int v58; // edx
  __int64 v59; // rax
  signed __int64 v60; // rbx
  int v61; // edx
  int v62; // r13d
  unsigned __int8 *v63; // rax
  __int64 v64; // r13
  __int64 v65; // r14
  __int64 i; // rbx
  __int64 v67; // rsi
  __int64 v68; // rdx
  __int64 v69; // rdi
  int *v70; // r13
  __int64 v71; // rbx
  __int64 (__fastcall *v72)(__int64, __int64, __int64, __int64 *, _QWORD); // rax
  __int64 v73; // rax
  __int64 v74; // rdi
  _DWORD *v75; // r13
  __int64 v76; // rbx
  __int64 (__fastcall *v77)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64); // rax
  __int64 v78; // rax
  __int64 *v79; // rdi
  __int64 v80; // r9
  __int64 v81; // rax
  int v82; // edx
  __int64 *v83; // rdi
  __int64 v84; // r9
  __int64 v85; // rax
  int v86; // edx
  int *v87; // rcx
  int *v88; // rax
  char v89; // al
  __int64 v90; // rsi
  int *v91; // rdi
  char *v92; // rcx
  signed int *v93; // rax
  char v94; // al
  signed __int64 v95; // rdx
  _QWORD *v96; // rax
  __int64 v97; // r13
  __int64 v98; // r12
  __int64 v99; // r13
  __int64 v100; // rbx
  __int64 v101; // rdx
  unsigned int v102; // esi
  _QWORD *v103; // rax
  __int64 v104; // rdx
  __int64 v105; // r13
  __int64 v106; // r12
  __int64 v107; // r13
  __int64 v108; // rbx
  __int64 v109; // rdx
  unsigned int v110; // esi
  signed __int64 v111; // rdx
  bool v112; // cc
  unsigned __int64 v113; // rax
  unsigned __int64 v114; // rax
  __int64 v115; // [rsp-218h] [rbp-218h]
  __int64 v116; // [rsp-218h] [rbp-218h]
  int v117; // [rsp-1FCh] [rbp-1FCh]
  char v118; // [rsp-1F8h] [rbp-1F8h]
  __int64 v119; // [rsp-1F0h] [rbp-1F0h]
  char v120; // [rsp-1F0h] [rbp-1F0h]
  char v121; // [rsp-1F0h] [rbp-1F0h]
  char v122; // [rsp-1F0h] [rbp-1F0h]
  __int64 v123; // [rsp-1E8h] [rbp-1E8h]
  char v124; // [rsp-1E8h] [rbp-1E8h]
  int *v125; // [rsp-1E8h] [rbp-1E8h]
  char v126; // [rsp-1E8h] [rbp-1E8h]
  char v127; // [rsp-1E0h] [rbp-1E0h]
  int *v128; // [rsp-1E0h] [rbp-1E0h]
  int *v129; // [rsp-1D8h] [rbp-1D8h]
  char v130; // [rsp-1D8h] [rbp-1D8h]
  _BYTE *v131; // [rsp-1D0h] [rbp-1D0h]
  __int64 v132; // [rsp-1B8h] [rbp-1B8h]
  __int64 v133; // [rsp-1B0h] [rbp-1B0h]
  __int64 v134; // [rsp-1A8h] [rbp-1A8h]
  __int64 v135; // [rsp-1A0h] [rbp-1A0h]
  __int64 v136; // [rsp-198h] [rbp-198h]
  __int64 v137; // [rsp-190h] [rbp-190h]
  __int64 v138; // [rsp-188h] [rbp-188h]
  signed __int64 v139; // [rsp-188h] [rbp-188h]
  __int64 v140; // [rsp-180h] [rbp-180h]
  int v141; // [rsp-178h] [rbp-178h]
  unsigned int v142; // [rsp-178h] [rbp-178h]
  char v143; // [rsp-171h] [rbp-171h]
  char v144; // [rsp-171h] [rbp-171h]
  unsigned __int8 v145; // [rsp-171h] [rbp-171h]
  __int64 v146; // [rsp-170h] [rbp-170h]
  __int64 v147; // [rsp-168h] [rbp-168h]
  unsigned __int8 v148; // [rsp-168h] [rbp-168h]
  _QWORD *v149; // [rsp-168h] [rbp-168h]
  _QWORD *v150; // [rsp-168h] [rbp-168h]
  __int64 v151; // [rsp-168h] [rbp-168h]
  __int64 v152; // [rsp-160h] [rbp-160h]
  int v153; // [rsp-160h] [rbp-160h]
  __int64 v154; // [rsp-158h] [rbp-158h]
  _QWORD *v155; // [rsp-158h] [rbp-158h]
  _QWORD *v156; // [rsp-158h] [rbp-158h]
  __int64 v157; // [rsp-158h] [rbp-158h]
  __int64 v158; // [rsp-150h] [rbp-150h]
  unsigned __int8 v159; // [rsp-150h] [rbp-150h]
  unsigned __int8 v160; // [rsp-150h] [rbp-150h]
  int v161[8]; // [rsp-148h] [rbp-148h] BYREF
  __int16 v162; // [rsp-128h] [rbp-128h]
  int **v163; // [rsp-118h] [rbp-118h] BYREF
  _DWORD **v164; // [rsp-110h] [rbp-110h]
  __int16 v165; // [rsp-F8h] [rbp-F8h]
  _DWORD *v166; // [rsp-E8h] [rbp-E8h] BYREF
  __int64 v167; // [rsp-E0h] [rbp-E0h]
  _DWORD v168[16]; // [rsp-D8h] [rbp-D8h] BYREF
  __int64 *v169; // [rsp-98h] [rbp-98h] BYREF
  unsigned __int64 v170; // [rsp-90h] [rbp-90h]
  __int64 v171; // [rsp-88h] [rbp-88h] BYREF
  int v172; // [rsp-80h] [rbp-80h]
  char v173; // [rsp-7Ch] [rbp-7Ch]
  __int64 v174; // [rsp-78h] [rbp-78h] BYREF

  if ( *a2 != 92 )
    return 0;
  v2 = (__int64)a2;
  v158 = *((_QWORD *)a2 - 8);
  v3 = *(_QWORD *)(v158 + 16);
  if ( !v3 )
    return 0;
  if ( *(_QWORD *)(v3 + 8) )
    return 0;
  v6 = *(_BYTE *)v158;
  if ( (unsigned __int8)(*(_BYTE *)v158 - 42) > 0x11u )
    return 0;
  v7 = (_BYTE *)*((_QWORD *)a2 - 4);
  if ( (unsigned __int8)(*v7 - 12) > 1u )
  {
    if ( (unsigned __int8)(*v7 - 9) > 2u )
      return 0;
    v169 = 0;
    v170 = (unsigned __int64)&v174;
    v166 = v168;
    v167 = 0x800000000LL;
    v171 = 8;
    v172 = 0;
    v173 = 1;
    v163 = (int **)&v169;
    v164 = &v166;
    v9 = sub_AA8FD0(&v163, (__int64)v7);
    if ( v9 )
    {
      while ( 1 )
      {
        v10 = v166;
        if ( !(_DWORD)v167 )
          break;
        v67 = *(_QWORD *)&v166[2 * (unsigned int)v167 - 2];
        LODWORD(v167) = v167 - 1;
        if ( !(unsigned __int8)sub_AA8FD0(&v163, v67) )
          goto LABEL_11;
      }
    }
    else
    {
LABEL_11:
      v10 = v166;
      v9 = 0;
    }
    if ( v10 != v168 )
      _libc_free((unsigned __int64)v10);
    if ( !v173 )
      _libc_free(v170);
    if ( !v9 )
      return 0;
    v6 = *(_BYTE *)v158;
  }
  v11 = *(int **)(v2 + 72);
  v141 = v6;
  v152 = *(unsigned int *)(v2 + 80);
  v12 = (unsigned int)v6 - 48 <= 1 || (unsigned __int8)(v6 - 51) <= 1u;
  if ( v12 && &v11[v152] != sub_2C4D170(v11, (__int64)&v11[v152], &dword_43A18EC) )
    return 0;
  v133 = 0;
  v135 = 0;
  v13 = *(_QWORD *)(v158 - 64);
  v14 = *(_QWORD *)(v13 + 16);
  if ( !v14 || (v140 = *(_QWORD *)(v14 + 8)) != 0 )
  {
    v15 = 0;
    v140 = 0;
    v16 = *(_QWORD *)(*(_QWORD *)(v158 - 32) + 16LL);
    v154 = *(_QWORD *)(v158 - 32);
    if ( !v16 )
      return 0;
  }
  else
  {
    v15 = 0;
    if ( *(_BYTE *)v13 == 92 && *(_QWORD *)(v13 - 64) )
    {
      if ( *(_QWORD *)(v13 - 32) )
      {
        v131 = *(_BYTE **)(v13 - 32);
        v15 = 1;
        v136 = *(_QWORD *)(v13 - 64);
        v140 = *(_QWORD *)(v13 + 72);
        v133 = *(unsigned int *)(v13 + 80);
      }
      else
      {
        v136 = *(_QWORD *)(v13 - 64);
      }
    }
    v16 = *(_QWORD *)(*(_QWORD *)(v158 - 32) + 16LL);
    v154 = *(_QWORD *)(v158 - 32);
    if ( !v16 )
      goto LABEL_23;
  }
  if ( !*(_QWORD *)(v16 + 8) && *(_BYTE *)v154 == 92 )
  {
    v146 = *(_QWORD *)(v154 - 64);
    if ( v146 )
    {
      if ( *(_QWORD *)(v154 - 32) )
      {
        v134 = *(_QWORD *)(v154 + 72);
        v135 = *(unsigned int *)(v154 + 80);
        if ( v15 )
        {
          v143 = v15;
          v154 = *(_QWORD *)(v154 - 32);
        }
        else
        {
          v136 = *(_QWORD *)(v158 - 64);
          v131 = (_BYTE *)v136;
          v154 = *(_QWORD *)(v154 - 32);
          v143 = 1;
        }
        goto LABEL_25;
      }
    }
  }
LABEL_23:
  if ( !v15 )
    return 0;
  v143 = 0;
  v134 = 0;
  v146 = v154;
LABEL_25:
  v17 = 0;
  v137 = *(_QWORD *)(v2 + 8);
  if ( *(_BYTE *)(v137 + 8) != 17 )
  {
    v137 = 0;
    v17 = 1;
  }
  v138 = *(_QWORD *)(v158 + 8);
  if ( *(_BYTE *)(v138 + 8) != 17 )
  {
    v138 = 0;
    v17 = 1;
  }
  v18 = *(_QWORD *)(v136 + 8);
  if ( *(_BYTE *)(v18 + 8) != 17 )
    v18 = 0;
  v132 = *(_QWORD *)(v146 + 8);
  v147 = v18;
  if ( *(_BYTE *)(v132 + 8) != 17 || v17 || !v18 )
    return 0;
  v19 = *(_DWORD *)(v138 + 32);
  if ( !v12 )
  {
    v68 = (*(_BYTE *)(v2 + 7) & 0x40) != 0 ? *(_QWORD *)(v2 - 8) : v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF);
    if ( **(_BYTE **)(v68 + 32) == 13 )
    {
      v23 = &v11[v152];
      goto LABEL_43;
    }
  }
  v20 = &v11[v152];
  v21 = (4 * v152) >> 2;
  v22 = (4 * v152) >> 4;
  if ( v22 )
  {
    v23 = v11;
    v24 = &v11[4 * v22];
    while ( 1 )
    {
      if ( v19 <= *v23 )
        goto LABEL_42;
      if ( v19 <= v23[1] )
      {
        ++v23;
        goto LABEL_42;
      }
      if ( v19 <= v23[2] )
      {
        v23 += 2;
        goto LABEL_42;
      }
      if ( v19 <= v23[3] )
        break;
      v23 += 4;
      if ( v24 == v23 )
      {
        v21 = v20 - v23;
        goto LABEL_152;
      }
    }
    v23 += 3;
LABEL_42:
    if ( v20 == v23 )
      goto LABEL_43;
    return 0;
  }
  v23 = v11;
LABEL_152:
  if ( v21 != 2 )
  {
    if ( v21 != 3 )
    {
      if ( v21 != 1 )
        goto LABEL_155;
      goto LABEL_178;
    }
    if ( *v23 >= v19 )
      goto LABEL_42;
    ++v23;
  }
  if ( *v23 >= v19 )
    goto LABEL_42;
  ++v23;
LABEL_178:
  if ( v19 <= *v23 )
    goto LABEL_42;
LABEL_155:
  v23 = &v11[v152];
LABEL_43:
  v166 = v168;
  v167 = 0xC00000000LL;
  v169 = &v171;
  v170 = 0xC00000000LL;
  if ( v11 == v23 )
  {
    v34 = *(_DWORD *)(v147 + 32);
    if ( v147 != v137 )
    {
LABEL_63:
      v130 = 0;
      goto LABEL_64;
    }
    v91 = v168;
    v90 = (unsigned int)v167;
    v93 = v168;
    v92 = (char *)&v168[(unsigned int)v167];
LABEL_182:
    v95 = v92 - (char *)v93;
    if ( v92 - (char *)v93 != 8 )
    {
      if ( v95 != 12 )
      {
        if ( v95 != 4 )
          goto LABEL_173;
        goto LABEL_185;
      }
      if ( v34 <= *v93 )
        goto LABEL_172;
      ++v93;
    }
    if ( v34 <= *v93 )
      goto LABEL_172;
    ++v93;
LABEL_185:
    if ( v34 <= *v93 )
      goto LABEL_172;
    goto LABEL_173;
  }
  v123 = a1;
  v25 = 0;
  v26 = 12;
  v119 = v2;
  v27 = v15;
  v28 = v19;
  v29 = v11;
  v129 = v11;
  while ( 1 )
  {
    v31 = *v29;
    if ( (int)v31 >= 0 && (int)v31 < v28 )
      break;
    if ( v25 + 1 > v26 )
    {
      sub_C8D5F0((__int64)&v166, v168, v25 + 1, 4u, (__int64)v11, v25 + 1);
      v25 = (unsigned int)v167;
    }
    v166[v25] = -1;
    v33 = (unsigned int)v170;
    LODWORD(v167) = v167 + 1;
    if ( (unsigned __int64)(unsigned int)v170 + 1 > HIDWORD(v170) )
    {
      sub_C8D5F0((__int64)&v169, &v171, (unsigned int)v170 + 1LL, 4u, (__int64)v11, (unsigned int)v170 + 1LL);
      v33 = (unsigned int)v170;
    }
    ++v29;
    *((_DWORD *)v169 + v33) = -1;
    LODWORD(v170) = v170 + 1;
    if ( v23 == v29 )
      goto LABEL_62;
LABEL_54:
    v25 = (unsigned int)v167;
    v26 = HIDWORD(v167);
  }
  v30 = (unsigned int)v31;
  if ( v27 )
    v30 = *(unsigned int *)(v140 + 4LL * (int)v31);
  v11 = (int *)(v25 + 1);
  if ( v25 + 1 > v26 )
  {
    v117 = v30;
    sub_C8D5F0((__int64)&v166, v168, v25 + 1, 4u, (__int64)v11, v30);
    v25 = (unsigned int)v167;
    LODWORD(v30) = v117;
  }
  v166[v25] = v30;
  LODWORD(v167) = v167 + 1;
  if ( v143 )
    LODWORD(v31) = *(_DWORD *)(v134 + 4 * v31);
  v32 = (unsigned int)v170;
  if ( (unsigned __int64)(unsigned int)v170 + 1 > HIDWORD(v170) )
  {
    sub_C8D5F0((__int64)&v169, &v171, (unsigned int)v170 + 1LL, 4u, (__int64)v11, (unsigned int)v170 + 1LL);
    v32 = (unsigned int)v170;
  }
  ++v29;
  *((_DWORD *)v169 + v32) = v31;
  LODWORD(v170) = v170 + 1;
  if ( v23 != v29 )
    goto LABEL_54;
LABEL_62:
  v15 = v27;
  v11 = v129;
  v17 = 0;
  a1 = v123;
  v2 = v119;
  v34 = *(_DWORD *)(v147 + 32);
  if ( v147 != v137 )
    goto LABEL_63;
  v90 = (unsigned int)v167;
  v91 = v166;
  v92 = (char *)&v166[(unsigned int)v167];
  if ( !((4LL * (unsigned int)v167) >> 4) )
  {
    v93 = v166;
    goto LABEL_182;
  }
  v93 = v166;
  while ( *v93 < v34 )
  {
    if ( v93[1] >= v34 )
    {
      ++v93;
      break;
    }
    if ( v93[2] >= v34 )
    {
      v93 += 2;
      break;
    }
    if ( v93[3] >= v34 )
    {
      v93 += 3;
      break;
    }
    v93 += 4;
    if ( &v166[4 * ((4LL * (unsigned int)v167) >> 4)] == v93 )
      goto LABEL_182;
  }
LABEL_172:
  if ( v92 != (char *)v93 )
    goto LABEL_63;
LABEL_173:
  v122 = v17;
  v126 = v15;
  v128 = v11;
  v94 = sub_B4ED80(v91, v90, v34);
  v17 = v122;
  v15 = v126;
  v130 = v94;
  v11 = v128;
LABEL_64:
  if ( v132 != v137 )
  {
LABEL_65:
    v127 = 0;
    goto LABEL_66;
  }
  v87 = (int *)v169 + (unsigned int)v170;
  if ( (4LL * (unsigned int)v170) >> 4 )
  {
    v88 = (int *)v169;
    while ( v34 > *v88 )
    {
      if ( v34 <= v88[1] )
      {
        ++v88;
        goto LABEL_163;
      }
      if ( v34 <= v88[2] )
      {
        v88 += 2;
        goto LABEL_163;
      }
      if ( v34 <= v88[3] )
      {
        v88 += 3;
        goto LABEL_163;
      }
      v88 += 4;
      if ( &v169[2 * ((4LL * (unsigned int)v170) >> 4)] == (__int64 *)v88 )
        goto LABEL_202;
    }
    goto LABEL_163;
  }
  v88 = (int *)v169;
LABEL_202:
  v111 = (char *)v87 - (char *)v88;
  if ( (char *)v87 - (char *)v88 == 8 )
    goto LABEL_216;
  if ( v111 != 12 )
  {
    if ( v111 == 4 )
      goto LABEL_205;
    goto LABEL_164;
  }
  if ( v34 <= *v88 )
    goto LABEL_163;
  ++v88;
LABEL_216:
  if ( v34 <= *v88 )
    goto LABEL_163;
  ++v88;
LABEL_205:
  if ( v34 <= *v88 )
  {
LABEL_163:
    if ( v87 != v88 )
      goto LABEL_65;
  }
LABEL_164:
  v118 = v17;
  v121 = v15;
  v125 = v11;
  v89 = sub_B4ED80((int *)v169, (unsigned int)v170, v34);
  v17 = v118;
  v15 = v121;
  v127 = v89;
  v11 = v125;
LABEL_66:
  v35 = *(__int64 **)(a1 + 152);
  v36 = *(unsigned int *)(a1 + 192);
  v120 = v17;
  v124 = v15;
  v163 = (int **)v158;
  v142 = v141 - 29;
  v37 = sub_DFBC30(v35, 7, v138, (__int64)v11, v152, v36, 0, 0, (__int64)&v163, 1, v2);
  v39 = v38;
  v40 = v37;
  v41 = sub_DFD800(*(_QWORD *)(a1 + 152), v142, v138, *(_DWORD *)(a1 + 192), 0, 0, 0, 0, 0, 0);
  v43 = v42;
  v44 = 1;
  v45 = v120;
  if ( v39 != 1 )
    v44 = v43;
  v46 = __OFADD__(v40, v41);
  v47 = v40 + v41;
  v153 = v44;
  if ( v46 )
  {
    v47 = 0x8000000000000000LL;
    if ( v40 > 0 )
      v47 = 0x7FFFFFFFFFFFFFFFLL;
  }
  v139 = v47;
  if ( v124 )
  {
    v48 = *(__int64 **)(a1 + 152);
    v115 = *(_QWORD *)(v158 - 64);
    v49 = *(unsigned int *)(a1 + 192);
    v163 = (int **)v136;
    v164 = (_DWORD **)v131;
    v50 = sub_DFBC30(v48, 6, v147, v140, v133, v49, 0, 0, (__int64)&v163, 2, v115);
    v45 = v120;
    v52 = v51 == 1;
    v53 = 1;
    if ( !v52 )
      v53 = v153;
    v153 = v53;
    if ( __OFADD__(v50, v139) )
    {
      v112 = v50 <= 0;
      v113 = 0x8000000000000000LL;
      if ( !v112 )
        v113 = 0x7FFFFFFFFFFFFFFFLL;
      v139 = v113;
    }
    else
    {
      v139 += v50;
    }
  }
  if ( v143 )
  {
    v54 = *(__int64 **)(a1 + 152);
    v144 = v45;
    v116 = *(_QWORD *)(v158 - 32);
    v55 = *(unsigned int *)(a1 + 192);
    v163 = (int **)v146;
    v164 = (_DWORD **)v154;
    v56 = sub_DFBC30(v54, 6, v132, v134, v135, v55, 0, 0, (__int64)&v163, 2, v116);
    v45 = v144;
    v52 = v57 == 1;
    v58 = 1;
    if ( !v52 )
      v58 = v153;
    v153 = v58;
    if ( __OFADD__(v56, v139) )
    {
      v112 = v56 <= 0;
      v114 = 0x8000000000000000LL;
      if ( !v112 )
        v114 = 0x7FFFFFFFFFFFFFFFLL;
      v139 = v114;
    }
    else
    {
      v139 += v56;
    }
  }
  v145 = v45;
  v59 = sub_DFD800(*(_QWORD *)(a1 + 152), v142, v137, *(_DWORD *)(a1 + 192), 0, 0, 0, 0, 0, 0);
  v4 = v145;
  v60 = v59;
  v62 = v61;
  if ( !v130 )
  {
    v79 = *(__int64 **)(a1 + 152);
    v80 = *(unsigned int *)(a1 + 192);
    v163 = (int **)v136;
    v164 = (_DWORD **)v131;
    v81 = sub_DFBC30(v79, 6, v147, (__int64)v166, (unsigned int)v167, v80, 0, 0, (__int64)&v163, 2, 0);
    v4 = v145;
    if ( v82 == 1 )
      v62 = 1;
    v46 = __OFADD__(v81, v60);
    v60 += v81;
    if ( v46 )
    {
      v60 = 0x7FFFFFFFFFFFFFFFLL;
      if ( v81 <= 0 )
        v60 = 0x8000000000000000LL;
    }
  }
  if ( !v127 )
  {
    v83 = *(__int64 **)(a1 + 152);
    v84 = *(unsigned int *)(a1 + 192);
    v163 = (int **)v146;
    v148 = v4;
    v164 = (_DWORD **)v154;
    v85 = sub_DFBC30(v83, 6, v132, (__int64)v169, (unsigned int)v170, v84, 0, 0, (__int64)&v163, 2, 0);
    v4 = v148;
    if ( v86 == 1 )
      v62 = 1;
    v46 = __OFADD__(v85, v60);
    v60 += v85;
    if ( v46 )
    {
      v60 = 0x7FFFFFFFFFFFFFFFLL;
      if ( v85 <= 0 )
        v60 = 0x8000000000000000LL;
    }
  }
  if ( v62 == v153 )
  {
    if ( v60 > v139 )
      goto LABEL_98;
LABEL_83:
    if ( v130 )
    {
LABEL_84:
      if ( v127 )
      {
LABEL_85:
        v165 = 257;
        v63 = (unsigned __int8 *)sub_2C51350(
                                   (__int64 *)(a1 + 8),
                                   v142,
                                   (unsigned __int8 *)v136,
                                   (unsigned __int8 *)v146,
                                   v161[0],
                                   0,
                                   (__int64)&v163,
                                   0);
        v64 = (__int64)v63;
        if ( *v63 > 0x1Cu )
          sub_B45260(v63, v158, 1);
        v65 = a1 + 200;
        if ( *(_BYTE *)v136 > 0x1Cu )
          sub_F15FC0(v65, v136);
        if ( *(_BYTE *)v146 > 0x1Cu )
          sub_F15FC0(v65, v146);
        sub_BD84D0(v2, v64);
        if ( *(_BYTE *)v64 > 0x1Cu )
        {
          sub_BD6B90((unsigned __int8 *)v64, (unsigned __int8 *)v2);
          for ( i = *(_QWORD *)(v64 + 16); i; i = *(_QWORD *)(i + 8) )
            sub_F15FC0(v65, *(_QWORD *)(i + 24));
          if ( *(_BYTE *)v64 > 0x1Cu )
            sub_F15FC0(v65, v64);
        }
        v4 = 1;
        if ( *(_BYTE *)v2 > 0x1Cu )
        {
          sub_F15FC0(v65, v2);
          v4 = 1;
        }
        goto LABEL_98;
      }
      v69 = *(_QWORD *)(a1 + 88);
      v70 = (int *)v169;
      v162 = 257;
      v71 = (unsigned int)v170;
      v72 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64 *, _QWORD))(*(_QWORD *)v69 + 112LL);
      if ( (char *)v72 == (char *)sub_9B6630 )
      {
        if ( *(_BYTE *)v146 > 0x15u || *(_BYTE *)v154 > 0x15u )
          goto LABEL_193;
        v73 = sub_AD5CE0(v146, v154, v169, (unsigned int)v170, 0);
      }
      else
      {
        v73 = v72(v69, v146, v154, v169, (unsigned int)v170);
      }
      if ( v73 )
      {
LABEL_126:
        v146 = v73;
        goto LABEL_85;
      }
LABEL_193:
      v165 = 257;
      v103 = sub_BD2C40(112, unk_3F1FE60);
      if ( v103 )
      {
        v104 = v154;
        v155 = v103;
        sub_B4E9E0((__int64)v103, v146, v104, v70, v71, (__int64)&v163, 0, 0);
        v103 = v155;
      }
      v156 = v103;
      (*(void (__fastcall **)(_QWORD, _QWORD *, int *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 96) + 16LL))(
        *(_QWORD *)(a1 + 96),
        v103,
        v161,
        *(_QWORD *)(a1 + 64),
        *(_QWORD *)(a1 + 72));
      v105 = *(_QWORD *)(a1 + 8);
      v73 = (__int64)v156;
      if ( v105 != v105 + 16LL * *(unsigned int *)(a1 + 16) )
      {
        v157 = v2;
        v106 = *(_QWORD *)(a1 + 8);
        v107 = v105 + 16LL * *(unsigned int *)(a1 + 16);
        v108 = v73;
        do
        {
          v109 = *(_QWORD *)(v106 + 8);
          v110 = *(_DWORD *)v106;
          v106 += 16;
          sub_B99FD0(v108, v110, v109);
        }
        while ( v107 != v106 );
        v2 = v157;
        v73 = v108;
      }
      goto LABEL_126;
    }
    v74 = *(_QWORD *)(a1 + 88);
    v75 = v166;
    v162 = 257;
    v76 = (unsigned int)v167;
    v77 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)v74 + 112LL);
    if ( v77 == sub_9B6630 )
    {
      if ( *(_BYTE *)v136 > 0x15u || *v131 > 0x15u )
        goto LABEL_187;
      v78 = sub_AD5CE0(v136, (__int64)v131, v166, (unsigned int)v167, 0);
    }
    else
    {
      v78 = ((__int64 (__fastcall *)(__int64, __int64, _BYTE *, _DWORD *, _QWORD))v77)(
              v74,
              v136,
              v131,
              v166,
              (unsigned int)v167);
    }
    if ( v78 )
    {
LABEL_132:
      v136 = v78;
      goto LABEL_84;
    }
LABEL_187:
    v165 = 257;
    v96 = sub_BD2C40(112, unk_3F1FE60);
    if ( v96 )
    {
      v149 = v96;
      sub_B4E9E0((__int64)v96, v136, (__int64)v131, v75, v76, (__int64)&v163, 0, 0);
      v96 = v149;
    }
    v150 = v96;
    (*(void (__fastcall **)(_QWORD, _QWORD *, int *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 96) + 16LL))(
      *(_QWORD *)(a1 + 96),
      v96,
      v161,
      *(_QWORD *)(a1 + 64),
      *(_QWORD *)(a1 + 72));
    v97 = *(_QWORD *)(a1 + 8);
    v78 = (__int64)v150;
    if ( v97 != v97 + 16LL * *(unsigned int *)(a1 + 16) )
    {
      v151 = v2;
      v98 = *(_QWORD *)(a1 + 8);
      v99 = v97 + 16LL * *(unsigned int *)(a1 + 16);
      v100 = v78;
      do
      {
        v101 = *(_QWORD *)(v98 + 8);
        v102 = *(_DWORD *)v98;
        v98 += 16;
        sub_B99FD0(v100, v102, v101);
      }
      while ( v99 != v98 );
      v2 = v151;
      v78 = v100;
    }
    goto LABEL_132;
  }
  if ( v153 >= v62 )
    goto LABEL_83;
LABEL_98:
  if ( v169 != &v171 )
  {
    v159 = v4;
    _libc_free((unsigned __int64)v169);
    v4 = v159;
  }
  if ( v166 != v168 )
  {
    v160 = v4;
    _libc_free((unsigned __int64)v166);
    return v160;
  }
  return v4;
}
