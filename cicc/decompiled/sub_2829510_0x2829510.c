// Function: sub_2829510
// Address: 0x2829510
//
__int64 __fastcall sub_2829510(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  unsigned __int64 *v6; // rax
  unsigned int v7; // r14d
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  unsigned int v12; // eax
  int v13; // edx
  unsigned __int64 v14; // rbx
  unsigned __int64 v15; // rax
  _QWORD *v16; // rax
  unsigned int v17; // edx
  unsigned int v18; // edx
  bool v19; // al
  _BYTE *v20; // rsi
  __int64 v21; // r8
  __int64 v22; // r9
  unsigned int v23; // ebx
  __int64 v24; // rax
  unsigned int *v25; // rbx
  unsigned int *v26; // r14
  __int64 v27; // rsi
  _BYTE *v28; // rsi
  unsigned __int8 *v29; // rdi
  unsigned __int8 *v30; // rcx
  __int64 v31; // r14
  __int64 *v32; // rbx
  int v33; // ecx
  __int64 v34; // rsi
  __int64 *v35; // rdx
  _QWORD *v36; // rax
  unsigned int v37; // esi
  unsigned int v38; // edx
  __int64 *v39; // rax
  unsigned int v40; // ecx
  __int64 v41; // rdx
  unsigned int v42; // edx
  __int64 v43; // rsi
  __int64 v44; // rcx
  __int64 v45; // r13
  int v46; // eax
  __int64 v47; // rbx
  __int64 v48; // r12
  int v49; // edx
  unsigned int v50; // eax
  __int64 v51; // rsi
  __int64 *v52; // rax
  __int64 *v53; // rdx
  __int64 v54; // rbx
  __int64 *v55; // r14
  __int64 v56; // rax
  unsigned int v57; // edx
  unsigned __int64 v58; // rax
  char *v59; // rax
  unsigned int v60; // edx
  int v61; // eax
  unsigned int v62; // edx
  unsigned __int64 v63; // rax
  char *v64; // rax
  unsigned int v65; // edx
  int v66; // eax
  unsigned int v67; // edx
  __int64 v68; // rax
  _QWORD *v69; // rdx
  unsigned __int64 v70; // rax
  unsigned __int16 v71; // cx
  unsigned __int8 v72; // r13
  __int64 *v73; // rax
  __int64 v74; // rax
  __int64 v75; // rdx
  __int64 *v76; // rdi
  int v77; // esi
  unsigned int v78; // edx
  __int64 *v79; // rax
  int v80; // edi
  unsigned int v81; // eax
  __int64 v82; // rsi
  unsigned int v83; // esi
  unsigned int v84; // eax
  unsigned int v85; // edx
  unsigned int v86; // edi
  int i; // edi
  int v88; // r10d
  __int64 *v89; // rdi
  int v90; // ecx
  unsigned int v91; // edx
  __int64 v92; // rsi
  __int64 v93; // rax
  int v94; // eax
  __int64 *v95; // rdi
  int v96; // ecx
  unsigned int v97; // edx
  __int64 v98; // rsi
  __int64 *v100; // rax
  __int64 v101; // rdx
  __int64 *v102; // r12
  __int64 v103; // rsi
  __int64 *v104; // rbx
  __int64 v105; // rcx
  __int64 *v106; // rax
  __int64 *v107; // rax
  int v108; // r10d
  int v109; // r10d
  unsigned int v110; // ecx
  __int64 v111; // rsi
  int v112; // r11d
  __int64 *v113; // rdx
  int v114; // r10d
  unsigned int v115; // ecx
  __int64 v116; // rsi
  int v117; // r11d
  const void **v119; // [rsp+18h] [rbp-2C8h]
  unsigned __int8 *v120; // [rsp+18h] [rbp-2C8h]
  bool v121; // [rsp+30h] [rbp-2B0h]
  unsigned __int8 *v122; // [rsp+30h] [rbp-2B0h]
  int v123; // [rsp+30h] [rbp-2B0h]
  char *v124; // [rsp+30h] [rbp-2B0h]
  unsigned __int8 *v125; // [rsp+38h] [rbp-2A8h]
  __int64 v126; // [rsp+38h] [rbp-2A8h]
  unsigned int v127; // [rsp+38h] [rbp-2A8h]
  char *v128; // [rsp+38h] [rbp-2A8h]
  unsigned int v129; // [rsp+38h] [rbp-2A8h]
  unsigned int v130; // [rsp+48h] [rbp-298h]
  bool v132; // [rsp+50h] [rbp-290h]
  unsigned int v133; // [rsp+50h] [rbp-290h]
  char v134; // [rsp+50h] [rbp-290h]
  unsigned int v135; // [rsp+5Ch] [rbp-284h]
  unsigned __int8 v136; // [rsp+5Ch] [rbp-284h]
  __int64 *v137; // [rsp+60h] [rbp-280h]
  __int64 v138; // [rsp+68h] [rbp-278h]
  unsigned __int8 *v139; // [rsp+68h] [rbp-278h]
  __int64 v140; // [rsp+78h] [rbp-268h]
  __int64 *v141; // [rsp+78h] [rbp-268h]
  _QWORD *v142; // [rsp+80h] [rbp-260h] BYREF
  unsigned int v143; // [rsp+88h] [rbp-258h]
  char *v144; // [rsp+90h] [rbp-250h] BYREF
  unsigned int v145; // [rsp+98h] [rbp-248h]
  char *v146; // [rsp+A0h] [rbp-240h] BYREF
  __int64 v147; // [rsp+A8h] [rbp-238h]
  __int64 v148; // [rsp+B0h] [rbp-230h] BYREF
  __int64 v149; // [rsp+B8h] [rbp-228h]
  __int64 v150; // [rsp+C0h] [rbp-220h]
  __int64 v151; // [rsp+C8h] [rbp-218h]
  __int64 *v152; // [rsp+D0h] [rbp-210h]
  __int64 v153; // [rsp+D8h] [rbp-208h]
  __int64 v154; // [rsp+E0h] [rbp-200h] BYREF
  __int64 v155; // [rsp+E8h] [rbp-1F8h]
  __int64 v156; // [rsp+F0h] [rbp-1F0h]
  __int64 v157; // [rsp+F8h] [rbp-1E8h]
  __int64 *v158; // [rsp+100h] [rbp-1E0h]
  __int64 v159; // [rsp+108h] [rbp-1D8h]
  __int64 v160; // [rsp+110h] [rbp-1D0h] BYREF
  __int64 v161; // [rsp+118h] [rbp-1C8h]
  __int64 *v162; // [rsp+120h] [rbp-1C0h] BYREF
  unsigned int v163; // [rsp+128h] [rbp-1B8h]
  unsigned int *v164; // [rsp+160h] [rbp-180h] BYREF
  __int64 v165; // [rsp+168h] [rbp-178h]
  _BYTE v166[64]; // [rsp+170h] [rbp-170h] BYREF
  _QWORD *v167; // [rsp+1B0h] [rbp-130h] BYREF
  __int64 *v168; // [rsp+1B8h] [rbp-128h]
  __int64 v169; // [rsp+1C0h] [rbp-120h]
  int v170; // [rsp+1C8h] [rbp-118h]
  char v171; // [rsp+1CCh] [rbp-114h]
  char v172; // [rsp+1D0h] [rbp-110h] BYREF
  char *v173; // [rsp+210h] [rbp-D0h] BYREF
  __int64 *v174; // [rsp+218h] [rbp-C8h]
  __int64 v175; // [rsp+220h] [rbp-C0h]
  int v176; // [rsp+228h] [rbp-B8h]
  unsigned __int8 v177; // [rsp+22Ch] [rbp-B4h]
  char v178; // [rsp+230h] [rbp-B0h] BYREF

  v152 = &v154;
  v158 = &v160;
  v6 = (unsigned __int64 *)&v162;
  v148 = 0;
  v149 = 0;
  v150 = 0;
  v151 = 0;
  v153 = 0;
  v154 = 0;
  v155 = 0;
  v156 = 0;
  v157 = 0;
  v159 = 0;
  v160 = 0;
  v161 = 1;
  do
  {
    *v6 = -4096;
    v6 += 2;
  }
  while ( v6 != (unsigned __int64 *)&v164 );
  v164 = (unsigned int *)v166;
  v165 = 0x1000000000LL;
  v130 = *(_DWORD *)(a2 + 8);
  if ( !v130 )
    goto LABEL_271;
  v135 = 1;
  v138 = 0;
  while ( 2 )
  {
    v7 = v138;
    v140 = 8 * v138;
    v8 = *(_QWORD *)(*(_QWORD *)a2 + 8 * v138);
    v125 = *(unsigned __int8 **)(v8 - 64);
    v9 = *(_QWORD *)(*(_QWORD *)(sub_DD8400(*(_QWORD *)(a1 + 32), *(_QWORD *)(v8 - 32))[4] + 8) + 32LL);
    LODWORD(v147) = *(_DWORD *)(v9 + 32);
    if ( (unsigned int)v147 > 0x40 )
      sub_C43780((__int64)&v146, (const void **)(v9 + 24));
    else
      v146 = *(char **)(v9 + 24);
    v10 = sub_9208B0(*(_QWORD *)(a1 + 56), *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)a2 + 8 * v138) - 64LL) + 8LL));
    v174 = (__int64 *)v11;
    v173 = (char *)((unsigned __int64)(v10 + 7) >> 3);
    v12 = sub_CA1930(&v173);
    v13 = v147;
    v14 = v12;
    if ( (unsigned int)v147 > 0x40 )
    {
      v123 = v147;
      if ( v123 - (unsigned int)sub_C444A0((__int64)&v146) <= 0x40 && v14 == *(_QWORD *)v146 )
        goto LABEL_66;
      LODWORD(v168) = v123;
      sub_C43780((__int64)&v167, (const void **)&v146);
      v13 = (int)v168;
      if ( (unsigned int)v168 > 0x40 )
      {
        sub_C43D10((__int64)&v167);
        goto LABEL_13;
      }
      v15 = (unsigned __int64)v167;
    }
    else
    {
      v15 = (unsigned __int64)v146;
      if ( (char *)v14 == v146 )
        goto LABEL_66;
      LODWORD(v168) = v147;
    }
    v16 = (_QWORD *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v13) & ~v15);
    if ( !v13 )
      v16 = 0;
    v167 = v16;
LABEL_13:
    sub_C46250((__int64)&v167);
    v17 = (unsigned int)v168;
    LODWORD(v168) = 0;
    LODWORD(v174) = v17;
    v173 = (char *)v167;
    if ( v17 <= 0x40 )
    {
      if ( (_QWORD *)v14 == v167 )
        goto LABEL_66;
    }
    else
    {
      v119 = (const void **)v167;
      v18 = v17 - sub_C444A0((__int64)&v173);
      v19 = 0;
      if ( v18 <= 0x40 )
        v19 = v14 == (_QWORD)*v119;
      if ( v173 )
      {
        v121 = v19;
        j_j___libc_free_0_0((unsigned __int64)v173);
        v19 = v121;
        if ( (unsigned int)v168 > 0x40 )
        {
          if ( v167 )
          {
            j_j___libc_free_0_0((unsigned __int64)v167);
            v19 = v121;
          }
        }
      }
      if ( v19 )
      {
LABEL_66:
        sub_2829290((__int64)&v148, (__int64 *)(*(_QWORD *)a2 + v140));
        if ( (unsigned int)v147 > 0x40 )
          goto LABEL_46;
        goto LABEL_48;
      }
    }
    v20 = *(_BYTE **)(a1 + 56);
    if ( a4 == 1 )
    {
      v122 = 0;
      v120 = (unsigned __int8 *)sub_98A180(v125, (__int64)v20);
    }
    else
    {
      v120 = 0;
      v122 = (unsigned __int8 *)sub_281EA60((__int64)v125, v20);
    }
    LODWORD(v165) = 0;
    if ( v130 <= v135 )
    {
      if ( !v138 )
        goto LABEL_45;
      v24 = 0;
      goto LABEL_29;
    }
    v23 = v135;
    v24 = 0;
    do
    {
      if ( v24 + 1 > (unsigned __int64)HIDWORD(v165) )
      {
        sub_C8D5F0((__int64)&v164, v166, v24 + 1, 4u, v21, v22);
        v24 = (unsigned int)v165;
      }
      v164[v24] = v23++;
      v24 = (unsigned int)(v165 + 1);
      LODWORD(v165) = v165 + 1;
    }
    while ( v23 < v130 );
    if ( v138 )
    {
      do
      {
LABEL_29:
        --v7;
        if ( v24 + 1 > (unsigned __int64)HIDWORD(v165) )
        {
          sub_C8D5F0((__int64)&v164, v166, v24 + 1, 4u, v21, v22);
          v24 = (unsigned int)v165;
        }
        v164[v24] = v7;
        v24 = (unsigned int)(v165 + 1);
        LODWORD(v165) = v165 + 1;
      }
      while ( v7 );
    }
    v25 = v164;
    v26 = &v164[v24];
    if ( v26 == v164 )
      goto LABEL_45;
    while ( 1 )
    {
      v27 = *(_QWORD *)(*(_QWORD *)(sub_DD8400(
                                      *(_QWORD *)(a1 + 32),
                                      *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a2 + 8LL * *v25) - 32LL))[4]
                                  + 8)
                      + 32LL);
      LODWORD(v174) = *(_DWORD *)(v27 + 32);
      if ( (unsigned int)v174 > 0x40 )
        sub_C43780((__int64)&v173, (const void **)(v27 + 24));
      else
        v173 = *(char **)(v27 + 24);
      if ( (unsigned int)v147 <= 0x40 )
      {
        if ( v146 != v173 )
          goto LABEL_41;
      }
      else if ( !sub_C43C50((__int64)&v146, (const void **)&v173) )
      {
        goto LABEL_41;
      }
      v28 = *(_BYTE **)(a1 + 56);
      v29 = *(unsigned __int8 **)(*(_QWORD *)(*(_QWORD *)a2 + 8LL * *v25) - 64LL);
      if ( a4 == 1 )
        break;
      v126 = sub_281EA60((__int64)v29, v28);
      if ( sub_D35390(
             *(char **)(*(_QWORD *)a2 + 8 * v138),
             *(char **)(*(_QWORD *)a2 + 8LL * *v25),
             *(_QWORD *)(a1 + 56),
             *(_QWORD *)(a1 + 32),
             0) )
      {
        v30 = v122;
        if ( (unsigned int)*v122 - 12 <= 1 )
          goto LABEL_55;
LABEL_40:
        if ( (unsigned __int8 *)v126 == v30 )
          goto LABEL_55;
      }
LABEL_41:
      if ( (unsigned int)v174 > 0x40 && v173 )
        j_j___libc_free_0_0((unsigned __int64)v173);
      if ( v26 == ++v25 )
        goto LABEL_45;
    }
    v126 = sub_98A180(v29, (__int64)v28);
    if ( !sub_D35390(
            *(char **)(*(_QWORD *)a2 + 8 * v138),
            *(char **)(*(_QWORD *)a2 + 8LL * *v25),
            *(_QWORD *)(a1 + 56),
            *(_QWORD *)(a1 + 32),
            0) )
      goto LABEL_41;
    v30 = v120;
    if ( (unsigned int)*v120 - 12 > 1 )
      goto LABEL_40;
LABEL_55:
    sub_2829290((__int64)&v154, (__int64 *)(*(_QWORD *)a2 + 8LL * *v25));
    sub_2829290((__int64)&v148, (__int64 *)(*(_QWORD *)a2 + v140));
    v31 = *(_QWORD *)(*(_QWORD *)a2 + 8LL * *v25);
    v32 = (__int64 *)(*(_QWORD *)a2 + v140);
    if ( (v161 & 1) != 0 )
    {
      v21 = (__int64)&v162;
      v33 = 3;
      goto LABEL_57;
    }
    v37 = v163;
    v21 = (__int64)v162;
    v33 = v163 - 1;
    if ( !v163 )
    {
      v38 = v161;
      ++v160;
      v39 = 0;
      v40 = ((unsigned int)v161 >> 1) + 1;
      goto LABEL_70;
    }
LABEL_57:
    v34 = v33 & (((unsigned int)*v32 >> 9) ^ ((unsigned int)*v32 >> 4));
    v35 = (__int64 *)(v21 + 16 * v34);
    v22 = *v35;
    if ( *v32 == *v35 )
    {
LABEL_58:
      v36 = v35 + 1;
      goto LABEL_59;
    }
    v108 = 1;
    v39 = 0;
    while ( v22 != -4096 )
    {
      if ( !v39 && v22 == -8192 )
        v39 = v35;
      LODWORD(v34) = v33 & (v108 + v34);
      v35 = (__int64 *)(v21 + 16LL * (unsigned int)v34);
      v22 = *v35;
      if ( *v32 == *v35 )
        goto LABEL_58;
      ++v108;
    }
    v21 = 12;
    v37 = 4;
    if ( !v39 )
      v39 = v35;
    v38 = v161;
    ++v160;
    v40 = ((unsigned int)v161 >> 1) + 1;
    if ( (v161 & 1) == 0 )
    {
      v37 = v163;
LABEL_70:
      v21 = 3 * v37;
    }
    if ( 4 * v40 >= (unsigned int)v21 )
    {
      sub_2828910((__int64)&v160, 2 * v37);
      if ( (v161 & 1) != 0 )
      {
        v21 = (__int64)&v162;
        v109 = 3;
      }
      else
      {
        v21 = (__int64)v162;
        if ( !v163 )
          goto LABEL_295;
        v109 = v163 - 1;
      }
      v38 = v161;
      v110 = v109 & (((unsigned int)*v32 >> 9) ^ ((unsigned int)*v32 >> 4));
      v39 = (__int64 *)(v21 + 16LL * v110);
      v111 = *v39;
      if ( *v39 == *v32 )
        goto LABEL_73;
      v112 = 1;
      v113 = 0;
      while ( v111 != -4096 )
      {
        if ( v111 == -8192 && !v113 )
          v113 = v39;
        v22 = (unsigned int)(v112 + 1);
        v110 = v109 & (v112 + v110);
        v39 = (__int64 *)(v21 + 16LL * v110);
        v111 = *v39;
        if ( *v32 == *v39 )
          goto LABEL_270;
        ++v112;
      }
LABEL_268:
      if ( v113 )
        v39 = v113;
LABEL_270:
      v38 = v161;
      goto LABEL_73;
    }
    if ( v37 - HIDWORD(v161) - v40 <= v37 >> 3 )
    {
      sub_2828910((__int64)&v160, v37);
      if ( (v161 & 1) != 0 )
      {
        v21 = (__int64)&v162;
        v114 = 3;
LABEL_265:
        v38 = v161;
        v115 = v114 & (((unsigned int)*v32 >> 9) ^ ((unsigned int)*v32 >> 4));
        v39 = (__int64 *)(v21 + 16LL * v115);
        v116 = *v39;
        if ( *v32 == *v39 )
          goto LABEL_73;
        v117 = 1;
        v113 = 0;
        while ( v116 != -4096 )
        {
          if ( v116 == -8192 && !v113 )
            v113 = v39;
          v22 = (unsigned int)(v117 + 1);
          v115 = v114 & (v117 + v115);
          v39 = (__int64 *)(v21 + 16LL * v115);
          v116 = *v39;
          if ( *v32 == *v39 )
            goto LABEL_270;
          ++v117;
        }
        goto LABEL_268;
      }
      v21 = (__int64)v162;
      if ( v163 )
      {
        v114 = v163 - 1;
        goto LABEL_265;
      }
LABEL_295:
      LODWORD(v161) = (2 * ((unsigned int)v161 >> 1) + 2) | v161 & 1;
      BUG();
    }
LABEL_73:
    LODWORD(v161) = (2 * (v38 >> 1) + 2) | v38 & 1;
    if ( *v39 != -4096 )
      --HIDWORD(v161);
    v41 = *v32;
    v36 = v39 + 1;
    *v36 = 0;
    *(v36 - 1) = v41;
LABEL_59:
    *v36 = v31;
    if ( (unsigned int)v174 > 0x40 && v173 )
      j_j___libc_free_0_0((unsigned __int64)v173);
LABEL_45:
    if ( (unsigned int)v147 > 0x40 )
    {
LABEL_46:
      if ( v146 )
        j_j___libc_free_0_0((unsigned __int64)v146);
    }
LABEL_48:
    ++v138;
    if ( v130 > v135 )
    {
      ++v135;
      continue;
    }
    break;
  }
  v173 = 0;
  v175 = 16;
  v176 = 0;
  v137 = &v152[(unsigned int)v153];
  v174 = (__int64 *)&v178;
  v177 = 1;
  if ( v137 == v152 )
  {
LABEL_271:
    v136 = 0;
    goto LABEL_202;
  }
  v141 = v152;
  v136 = 0;
  while ( 2 )
  {
    v44 = v155;
    v45 = *v141;
    v46 = v157;
    if ( (_DWORD)v157 )
    {
      v42 = (v157 - 1) & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
      v43 = *(_QWORD *)(v155 + 8LL * v42);
      if ( v45 == v43 )
        goto LABEL_83;
      v21 = 1;
      while ( v43 != -4096 )
      {
        v22 = (unsigned int)(v21 + 1);
        v42 = (v157 - 1) & (v21 + v42);
        v43 = *(_QWORD *)(v155 + 8LL * v42);
        if ( v45 == v43 )
          goto LABEL_83;
        v21 = (unsigned int)v22;
      }
    }
    v47 = *v141;
    v48 = 0;
    v167 = 0;
    v169 = 8;
    v168 = (__int64 *)&v172;
    v170 = 0;
    v171 = 1;
    if ( !(_DWORD)v157 )
      goto LABEL_144;
    while ( 2 )
    {
      v49 = v46 - 1;
      v50 = (v46 - 1) & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
      v51 = *(_QWORD *)(v44 + 8LL * v50);
      if ( v51 != v47 )
      {
        for ( i = 1; ; ++i )
        {
          if ( v51 == -4096 )
            goto LABEL_144;
          v21 = (unsigned int)(i + 1);
          v50 = v49 & (i + v50);
          v51 = *(_QWORD *)(v44 + 8LL * v50);
          if ( v51 == v47 )
            break;
        }
      }
LABEL_87:
      if ( v177 )
      {
        v52 = v174;
        v53 = &v174[HIDWORD(v175)];
        if ( v174 != v53 )
        {
          do
          {
            if ( *v52 == v47 )
              goto LABEL_92;
            ++v52;
          }
          while ( v53 != v52 );
        }
      }
      else if ( sub_C8CA60((__int64)&v173, v47) )
      {
        goto LABEL_92;
      }
      if ( !v171 )
        goto LABEL_151;
      v73 = v168;
      v44 = HIDWORD(v169);
      v53 = &v168[HIDWORD(v169)];
      if ( v168 == v53 )
      {
LABEL_152:
        if ( HIDWORD(v169) < (unsigned int)v169 )
        {
          ++HIDWORD(v169);
          *v53 = v47;
          v167 = (_QWORD *)((char *)v167 + 1);
          goto LABEL_139;
        }
LABEL_151:
        sub_C8CC70((__int64)&v167, v47, (__int64)v53, v44, v21, v22);
        goto LABEL_139;
      }
      while ( *v73 != v47 )
      {
        if ( v53 == ++v73 )
          goto LABEL_152;
      }
LABEL_139:
      v74 = sub_9208B0(*(_QWORD *)(a1 + 56), *(_QWORD *)(*(_QWORD *)(v47 - 64) + 8LL));
      v147 = v75;
      v146 = (char *)((unsigned __int64)(v74 + 7) >> 3);
      v48 = (unsigned int)sub_CA1930(&v146) + (unsigned int)v48;
      if ( (v161 & 1) != 0 )
      {
        v76 = (__int64 *)&v162;
        v77 = 3;
        goto LABEL_141;
      }
      v83 = v163;
      v76 = v162;
      if ( !v163 )
      {
        v84 = v161;
        ++v160;
        v21 = 0;
        v85 = ((unsigned int)v161 >> 1) + 1;
        goto LABEL_155;
      }
      v77 = v163 - 1;
LABEL_141:
      v78 = v77 & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
      v79 = &v76[2 * v78];
      v22 = *v79;
      if ( *v79 == v47 )
        goto LABEL_142;
      v88 = 1;
      v21 = 0;
      while ( 2 )
      {
        if ( v22 == -4096 )
        {
          v86 = 12;
          v83 = 4;
          if ( !v21 )
            v21 = (__int64)v79;
          v84 = v161;
          ++v160;
          v85 = ((unsigned int)v161 >> 1) + 1;
          if ( (v161 & 1) == 0 )
          {
            v83 = v163;
LABEL_155:
            v86 = 3 * v83;
          }
          if ( 4 * v85 < v86 )
          {
            if ( v83 - HIDWORD(v161) - v85 > v83 >> 3 )
            {
LABEL_158:
              LODWORD(v161) = (2 * (v84 >> 1) + 2) | v84 & 1;
              if ( *(_QWORD *)v21 != -4096 )
                --HIDWORD(v161);
              *(_QWORD *)v21 = v47;
              v47 = 0;
              *(_QWORD *)(v21 + 8) = 0;
              goto LABEL_143;
            }
            sub_2828910((__int64)&v160, v83);
            if ( (v161 & 1) != 0 )
            {
              v95 = (__int64 *)&v162;
              v96 = 3;
              goto LABEL_185;
            }
            v95 = v162;
            if ( v163 )
            {
              v96 = v163 - 1;
LABEL_185:
              v84 = v161;
              v97 = v96 & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
              v21 = (__int64)&v95[2 * v97];
              v98 = *(_QWORD *)v21;
              if ( *(_QWORD *)v21 == v47 )
                goto LABEL_158;
              v22 = 1;
              v93 = 0;
              while ( v98 != -4096 )
              {
                if ( !v93 && v98 == -8192 )
                  v93 = v21;
                v97 = v96 & (v22 + v97);
                v21 = (__int64)&v95[2 * v97];
                v98 = *(_QWORD *)v21;
                if ( *(_QWORD *)v21 == v47 )
                  goto LABEL_178;
                v22 = (unsigned int)(v22 + 1);
              }
LABEL_176:
              if ( v93 )
                v21 = v93;
LABEL_178:
              v84 = v161;
              goto LABEL_158;
            }
LABEL_296:
            LODWORD(v161) = (2 * ((unsigned int)v161 >> 1) + 2) | v161 & 1;
            BUG();
          }
          sub_2828910((__int64)&v160, 2 * v83);
          if ( (v161 & 1) != 0 )
          {
            v89 = (__int64 *)&v162;
            v90 = 3;
          }
          else
          {
            v89 = v162;
            if ( !v163 )
              goto LABEL_296;
            v90 = v163 - 1;
          }
          v84 = v161;
          v91 = v90 & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
          v21 = (__int64)&v89[2 * v91];
          v92 = *(_QWORD *)v21;
          if ( *(_QWORD *)v21 == v47 )
            goto LABEL_158;
          v22 = 1;
          v93 = 0;
          while ( v92 != -4096 )
          {
            if ( !v93 && v92 == -8192 )
              v93 = v21;
            v91 = v90 & (v22 + v91);
            v21 = (__int64)&v89[2 * v91];
            v92 = *(_QWORD *)v21;
            if ( *(_QWORD *)v21 == v47 )
              goto LABEL_178;
            v22 = (unsigned int)(v22 + 1);
          }
          goto LABEL_176;
        }
        if ( v22 != -8192 || v21 )
          v79 = (__int64 *)v21;
        v21 = (unsigned int)(v88 + 1);
        v78 = v77 & (v88 + v78);
        v22 = v76[2 * v78];
        if ( v22 != v47 )
        {
          ++v88;
          v21 = (__int64)v79;
          v79 = &v76[2 * v78];
          continue;
        }
        break;
      }
      v79 = &v76[2 * v78];
LABEL_142:
      v47 = v79[1];
LABEL_143:
      v46 = v157;
      v44 = v155;
      if ( (_DWORD)v157 )
        continue;
      break;
    }
LABEL_144:
    v44 = v149;
    if ( !(_DWORD)v151 )
      goto LABEL_92;
    v80 = 1;
    v81 = (v151 - 1) & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
    v82 = *(_QWORD *)(v149 + 8LL * v81);
    if ( v82 == v47 )
      goto LABEL_87;
    while ( v82 != -4096 )
    {
      v21 = (unsigned int)(v80 + 1);
      v81 = (v151 - 1) & (v80 + v81);
      v82 = *(_QWORD *)(v149 + 8LL * v81);
      if ( v82 == v47 )
        goto LABEL_87;
      ++v80;
    }
LABEL_92:
    v54 = *(_QWORD *)(v45 - 32);
    v139 = *(unsigned __int8 **)(v45 - 64);
    v55 = sub_DD8400(*(_QWORD *)(a1 + 32), v54);
    v56 = *(_QWORD *)(*(_QWORD *)(v55[4] + 8) + 32LL);
    v57 = *(_DWORD *)(v56 + 32);
    v143 = v57;
    if ( v57 <= 0x40 )
    {
      v142 = *(_QWORD **)(v56 + 24);
      goto LABEL_94;
    }
    sub_C43780((__int64)&v142, (const void **)(v56 + 24));
    v57 = v143;
    if ( v143 <= 0x40 )
    {
LABEL_94:
      v58 = (unsigned __int64)v142;
      if ( v142 == (_QWORD *)v48 )
      {
        v62 = v143;
        goto LABEL_113;
      }
      v145 = v57;
    }
    else
    {
      v129 = v143;
      v94 = sub_C444A0((__int64)&v142);
      v62 = v129;
      if ( v129 - v94 <= 0x40 && *v142 == v48 )
        goto LABEL_113;
      v145 = v129;
      sub_C43780((__int64)&v144, (const void **)&v142);
      v57 = v145;
      if ( v145 > 0x40 )
      {
        sub_C43D10((__int64)&v144);
        goto LABEL_99;
      }
      v58 = (unsigned __int64)v144;
    }
    v59 = (char *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v57) & ~v58);
    if ( !v57 )
      v59 = 0;
    v144 = v59;
LABEL_99:
    sub_C46250((__int64)&v144);
    v60 = v145;
    v145 = 0;
    v22 = (__int64)&v144;
    LODWORD(v147) = v60;
    v146 = v144;
    v127 = v60;
    if ( v60 <= 0x40 )
    {
      v62 = v143;
      if ( v144 == (char *)v48 )
        goto LABEL_113;
LABEL_107:
      if ( v62 > 0x40 && v142 )
        j_j___libc_free_0_0((unsigned __int64)v142);
      if ( !v171 )
        _libc_free((unsigned __int64)v168);
    }
    else
    {
      v124 = v144;
      v61 = sub_C444A0((__int64)&v146);
      v132 = 1;
      v22 = (__int64)&v144;
      if ( v127 - v61 <= 0x40 )
        v132 = *(_QWORD *)v124 != v48;
      if ( v146 )
      {
        j_j___libc_free_0_0((unsigned __int64)v146);
        v22 = (__int64)&v144;
        if ( v145 > 0x40 )
        {
          if ( v144 )
          {
            j_j___libc_free_0_0((unsigned __int64)v144);
            v22 = (__int64)&v144;
          }
        }
      }
      v62 = v143;
      if ( v132 )
        goto LABEL_107;
LABEL_113:
      v145 = v62;
      if ( v62 <= 0x40 )
      {
        v63 = (unsigned __int64)v142;
        goto LABEL_115;
      }
      sub_C43780((__int64)&v144, (const void **)&v142);
      v62 = v145;
      if ( v145 <= 0x40 )
      {
        v63 = (unsigned __int64)v144;
LABEL_115:
        v64 = (char *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v62) & ~v63);
        if ( !v62 )
          v64 = 0;
        v144 = v64;
      }
      else
      {
        sub_C43D10((__int64)&v144);
      }
      sub_C46250((__int64)&v144);
      v65 = v145;
      v145 = 0;
      LODWORD(v147) = v65;
      v146 = v144;
      v133 = v65;
      if ( v65 <= 0x40 )
      {
        v134 = v144 == (char *)v48;
      }
      else
      {
        v128 = v144;
        v66 = sub_C444A0((__int64)&v146);
        v67 = v133;
        v134 = 0;
        if ( v67 - v66 <= 0x40 )
          v134 = *(_QWORD *)v128 == v48;
        if ( v128 )
        {
          j_j___libc_free_0_0((unsigned __int64)v128);
          if ( v145 > 0x40 )
          {
            if ( v144 )
              j_j___libc_free_0_0((unsigned __int64)v144);
          }
        }
      }
      v68 = sub_AE4570(*(_QWORD *)(a1 + 56), *(_QWORD *)(v54 + 8));
      v69 = sub_DA2C50(*(_QWORD *)(a1 + 32), v68, v48, 0);
      _BitScanReverse64(&v70, 1LL << (*(_WORD *)(v45 + 2) >> 1));
      LOBYTE(v71) = 63 - (v70 ^ 0x3F);
      HIBYTE(v71) = 1;
      v72 = sub_2820F60(a1, v54, (__int64)v69, v71, v139, v45, (__int64)&v167, (__int64)v55, a3, v134, 0);
      if ( v72 )
      {
        v100 = v168;
        if ( v171 )
        {
          v101 = HIDWORD(v169);
          v102 = &v168[HIDWORD(v169)];
        }
        else
        {
          v101 = (unsigned int)v169;
          v102 = &v168[(unsigned int)v169];
        }
        if ( v102 != v168 )
        {
          while ( 1 )
          {
            v103 = *v100;
            v104 = v100;
            if ( (unsigned __int64)*v100 < 0xFFFFFFFFFFFFFFFELL )
              break;
            if ( v102 == ++v100 )
              goto LABEL_127;
          }
          if ( v102 != v100 )
          {
            v105 = v177;
            if ( v177 )
            {
LABEL_219:
              v106 = v174;
              v101 = (__int64)&v174[HIDWORD(v175)];
              if ( v174 != (__int64 *)v101 )
              {
                do
                {
                  if ( v103 == *v106 )
                    goto LABEL_223;
                  ++v106;
                }
                while ( (__int64 *)v101 != v106 );
              }
              if ( HIDWORD(v175) < (unsigned int)v175 )
              {
                ++HIDWORD(v175);
                *(_QWORD *)v101 = v103;
                v105 = v177;
                ++v173;
                goto LABEL_223;
              }
            }
            while ( 1 )
            {
              sub_C8CC70((__int64)&v173, v103, v101, v105, v21, v22);
              v105 = v177;
LABEL_223:
              v107 = v104 + 1;
              if ( v102 == v104 + 1 )
                break;
              while ( 1 )
              {
                v103 = *v107;
                v104 = v107;
                if ( (unsigned __int64)*v107 < 0xFFFFFFFFFFFFFFFELL )
                  break;
                if ( v102 == ++v107 )
                  goto LABEL_127;
              }
              if ( v102 == v107 )
                break;
              if ( (_BYTE)v105 )
                goto LABEL_219;
            }
          }
        }
      }
      else
      {
        v72 = v136;
      }
LABEL_127:
      if ( v143 > 0x40 && v142 )
        j_j___libc_free_0_0((unsigned __int64)v142);
      if ( !v171 )
        _libc_free((unsigned __int64)v168);
      v136 = v72;
    }
LABEL_83:
    if ( v137 != ++v141 )
      continue;
    break;
  }
  if ( !v177 )
    _libc_free((unsigned __int64)v174);
LABEL_202:
  if ( v164 != (unsigned int *)v166 )
    _libc_free((unsigned __int64)v164);
  if ( (v161 & 1) == 0 )
    sub_C7D6A0((__int64)v162, 16LL * v163, 8);
  if ( v158 != &v160 )
    _libc_free((unsigned __int64)v158);
  sub_C7D6A0(v155, 8LL * (unsigned int)v157, 8);
  if ( v152 != &v154 )
    _libc_free((unsigned __int64)v152);
  sub_C7D6A0(v149, 8LL * (unsigned int)v151, 8);
  return v136;
}
