// Function: sub_3863FF0
// Address: 0x3863ff0
//
__int64 __fastcall sub_3863FF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5, __m128i a6)
{
  __int64 v6; // r15
  __int64 *v7; // rax
  __int64 *v8; // rcx
  __int64 v9; // rdx
  __int64 *v10; // rdi
  __int64 v11; // r14
  __int64 v12; // r13
  __int64 *v13; // rbx
  _QWORD *v14; // rbx
  _QWORD *v15; // rax
  _QWORD *v16; // r12
  _QWORD *v17; // rdx
  __int64 v18; // rsi
  __int64 v19; // rcx
  __int64 v20; // rsi
  __int64 v21; // r13
  _QWORD *v22; // rdi
  __int64 v23; // rcx
  __int64 v24; // rdx
  __int64 v25; // r13
  __int64 v26; // r8
  unsigned int v27; // esi
  unsigned int v28; // r9d
  __int64 v29; // r10
  unsigned __int64 v30; // rdi
  unsigned int v31; // edx
  _QWORD *v32; // rax
  __int64 v33; // r8
  _QWORD *v34; // rcx
  unsigned int *v35; // rbx
  unsigned int v36; // esi
  unsigned int v37; // r9d
  __int64 v38; // r12
  unsigned __int64 v39; // rcx
  unsigned int v40; // r10d
  _QWORD *v41; // rax
  __int64 v42; // rdi
  _QWORD *v43; // r8
  unsigned int *v44; // r11
  unsigned int *v45; // r14
  __int64 v46; // r15
  __int64 *v47; // rcx
  unsigned int v48; // r12d
  __int64 *v49; // rsi
  unsigned int v50; // ebx
  int v51; // r13d
  int v52; // r8d
  int v53; // r9d
  unsigned int v54; // eax
  bool v55; // zf
  __int64 v56; // rsi
  _QWORD *v57; // r10
  __int64 *v58; // r9
  __int64 *v59; // rsi
  __int64 *v60; // rdx
  __int64 v61; // rax
  bool v62; // cc
  unsigned int v63; // r9d
  _QWORD *v65; // r9
  __int64 *v66; // rdx
  _QWORD *v67; // rdx
  __int64 v68; // r11
  unsigned int v69; // r14d
  int v70; // r15d
  int v71; // eax
  int v72; // eax
  __int64 v73; // rax
  int v74; // eax
  int v75; // r15d
  __int64 v76; // r10
  unsigned __int64 v77; // r8
  int v78; // eax
  unsigned int v79; // ecx
  _QWORD *v80; // rdx
  __int64 v81; // rdi
  __int64 v82; // rax
  int v83; // r15d
  int v84; // eax
  int v85; // eax
  int v86; // r15d
  __int64 v87; // r10
  _QWORD *v88; // rsi
  int v89; // r12d
  unsigned __int64 v90; // r8
  unsigned int v91; // ecx
  __int64 v92; // rdi
  int v93; // eax
  int v94; // r10d
  __int64 v95; // r12
  unsigned __int64 v96; // r9
  unsigned int v97; // ecx
  __int64 v98; // r8
  _QWORD *v99; // rsi
  int v100; // edi
  int v101; // eax
  int v102; // r10d
  __int64 v103; // r12
  unsigned __int64 v104; // r9
  int v105; // edi
  unsigned int v106; // ecx
  __int64 v107; // r8
  _QWORD *v108; // r15
  _QWORD *v109; // r8
  _QWORD *v110; // rdx
  _QWORD *v111; // rcx
  _QWORD *v112; // rsi
  _BYTE *v113; // rdi
  _BYTE *v114; // rax
  _QWORD *v115; // rbx
  __int64 v116; // r11
  unsigned int v117; // r15d
  int v118; // r14d
  int v119; // eax
  int v120; // eax
  __int64 v121; // rax
  int v122; // edx
  __int64 v123; // rdx
  _QWORD *v124; // r11
  int v125; // r12d
  int v126; // edi
  int v127; // r12d
  const void *v128; // [rsp+8h] [rbp-138h]
  __int64 v129; // [rsp+10h] [rbp-130h]
  unsigned int *v130; // [rsp+20h] [rbp-120h]
  unsigned __int64 v132; // [rsp+30h] [rbp-110h]
  _QWORD *v133; // [rsp+30h] [rbp-110h]
  unsigned __int64 v134; // [rsp+38h] [rbp-108h]
  _QWORD *v135; // [rsp+38h] [rbp-108h]
  __int64 *v136; // [rsp+48h] [rbp-F8h]
  _QWORD *v137; // [rsp+50h] [rbp-F0h]
  unsigned __int64 *v138; // [rsp+58h] [rbp-E8h]
  _QWORD *v139; // [rsp+58h] [rbp-E8h]
  unsigned int *v141; // [rsp+68h] [rbp-D8h]
  unsigned int *v142; // [rsp+68h] [rbp-D8h]
  unsigned int *v143; // [rsp+68h] [rbp-D8h]
  _QWORD *v144; // [rsp+68h] [rbp-D8h]
  unsigned int *v145; // [rsp+70h] [rbp-D0h]
  _QWORD *v146; // [rsp+70h] [rbp-D0h]
  __int64 *v147; // [rsp+78h] [rbp-C8h]
  unsigned __int8 v148; // [rsp+78h] [rbp-C8h]
  _QWORD *v149; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v150; // [rsp+88h] [rbp-B8h]
  __int64 v151; // [rsp+90h] [rbp-B0h]
  __int64 v152; // [rsp+A0h] [rbp-A0h] BYREF
  __int64 *v153; // [rsp+A8h] [rbp-98h]
  __int64 *v154; // [rsp+B0h] [rbp-90h]
  __int64 v155; // [rsp+B8h] [rbp-88h]
  int v156; // [rsp+C0h] [rbp-80h]
  _BYTE v157[120]; // [rsp+C8h] [rbp-78h] BYREF

  v6 = a1;
  v7 = (__int64 *)v157;
  *(_QWORD *)(a1 + 200) = -1;
  v8 = *(__int64 **)a3;
  v9 = *(unsigned int *)(a3 + 8);
  v152 = 0;
  v153 = (__int64 *)v157;
  v154 = (__int64 *)v157;
  v155 = 8;
  v156 = 0;
  v136 = &v8[v9];
  if ( v8 == v136 )
  {
    v10 = (__int64 *)v157;
    goto LABEL_93;
  }
  v147 = v8;
  v10 = (__int64 *)v157;
  v11 = v6;
  while ( 2 )
  {
    v12 = *v147;
    if ( v7 == v10 )
    {
      v13 = &v7[HIDWORD(v155)];
      if ( v13 == v7 )
      {
        v66 = v7;
      }
      else
      {
        do
        {
          if ( *v7 == v12 )
            break;
          ++v7;
        }
        while ( v13 != v7 );
        v66 = v13;
      }
      goto LABEL_61;
    }
    v13 = &v10[(unsigned int)v155];
    v7 = sub_16CC9F0((__int64)&v152, *v147);
    if ( *v7 == v12 )
    {
      if ( v154 == v153 )
        v66 = &v154[HIDWORD(v155)];
      else
        v66 = &v154[(unsigned int)v155];
LABEL_61:
      while ( v66 != v7 && (unsigned __int64)*v7 >= 0xFFFFFFFFFFFFFFFELL )
        ++v7;
      goto LABEL_9;
    }
    if ( v154 == v153 )
    {
      v66 = &v154[HIDWORD(v155)];
      v7 = v66;
      goto LABEL_61;
    }
    v7 = &v154[(unsigned int)v155];
LABEL_9:
    if ( v13 != v7 )
      goto LABEL_3;
    v151 = v12;
    v149 = &v149;
    v150 = 1;
    v14 = (_QWORD *)(a2 + 8);
    v15 = *(_QWORD **)(a2 + 16);
    if ( !v15 )
      goto LABEL_223;
    v16 = (_QWORD *)(a2 + 8);
    v17 = *(_QWORD **)(a2 + 16);
    do
    {
      while ( 1 )
      {
        v18 = v17[2];
        v19 = v17[3];
        if ( v12 <= v17[6] )
          break;
        v17 = (_QWORD *)v17[3];
        if ( !v19 )
          goto LABEL_15;
      }
      v16 = v17;
      v17 = (_QWORD *)v17[2];
    }
    while ( v18 );
LABEL_15:
    if ( v16 == v14 || (v20 = v16[6], v12 < v20) )
LABEL_223:
      BUG();
    if ( (v16[5] & 1) != 0 )
      goto LABEL_20;
    v21 = v16[4];
    if ( (*(_BYTE *)(v21 + 8) & 1) != 0 )
    {
      v20 = *(_QWORD *)(v21 + 16);
      goto LABEL_20;
    }
    v57 = *(_QWORD **)v21;
    if ( (*(_BYTE *)(*(_QWORD *)v21 + 8LL) & 1) == 0 )
    {
      v65 = (_QWORD *)*v57;
      if ( (*(_BYTE *)(*v57 + 8LL) & 1) != 0 )
      {
        v57 = (_QWORD *)*v57;
      }
      else
      {
        v108 = (_QWORD *)*v65;
        if ( (*(_BYTE *)(*v65 + 8LL) & 1) == 0 )
        {
          v109 = (_QWORD *)*v108;
          if ( (*(_BYTE *)(*v108 + 8LL) & 1) != 0 )
          {
            v108 = (_QWORD *)*v108;
          }
          else
          {
            v110 = (_QWORD *)*v109;
            if ( (*(_BYTE *)(*v109 + 8LL) & 1) == 0 )
            {
              v111 = (_QWORD *)*v110;
              if ( (*(_BYTE *)(*v110 + 8LL) & 1) == 0 )
              {
                v112 = (_QWORD *)*v111;
                if ( (*(_BYTE *)(*v111 + 8LL) & 1) == 0 )
                {
                  v113 = (_BYTE *)*v112;
                  v133 = (_QWORD *)*v111;
                  if ( (*(_BYTE *)(*v112 + 8LL) & 1) == 0 )
                  {
                    v135 = (_QWORD *)*v110;
                    v137 = (_QWORD *)*v109;
                    v139 = (_QWORD *)*v108;
                    v144 = (_QWORD *)*v57;
                    v146 = *(_QWORD **)v21;
                    v114 = sub_3863620(v113);
                    v111 = v135;
                    v110 = v137;
                    v109 = v139;
                    v113 = v114;
                    *v133 = v114;
                    v65 = v144;
                    v57 = v146;
                  }
                  *v111 = v113;
                  v112 = v113;
                }
                *v110 = v112;
                v111 = v112;
              }
              *v109 = v111;
              v110 = v111;
            }
            *v108 = v110;
            v108 = v110;
          }
          *v65 = v108;
        }
        *v57 = v108;
        v57 = v108;
      }
      *(_QWORD *)v21 = v57;
    }
    v16[4] = v57;
    v20 = v57[2];
    v15 = *(_QWORD **)(a2 + 16);
    if ( v15 )
    {
LABEL_20:
      v22 = (_QWORD *)(a2 + 8);
      do
      {
        while ( 1 )
        {
          v23 = v15[2];
          v24 = v15[3];
          if ( v15[6] >= v20 )
            break;
          v15 = (_QWORD *)v15[3];
          if ( !v24 )
            goto LABEL_24;
        }
        v22 = v15;
        v15 = (_QWORD *)v15[2];
      }
      while ( v23 );
LABEL_24:
      if ( v14 != v22 && v22[6] <= v20 )
        v14 = v22;
    }
    if ( (v14[5] & 1) == 0 )
      goto LABEL_3;
    v25 = v11;
    v132 = (unsigned __int64)(v14 + 4);
    do
    {
      v26 = *(_QWORD *)(v132 + 16);
      v7 = v153;
      v138 = (unsigned __int64 *)(v132 + 16);
      v10 = v154;
      if ( v154 != v153 )
        goto LABEL_30;
      v58 = &v154[HIDWORD(v155)];
      if ( v154 != v58 )
      {
        v59 = 0;
        v60 = v154;
        while ( v26 != *v60 )
        {
          if ( *v60 == -2 )
            v59 = v60;
          if ( v58 == ++v60 )
          {
            if ( !v59 )
              goto LABEL_181;
            *v59 = v26;
            v10 = v154;
            --v156;
            v7 = v153;
            ++v152;
            goto LABEL_31;
          }
        }
        goto LABEL_31;
      }
LABEL_181:
      if ( HIDWORD(v155) < (unsigned int)v155 )
      {
        ++HIDWORD(v155);
        *v58 = v26;
        v7 = v153;
        ++v152;
        v10 = v154;
      }
      else
      {
LABEL_30:
        sub_16CCBA0((__int64)&v152, v26);
        v10 = v154;
        v7 = v153;
      }
LABEL_31:
      v134 = *(_QWORD *)(v132 + 8) & 0xFFFFFFFFFFFFFFFELL;
      if ( !v134 )
      {
        v11 = v25;
        goto LABEL_4;
      }
      v129 = v25 + 16;
      v128 = (const void *)(v25 + 240);
      while ( 2 )
      {
        v27 = *(_DWORD *)(v25 + 40);
        if ( !v27 )
        {
          ++*(_QWORD *)(v25 + 16);
LABEL_171:
          v27 *= 2;
LABEL_172:
          sub_3863DD0(v129, v27);
          sub_38632B0(v129, v138, &v149);
          v34 = v149;
          v120 = *(_DWORD *)(v25 + 32) + 1;
          goto LABEL_161;
        }
        v28 = v27 - 1;
        v29 = *(_QWORD *)(v25 + 24);
        v30 = *(_QWORD *)(v132 + 16);
        v31 = (v27 - 1) & (v30 ^ (v30 >> 9));
        v32 = (_QWORD *)(v29 + 32LL * v31);
        v33 = *v32;
        v34 = v32;
        if ( v30 == *v32 )
        {
LABEL_35:
          v35 = (unsigned int *)v34[1];
          goto LABEL_36;
        }
        v115 = 0;
        v116 = *v32;
        v117 = (v27 - 1) & (v30 ^ (*(_QWORD *)(v132 + 16) >> 9));
        v118 = 1;
        while ( v116 != -4 )
        {
          if ( !v115 && v116 == -16 )
            v115 = v34;
          v117 = v28 & (v118 + v117);
          v34 = (_QWORD *)(v29 + 32LL * v117);
          v116 = *v34;
          if ( v30 == *v34 )
            goto LABEL_35;
          ++v118;
        }
        v119 = *(_DWORD *)(v25 + 32);
        if ( v115 )
          v34 = v115;
        ++*(_QWORD *)(v25 + 16);
        v120 = v119 + 1;
        if ( 4 * v120 >= 3 * v27 )
          goto LABEL_171;
        if ( v27 - *(_DWORD *)(v25 + 36) - v120 <= v27 >> 3 )
          goto LABEL_172;
LABEL_161:
        *(_DWORD *)(v25 + 32) = v120;
        if ( *v34 != -4 )
          --*(_DWORD *)(v25 + 36);
        v121 = *(_QWORD *)(v132 + 16);
        v34[1] = 0;
        v34[2] = 0;
        *v34 = v121;
        v34[3] = 0;
        v27 = *(_DWORD *)(v25 + 40);
        if ( !v27 )
        {
          ++*(_QWORD *)(v25 + 16);
          v35 = 0;
          goto LABEL_165;
        }
        v28 = v27 - 1;
        v29 = *(_QWORD *)(v25 + 24);
        v35 = 0;
        v30 = *(_QWORD *)(v132 + 16);
        v31 = (v27 - 1) & (v30 ^ (v30 >> 9));
        v32 = (_QWORD *)(v29 + 32LL * v31);
        v33 = *v32;
LABEL_36:
        if ( v30 == v33 )
        {
LABEL_37:
          v130 = (unsigned int *)v32[2];
          goto LABEL_38;
        }
        v124 = 0;
        v125 = 1;
        while ( v33 != -4 )
        {
          if ( v33 == -16 && !v124 )
            v124 = v32;
          v31 = v28 & (v125 + v31);
          v32 = (_QWORD *)(v29 + 32LL * v31);
          v33 = *v32;
          if ( v30 == *v32 )
            goto LABEL_37;
          ++v125;
        }
        v126 = *(_DWORD *)(v25 + 32);
        if ( v124 )
          v32 = v124;
        ++*(_QWORD *)(v25 + 16);
        v122 = v126 + 1;
        if ( 4 * (v126 + 1) < 3 * v27 )
        {
          if ( v27 - (v122 + *(_DWORD *)(v25 + 36)) > v27 >> 3 )
            goto LABEL_167;
          goto LABEL_166;
        }
LABEL_165:
        v27 *= 2;
LABEL_166:
        sub_3863DD0(v129, v27);
        sub_38632B0(v129, v138, &v149);
        v32 = v149;
        v122 = *(_DWORD *)(v25 + 32) + 1;
LABEL_167:
        *(_DWORD *)(v25 + 32) = v122;
        if ( *v32 != -4 )
          --*(_DWORD *)(v25 + 36);
        v130 = 0;
        v123 = *(_QWORD *)(v132 + 16);
        v32[1] = 0;
        v32[2] = 0;
        *v32 = v123;
        v32[3] = 0;
LABEL_38:
        v145 = v35;
        if ( v130 == v35 )
          goto LABEL_85;
        do
        {
          v36 = *(_DWORD *)(v25 + 40);
          if ( !v36 )
          {
            ++*(_QWORD *)(v25 + 16);
            goto LABEL_126;
          }
          v37 = v36 - 1;
          v38 = *(_QWORD *)(v25 + 24);
          v39 = *(_QWORD *)(v134 + 16);
          v40 = (v36 - 1) & (v39 ^ (v39 >> 9));
          v41 = (_QWORD *)(v38 + 32LL * v40);
          v42 = *v41;
          v43 = v41;
          if ( *v41 != v39 )
          {
            v67 = 0;
            v68 = *v41;
            v69 = (v36 - 1) & (v39 ^ (*(_QWORD *)(v134 + 16) >> 9));
            v70 = 1;
            while ( v68 != -4 )
            {
              if ( !v67 && v68 == -16 )
                v67 = v43;
              v69 = v37 & (v70 + v69);
              v43 = (_QWORD *)(v38 + 32LL * v69);
              v68 = *v43;
              if ( v39 == *v43 )
                goto LABEL_41;
              ++v70;
            }
            v71 = *(_DWORD *)(v25 + 32);
            if ( !v67 )
              v67 = v43;
            ++*(_QWORD *)(v25 + 16);
            v72 = v71 + 1;
            if ( 4 * v72 < 3 * v36 )
            {
              if ( v36 - *(_DWORD *)(v25 + 36) - v72 <= v36 >> 3 )
              {
                sub_3863DD0(v129, v36);
                v101 = *(_DWORD *)(v25 + 40);
                if ( !v101 )
                {
LABEL_225:
                  ++*(_DWORD *)(v25 + 32);
                  BUG();
                }
                v102 = v101 - 1;
                v103 = *(_QWORD *)(v25 + 24);
                v99 = 0;
                v104 = *(_QWORD *)(v134 + 16);
                v105 = 1;
                v72 = *(_DWORD *)(v25 + 32) + 1;
                v106 = v102 & (v104 ^ (v104 >> 9));
                v67 = (_QWORD *)(v103 + 32LL * v106);
                v107 = *v67;
                if ( *v67 != v104 )
                {
                  while ( v107 != -4 )
                  {
                    if ( !v99 && v107 == -16 )
                      v99 = v67;
                    v106 = v102 & (v105 + v106);
                    v67 = (_QWORD *)(v103 + 32LL * v106);
                    v107 = *v67;
                    if ( v104 == *v67 )
                      goto LABEL_103;
                    ++v105;
                  }
LABEL_138:
                  if ( v99 )
                    v67 = v99;
                  goto LABEL_103;
                }
              }
              goto LABEL_103;
            }
LABEL_126:
            sub_3863DD0(v129, 2 * v36);
            v93 = *(_DWORD *)(v25 + 40);
            if ( !v93 )
              goto LABEL_225;
            v94 = v93 - 1;
            v95 = *(_QWORD *)(v25 + 24);
            v96 = *(_QWORD *)(v134 + 16);
            v72 = *(_DWORD *)(v25 + 32) + 1;
            v97 = v94 & (v96 ^ (v96 >> 9));
            v67 = (_QWORD *)(v95 + 32LL * v97);
            v98 = *v67;
            if ( *v67 != v96 )
            {
              v99 = 0;
              v100 = 1;
              while ( v98 != -4 )
              {
                if ( !v99 && v98 == -16 )
                  v99 = v67;
                v97 = v94 & (v100 + v97);
                v67 = (_QWORD *)(v95 + 32LL * v97);
                v98 = *v67;
                if ( v96 == *v67 )
                  goto LABEL_103;
                ++v100;
              }
              goto LABEL_138;
            }
LABEL_103:
            *(_DWORD *)(v25 + 32) = v72;
            if ( *v67 != -4 )
              --*(_DWORD *)(v25 + 36);
            v73 = *(_QWORD *)(v134 + 16);
            v67[1] = 0;
            v67[2] = 0;
            *v67 = v73;
            v67[3] = 0;
            v36 = *(_DWORD *)(v25 + 40);
            if ( v36 )
            {
              v37 = v36 - 1;
              v38 = *(_QWORD *)(v25 + 24);
              v44 = 0;
              v39 = *(_QWORD *)(v134 + 16);
              v40 = (v36 - 1) & (v39 ^ (v39 >> 9));
              v41 = (_QWORD *)(v38 + 32LL * v40);
              v42 = *v41;
              if ( *v41 == v39 )
                goto LABEL_42;
LABEL_113:
              v80 = 0;
              v83 = 1;
              while ( v42 != -4 )
              {
                if ( v42 == -16 && !v80 )
                  v80 = v41;
                v40 = v37 & (v83 + v40);
                v41 = (_QWORD *)(v38 + 32LL * v40);
                v42 = *v41;
                if ( *v41 == v39 )
                  goto LABEL_42;
                ++v83;
              }
              if ( !v80 )
                v80 = v41;
              v84 = *(_DWORD *)(v25 + 32);
              ++*(_QWORD *)(v25 + 16);
              v78 = v84 + 1;
              if ( 4 * v78 < 3 * v36 )
              {
                if ( v36 - (v78 + *(_DWORD *)(v25 + 36)) > v36 >> 3 )
                  goto LABEL_109;
                v143 = v44;
                sub_3863DD0(v129, v36);
                v85 = *(_DWORD *)(v25 + 40);
                if ( !v85 )
                {
LABEL_224:
                  ++*(_DWORD *)(v25 + 32);
                  BUG();
                }
                v86 = v85 - 1;
                v87 = *(_QWORD *)(v25 + 24);
                v88 = 0;
                v44 = v143;
                v89 = 1;
                v78 = *(_DWORD *)(v25 + 32) + 1;
                v90 = *(_QWORD *)(v134 + 16);
                v91 = v86 & (v90 ^ (v90 >> 9));
                v80 = (_QWORD *)(v87 + 32LL * v91);
                v92 = *v80;
                if ( v90 == *v80 )
                  goto LABEL_109;
                while ( v92 != -4 )
                {
                  if ( !v88 && v92 == -16 )
                    v88 = v80;
                  v91 = v86 & (v89 + v91);
                  v80 = (_QWORD *)(v87 + 32LL * v91);
                  v92 = *v80;
                  if ( v90 == *v80 )
                    goto LABEL_109;
                  ++v89;
                }
                goto LABEL_122;
              }
            }
            else
            {
              ++*(_QWORD *)(v25 + 16);
              v44 = 0;
            }
            v142 = v44;
            sub_3863DD0(v129, 2 * v36);
            v74 = *(_DWORD *)(v25 + 40);
            if ( !v74 )
              goto LABEL_224;
            v75 = v74 - 1;
            v76 = *(_QWORD *)(v25 + 24);
            v44 = v142;
            v77 = *(_QWORD *)(v134 + 16);
            v78 = *(_DWORD *)(v25 + 32) + 1;
            v79 = v75 & (v77 ^ (v77 >> 9));
            v80 = (_QWORD *)(v76 + 32LL * v79);
            v81 = *v80;
            if ( *v80 == v77 )
              goto LABEL_109;
            v88 = 0;
            v127 = 1;
            while ( v81 != -4 )
            {
              if ( !v88 && v81 == -16 )
                v88 = v80;
              v79 = v75 & (v127 + v79);
              v80 = (_QWORD *)(v76 + 32LL * v79);
              v81 = *v80;
              if ( v77 == *v80 )
                goto LABEL_109;
              ++v127;
            }
LABEL_122:
            if ( v88 )
              v80 = v88;
LABEL_109:
            *(_DWORD *)(v25 + 32) = v78;
            if ( *v80 != -4 )
              --*(_DWORD *)(v25 + 36);
            v141 = 0;
            v82 = *(_QWORD *)(v134 + 16);
            v80[1] = 0;
            v80[2] = 0;
            *v80 = v82;
            v80[3] = 0;
            goto LABEL_43;
          }
LABEL_41:
          v44 = (unsigned int *)v43[1];
          if ( v42 != v39 )
            goto LABEL_113;
LABEL_42:
          v141 = (unsigned int *)v41[2];
LABEL_43:
          if ( v44 == v141 )
            goto LABEL_84;
          v45 = v44;
          v46 = v25;
          do
          {
            v50 = *v45;
            v48 = *v145;
            if ( *v45 < *v145 )
            {
              v47 = (__int64 *)(v132 + 16);
              v48 = *v45;
              v49 = (__int64 *)(v134 + 16);
              v50 = *v145;
            }
            else
            {
              v47 = (__int64 *)(v134 + 16);
              v49 = (__int64 *)(v132 + 16);
            }
            v51 = sub_385F910(v46, v49, a5, a6, v48, v47, v50, a4);
            v54 = sub_385F800(v51);
            LOBYTE(v54) = *(_BYTE *)(v46 + 217) & v54;
            v55 = *(_BYTE *)(v46 + 218) == 0;
            *(_BYTE *)(v46 + 217) = v54;
            if ( !v55 )
            {
              v56 = *(unsigned int *)(v46 + 232);
              if ( !v51 )
              {
                if ( (unsigned int)v56 < dword_5052200 )
                  goto LABEL_51;
                goto LABEL_49;
              }
              v149 = (_QWORD *)__PAIR64__(v50, v48);
              LODWORD(v150) = v51;
              if ( (unsigned int)v56 >= *(_DWORD *)(v46 + 236) )
              {
                sub_16CD150(v46 + 224, v128, 0, 12, v52, v53);
                v56 = *(unsigned int *)(v46 + 232);
              }
              v61 = *(_QWORD *)(v46 + 224) + 12 * v56;
              *(_QWORD *)v61 = v149;
              *(_DWORD *)(v61 + 8) = v150;
              LODWORD(v61) = *(_DWORD *)(v46 + 232) + 1;
              v62 = dword_5052200 <= (unsigned int)v61;
              *(_DWORD *)(v46 + 232) = v61;
              if ( v62 )
              {
                v54 = *(unsigned __int8 *)(v46 + 217);
LABEL_49:
                *(_BYTE *)(v46 + 218) = 0;
                *(_DWORD *)(v46 + 232) = 0;
                goto LABEL_50;
              }
              if ( *(_BYTE *)(v46 + 218) )
                goto LABEL_51;
              v54 = *(unsigned __int8 *)(v46 + 217);
            }
LABEL_50:
            if ( !(_BYTE)v54 )
            {
              v63 = v54;
              v10 = v154;
              v7 = v153;
              goto LABEL_89;
            }
LABEL_51:
            ++v45;
          }
          while ( v141 != v45 );
          v25 = v46;
LABEL_84:
          ++v145;
        }
        while ( v130 != v145 );
LABEL_85:
        v134 = *(_QWORD *)(v134 + 8) & 0xFFFFFFFFFFFFFFFELL;
        if ( v134 )
          continue;
        break;
      }
      v132 = *(_QWORD *)(v132 + 8) & 0xFFFFFFFFFFFFFFFELL;
    }
    while ( v132 );
    v11 = v25;
LABEL_3:
    v10 = v154;
    v7 = v153;
LABEL_4:
    if ( v136 != ++v147 )
      continue;
    break;
  }
  v6 = v11;
LABEL_93:
  v63 = *(unsigned __int8 *)(v6 + 217);
LABEL_89:
  if ( v10 != v7 )
  {
    v148 = v63;
    _libc_free((unsigned __int64)v10);
    return v148;
  }
  return v63;
}
