// Function: sub_399B1E0
// Address: 0x399b1e0
//
void __fastcall sub_399B1E0(__int64 a1)
{
  char v1; // r13
  __int64 v2; // rdi
  __int64 v3; // rbx
  int v4; // eax
  int v5; // r12d
  __int64 v6; // rax
  int v7; // eax
  int v8; // ebx
  bool v9; // al
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r9
  char *v13; // r12
  char *v14; // r11
  __int64 v15; // r10
  int v16; // r8d
  unsigned int v17; // ecx
  _QWORD *v18; // rbx
  __int64 v19; // rdx
  unsigned int v20; // edx
  __int64 v21; // r13
  __int64 *v22; // rax
  __int64 v23; // r13
  __int64 v24; // r14
  unsigned int v25; // ecx
  unsigned int v26; // eax
  __int64 v27; // rdi
  int v28; // edx
  __int64 v29; // rbx
  __int64 v30; // rdi
  __int64 v31; // rbx
  __int64 v32; // rdi
  __int64 v33; // rbx
  int v34; // eax
  __int64 *v35; // r15
  __int64 v36; // r11
  __int64 v37; // rax
  __int64 *v38; // r13
  __int64 *v39; // r15
  int v40; // r9d
  int v41; // r8d
  unsigned int v42; // edi
  _QWORD *v43; // rbx
  __int64 v44; // rcx
  __int64 v45; // r12
  __int64 v46; // rax
  _QWORD *v47; // rax
  __int64 v48; // r12
  __int64 v49; // rdx
  __int64 v50; // r14
  unsigned int v51; // eax
  __int64 v52; // rdi
  int v53; // ecx
  __int64 v54; // rax
  unsigned int v55; // esi
  unsigned __int64 v56; // rax
  unsigned int v57; // ebx
  unsigned int v58; // edi
  _QWORD *v59; // rcx
  __int64 v60; // rdx
  __int64 v61; // r10
  unsigned int v62; // edx
  _QWORD *v63; // r9
  __int64 v64; // rsi
  int v65; // eax
  unsigned int v66; // edx
  __int64 v67; // rax
  __m128i *v68; // r12
  __int64 v69; // rbx
  unsigned __int64 v70; // rax
  __int64 v71; // r12
  const __m128i *v72; // rbx
  bool v73; // al
  __int64 v74; // r13
  __int64 *v75; // r14
  __m128i *v76; // rdx
  __m128i *v77; // rbx
  __int64 v78; // rax
  __m128i *v79; // rdi
  __int64 v80; // rbx
  const __m128i *v81; // rsi
  __m128i *v82; // rdi
  const __m128i *v83; // rax
  __int64 v84; // rax
  __int64 v85; // rbx
  __int64 v86; // rdx
  __int64 *v87; // r12
  __int64 v88; // rsi
  __int64 v89; // r12
  __int64 v90; // rdx
  unsigned __int8 **v91; // rbx
  unsigned __int8 *v92; // rsi
  unsigned __int8 v93; // al
  __int64 v94; // rbx
  __int64 v95; // rax
  __int64 *v96; // r12
  __int64 v97; // rdx
  _QWORD *v98; // rbx
  _QWORD *v99; // r12
  unsigned __int64 v100; // rdi
  int v101; // r10d
  int v102; // r8d
  _QWORD *v103; // rdi
  unsigned int v104; // ebx
  __int64 v105; // rcx
  int v106; // r11d
  _QWORD *v107; // r10
  _QWORD *v108; // r8
  int v109; // r9d
  unsigned int v110; // eax
  __int64 v111; // rsi
  int v112; // r8d
  __int64 v113; // r15
  _QWORD *v114; // rdi
  __int64 v115; // rcx
  int v116; // r9d
  _QWORD *v117; // r8
  int v118; // edx
  unsigned int v119; // eax
  __int64 v120; // rdi
  _QWORD *v121; // r10
  int v122; // esi
  _QWORD *v123; // rcx
  unsigned int v124; // ebx
  __int64 v125; // rcx
  _QWORD *v126; // r9
  int v127; // eax
  _QWORD *v128; // rdi
  int v129; // r9d
  _QWORD *v130; // r8
  int v131; // r10d
  _QWORD *v132; // r8
  int v133; // r15d
  int v134; // [rsp+Ch] [rbp-144h]
  __int64 v136; // [rsp+28h] [rbp-128h]
  __int64 v137; // [rsp+30h] [rbp-120h]
  __int64 v138; // [rsp+38h] [rbp-118h]
  __int64 v139; // [rsp+40h] [rbp-110h]
  _QWORD *v140; // [rsp+48h] [rbp-108h]
  __int64 v141; // [rsp+50h] [rbp-100h]
  __int64 v142; // [rsp+50h] [rbp-100h]
  __int64 v143; // [rsp+58h] [rbp-F8h]
  const __m128i *v144; // [rsp+58h] [rbp-F8h]
  char *v145; // [rsp+60h] [rbp-F0h]
  __m128i *v146; // [rsp+60h] [rbp-F0h]
  char *v147; // [rsp+60h] [rbp-F0h]
  char *v148; // [rsp+60h] [rbp-F0h]
  __int64 v149; // [rsp+68h] [rbp-E8h]
  __int64 v150; // [rsp+68h] [rbp-E8h]
  __int64 v151; // [rsp+68h] [rbp-E8h]
  int v152; // [rsp+68h] [rbp-E8h]
  __int64 v153; // [rsp+68h] [rbp-E8h]
  __int64 v154; // [rsp+70h] [rbp-E0h]
  __int64 v155; // [rsp+78h] [rbp-D8h]
  __int64 *v156; // [rsp+78h] [rbp-D8h]
  int v157[2]; // [rsp+88h] [rbp-C8h] BYREF
  __int64 v158; // [rsp+90h] [rbp-C0h] BYREF
  int v159; // [rsp+98h] [rbp-B8h]
  char v160[8]; // [rsp+A0h] [rbp-B0h] BYREF
  unsigned __int64 v161; // [rsp+A8h] [rbp-A8h]
  char v162; // [rsp+B0h] [rbp-A0h]
  __int64 v163; // [rsp+C0h] [rbp-90h] BYREF
  unsigned __int64 v164; // [rsp+C8h] [rbp-88h]
  bool v165; // [rsp+D0h] [rbp-80h]
  __int64 v166; // [rsp+E0h] [rbp-70h] BYREF
  _QWORD *v167; // [rsp+E8h] [rbp-68h]
  __int64 v168; // [rsp+F0h] [rbp-60h]
  unsigned int v169; // [rsp+F8h] [rbp-58h]
  char *v170; // [rsp+100h] [rbp-50h] BYREF
  unsigned __int64 v171; // [rsp+108h] [rbp-48h]
  __int64 v172; // [rsp+110h] [rbp-40h] BYREF
  __int64 v173; // [rsp+118h] [rbp-38h]

  sub_16D8B50(
    (__m128i **)v157,
    (unsigned __int8 *)"writer",
    6u,
    (__int64)"DWARF Debug Writer",
    18,
    unk_4F9E388,
    (unsigned __int8 *)"dwarf",
    5u,
    "DWARF Emission",
    (double *)0xE);
  v1 = byte_5057420;
  if ( byte_5057420 )
    goto LABEL_2;
  v2 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 1688LL);
  v170 = "llvm.dbg.cu";
  v141 = v2;
  LOWORD(v172) = 259;
  v3 = sub_1632310(v2, (__int64)&v170);
  v4 = 0;
  if ( v3 )
    v4 = sub_161F520(v3);
  LODWORD(v171) = v4;
  v170 = (char *)v3;
  sub_1632FD0((__int64)&v170);
  v5 = v171;
  LOWORD(v172) = 259;
  v170 = "llvm.dbg.cu";
  v6 = sub_1632310(v2, (__int64)&v170);
  LODWORD(v171) = 0;
  v170 = (char *)v6;
  sub_1632FD0((__int64)&v170);
  v7 = v171;
  if ( v5 == (_DWORD)v171 )
  {
    v9 = 0;
  }
  else
  {
    v8 = 0;
    do
    {
      ++v8;
      LODWORD(v171) = v7 + 1;
      sub_1632FD0((__int64)&v170);
      v7 = v171;
    }
    while ( v5 != (_DWORD)v171 );
    v1 = v8 != 0;
    v9 = v8 == 1;
  }
  v168 = 0;
  v169 = 0;
  v10 = *(_QWORD *)(a1 + 16);
  v166 = 0;
  v167 = 0;
  *(_BYTE *)(v10 + 1744) = v1;
  *(_BYTE *)(a1 + 5408) = v9;
  v143 = v2 + 8;
  v155 = *(_QWORD *)(v2 + 16);
  if ( v155 != v2 + 8 )
  {
    while ( 1 )
    {
      v11 = 0;
      if ( v155 )
        v11 = v155 - 56;
      v149 = v11;
      v170 = (char *)&v172;
      v171 = 0x100000000LL;
      sub_1626700(v11, (__int64)&v170);
      v13 = v170;
      v14 = &v170[8 * (unsigned int)v171];
      if ( v170 != v14 )
        break;
LABEL_28:
      if ( v14 != (char *)&v172 )
        _libc_free((unsigned __int64)v14);
      v155 = *(_QWORD *)(v155 + 8);
      if ( v143 == v155 )
        goto LABEL_31;
    }
    v15 = v149;
    while ( 1 )
    {
      v23 = *(_QWORD *)v13;
      v24 = *(_QWORD *)(*(_QWORD *)v13 - 8LL * *(unsigned int *)(*(_QWORD *)v13 + 8LL));
      if ( v169 )
      {
        v16 = v169 - 1;
        v17 = (v169 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
        v18 = &v167[5 * v17];
        v19 = *v18;
        if ( v24 == *v18 )
        {
LABEL_17:
          v20 = *((_DWORD *)v18 + 4);
          v21 = *(_QWORD *)(v23 + 8 * (1LL - *(unsigned int *)(*(_QWORD *)v13 + 8LL)));
          if ( v20 >= *((_DWORD *)v18 + 5) )
          {
            v147 = v14;
            v151 = v15;
            sub_16CD150((__int64)(v18 + 1), v18 + 3, 0, 16, v16, v12);
            v20 = *((_DWORD *)v18 + 4);
            v14 = v147;
            v15 = v151;
          }
          goto LABEL_19;
        }
        v152 = 1;
        v12 = 0;
        while ( v19 != -8 )
        {
          if ( v19 != -16 || v12 )
            v18 = (_QWORD *)v12;
          LODWORD(v12) = v152 + 1;
          v17 = v16 & (v152 + v17);
          v19 = v167[5 * v17];
          if ( v24 == v19 )
          {
            v18 = &v167[5 * v17];
            goto LABEL_17;
          }
          ++v152;
          v12 = (__int64)v18;
          v18 = &v167[5 * v17];
        }
        if ( v12 )
          v18 = (_QWORD *)v12;
        ++v166;
        v28 = v168 + 1;
        if ( 4 * ((int)v168 + 1) < 3 * v169 )
        {
          if ( v169 - HIDWORD(v168) - v28 <= v169 >> 3 )
          {
            v148 = v14;
            v153 = v15;
            sub_3992510((__int64)&v166, v169);
            if ( !v169 )
            {
LABEL_269:
              LODWORD(v168) = v168 + 1;
              BUG();
            }
            v112 = 1;
            LODWORD(v113) = (v169 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
            v15 = v153;
            v14 = v148;
            v18 = &v167[5 * (unsigned int)v113];
            v28 = v168 + 1;
            v114 = 0;
            v115 = *v18;
            if ( v24 != *v18 )
            {
              while ( v115 != -8 )
              {
                if ( v115 == -16 && !v114 )
                  v114 = v18;
                LODWORD(v12) = v112 + 1;
                v113 = (v169 - 1) & ((_DWORD)v113 + v112);
                v18 = &v167[5 * v113];
                v115 = *v18;
                if ( v24 == *v18 )
                  goto LABEL_24;
                ++v112;
              }
              if ( v114 )
                v18 = v114;
            }
          }
          goto LABEL_24;
        }
      }
      else
      {
        ++v166;
      }
      v145 = v14;
      v150 = v15;
      sub_3992510((__int64)&v166, 2 * v169);
      if ( !v169 )
        goto LABEL_269;
      v25 = v169 - 1;
      v15 = v150;
      v14 = v145;
      v26 = (v169 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v18 = &v167[5 * v26];
      v27 = *v18;
      v28 = v168 + 1;
      if ( v24 != *v18 )
      {
        LODWORD(v12) = 1;
        v130 = 0;
        while ( v27 != -8 )
        {
          if ( !v130 && v27 == -16 )
            v130 = v18;
          v133 = v12 + 1;
          LODWORD(v12) = v26 + v12;
          v26 = v25 & v12;
          v18 = &v167[5 * (v25 & (unsigned int)v12)];
          v27 = *v18;
          if ( v24 == *v18 )
            goto LABEL_24;
          LODWORD(v12) = v133;
        }
        if ( v130 )
          v18 = v130;
      }
LABEL_24:
      LODWORD(v168) = v28;
      if ( *v18 != -8 )
        --HIDWORD(v168);
      *v18 = v24;
      v18[1] = v18 + 3;
      v18[2] = 0x100000000LL;
      v20 = 0;
      v21 = *(_QWORD *)(v23 + 8 * (1LL - *(unsigned int *)(v23 + 8)));
LABEL_19:
      v13 += 8;
      v22 = (__int64 *)(v18[1] + 16LL * v20);
      *v22 = v15;
      v22[1] = v21;
      ++*((_DWORD *)v18 + 4);
      if ( v14 == v13 )
      {
        v14 = v170;
        goto LABEL_28;
      }
    }
  }
LABEL_31:
  if ( *(_BYTE *)(a1 + 4514) )
  {
    v29 = a1 + 4040;
    if ( *(_BYTE *)(a1 + 4513) )
      v29 = a1 + 4520;
    LOWORD(v172) = 259;
    v30 = *(_QWORD *)(a1 + 8);
    v170 = "str_offsets_base";
    *(_QWORD *)(v29 + 248) = sub_396F530(v30, (__int64)&v170);
  }
  if ( (unsigned __int16)sub_398C0A0(a1) > 4u )
  {
    v31 = a1 + 4040;
    if ( *(_BYTE *)(a1 + 4513) )
      v31 = a1 + 4520;
    LOWORD(v172) = 259;
    v32 = *(_QWORD *)(a1 + 8);
    v170 = "rnglists_table_base";
    *(_QWORD *)(v31 + 256) = sub_396F530(v32, (__int64)&v170);
  }
  v170 = "llvm.dbg.cu";
  LOWORD(v172) = 259;
  v33 = sub_1632310(v141, (__int64)&v170);
  v34 = 0;
  if ( v33 )
    v34 = sub_161F520(v33);
  v35 = &v163;
  LODWORD(v171) = v34;
  v170 = (char *)v33;
  sub_1632FD0((__int64)&v170);
  v163 = v33;
  LODWORD(v164) = 0;
  sub_1632FD0((__int64)&v163);
  v134 = v171;
  v158 = v163;
  v159 = v164;
  if ( (_DWORD)v164 != (_DWORD)v171 )
  {
    while ( 1 )
    {
      v137 = sub_1632FB0((__int64)&v158);
      v136 = sub_3999410(a1, v137);
      v36 = *(_QWORD *)(v137 + 8 * (6LL - *(unsigned int *)(v137 + 8)));
      if ( v36 )
      {
        v37 = 8LL * *(unsigned int *)(v36 + 8);
        v38 = (__int64 *)(v36 - v37);
        if ( v36 - v37 != v36 )
        {
          v156 = v35;
          v39 = *(__int64 **)(v137 + 8 * (6LL - *(unsigned int *)(v137 + 8)));
          while ( 1 )
          {
            v48 = *v38;
            v49 = *(unsigned int *)(*v38 + 8);
            v50 = *(_QWORD *)(*v38 - 8 * v49);
            if ( !v169 )
              break;
            v40 = v169 - 1;
            v41 = (int)v167;
            v42 = (v169 - 1) & (((unsigned int)v50 >> 4) ^ ((unsigned int)v50 >> 9));
            v43 = &v167[5 * v42];
            v44 = *v43;
            if ( v50 != *v43 )
            {
              v106 = 1;
              v107 = 0;
              while ( v44 != -8 )
              {
                if ( !v107 && v44 == -16 )
                  v107 = v43;
                v42 = v40 & (v106 + v42);
                v43 = &v167[5 * v42];
                v44 = *v43;
                if ( v50 == *v43 )
                  goto LABEL_46;
                ++v106;
              }
              if ( v107 )
                v43 = v107;
              ++v166;
              v53 = v168 + 1;
              if ( 4 * ((int)v168 + 1) < 3 * v169 )
              {
                if ( v169 - HIDWORD(v168) - v53 <= v169 >> 3 )
                {
                  sub_3992510((__int64)&v166, v169);
                  if ( !v169 )
                  {
LABEL_271:
                    LODWORD(v168) = v168 + 1;
                    BUG();
                  }
                  v108 = 0;
                  v109 = 1;
                  v110 = (v169 - 1) & (((unsigned int)v50 >> 4) ^ ((unsigned int)v50 >> 9));
                  v43 = &v167[5 * v110];
                  v111 = *v43;
                  v53 = v168 + 1;
                  if ( v50 != *v43 )
                  {
                    while ( v111 != -8 )
                    {
                      if ( !v108 && v111 == -16 )
                        v108 = v43;
                      v110 = (v169 - 1) & (v110 + v109);
                      v43 = &v167[5 * v110];
                      v111 = *v43;
                      if ( v50 == *v43 )
                        goto LABEL_58;
                      ++v109;
                    }
LABEL_152:
                    if ( v108 )
                      v43 = v108;
                  }
                }
LABEL_58:
                LODWORD(v168) = v53;
                if ( *v43 != -8 )
                  --HIDWORD(v168);
                *v43 = v50;
                v43[1] = v43 + 3;
                v43[2] = 0x100000000LL;
                v45 = *(_QWORD *)(v48 + 8 * (1LL - *(unsigned int *)(v48 + 8)));
                v46 = 0;
                goto LABEL_52;
              }
LABEL_56:
              sub_3992510((__int64)&v166, 2 * v169);
              if ( !v169 )
                goto LABEL_271;
              v51 = (v169 - 1) & (((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4));
              v43 = &v167[5 * v51];
              v52 = *v43;
              v53 = v168 + 1;
              if ( v50 != *v43 )
              {
                v129 = 1;
                v108 = 0;
                while ( v52 != -8 )
                {
                  if ( !v108 && v52 == -16 )
                    v108 = v43;
                  v51 = (v169 - 1) & (v51 + v129);
                  v43 = &v167[5 * v51];
                  v52 = *v43;
                  if ( v50 == *v43 )
                    goto LABEL_58;
                  ++v129;
                }
                goto LABEL_152;
              }
              goto LABEL_58;
            }
LABEL_46:
            v45 = *(_QWORD *)(v48 + 8 * (1 - v49));
            v46 = *((unsigned int *)v43 + 4);
            if ( (_DWORD)v46 )
            {
              if ( !v45 || !(unsigned __int8)sub_15B1550(v45, v169, v49) )
                goto LABEL_53;
              v46 = *((unsigned int *)v43 + 4);
            }
            if ( (unsigned int)v46 >= *((_DWORD *)v43 + 5) )
            {
              sub_16CD150((__int64)(v43 + 1), v43 + 3, 0, 16, v41, v40);
              v46 = *((unsigned int *)v43 + 4);
            }
LABEL_52:
            v47 = (_QWORD *)(v43[1] + 16 * v46);
            *v47 = 0;
            v47[1] = v45;
            ++*((_DWORD *)v43 + 4);
LABEL_53:
            if ( v39 == ++v38 )
            {
              v35 = v156;
              goto LABEL_62;
            }
          }
          ++v166;
          goto LABEL_56;
        }
      }
LABEL_62:
      v170 = 0;
      v171 = 0;
      v172 = 0;
      v173 = 0;
      v138 = *(_QWORD *)(v137 + 8 * (6LL - *(unsigned int *)(v137 + 8)));
      if ( v138 )
      {
        v54 = 8LL * *(unsigned int *)(v138 + 8);
        if ( v138 - v54 != v138 )
        {
          v154 = v138 - v54;
          v55 = 0;
          v56 = 0;
          while ( 1 )
          {
            v61 = *(_QWORD *)(*(_QWORD *)v154 - 8LL * *(unsigned int *)(*(_QWORD *)v154 + 8LL));
            v142 = v61;
            if ( !v55 )
              break;
            v57 = ((unsigned int)v61 >> 9) ^ ((unsigned int)v61 >> 4);
            v58 = (v55 - 1) & v57;
            v59 = (_QWORD *)(v56 + 8LL * v58);
            v60 = *v59;
            if ( v61 == *v59 )
              goto LABEL_66;
            v101 = 1;
            v63 = 0;
            while ( v60 != -8 )
            {
              if ( v63 || v60 != -16 )
                v59 = v63;
              v58 = (v55 - 1) & (v101 + v58);
              v60 = *(_QWORD *)(v56 + 8LL * v58);
              if ( v142 == v60 )
                goto LABEL_66;
              ++v101;
              v63 = v59;
              v59 = (_QWORD *)(v56 + 8LL * v58);
            }
            if ( !v63 )
              v63 = v59;
            ++v170;
            v65 = v172 + 1;
            if ( 4 * ((int)v172 + 1) >= 3 * v55 )
              goto LABEL_70;
            if ( v55 - (v65 + HIDWORD(v172)) <= v55 >> 3 )
            {
              sub_3992800((__int64)&v170, v55);
              if ( !(_DWORD)v173 )
              {
LABEL_268:
                LODWORD(v172) = v172 + 1;
                BUG();
              }
              v102 = 1;
              v103 = 0;
              v104 = (v173 - 1) & v57;
              v63 = (_QWORD *)(v171 + 8LL * v104);
              v105 = *v63;
              v65 = v172 + 1;
              if ( v142 != *v63 )
              {
                while ( v105 != -8 )
                {
                  if ( !v103 && v105 == -16 )
                    v103 = v63;
                  v104 = (v173 - 1) & (v102 + v104);
                  v63 = (_QWORD *)(v171 + 8LL * v104);
                  v105 = *v63;
                  if ( v142 == *v63 )
                    goto LABEL_72;
                  ++v102;
                }
                if ( v103 )
                  v63 = v103;
              }
            }
LABEL_72:
            LODWORD(v172) = v65;
            if ( *v63 != -8 )
              --HIDWORD(v172);
            *v63 = v142;
            if ( v169 )
            {
              v66 = (v169 - 1) & (((unsigned int)v142 >> 9) ^ ((unsigned int)v142 >> 4));
              v140 = &v167[5 * v66];
              v67 = *v140;
              if ( v142 == *v140 )
              {
LABEL_76:
                v68 = (__m128i *)v140[1];
                v69 = *((unsigned int *)v140 + 4);
                v144 = &v68[v69];
                if ( v68 == &v68[v69] )
                {
                  v76 = &v68[v69];
                }
                else
                {
                  _BitScanReverse64(&v70, (v69 * 16) >> 4);
                  sub_3988120(v68, v68[v69].m128i_i64, 2LL * (int)(63 - (v70 ^ 0x3F)));
                  if ( (unsigned __int64)v69 <= 16 )
                  {
                    sub_3986A20(v68, v144);
                    goto LABEL_88;
                  }
                  v146 = v68 + 16;
                  sub_3986A20(v68, v68 + 16);
                  if ( &v68[v69] != &v68[16] )
                  {
                    do
                    {
                      v71 = v146->m128i_i64[1];
                      v72 = v146;
                      v139 = v146->m128i_i64[0];
                      while ( 1 )
                      {
                        v74 = v72[-1].m128i_i64[1];
                        v75 = (__int64 *)v72;
                        if ( !v74 || !v71 )
                        {
                          v73 = v74 != 0;
                          goto LABEL_81;
                        }
                        sub_15B1350((__int64)v160, *(unsigned __int64 **)(v71 + 24), *(unsigned __int64 **)(v71 + 32));
                        sub_15B1350((__int64)v35, *(unsigned __int64 **)(v74 + 24), *(unsigned __int64 **)(v74 + 32));
                        if ( !v162 )
                        {
                          v73 = v165;
                          goto LABEL_81;
                        }
                        if ( !v165 )
                          break;
                        v73 = v161 < v164;
LABEL_81:
                        --v72;
                        if ( !v73 )
                          break;
                        v72[1] = _mm_loadu_si128(v72);
                      }
                      ++v146;
                      v75[1] = v71;
                      *v75 = v139;
                    }
                    while ( v144 != v146 );
                  }
LABEL_88:
                  v144 = (const __m128i *)v140[1];
                  v76 = (__m128i *)&v144[*((unsigned int *)v140 + 4)];
                  if ( v144 != v76 )
                  {
                    v77 = (__m128i *)v140[1];
                    while ( 1 )
                    {
                      v79 = v77++;
                      if ( v77 == v76 )
                        break;
                      v78 = v77[-1].m128i_i64[1];
                      if ( v78 == v77->m128i_i64[1] )
                      {
                        if ( v79 == v76 )
                          goto LABEL_92;
                        v81 = v79 + 2;
                        if ( &v79[2] != v76 )
                        {
                          while ( 1 )
                          {
                            if ( v81->m128i_i64[1] != v78 )
                              *++v79 = _mm_loadu_si128(v81);
                            if ( ++v81 == v76 )
                              break;
                            v78 = v79->m128i_i64[1];
                          }
                          v82 = v79 + 1;
                          v144 = (const __m128i *)v140[1];
                          v83 = &v144[*((unsigned int *)v140 + 4)];
                          v77 = (__m128i *)((char *)v82 + (char *)v83 - (char *)v81);
                          if ( v81 != v83 )
                          {
                            memmove(v82, v81, (char *)v83 - (char *)v81);
                            v144 = (const __m128i *)v140[1];
                          }
                        }
                        goto LABEL_93;
                      }
                    }
                  }
                }
                goto LABEL_92;
              }
              v116 = 1;
              v117 = 0;
              while ( v67 != -8 )
              {
                if ( v67 == -16 && !v117 )
                  v117 = v140;
                v66 = (v169 - 1) & (v116 + v66);
                v140 = &v167[5 * v66];
                v67 = *v140;
                if ( v142 == *v140 )
                  goto LABEL_76;
                ++v116;
              }
              if ( !v117 )
                v117 = v140;
              ++v166;
              v118 = v168 + 1;
              v140 = v117;
              if ( 4 * ((int)v168 + 1) < 3 * v169 )
              {
                if ( v169 - HIDWORD(v168) - v118 <= v169 >> 3 )
                {
                  sub_3992510((__int64)&v166, v169);
                  if ( !v169 )
                  {
LABEL_270:
                    LODWORD(v168) = v168 + 1;
                    BUG();
                  }
                  v124 = (v169 - 1) & (((unsigned int)v142 >> 9) ^ ((unsigned int)v142 >> 4));
                  v140 = &v167[5 * v124];
                  v125 = *v140;
                  v118 = v168 + 1;
                  if ( v142 != *v140 )
                  {
                    v126 = &v167[5 * v124];
                    v127 = 1;
                    v128 = 0;
                    while ( v125 != -8 )
                    {
                      if ( v125 == -16 && !v128 )
                        v128 = v126;
                      v124 = (v169 - 1) & (v127 + v124);
                      v126 = &v167[5 * v124];
                      v125 = *v126;
                      if ( v142 == *v126 )
                      {
                        v140 = &v167[5 * v124];
                        goto LABEL_173;
                      }
                      ++v127;
                    }
                    if ( !v128 )
                      v128 = v126;
                    v140 = v128;
                  }
                }
                goto LABEL_173;
              }
            }
            else
            {
              ++v166;
            }
            sub_3992510((__int64)&v166, 2 * v169);
            if ( !v169 )
              goto LABEL_270;
            v119 = (v169 - 1) & (((unsigned int)v142 >> 9) ^ ((unsigned int)v142 >> 4));
            v118 = v168 + 1;
            v140 = &v167[5 * v119];
            v120 = *v140;
            if ( v142 != *v140 )
            {
              v121 = &v167[5 * ((v169 - 1) & (((unsigned int)v142 >> 9) ^ ((unsigned int)v142 >> 4)))];
              v122 = 1;
              v123 = 0;
              while ( v120 != -8 )
              {
                if ( v120 == -16 && !v123 )
                  v123 = v121;
                v119 = (v169 - 1) & (v122 + v119);
                v121 = &v167[5 * v119];
                v120 = *v121;
                if ( v142 == *v121 )
                {
                  v140 = &v167[5 * v119];
                  goto LABEL_173;
                }
                ++v122;
              }
              if ( !v123 )
                v123 = v121;
              v140 = v123;
            }
LABEL_173:
            LODWORD(v168) = v118;
            if ( *v140 != -8 )
              --HIDWORD(v168);
            v76 = (__m128i *)(v140 + 3);
            *v140 = v142;
            v144 = (const __m128i *)(v140 + 3);
            v140[1] = v140 + 3;
            v140[2] = 0x100000000LL;
LABEL_92:
            v77 = v76;
LABEL_93:
            v80 = v77 - v144;
            *((_DWORD *)v140 + 4) = v80;
            sub_39CBE70(v136, v142, v144, (unsigned int)v80);
LABEL_66:
            v154 += 8;
            if ( v138 == v154 )
              goto LABEL_105;
            v56 = v171;
            v55 = v173;
          }
          ++v170;
LABEL_70:
          sub_3992800((__int64)&v170, 2 * v55);
          if ( !(_DWORD)v173 )
            goto LABEL_268;
          v62 = (v173 - 1) & (((unsigned int)v142 >> 9) ^ ((unsigned int)v142 >> 4));
          v63 = (_QWORD *)(v171 + 8LL * v62);
          v64 = *v63;
          v65 = v172 + 1;
          if ( v142 != *v63 )
          {
            v131 = 1;
            v132 = 0;
            while ( v64 != -8 )
            {
              if ( !v132 && v64 == -16 )
                v132 = v63;
              v62 = (v173 - 1) & (v131 + v62);
              v63 = (_QWORD *)(v171 + 8LL * v62);
              v64 = *v63;
              if ( v142 == *v63 )
                goto LABEL_72;
              ++v131;
            }
            if ( v132 )
              v63 = v132;
          }
          goto LABEL_72;
        }
      }
LABEL_105:
      sub_39A3B30(v136);
      v84 = *(unsigned int *)(v137 + 8);
      v85 = *(_QWORD *)(v137 + 8 * (4 - v84));
      if ( v85 )
      {
        v86 = 8LL * *(unsigned int *)(v85 + 8);
        v87 = (__int64 *)(v85 - v86);
        if ( v85 - v86 != v85 )
        {
          do
          {
            v88 = *v87++;
            sub_39A64F0(v136, v88);
          }
          while ( (__int64 *)v85 != v87 );
          v84 = *(unsigned int *)(v137 + 8);
        }
      }
      v89 = *(_QWORD *)(v137 + 8 * (5 - v84));
      if ( v89 )
      {
        v90 = 8LL * *(unsigned int *)(v89 + 8);
        v91 = (unsigned __int8 **)(v89 - v90);
        if ( v89 - v90 != v89 )
          break;
      }
LABEL_118:
      v94 = *(_QWORD *)(v137 + 8 * (7 - v84));
      if ( v94 )
      {
        v95 = 8LL * *(unsigned int *)(v94 + 8);
        v96 = (__int64 *)(v94 - v95);
        if ( v94 - v95 != v94 )
        {
          do
          {
            v97 = *v96++;
            sub_3989CD0(a1, v136, v97);
          }
          while ( (__int64 *)v94 != v96 );
        }
      }
      j___libc_free_0(v171);
      ++v159;
      sub_1632FD0((__int64)&v158);
      if ( v159 == v134 )
        goto LABEL_122;
    }
    while ( 1 )
    {
      v92 = *v91;
      v93 = **v91;
      if ( v93 <= 0xEu )
      {
        if ( v93 <= 0xAu )
          goto LABEL_113;
LABEL_116:
        ++v91;
        sub_39A64F0(v136, v92);
        if ( (unsigned __int8 **)v89 == v91 )
        {
LABEL_117:
          v84 = *(unsigned int *)(v137 + 8);
          goto LABEL_118;
        }
      }
      else
      {
        if ( (unsigned __int8)(v93 - 32) <= 1u )
          goto LABEL_116;
LABEL_113:
        if ( (unsigned __int8 **)v89 == ++v91 )
          goto LABEL_117;
      }
    }
  }
LABEL_122:
  if ( v169 )
  {
    v98 = v167;
    v99 = &v167[5 * v169];
    do
    {
      if ( *v98 != -16 && *v98 != -8 )
      {
        v100 = v98[1];
        if ( (_QWORD *)v100 != v98 + 3 )
          _libc_free(v100);
      }
      v98 += 5;
    }
    while ( v99 != v98 );
  }
  j___libc_free_0((unsigned __int64)v167);
LABEL_2:
  if ( *(_QWORD *)v157 )
    sub_16D7950(*(__int64 *)v157);
}
