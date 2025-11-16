// Function: sub_30E58F0
// Address: 0x30e58f0
//
__int64 __fastcall sub_30E58F0(
        __int64 a1,
        unsigned __int8 (__fastcall *a2)(__int64, __int64, __int64, __int64 *),
        __int64 a3)
{
  __int64 v3; // r15
  __int64 v4; // rax
  __int64 *v5; // rbx
  __int64 *v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // rax
  void (__fastcall *v9)(__m128i *, __int64, __int64); // r12
  __int64 *v10; // r15
  int v11; // r11d
  __int64 v12; // r8
  __int64 *v13; // rdx
  unsigned int v14; // edi
  __int64 *v15; // rax
  __int64 v16; // rcx
  void (__fastcall *v17)(__m128i *, __int64, __int64); // r12
  __int64 v18; // r13
  unsigned int v19; // esi
  __int64 v20; // r10
  int v21; // r11d
  __int64 v22; // r8
  __int64 *v23; // rdx
  unsigned int v24; // edi
  __int64 *v25; // rax
  __int64 v26; // rcx
  void (__fastcall *v27)(__m128i *, __int64, __int64); // r12
  __int64 v28; // r13
  unsigned int v29; // esi
  __int64 v30; // r10
  int v31; // r11d
  __int64 v32; // r8
  __int64 *v33; // rdx
  unsigned int v34; // edi
  __int64 *v35; // rax
  __int64 v36; // rcx
  void (__fastcall *v37)(__m128i *, __int64, __int64); // r12
  __int64 v38; // r13
  unsigned int v39; // esi
  __int64 v40; // r10
  int v41; // r11d
  __int64 v42; // r8
  __int64 *v43; // rdx
  unsigned int v44; // edi
  __int64 *v45; // rax
  __int64 v46; // rcx
  unsigned int v47; // esi
  __int64 v48; // r13
  int v49; // eax
  int v50; // edi
  __int64 v51; // rsi
  unsigned int v52; // eax
  int v53; // ecx
  __int64 v54; // r8
  int v55; // r11d
  __int64 *v56; // r9
  int v57; // eax
  __int64 *v58; // r12
  __int64 v59; // r14
  int v60; // r11d
  __int64 v61; // r9
  __int64 *v62; // rcx
  unsigned int v63; // r8d
  __int64 *v64; // rax
  __int64 v65; // rdi
  __int64 v66; // rdx
  unsigned int v67; // esi
  __int64 v68; // r13
  int v69; // esi
  int v70; // esi
  __int64 v71; // r8
  unsigned int v72; // edx
  int v73; // eax
  __int64 v74; // rdi
  int v75; // eax
  int v76; // ecx
  int v77; // eax
  int v78; // ecx
  int v79; // eax
  int v80; // ecx
  int v81; // eax
  int v82; // esi
  __int64 v83; // rdi
  unsigned int v84; // eax
  __int64 v85; // r8
  int v86; // r10d
  __int64 *v87; // r9
  int v88; // eax
  int v89; // edi
  __int64 v90; // rsi
  unsigned int v91; // eax
  __int64 v92; // r8
  int v93; // r10d
  __int64 *v94; // r9
  int v95; // eax
  int v96; // edi
  __int64 v97; // rsi
  unsigned int v98; // eax
  __int64 v99; // r8
  int v100; // r10d
  __int64 *v101; // r9
  int v102; // eax
  int v103; // eax
  int v104; // r9d
  __int64 *v105; // r8
  __int64 v106; // rdi
  unsigned int v107; // r14d
  __int64 v108; // rsi
  int v109; // eax
  int v110; // eax
  int v111; // r9d
  __int64 *v112; // r8
  __int64 v113; // rdi
  __int64 v114; // r14
  __int64 v115; // rsi
  int v116; // eax
  int v117; // eax
  int v118; // r10d
  __int64 *v119; // r8
  __int64 v120; // rdi
  unsigned int v121; // r14d
  __int64 v122; // rsi
  int v123; // eax
  int v124; // eax
  int v125; // r10d
  __int64 *v126; // r8
  __int64 v127; // rdi
  unsigned int v128; // r14d
  __int64 v129; // rsi
  __int64 v130; // r12
  __int64 v131; // r13
  void (__fastcall *v132)(__m128i *, __int64, __int64); // rax
  __int64 v133; // rsi
  __int64 v134; // rbx
  unsigned __int64 v135; // rdx
  __m128i v136; // xmm0
  __m128i v137; // xmm1
  __int64 v138; // r13
  __int64 i; // rbx
  __int64 v140; // rcx
  __int64 result; // rax
  int v142; // eax
  int v143; // esi
  int v144; // esi
  __int64 *v145; // r9
  __int64 v146; // r8
  int v147; // r10d
  unsigned int v148; // edx
  __int64 v149; // rdi
  __int64 *v150; // r11
  int v151; // r10d
  int v152; // r8d
  unsigned int v153; // r10d
  __int64 *v154; // r9
  int v155; // r8d
  unsigned int v156; // r10d
  __int64 *v157; // r9
  unsigned int v159; // [rsp+10h] [rbp-B0h]
  __int64 v161; // [rsp+18h] [rbp-A8h]
  __int64 *src; // [rsp+28h] [rbp-98h]
  void *srca; // [rsp+28h] [rbp-98h]
  __m128i v165; // [rsp+30h] [rbp-90h] BYREF
  __int64 (__fastcall *v166)(__m128i *, __m128i *, __int64); // [rsp+40h] [rbp-80h]
  __int64 v167; // [rsp+48h] [rbp-78h]
  __m128i v168; // [rsp+50h] [rbp-70h] BYREF
  void (__fastcall *v169)(__m128i *, __int64, __int64); // [rsp+60h] [rbp-60h]
  __int64 v170; // [rsp+68h] [rbp-58h]
  __m128i v171; // [rsp+70h] [rbp-50h] BYREF
  void (__fastcall *v172)(__m128i *, __int64, __int64); // [rsp+80h] [rbp-40h]
  __int64 v173; // [rsp+88h] [rbp-38h]

  v3 = a1;
  v4 = *(unsigned int *)(a1 + 16);
  v5 = *(__int64 **)(a1 + 8);
  v171.m128i_i64[1] = a3;
  v4 *= 8;
  v6 = (__int64 *)((char *)v5 + v4);
  v171.m128i_i64[0] = (__int64)a2;
  v7 = v4 >> 3;
  v8 = v4 >> 5;
  src = v6;
  v172 = (void (__fastcall *)(__m128i *, __int64, __int64))a1;
  if ( !v8 )
  {
LABEL_145:
    if ( v7 != 2 )
    {
      if ( v7 != 3 )
      {
        if ( v7 != 1 )
        {
LABEL_148:
          v5 = src;
          goto LABEL_149;
        }
LABEL_187:
        if ( (unsigned __int8)sub_30E5660((__int64)&v171, *v5) )
          goto LABEL_39;
        goto LABEL_148;
      }
      if ( (unsigned __int8)sub_30E5660((__int64)&v171, *v5) )
        goto LABEL_39;
      ++v5;
    }
    if ( (unsigned __int8)sub_30E5660((__int64)&v171, *v5) )
      goto LABEL_39;
    ++v5;
    goto LABEL_187;
  }
  v9 = (_DWORD *)a1;
  v10 = &v5[4 * v8];
  while ( 1 )
  {
    v47 = v9[52];
    v48 = *v5;
    if ( !v47 )
    {
      ++*((_QWORD *)v9 + 23);
LABEL_18:
      sub_30E3EE0((__int64)(v9 + 46), 2 * v47);
      v49 = v9[52];
      if ( !v49 )
        goto LABEL_246;
      v50 = v49 - 1;
      v51 = *((_QWORD *)v9 + 24);
      v52 = (v49 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
      v53 = v9[50] + 1;
      v13 = (__int64 *)(v51 + 16LL * v52);
      v54 = *v13;
      if ( v48 != *v13 )
      {
        v55 = 1;
        v56 = 0;
        while ( v54 != -4096 )
        {
          if ( v56 || v54 != -8192 )
            v13 = v56;
          v52 = v50 & (v55 + v52);
          v54 = *(_QWORD *)(v51 + 16LL * v52);
          if ( v48 == v54 )
          {
            v13 = (__int64 *)(v51 + 16LL * v52);
            goto LABEL_35;
          }
          ++v55;
          v56 = v13;
          v13 = (__int64 *)(v51 + 16LL * v52);
        }
        if ( v56 )
          v13 = v56;
      }
      goto LABEL_35;
    }
    v11 = 1;
    v12 = *((_QWORD *)v9 + 24);
    v13 = 0;
    v14 = (v47 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
    v15 = (__int64 *)(v12 + 16LL * v14);
    v16 = *v15;
    if ( v48 != *v15 )
      break;
LABEL_4:
    if ( ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v171.m128i_i64[0])(
           v171.m128i_i64[1],
           v48,
           *((unsigned int *)v15 + 2)) )
    {
      goto LABEL_38;
    }
LABEL_5:
    v17 = v172;
    v18 = v5[1];
    v19 = *((_DWORD *)v172 + 52);
    v20 = (__int64)v172 + 184;
    if ( !v19 )
    {
      ++*((_QWORD *)v172 + 23);
LABEL_97:
      sub_30E3EE0(v20, 2 * v19);
      v81 = v17[52];
      if ( !v81 )
        goto LABEL_245;
      v82 = v81 - 1;
      v83 = *((_QWORD *)v17 + 24);
      v84 = (v81 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v76 = v17[50] + 1;
      v23 = (__int64 *)(v83 + 16LL * v84);
      v85 = *v23;
      if ( v18 != *v23 )
      {
        v86 = 1;
        v87 = 0;
        while ( v85 != -4096 )
        {
          if ( !v87 && v85 == -8192 )
            v87 = v23;
          v84 = v82 & (v86 + v84);
          v23 = (__int64 *)(v83 + 16LL * v84);
          v85 = *v23;
          if ( v18 == *v23 )
            goto LABEL_64;
          ++v86;
        }
        if ( v87 )
          v23 = v87;
      }
      goto LABEL_64;
    }
    v21 = 1;
    v22 = *((_QWORD *)v172 + 24);
    v23 = 0;
    v24 = (v19 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
    v25 = (__int64 *)(v22 + 16LL * v24);
    v26 = *v25;
    if ( v18 == *v25 )
    {
LABEL_7:
      if ( ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v171.m128i_i64[0])(
             v171.m128i_i64[1],
             v18,
             *((unsigned int *)v25 + 2)) )
      {
        goto LABEL_67;
      }
      goto LABEL_8;
    }
    while ( v26 != -4096 )
    {
      if ( !v23 && v26 == -8192 )
        v23 = v25;
      v24 = (v19 - 1) & (v21 + v24);
      v25 = (__int64 *)(v22 + 16LL * v24);
      v26 = *v25;
      if ( v18 == *v25 )
        goto LABEL_7;
      ++v21;
    }
    if ( !v23 )
      v23 = v25;
    v75 = *((_DWORD *)v172 + 50);
    ++*((_QWORD *)v172 + 23);
    v76 = v75 + 1;
    if ( 4 * (v75 + 1) >= 3 * v19 )
      goto LABEL_97;
    if ( v19 - v17[51] - v76 <= v19 >> 3 )
    {
      sub_30E3EE0(v20, v19);
      v109 = v17[52];
      if ( !v109 )
      {
LABEL_245:
        ++v17[50];
        BUG();
      }
      v110 = v109 - 1;
      v111 = 1;
      v112 = 0;
      v113 = *((_QWORD *)v17 + 24);
      LODWORD(v114) = v110 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v76 = v17[50] + 1;
      v23 = (__int64 *)(v113 + 16LL * (unsigned int)v114);
      v115 = *v23;
      if ( v18 != *v23 )
      {
        while ( v115 != -4096 )
        {
          if ( !v112 && v115 == -8192 )
            v112 = v23;
          v114 = v110 & (unsigned int)(v114 + v111);
          v23 = (__int64 *)(v113 + 16 * v114);
          v115 = *v23;
          if ( v18 == *v23 )
            goto LABEL_64;
          ++v111;
        }
        if ( v112 )
          v23 = v112;
      }
    }
LABEL_64:
    v17[50] = v76;
    if ( *v23 != -4096 )
      --v17[51];
    *v23 = v18;
    *((_DWORD *)v23 + 2) = 0;
    if ( ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v171.m128i_i64[0])(v171.m128i_i64[1], v18, 0) )
    {
LABEL_67:
      v3 = a1;
      ++v5;
      goto LABEL_39;
    }
LABEL_8:
    v27 = v172;
    v28 = v5[2];
    v29 = *((_DWORD *)v172 + 52);
    v30 = (__int64)v172 + 184;
    if ( !v29 )
    {
      ++*((_QWORD *)v172 + 23);
LABEL_105:
      sub_30E3EE0(v30, 2 * v29);
      v88 = v27[52];
      if ( !v88 )
        goto LABEL_247;
      v89 = v88 - 1;
      v90 = *((_QWORD *)v27 + 24);
      v91 = (v88 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
      v78 = v27[50] + 1;
      v33 = (__int64 *)(v90 + 16LL * v91);
      v92 = *v33;
      if ( v28 != *v33 )
      {
        v93 = 1;
        v94 = 0;
        while ( v92 != -4096 )
        {
          if ( !v94 && v92 == -8192 )
            v94 = v33;
          v91 = v89 & (v93 + v91);
          v33 = (__int64 *)(v90 + 16LL * v91);
          v92 = *v33;
          if ( v28 == *v33 )
            goto LABEL_78;
          ++v93;
        }
        if ( v94 )
          v33 = v94;
      }
      goto LABEL_78;
    }
    v31 = 1;
    v32 = *((_QWORD *)v172 + 24);
    v33 = 0;
    v34 = (v29 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
    v35 = (__int64 *)(v32 + 16LL * v34);
    v36 = *v35;
    if ( v28 == *v35 )
    {
LABEL_10:
      if ( ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v171.m128i_i64[0])(
             v171.m128i_i64[1],
             v28,
             *((unsigned int *)v35 + 2)) )
      {
        goto LABEL_81;
      }
      goto LABEL_11;
    }
    while ( v36 != -4096 )
    {
      if ( !v33 && v36 == -8192 )
        v33 = v35;
      v34 = (v29 - 1) & (v31 + v34);
      v35 = (__int64 *)(v32 + 16LL * v34);
      v36 = *v35;
      if ( v28 == *v35 )
        goto LABEL_10;
      ++v31;
    }
    if ( !v33 )
      v33 = v35;
    v77 = *((_DWORD *)v172 + 50);
    ++*((_QWORD *)v172 + 23);
    v78 = v77 + 1;
    if ( 4 * (v77 + 1) >= 3 * v29 )
      goto LABEL_105;
    if ( v29 - v27[51] - v78 <= v29 >> 3 )
    {
      sub_30E3EE0(v30, v29);
      v116 = v27[52];
      if ( !v116 )
      {
LABEL_247:
        ++v27[50];
        BUG();
      }
      v117 = v116 - 1;
      v118 = 1;
      v119 = 0;
      v120 = *((_QWORD *)v27 + 24);
      v121 = v117 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
      v78 = v27[50] + 1;
      v33 = (__int64 *)(v120 + 16LL * v121);
      v122 = *v33;
      if ( v28 != *v33 )
      {
        while ( v122 != -4096 )
        {
          if ( v119 || v122 != -8192 )
            v33 = v119;
          v155 = v118 + 1;
          v156 = v117 & (v121 + v118);
          v121 = v156;
          v157 = (__int64 *)(v120 + 16LL * v156);
          v122 = *v157;
          if ( v28 == *v157 )
          {
            v33 = (__int64 *)(v120 + 16LL * v156);
            goto LABEL_78;
          }
          v118 = v155;
          v119 = v33;
          v33 = v157;
        }
        if ( v119 )
          v33 = v119;
      }
    }
LABEL_78:
    v27[50] = v78;
    if ( *v33 != -4096 )
      --v27[51];
    *v33 = v28;
    *((_DWORD *)v33 + 2) = 0;
    if ( ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v171.m128i_i64[0])(v171.m128i_i64[1], v28, 0) )
    {
LABEL_81:
      v3 = a1;
      v5 += 2;
      goto LABEL_39;
    }
LABEL_11:
    v37 = v172;
    v38 = v5[3];
    v39 = *((_DWORD *)v172 + 52);
    v40 = (__int64)v172 + 184;
    if ( !v39 )
    {
      ++*((_QWORD *)v172 + 23);
LABEL_113:
      sub_30E3EE0(v40, 2 * v39);
      v95 = v37[52];
      if ( !v95 )
        goto LABEL_248;
      v96 = v95 - 1;
      v97 = *((_QWORD *)v37 + 24);
      v98 = (v95 - 1) & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
      v80 = v37[50] + 1;
      v43 = (__int64 *)(v97 + 16LL * v98);
      v99 = *v43;
      if ( v38 != *v43 )
      {
        v100 = 1;
        v101 = 0;
        while ( v99 != -4096 )
        {
          if ( !v101 && v99 == -8192 )
            v101 = v43;
          v98 = v96 & (v100 + v98);
          v43 = (__int64 *)(v97 + 16LL * v98);
          v99 = *v43;
          if ( v38 == *v43 )
            goto LABEL_92;
          ++v100;
        }
        if ( v101 )
          v43 = v101;
      }
      goto LABEL_92;
    }
    v41 = 1;
    v42 = *((_QWORD *)v172 + 24);
    v43 = 0;
    v44 = (v39 - 1) & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
    v45 = (__int64 *)(v42 + 16LL * v44);
    v46 = *v45;
    if ( v38 == *v45 )
    {
LABEL_13:
      if ( ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v171.m128i_i64[0])(
             v171.m128i_i64[1],
             v38,
             *((unsigned int *)v45 + 2)) )
      {
        goto LABEL_95;
      }
      goto LABEL_14;
    }
    while ( v46 != -4096 )
    {
      if ( v46 == -8192 && !v43 )
        v43 = v45;
      v44 = (v39 - 1) & (v41 + v44);
      v45 = (__int64 *)(v42 + 16LL * v44);
      v46 = *v45;
      if ( v38 == *v45 )
        goto LABEL_13;
      ++v41;
    }
    if ( !v43 )
      v43 = v45;
    v79 = *((_DWORD *)v172 + 50);
    ++*((_QWORD *)v172 + 23);
    v80 = v79 + 1;
    if ( 4 * (v79 + 1) >= 3 * v39 )
      goto LABEL_113;
    if ( v39 - v37[51] - v80 <= v39 >> 3 )
    {
      sub_30E3EE0(v40, v39);
      v123 = v37[52];
      if ( !v123 )
      {
LABEL_248:
        ++v37[50];
        BUG();
      }
      v124 = v123 - 1;
      v125 = 1;
      v126 = 0;
      v127 = *((_QWORD *)v37 + 24);
      v128 = v124 & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
      v80 = v37[50] + 1;
      v43 = (__int64 *)(v127 + 16LL * v128);
      v129 = *v43;
      if ( v38 != *v43 )
      {
        while ( v129 != -4096 )
        {
          if ( v129 != -8192 || v126 )
            v43 = v126;
          v152 = v125 + 1;
          v153 = v124 & (v128 + v125);
          v128 = v153;
          v154 = (__int64 *)(v127 + 16LL * v153);
          v129 = *v154;
          if ( v38 == *v154 )
          {
            v43 = (__int64 *)(v127 + 16LL * v153);
            goto LABEL_92;
          }
          v125 = v152;
          v126 = v43;
          v43 = v154;
        }
        if ( v126 )
          v43 = v126;
      }
    }
LABEL_92:
    v37[50] = v80;
    if ( *v43 != -4096 )
      --v37[51];
    *v43 = v38;
    *((_DWORD *)v43 + 2) = 0;
    if ( ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v171.m128i_i64[0])(v171.m128i_i64[1], v38, 0) )
    {
LABEL_95:
      v3 = a1;
      v5 += 3;
      goto LABEL_39;
    }
LABEL_14:
    v5 += 4;
    if ( v10 == v5 )
    {
      v3 = a1;
      v7 = src - v5;
      goto LABEL_145;
    }
    v9 = v172;
  }
  while ( v16 != -4096 )
  {
    if ( v16 == -8192 && !v13 )
      v13 = v15;
    v14 = (v47 - 1) & (v11 + v14);
    v15 = (__int64 *)(v12 + 16LL * v14);
    v16 = *v15;
    if ( v48 == *v15 )
      goto LABEL_4;
    ++v11;
  }
  if ( !v13 )
    v13 = v15;
  v57 = v9[50];
  ++*((_QWORD *)v9 + 23);
  v53 = v57 + 1;
  if ( 4 * (v57 + 1) >= 3 * v47 )
    goto LABEL_18;
  if ( v47 - v9[51] - v53 <= v47 >> 3 )
  {
    sub_30E3EE0((__int64)(v9 + 46), v47);
    v102 = v9[52];
    if ( !v102 )
    {
LABEL_246:
      ++v9[50];
      BUG();
    }
    v103 = v102 - 1;
    v104 = 1;
    v105 = 0;
    v106 = *((_QWORD *)v9 + 24);
    v107 = v103 & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
    v53 = v9[50] + 1;
    v13 = (__int64 *)(v106 + 16LL * v107);
    v108 = *v13;
    if ( v48 != *v13 )
    {
      while ( v108 != -4096 )
      {
        if ( !v105 && v108 == -8192 )
          v105 = v13;
        v107 = v103 & (v104 + v107);
        v13 = (__int64 *)(v106 + 16LL * v107);
        v108 = *v13;
        if ( v48 == *v13 )
          goto LABEL_35;
        ++v104;
      }
      if ( v105 )
        v13 = v105;
    }
  }
LABEL_35:
  v9[50] = v53;
  if ( *v13 != -4096 )
    --v9[51];
  *v13 = v48;
  *((_DWORD *)v13 + 2) = 0;
  if ( !((__int64 (__fastcall *)(__int64, __int64, _QWORD))v171.m128i_i64[0])(v171.m128i_i64[1], v48, 0) )
    goto LABEL_5;
LABEL_38:
  v3 = a1;
LABEL_39:
  if ( src != v5 )
  {
    v58 = v5 + 1;
    if ( src != v5 + 1 )
    {
      v59 = a3;
      v161 = v3 + 184;
      while ( 1 )
      {
        v67 = *(_DWORD *)(v3 + 208);
        v68 = *v58;
        if ( !v67 )
          break;
        v60 = 1;
        v61 = *(_QWORD *)(v3 + 192);
        v62 = 0;
        v63 = (v67 - 1) & (((unsigned int)v68 >> 9) ^ ((unsigned int)v68 >> 4));
        v64 = (__int64 *)(v61 + 16LL * v63);
        v65 = *v64;
        if ( v68 != *v64 )
        {
          while ( v65 != -4096 )
          {
            if ( !v62 && v65 == -8192 )
              v62 = v64;
            v63 = (v67 - 1) & (v60 + v63);
            v64 = (__int64 *)(v61 + 16LL * v63);
            v65 = *v64;
            if ( v68 == *v64 )
              goto LABEL_43;
            ++v60;
          }
          if ( !v62 )
            v62 = v64;
          v142 = *(_DWORD *)(v3 + 200);
          ++*(_QWORD *)(v3 + 184);
          v73 = v142 + 1;
          if ( 4 * v73 < 3 * v67 )
          {
            if ( v67 - *(_DWORD *)(v3 + 204) - v73 <= v67 >> 3 )
            {
              v159 = ((unsigned int)v68 >> 9) ^ ((unsigned int)v68 >> 4);
              sub_30E3EE0(v161, v67);
              v143 = *(_DWORD *)(v3 + 208);
              if ( !v143 )
              {
LABEL_244:
                ++*(_DWORD *)(v3 + 200);
                BUG();
              }
              v144 = v143 - 1;
              v145 = 0;
              v146 = *(_QWORD *)(v3 + 192);
              v147 = 1;
              v148 = v144 & v159;
              v73 = *(_DWORD *)(v3 + 200) + 1;
              v62 = (__int64 *)(v146 + 16LL * (v144 & v159));
              v149 = *v62;
              if ( v68 != *v62 )
              {
                while ( v149 != -4096 )
                {
                  if ( v145 || v149 != -8192 )
                    v62 = v145;
                  v148 = v144 & (v147 + v148);
                  v150 = (__int64 *)(v146 + 16LL * v148);
                  v149 = *v150;
                  if ( v68 == *v150 )
                  {
LABEL_182:
                    v62 = v150;
                    goto LABEL_51;
                  }
                  ++v147;
                  v145 = v62;
                  v62 = (__int64 *)(v146 + 16LL * v148);
                }
LABEL_191:
                if ( v145 )
                  v62 = v145;
              }
            }
LABEL_51:
            *(_DWORD *)(v3 + 200) = v73;
            if ( *v62 != -4096 )
              --*(_DWORD *)(v3 + 204);
            *v62 = v68;
            v66 = 0;
            *((_DWORD *)v62 + 2) = 0;
            goto LABEL_44;
          }
LABEL_49:
          sub_30E3EE0(v161, 2 * v67);
          v69 = *(_DWORD *)(v3 + 208);
          if ( !v69 )
            goto LABEL_244;
          v70 = v69 - 1;
          v71 = *(_QWORD *)(v3 + 192);
          v72 = v70 & (((unsigned int)v68 >> 9) ^ ((unsigned int)v68 >> 4));
          v73 = *(_DWORD *)(v3 + 200) + 1;
          v62 = (__int64 *)(v71 + 16LL * v72);
          v74 = *v62;
          if ( v68 != *v62 )
          {
            v151 = 1;
            v145 = 0;
            while ( v74 != -4096 )
            {
              if ( v74 != -8192 || v145 )
                v62 = v145;
              v72 = v70 & (v151 + v72);
              v150 = (__int64 *)(v71 + 16LL * v72);
              v74 = *v150;
              if ( v68 == *v150 )
                goto LABEL_182;
              ++v151;
              v145 = v62;
              v62 = (__int64 *)(v71 + 16LL * v72);
            }
            goto LABEL_191;
          }
          goto LABEL_51;
        }
LABEL_43:
        v66 = *((unsigned int *)v64 + 2);
LABEL_44:
        if ( !a2(v59, v68, v66, v62) )
          *v5++ = *v58;
        if ( src == ++v58 )
          goto LABEL_149;
      }
      ++*(_QWORD *)(v3 + 184);
      goto LABEL_49;
    }
  }
LABEL_149:
  v130 = *(_QWORD *)(v3 + 8);
  v131 = v130 + 8LL * *(unsigned int *)(v3 + 16) - (_QWORD)src;
  if ( src != (__int64 *)(v130 + 8LL * *(unsigned int *)(v3 + 16)) )
  {
    memmove(v5, src, v130 + 8LL * *(unsigned int *)(v3 + 16) - (_QWORD)src);
    v130 = *(_QWORD *)(v3 + 8);
  }
  v132 = *(void (__fastcall **)(__m128i *, __int64, __int64))(v3 + 168);
  v166 = 0;
  v133 = v167;
  v134 = ((__int64)v5 + v131 - v130) >> 3;
  *(_DWORD *)(v3 + 16) = v134;
  v135 = (unsigned int)v134;
  if ( v132 )
  {
    v132(&v165, v3 + 152, 2);
    v133 = *(_QWORD *)(v3 + 176);
    v132 = *(void (__fastcall **)(__m128i *, __int64, __int64))(v3 + 168);
    v130 = *(_QWORD *)(v3 + 8);
    v135 = *(unsigned int *)(v3 + 16);
  }
  v136 = _mm_loadu_si128(&v165);
  v169 = v132;
  v137 = _mm_loadu_si128(&v171);
  v166 = 0;
  v167 = v173;
  v170 = v133;
  v165 = v137;
  v171 = v136;
  v168 = v136;
  if ( v135 > 1 )
  {
    v138 = (__int64)(8 * v135) >> 3;
    for ( i = (v138 - 2) / 2; ; --i )
    {
      v140 = *(_QWORD *)(v130 + 8 * i);
      v172 = 0;
      if ( v132 )
      {
        srca = (void *)v140;
        v132(&v171, (__int64)&v168, 2);
        v140 = (__int64)srca;
        v173 = v170;
        v172 = v169;
      }
      sub_30E32A0(v130, i, v138, v140, (__int64)&v171);
      if ( v172 )
        v172(&v171, (__int64)&v171, 3);
      v132 = v169;
      if ( !i )
        break;
    }
  }
  if ( v132 )
    v132(&v168, (__int64)&v168, 3);
  result = (__int64)v166;
  if ( v166 )
    return v166(&v165, &v165, 3);
  return result;
}
