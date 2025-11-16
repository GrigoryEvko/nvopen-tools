// Function: sub_11B67C0
// Address: 0x11b67c0
//
unsigned __int8 *__fastcall sub_11B67C0(const __m128i *a1, __int64 a2)
{
  unsigned __int8 *v4; // rbx
  __int64 v5; // r15
  unsigned __int64 v6; // xmm2_8
  __m128i v7; // xmm0
  __m128i v8; // xmm1
  __m128i v9; // xmm3
  __int64 *v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // r14
  __int64 v14; // rdx
  _BYTE *v16; // rdx
  __int64 v17; // rax
  __int64 v18; // r12
  __int64 v19; // rbx
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // r12
  __int64 v23; // r12
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rax
  int v29; // eax
  __int64 v30; // r10
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // rax
  __int64 *v34; // r12
  unsigned int v35; // edx
  __int64 v36; // rsi
  __int64 v37; // rax
  char v38; // cl
  unsigned __int64 v39; // r9
  unsigned __int64 v40; // rax
  void *v41; // rax
  bool v42; // al
  __int64 v43; // r14
  __int64 v44; // r10
  __int64 v45; // rax
  __int64 v46; // r12
  __int64 v47; // r13
  __int64 v48; // r14
  unsigned __int8 *v49; // rax
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // r15
  unsigned __int8 *v53; // r12
  unsigned int v54; // r14d
  unsigned __int64 v55; // rax
  int v56; // eax
  unsigned int v57; // edx
  unsigned __int64 v58; // rcx
  unsigned int v59; // eax
  __int64 v60; // rsi
  __int64 v61; // rax
  __int64 v62; // rsi
  unsigned int v63; // ecx
  unsigned int v64; // r12d
  __int64 v65; // rdx
  __int64 v66; // rcx
  __int64 v67; // r8
  __int64 v68; // r9
  unsigned __int8 *v69; // r12
  __int64 v70; // rdi
  int v71; // eax
  __int64 v72; // rax
  _QWORD *v73; // rcx
  __int64 v74; // rdx
  __int64 v75; // rcx
  unsigned __int8 *v76; // rbx
  __int64 v77; // r8
  __int64 v78; // r9
  __int64 v79; // r12
  __int64 v80; // r14
  __int64 v81; // rax
  __int64 v82; // rax
  __int64 v83; // rdx
  __int64 v84; // r12
  __int64 v85; // r12
  __int64 v86; // rdx
  __int64 v87; // rcx
  __int64 v88; // r8
  __int64 v89; // r9
  __int64 v90; // rax
  __int64 v91; // r14
  __int64 v92; // r12
  __int64 v93; // r13
  __int64 v94; // r14
  char v95; // al
  __int64 v96; // rdx
  unsigned int **v97; // rdi
  __int64 v98; // rax
  __int64 v99; // rdx
  unsigned __int64 v100; // rax
  __int64 v101; // r12
  unsigned __int64 v102; // rcx
  unsigned int v103; // r12d
  __int64 v104; // rax
  __int64 v105; // rcx
  int v106; // eax
  unsigned int v107; // eax
  _QWORD *v108; // rax
  __int64 v109; // rax
  __int64 v110; // rax
  __int64 v111; // rax
  __int64 v112; // rbx
  unsigned int v113; // r14d
  unsigned __int64 v114; // rax
  unsigned int v115; // edx
  __int64 v116; // r12
  __int64 v117; // r14
  __int64 v118; // r13
  __int64 v119; // r8
  unsigned int v120; // ecx
  __int64 v121; // r9
  _QWORD *v122; // rax
  int v123; // r14d
  __int64 v124; // r13
  int v125; // eax
  _QWORD *v126; // rax
  __int64 v127; // rax
  __int64 v128; // r12
  _QWORD *v129; // rax
  __int64 v130; // r15
  __int64 v131; // r13
  __int64 v132; // rdx
  unsigned int v133; // esi
  _QWORD *v134; // rax
  __int64 v135; // rax
  __int64 v136; // rdx
  __int64 v137; // rbx
  __int64 v138; // r14
  __int64 v139; // rdx
  unsigned int v140; // esi
  _QWORD *v141; // rax
  __int64 v142; // rdx
  __int64 v143; // r13
  __int64 v144; // r15
  __int64 v145; // rdx
  unsigned int v146; // esi
  _QWORD *v147; // rax
  __int64 v148; // rbx
  __int64 v149; // r14
  __int64 v150; // rdx
  unsigned int v151; // esi
  _QWORD *v152; // rax
  __int64 v153; // r14
  __int64 v154; // r12
  __int64 v155; // rdx
  unsigned int v156; // esi
  __int64 v157; // rax
  __int64 v158; // rdx
  __int64 v159; // rdi
  unsigned __int8 *v160; // rdx
  unsigned __int8 *v161; // rdi
  unsigned int v162; // esi
  __int64 v163; // r14
  unsigned int **v164; // rdi
  _BYTE *v165; // rsi
  __int64 v166; // rax
  unsigned int v167; // eax
  __int64 v168; // rdx
  _BYTE *v169; // r14
  __int64 v170; // r8
  unsigned int **v171; // rdi
  _BYTE *v172; // rsi
  __int64 v173; // rax
  unsigned __int64 v174; // rdx
  char v175; // al
  __int64 v176; // rax
  __int64 *v177; // [rsp-10h] [rbp-100h]
  __int64 v178; // [rsp-8h] [rbp-F8h]
  __int64 v179; // [rsp+0h] [rbp-F0h]
  unsigned int v180; // [rsp+0h] [rbp-F0h]
  __int16 v181; // [rsp+8h] [rbp-E8h]
  char v182; // [rsp+8h] [rbp-E8h]
  int v183; // [rsp+8h] [rbp-E8h]
  unsigned int v184; // [rsp+8h] [rbp-E8h]
  __int64 v185; // [rsp+8h] [rbp-E8h]
  __int64 v186; // [rsp+10h] [rbp-E0h]
  __int64 v187; // [rsp+10h] [rbp-E0h]
  unsigned int v188; // [rsp+10h] [rbp-E0h]
  unsigned __int64 v189; // [rsp+10h] [rbp-E0h]
  __int64 v190; // [rsp+10h] [rbp-E0h]
  __int64 v191; // [rsp+10h] [rbp-E0h]
  bool v192; // [rsp+18h] [rbp-D8h]
  __int64 v193; // [rsp+18h] [rbp-D8h]
  __int64 v194; // [rsp+18h] [rbp-D8h]
  unsigned int v195; // [rsp+18h] [rbp-D8h]
  unsigned __int8 *v196; // [rsp+18h] [rbp-D8h]
  __int64 v197; // [rsp+18h] [rbp-D8h]
  unsigned int v198; // [rsp+18h] [rbp-D8h]
  __int64 v199; // [rsp+28h] [rbp-C8h]
  __int64 v200; // [rsp+28h] [rbp-C8h]
  unsigned __int8 *v201; // [rsp+28h] [rbp-C8h]
  __int64 v202; // [rsp+28h] [rbp-C8h]
  __int64 v203; // [rsp+28h] [rbp-C8h]
  __int64 v204; // [rsp+28h] [rbp-C8h]
  unsigned __int64 v205; // [rsp+30h] [rbp-C0h] BYREF
  unsigned int v206; // [rsp+38h] [rbp-B8h]
  unsigned __int64 v207; // [rsp+40h] [rbp-B0h] BYREF
  unsigned int v208; // [rsp+48h] [rbp-A8h]
  __int16 v209; // [rsp+60h] [rbp-90h]
  __m128i v210; // [rsp+70h] [rbp-80h] BYREF
  __m128i v211; // [rsp+80h] [rbp-70h] BYREF
  unsigned __int64 v212; // [rsp+90h] [rbp-60h]
  __int64 v213; // [rsp+98h] [rbp-58h]
  __m128i v214; // [rsp+A0h] [rbp-50h]
  __int64 v215; // [rsp+B0h] [rbp-40h]

  v4 = *(unsigned __int8 **)(a2 - 64);
  v5 = *(_QWORD *)(a2 - 32);
  v199 = a2;
  v6 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
  v7 = _mm_loadu_si128(a1 + 6);
  v8 = _mm_loadu_si128(a1 + 7);
  v9 = _mm_loadu_si128(a1 + 9);
  v215 = a1[10].m128i_i64[0];
  v212 = v6;
  v213 = a2;
  v210 = v7;
  v211 = v8;
  v214 = v9;
  v13 = sub_10049F0((__int64)v4, v5, (__int64)&v210);
  v14 = v13;
  if ( v13 )
    return sub_F162A0((__int64)a1, a2, v14);
  v16 = *(_BYTE **)(a2 - 64);
  if ( *v16 == 86 && *(_BYTE *)(*(_QWORD *)(*((_QWORD *)v16 - 12) + 8LL) + 8LL) == 12 && **(_BYTE **)(a2 - 32) <= 0x15u )
  {
    v41 = sub_F26350((__int64)a1, (_BYTE *)a2, (__int64)v16, 0);
    if ( v41 )
      return (unsigned __int8 *)v41;
  }
  if ( *(_BYTE *)v5 != 17 )
  {
    v192 = 0;
    v29 = *v4;
    v30 = 0;
    goto LABEL_17;
  }
  v17 = sub_11AEFE0(v5);
  if ( v17 )
  {
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      v18 = *(_QWORD *)(a2 - 8);
    else
      v18 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
    v19 = *(_QWORD *)(v18 + 32);
    v10 = (__int64 *)(v18 + 32);
    if ( v19 )
    {
      v20 = *(_QWORD *)(v18 + 40);
      **(_QWORD **)(v18 + 48) = v20;
      if ( v20 )
        *(_QWORD *)(v20 + 16) = *(_QWORD *)(v18 + 48);
    }
    *(_QWORD *)(v18 + 32) = v17;
    v21 = *(_QWORD *)(v17 + 16);
    *(_QWORD *)(v18 + 40) = v21;
    if ( v21 )
      *(_QWORD *)(v21 + 16) = v18 + 40;
    *(_QWORD *)(v18 + 48) = v17 + 16;
    *(_QWORD *)(v17 + 16) = v10;
    if ( *(_BYTE *)v19 <= 0x1Cu )
      return (unsigned __int8 *)v199;
    goto LABEL_14;
  }
  v35 = *(_DWORD *)(v5 + 32);
  v36 = v5 + 24;
  v37 = *(_QWORD *)(*(_QWORD *)(a2 - 64) + 8LL);
  v38 = *(_BYTE *)(v37 + 8);
  v39 = *(unsigned int *)(v37 + 32);
  if ( v35 <= 0x40 )
  {
    v40 = *(_QWORD *)(v5 + 24);
LABEL_41:
    v192 = v39 > v40;
    goto LABEL_42;
  }
  v180 = *(_DWORD *)(v5 + 32);
  v182 = *(_BYTE *)(v37 + 8);
  v189 = *(unsigned int *)(v37 + 32);
  v71 = sub_C444A0(v5 + 24);
  v35 = v180;
  v36 = v5 + 24;
  v39 = v189;
  v38 = v182;
  if ( v180 - v71 <= 0x40 )
  {
    v40 = **(_QWORD **)(v5 + 24);
    goto LABEL_41;
  }
  v192 = 0;
LABEL_42:
  if ( *v4 == 85
    && (v111 = *((_QWORD *)v4 - 4)) != 0
    && !*(_BYTE *)v111
    && *(_QWORD *)(v111 + 24) == *((_QWORD *)v4 + 10)
    && (*(_BYTE *)(v111 + 33) & 0x20) != 0
    && *(_DWORD *)(v111 + 36) == 345 )
  {
    if ( v192 )
    {
      v112 = *(_QWORD *)(a2 + 8);
      v113 = *(_DWORD *)(v112 + 8) >> 8;
      if ( v35 > 0x40 )
      {
        v115 = v35 - sub_C444A0(v36);
      }
      else
      {
        v114 = *(_QWORD *)(v5 + 24);
        if ( !v114 )
          goto LABEL_271;
        _BitScanReverse64(&v114, v114);
        v115 = 64 - (v114 ^ 0x3F);
      }
      if ( v113 < v115 )
      {
        v14 = sub_ACADE0((__int64 **)v112);
        return sub_F162A0((__int64)a1, a2, v14);
      }
LABEL_271:
      sub_C44AB0((__int64)&v210, v36, v113);
      v157 = sub_AD8D80(v112, (__int64)&v210);
      v14 = v157;
      if ( v210.m128i_i32[2] > 0x40u && v210.m128i_i64[0] )
      {
        v202 = v157;
        j_j___libc_free_0_0(v210.m128i_i64[0]);
        v14 = v202;
      }
      return sub_F162A0((__int64)a1, a2, v14);
    }
    if ( v38 != 18 )
      return 0;
  }
  else if ( v38 != 18 && !v192 )
  {
    return 0;
  }
  v41 = sub_11AFDB0((__int64)a1, a2);
  if ( v41 )
    return (unsigned __int8 *)v41;
  v29 = *v4;
  v30 = v5;
  if ( (_BYTE)v29 == 84 )
  {
    v41 = sub_11B5E90((__int64)a1, (__int64 *)a2, (__int64)v4, (__int64)v10, v11, v12);
    if ( v41 )
      return (unsigned __int8 *)v41;
    v29 = *v4;
    v30 = v5;
  }
LABEL_17:
  if ( (_BYTE)v29 == 41 )
  {
    v190 = v30;
    if ( sub_11AF2C0((__int64)v4, v5, (__int64)v16, (__int64)v10) )
    {
      v116 = a1[2].m128i_i64[0];
      v117 = *((_QWORD *)v4 - 4);
      v209 = 257;
      v118 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(v116 + 80) + 96LL))(
               *(_QWORD *)(v116 + 80),
               v117,
               v5);
      if ( !v118 )
      {
        LOWORD(v212) = 257;
        v152 = sub_BD2C40(72, 2u);
        v118 = (__int64)v152;
        if ( v152 )
          sub_B4DE80((__int64)v152, v117, v5, (__int64)&v210, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, unsigned __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v116 + 88) + 16LL))(
          *(_QWORD *)(v116 + 88),
          v118,
          &v207,
          *(_QWORD *)(v116 + 56),
          *(_QWORD *)(v116 + 64));
        v153 = *(_QWORD *)v116;
        v154 = *(_QWORD *)v116 + 16LL * *(unsigned int *)(v116 + 8);
        while ( v154 != v153 )
        {
          v155 = *(_QWORD *)(v153 + 8);
          v156 = *(_DWORD *)v153;
          v153 += 16;
          sub_B99FD0(v118, v156, v155);
        }
      }
      LOWORD(v212) = 257;
      v49 = (unsigned __int8 *)sub_B50340((unsigned int)*v4 - 29, v118, (__int64)&v210, 0, 0);
      goto LABEL_52;
    }
    v29 = *v4;
    v30 = v190;
  }
  if ( (unsigned __int8)(v29 - 42) > 0x11u )
    goto LABEL_19;
  v187 = v30;
  v42 = sub_11AF2C0((__int64)v4, v5, (unsigned int)(v29 - 42), (__int64)v10);
  v30 = v187;
  if ( !v42 )
  {
    v29 = *v4;
    goto LABEL_19;
  }
  if ( v192 || (v95 = sub_991A70(v4, 0, 0, 0, 0, 0, 0), v10 = v177, v95) )
  {
    v43 = a1[2].m128i_i64[0];
    v44 = *((_QWORD *)v4 - 8);
    v45 = *((_QWORD *)v4 - 4);
    v209 = 257;
    v194 = v44;
    v200 = v45;
    v46 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(v43 + 80) + 96LL))(
            *(_QWORD *)(v43 + 80),
            v44,
            v5);
    if ( !v46 )
    {
      LOWORD(v212) = 257;
      v134 = sub_BD2C40(72, 2u);
      v46 = (__int64)v134;
      if ( v134 )
        sub_B4DE80((__int64)v134, v194, v5, (__int64)&v210, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, unsigned __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v43 + 88) + 16LL))(
        *(_QWORD *)(v43 + 88),
        v46,
        &v207,
        *(_QWORD *)(v43 + 56),
        *(_QWORD *)(v43 + 64));
      v135 = *(_QWORD *)v43;
      v136 = 16LL * *(unsigned int *)(v43 + 8);
      if ( v135 != v135 + v136 )
      {
        v196 = v4;
        v137 = *(_QWORD *)v43;
        v138 = v135 + v136;
        do
        {
          v139 = *(_QWORD *)(v137 + 8);
          v140 = *(_DWORD *)v137;
          v137 += 16;
          sub_B99FD0(v46, v140, v139);
        }
        while ( v138 != v137 );
        v4 = v196;
      }
    }
    v47 = a1[2].m128i_i64[0];
    v209 = 257;
    v48 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(v47 + 80) + 96LL))(
            *(_QWORD *)(v47 + 80),
            v200,
            v5);
    if ( !v48 )
    {
      LOWORD(v212) = 257;
      v129 = sub_BD2C40(72, 2u);
      v48 = (__int64)v129;
      if ( v129 )
        sub_B4DE80((__int64)v129, v200, v5, (__int64)&v210, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, unsigned __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v47 + 88) + 16LL))(
        *(_QWORD *)(v47 + 88),
        v48,
        &v207,
        *(_QWORD *)(v47 + 56),
        *(_QWORD *)(v47 + 64));
      v130 = *(_QWORD *)v47;
      v131 = *(_QWORD *)v47 + 16LL * *(unsigned int *)(v47 + 8);
      while ( v131 != v130 )
      {
        v132 = *(_QWORD *)(v130 + 8);
        v133 = *(_DWORD *)v130;
        v130 += 16;
        sub_B99FD0(v48, v133, v132);
      }
    }
    LOWORD(v212) = 257;
    v49 = (unsigned __int8 *)sub_B504D0((unsigned int)*v4 - 29, v46, v48, (__int64)&v210, 0, 0);
LABEL_52:
    v199 = (__int64)v49;
    sub_B45260(v49, (__int64)v4, 1);
    return (unsigned __int8 *)v199;
  }
  v29 = *v4;
  v30 = v187;
LABEL_19:
  v21 = (unsigned int)(v29 - 82);
  if ( (unsigned __int8)(v29 - 82) <= 1u )
  {
    v193 = *((_QWORD *)v4 - 8);
    if ( !v193 || (v179 = *((_QWORD *)v4 - 4)) == 0 )
    {
LABEL_156:
      if ( (unsigned int)(unsigned __int8)v29 - 67 <= 0xC )
      {
        v96 = *((_QWORD *)v4 + 2);
        if ( v96 )
        {
          if ( !*(_QWORD *)(v96 + 8) && (_BYTE)v29 != 78 )
          {
            v97 = (unsigned int **)a1[2].m128i_i64[0];
            LOWORD(v212) = 257;
            v98 = sub_A837F0(v97, *((_BYTE **)v4 - 4), (_BYTE *)v5, (__int64)&v210);
            v99 = *(_QWORD *)(a2 + 8);
            LOWORD(v212) = 257;
            return (unsigned __int8 *)sub_B51D30((unsigned int)*v4 - 29, v98, v99, (__int64)&v210, 0, 0);
          }
        }
      }
LABEL_60:
      if ( !v30 )
        return 0;
      goto LABEL_61;
    }
    v186 = v30;
    v181 = sub_B53900((__int64)v4);
    if ( sub_11AF2C0((__int64)v4, v5, v31, v32) )
    {
      v91 = a1[2].m128i_i64[0];
      v209 = 257;
      v92 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(v91 + 80) + 96LL))(
              *(_QWORD *)(v91 + 80),
              v193,
              v5);
      if ( !v92 )
      {
        LOWORD(v212) = 257;
        v147 = sub_BD2C40(72, 2u);
        v92 = (__int64)v147;
        if ( v147 )
          sub_B4DE80((__int64)v147, v193, v5, (__int64)&v210, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, unsigned __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v91 + 88) + 16LL))(
          *(_QWORD *)(v91 + 88),
          v92,
          &v207,
          *(_QWORD *)(v91 + 56),
          *(_QWORD *)(v91 + 64));
        if ( *(_QWORD *)v91 != *(_QWORD *)v91 + 16LL * *(unsigned int *)(v91 + 8) )
        {
          v201 = v4;
          v148 = *(_QWORD *)v91;
          v149 = *(_QWORD *)v91 + 16LL * *(unsigned int *)(v91 + 8);
          do
          {
            v150 = *(_QWORD *)(v148 + 8);
            v151 = *(_DWORD *)v148;
            v148 += 16;
            sub_B99FD0(v92, v151, v150);
          }
          while ( v149 != v148 );
          v4 = v201;
        }
      }
      v93 = a1[2].m128i_i64[0];
      v209 = 257;
      v94 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(v93 + 80) + 96LL))(
              *(_QWORD *)(v93 + 80),
              v179,
              v5);
      if ( !v94 )
      {
        LOWORD(v212) = 257;
        v141 = sub_BD2C40(72, 2u);
        v94 = (__int64)v141;
        if ( v141 )
          sub_B4DE80((__int64)v141, v179, v5, (__int64)&v210, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, unsigned __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v93 + 88) + 16LL))(
          *(_QWORD *)(v93 + 88),
          v94,
          &v207,
          *(_QWORD *)(v93 + 56),
          *(_QWORD *)(v93 + 64));
        v142 = 16LL * *(unsigned int *)(v93 + 8);
        v143 = *(_QWORD *)v93;
        v144 = v143 + v142;
        while ( v144 != v143 )
        {
          v145 = *(_QWORD *)(v143 + 8);
          v146 = *(_DWORD *)v143;
          v143 += 16;
          sub_B99FD0(v94, v146, v145);
        }
      }
      LOWORD(v212) = 257;
      return sub_B527B0((unsigned int)*v4 - 29, v181, v92, v94, (__int64)v4, (__int64)&v210, 0, 0);
    }
    LOBYTE(v29) = *v4;
    v30 = v186;
  }
  if ( (unsigned __int8)v29 <= 0x1Cu )
    goto LABEL_60;
  if ( (_BYTE)v29 == 91 )
  {
    if ( **((_BYTE **)v4 - 4) <= 0x15u )
    {
      if ( !v30 )
        return 0;
      v33 = *((_QWORD *)v4 - 12);
      if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      {
        v34 = *(__int64 **)(a2 - 8);
      }
      else
      {
        v21 = 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
        v34 = (__int64 *)(a2 - v21);
      }
      v19 = *v34;
      if ( *v34 )
      {
        v10 = (__int64 *)v34[2];
        v21 = v34[1];
        *v10 = v21;
        if ( v21 )
        {
          v10 = (__int64 *)v34[2];
          *(_QWORD *)(v21 + 16) = v10;
        }
      }
      *v34 = v33;
      if ( v33 )
      {
        v21 = *(_QWORD *)(v33 + 16);
        v10 = (__int64 *)(v33 + 16);
        v34[1] = v21;
        if ( v21 )
          *(_QWORD *)(v21 + 16) = v34 + 1;
        v34[2] = (__int64)v10;
        *(_QWORD *)(v33 + 16) = v34;
      }
      if ( *(_BYTE *)v19 <= 0x1Cu )
        return (unsigned __int8 *)v199;
LABEL_14:
      v22 = a1[2].m128i_i64[1];
      v210.m128i_i64[0] = v19;
      v23 = v22 + 2096;
      sub_11B4E60(v23, v210.m128i_i64, v21, (__int64)v10, v11, v12);
      v28 = *(_QWORD *)(v19 + 16);
      if ( v28 )
      {
        if ( !*(_QWORD *)(v28 + 8) )
        {
          v210.m128i_i64[0] = *(_QWORD *)(v28 + 24);
          sub_11B4E60(v23, v210.m128i_i64, v24, v25, v26, v27);
        }
      }
      return (unsigned __int8 *)v199;
    }
    goto LABEL_60;
  }
  if ( (_BYTE)v29 != 63 )
  {
    if ( (_BYTE)v29 == 92 )
    {
      if ( *(_BYTE *)(*((_QWORD *)v4 + 1) + 8LL) == 17 && *(_BYTE *)v5 == 17 )
      {
        v122 = *(_QWORD **)(v5 + 24);
        if ( *(_DWORD *)(v5 + 32) > 0x40u )
          v122 = (_QWORD *)*v122;
        v123 = *(_DWORD *)(*((_QWORD *)v4 + 9) + 4LL * (unsigned int)v122);
        if ( v123 >= 0 )
        {
          v124 = *((_QWORD *)v4 - 8);
          v125 = *(_DWORD *)(*(_QWORD *)(v124 + 8) + 32LL);
          if ( v125 <= v123 )
          {
            v124 = *((_QWORD *)v4 - 4);
            v123 -= v125;
          }
          v126 = (_QWORD *)sub_BD5C60(a2);
          v127 = sub_BCB2E0(v126);
          LOWORD(v212) = 257;
          v128 = sub_AD64C0(v127, v123, 0);
          v199 = (__int64)sub_BD2C40(72, 2u);
          if ( v199 )
            sub_B4DE80(v199, v124, v128, (__int64)&v210, 0, 0);
          return (unsigned __int8 *)v199;
        }
        v14 = sub_ACADE0(*(__int64 ***)(a2 + 8));
        return sub_F162A0((__int64)a1, a2, v14);
      }
      goto LABEL_60;
    }
    goto LABEL_156;
  }
  if ( !v30 )
    return 0;
  v108 = *(_QWORD **)(v30 + 24);
  if ( *(_DWORD *)(v30 + 32) > 0x40u )
    v108 = (_QWORD *)*v108;
  v195 = *(_DWORD *)(*((_QWORD *)v4 + 1) + 32LL);
  if ( v195 <= (unsigned __int64)v108 )
  {
LABEL_61:
    v50 = *(_QWORD *)(*(_QWORD *)(a2 - 64) + 8LL);
    v188 = *(_DWORD *)(v50 + 32);
    if ( *(_BYTE *)(v50 + 8) == 18 || *(_DWORD *)(v50 + 32) == 1 )
      return 0;
    v51 = *((_QWORD *)v4 + 2);
    if ( !v51 )
      goto LABEL_64;
    v72 = *(_QWORD *)(v51 + 8);
    goto LABEL_127;
  }
  v109 = *((_QWORD *)v4 + 2);
  if ( !v109 )
  {
    v110 = *(_QWORD *)(*(_QWORD *)(a2 - 64) + 8LL);
    v188 = *(_DWORD *)(v110 + 32);
    if ( *(_BYTE *)(v110 + 8) == 18 || *(_DWORD *)(v110 + 32) == 1 )
      return 0;
    goto LABEL_65;
  }
  v72 = *(_QWORD *)(v109 + 8);
  if ( !v72 )
  {
    v159 = 32LL * (*((_DWORD *)v4 + 1) & 0x7FFFFFF);
    v160 = &v4[-v159];
    if ( (v4[7] & 0x40) != 0 )
      v160 = (unsigned __int8 *)*((_QWORD *)v4 - 1);
    v161 = &v160[v159];
    if ( v161 == v160 )
    {
      v176 = *(_QWORD *)(*(_QWORD *)(a2 - 64) + 8LL);
      v188 = *(_DWORD *)(v176 + 32);
      if ( *(_BYTE *)(v176 + 8) == 18 || *(_DWORD *)(v176 + 32) == 1 )
        return 0;
      goto LABEL_128;
    }
    v162 = 0;
    do
    {
      v162 += (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)v160 + 8LL) + 8LL) - 17 < 2;
      v160 += 32;
    }
    while ( v160 != v161 );
    v198 = v162;
    if ( v162 == 1 )
    {
      v163 = *(_QWORD *)&v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
      if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v163 + 8) + 8LL) - 17 <= 1 )
      {
        v164 = (unsigned int **)a1[2].m128i_i64[0];
        v165 = *(_BYTE **)&v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
        LOWORD(v212) = 257;
        v203 = v30;
        v166 = sub_A837F0(v164, v165, (_BYTE *)v30, (__int64)&v210);
        v30 = v203;
        v163 = v166;
      }
      v210.m128i_i64[0] = (__int64)&v211;
      v210.m128i_i64[1] = 0x600000000LL;
      v167 = *((_DWORD *)v4 + 1) & 0x7FFFFFF;
      if ( v167 != 1 )
      {
        v204 = v163;
        v168 = v167;
        v169 = (_BYTE *)v30;
        do
        {
          v170 = *(_QWORD *)&v4[32 * (v198 - v168)];
          if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v170 + 8) + 8LL) - 17 <= 1 )
          {
            v171 = (unsigned int **)a1[2].m128i_i64[0];
            v172 = *(_BYTE **)&v4[32 * (v198 - v168)];
            v209 = 257;
            v170 = sub_A837F0(v171, v172, v169, (__int64)&v207);
          }
          v173 = v210.m128i_u32[2];
          v174 = v210.m128i_u32[2] + 1LL;
          if ( v174 > v210.m128i_u32[3] )
          {
            v191 = v170;
            sub_C8D5F0((__int64)&v210, &v211, v174, 8u, v170, v12);
            v173 = v210.m128i_u32[2];
            v170 = v191;
          }
          *(_QWORD *)(v210.m128i_i64[0] + 8 * v173) = v170;
          ++v210.m128i_i32[2];
          ++v198;
          v168 = *((_DWORD *)v4 + 1) & 0x7FFFFFF;
        }
        while ( v198 != (_DWORD)v168 );
        v163 = v204;
      }
      v209 = 257;
      v199 = sub_9C6B10(*((_QWORD *)v4 + 9), v163, v210.m128i_i64[0], v210.m128i_u32[2], (__int64)&v207, v12, 0, 0);
      v175 = sub_B4DE30((__int64)v4);
      sub_B4DE00(v199, v175);
      if ( (__m128i *)v210.m128i_i64[0] != &v211 )
        _libc_free(v210.m128i_i64[0], v178);
      return (unsigned __int8 *)v199;
    }
  }
  v158 = *(_QWORD *)(*(_QWORD *)(a2 - 64) + 8LL);
  v188 = *(_DWORD *)(v158 + 32);
  if ( *(_BYTE *)(v158 + 8) == 18 || *(_DWORD *)(v158 + 32) == 1 )
    return 0;
LABEL_127:
  if ( v72 )
  {
LABEL_64:
    v195 = *(_DWORD *)(*((_QWORD *)v4 + 1) + 32LL);
LABEL_65:
    v206 = v195;
    if ( v195 > 0x40 )
      sub_C43690((__int64)&v205, 0, 0);
    else
      v205 = 0;
    v52 = *((_QWORD *)v4 + 2);
    if ( !v52 )
    {
      v64 = v206;
      goto LABEL_169;
    }
    while ( 1 )
    {
      v53 = *(unsigned __int8 **)(v52 + 24);
      if ( *v53 <= 0x1Cu )
      {
        v210.m128i_i32[2] = v195;
        if ( v195 > 0x40 )
        {
          sub_C43690((__int64)&v210, -1, 1);
        }
        else
        {
          v100 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v195;
          if ( !v195 )
            v100 = 0;
          v210.m128i_i64[0] = v100;
        }
        if ( v206 > 0x40 && v205 )
          j_j___libc_free_0_0(v205);
        v64 = v210.m128i_u32[2];
        v205 = v210.m128i_i64[0];
        v206 = v210.m128i_u32[2];
LABEL_169:
        if ( !v64 )
          return 0;
LABEL_92:
        if ( v64 > 0x40 )
        {
          if ( v64 != (unsigned int)sub_C445E0((__int64)&v205) )
            goto LABEL_94;
LABEL_56:
          if ( v206 > 0x40 && v205 )
            j_j___libc_free_0_0(v205);
          return 0;
        }
        if ( v205 == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v64) )
          return 0;
LABEL_94:
        v208 = v188;
        if ( v188 > 0x40 )
          sub_C43690((__int64)&v207, 0, 0);
        else
          v207 = 0;
        v210.m128i_i32[2] = v206;
        if ( v206 > 0x40 )
          sub_C43780((__int64)&v210, (const void **)&v205);
        else
          v210.m128i_i64[0] = v205;
        v69 = sub_11A3F30((__int64)a1, v4, (__int64)&v210, (__int64 *)&v207, 0, 1);
        if ( v210.m128i_i32[2] > 0x40u && v210.m128i_i64[0] )
          j_j___libc_free_0_0(v210.m128i_i64[0]);
        if ( v69 == v4 || !v69 )
        {
          if ( v208 > 0x40 && v207 )
            j_j___libc_free_0_0(v207);
          goto LABEL_56;
        }
        if ( *v4 > 0x1Cu )
        {
          v70 = a1[2].m128i_i64[1];
          v210.m128i_i64[0] = (__int64)v4;
          sub_11B4E60(v70 + 2096, v210.m128i_i64, v65, v66, v67, v68);
        }
        sub_BD84D0((__int64)v4, (__int64)v69);
LABEL_106:
        if ( v208 > 0x40 && v207 )
          j_j___libc_free_0_0(v207);
        if ( v206 > 0x40 && v205 )
          j_j___libc_free_0_0(v205);
        return (unsigned __int8 *)v199;
      }
      v54 = *(_DWORD *)(*((_QWORD *)v4 + 1) + 32LL);
      v208 = v54;
      if ( v54 > 0x40 )
      {
        sub_C43690((__int64)&v207, -1, 1);
        v106 = *v53;
        if ( v106 != 90 )
        {
          if ( v106 != 92 )
            goto LABEL_83;
          v107 = *(_DWORD *)(*((_QWORD *)v53 + 1) + 32LL);
          v210.m128i_i32[2] = v54;
          v184 = v107;
          sub_C43690((__int64)&v210, 0, 0);
          v59 = v184;
          if ( v208 > 0x40 && v207 )
          {
            j_j___libc_free_0_0(v207);
            v58 = v210.m128i_i64[0];
            v57 = v210.m128i_u32[2];
            v59 = v184;
          }
          else
          {
            v58 = v210.m128i_i64[0];
            v57 = v210.m128i_u32[2];
          }
          goto LABEL_75;
        }
      }
      else
      {
        v55 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v54;
        if ( !v54 )
          v55 = 0;
        v207 = v55;
        v56 = *v53;
        if ( v56 != 90 )
        {
          if ( v56 != 92 )
            goto LABEL_83;
          v57 = v54;
          v58 = 0;
          v59 = *(_DWORD *)(*((_QWORD *)v53 + 1) + 32LL);
LABEL_75:
          v207 = v58;
          v208 = v57;
          if ( v59 )
          {
            v60 = v59;
            v61 = 0;
            v62 = 4 * v60;
            do
            {
              v63 = *(_DWORD *)(*((_QWORD *)v53 + 9) + v61);
              if ( v63 < 2 * v54 )
              {
                if ( v4 == *((unsigned __int8 **)v53 - 8) && *((_QWORD *)v53 - 8) != 0 && v54 > v63 )
                {
                  v119 = 1LL << v63;
                  if ( v208 > 0x40 )
                    *(_QWORD *)(v207 + 8LL * (v63 >> 6)) |= v119;
                  else
                    v207 |= v119;
                }
                else if ( *((_QWORD *)v53 - 4) != 0 && v4 == *((unsigned __int8 **)v53 - 4) && v54 <= v63 )
                {
                  v120 = v63 - v54;
                  v121 = 1LL << v120;
                  if ( v208 > 0x40 )
                    *(_QWORD *)(v207 + 8LL * (v120 >> 6)) |= v121;
                  else
                    v207 |= v121;
                }
              }
              v61 += 4;
            }
            while ( v62 != v61 );
          }
          goto LABEL_83;
        }
      }
      v101 = *((_QWORD *)v53 - 4);
      if ( *(_BYTE *)v101 != 17 )
        goto LABEL_83;
      if ( *(_DWORD *)(v101 + 32) > 0x40u )
      {
        v183 = *(_DWORD *)(v101 + 32);
        if ( v183 - (unsigned int)sub_C444A0(v101 + 24) > 0x40 )
          goto LABEL_83;
        v102 = **(_QWORD **)(v101 + 24);
        if ( v54 <= v102 )
          goto LABEL_83;
      }
      else
      {
        v102 = *(_QWORD *)(v101 + 24);
        if ( v54 <= v102 )
          goto LABEL_83;
      }
      v103 = v102;
      v210.m128i_i32[2] = v54;
      v104 = 1LL << v102;
      v105 = 1LL << v102;
      if ( v54 <= 0x40 )
        break;
      v185 = v104;
      sub_C43690((__int64)&v210, 0, 0);
      v105 = v185;
      if ( v210.m128i_i32[2] <= 0x40u )
        goto LABEL_176;
      *(_QWORD *)(v210.m128i_i64[0] + 8LL * (v103 >> 6)) |= v185;
LABEL_177:
      if ( v208 > 0x40 && v207 )
        j_j___libc_free_0_0(v207);
      v207 = v210.m128i_i64[0];
      v208 = v210.m128i_u32[2];
LABEL_83:
      if ( v206 > 0x40 )
        sub_C43BD0(&v205, (__int64 *)&v207);
      else
        v205 |= v207;
      if ( v208 > 0x40 && v207 )
        j_j___libc_free_0_0(v207);
      v64 = v206;
      if ( !v206 )
        return 0;
      if ( v206 <= 0x40 )
      {
        if ( v205 == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v206) )
          goto LABEL_92;
      }
      else if ( v64 == (unsigned int)sub_C445E0((__int64)&v205) )
      {
        goto LABEL_92;
      }
      v52 = *(_QWORD *)(v52 + 8);
      if ( !v52 )
        goto LABEL_92;
    }
    v210.m128i_i64[0] = 0;
LABEL_176:
    v210.m128i_i64[0] |= v105;
    goto LABEL_177;
  }
LABEL_128:
  v206 = v188;
  if ( v188 > 0x40 )
  {
    v197 = v30;
    sub_C43690((__int64)&v205, 0, 0);
    v208 = v188;
    sub_C43690((__int64)&v207, 0, 0);
    v13 = v207;
    v30 = v197;
    v188 = v208;
  }
  else
  {
    v208 = v188;
    v205 = 0;
    v207 = 0;
  }
  v73 = *(_QWORD **)(v30 + 24);
  if ( *(_DWORD *)(v30 + 32) > 0x40u )
    v73 = (_QWORD *)*v73;
  v74 = 1LL << (char)v73;
  if ( v188 > 0x40 )
  {
    *(_QWORD *)(v13 + 8LL * ((unsigned int)v73 >> 6)) |= v74;
    v210.m128i_i32[2] = v208;
    if ( v208 > 0x40 )
    {
      sub_C43780((__int64)&v210, (const void **)&v207);
      goto LABEL_135;
    }
  }
  else
  {
    v210.m128i_i32[2] = v188;
    v207 = v74 | v13;
  }
  v210.m128i_i64[0] = v207;
LABEL_135:
  v76 = sub_11A3F30((__int64)a1, v4, (__int64)&v210, (__int64 *)&v205, 0, 0);
  if ( v210.m128i_i32[2] > 0x40u && v210.m128i_i64[0] )
    j_j___libc_free_0_0(v210.m128i_i64[0]);
  if ( v76 )
  {
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      v79 = *(_QWORD *)(a2 - 8);
    else
      v79 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
    v80 = *(_QWORD *)v79;
    if ( *(_QWORD *)v79 )
    {
      v81 = *(_QWORD *)(v79 + 8);
      **(_QWORD **)(v79 + 16) = v81;
      if ( v81 )
        *(_QWORD *)(v81 + 16) = *(_QWORD *)(v79 + 16);
    }
    *(_QWORD *)v79 = v76;
    v82 = *((_QWORD *)v76 + 2);
    v83 = (__int64)(v76 + 16);
    *(_QWORD *)(v79 + 8) = v82;
    if ( v82 )
    {
      v75 = v79 + 8;
      *(_QWORD *)(v82 + 16) = v79 + 8;
    }
    *(_QWORD *)(v79 + 16) = v83;
    *((_QWORD *)v76 + 2) = v79;
    if ( *(_BYTE *)v80 > 0x1Cu )
    {
      v84 = a1[2].m128i_i64[1];
      v210.m128i_i64[0] = v80;
      v85 = v84 + 2096;
      sub_11B4E60(v85, v210.m128i_i64, v83, v75, v77, v78);
      v90 = *(_QWORD *)(v80 + 16);
      if ( v90 )
      {
        if ( !*(_QWORD *)(v90 + 8) )
        {
          v210.m128i_i64[0] = *(_QWORD *)(v90 + 24);
          sub_11B4E60(v85, v210.m128i_i64, v86, v87, v88, v89);
        }
      }
    }
    goto LABEL_106;
  }
  sub_969240((__int64 *)&v207);
  sub_969240((__int64 *)&v205);
  return 0;
}
