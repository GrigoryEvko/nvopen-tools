// Function: sub_185E850
// Address: 0x185e850
//
__int64 __fastcall sub_185E850(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        __m128i a4,
        __m128i a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // r13
  __int64 i; // rbx
  __int64 v13; // rdx
  __int64 v14; // rcx
  _QWORD *v15; // rdi
  unsigned __int8 v16; // al
  __int64 result; // rax
  unsigned int v18; // ecx
  __int64 v19; // rax
  __int64 v20; // rax
  unsigned int v21; // ebx
  __int64 v22; // rsi
  char v23; // al
  _BYTE *v24; // rdi
  __int64 v25; // r15
  __int64 v26; // r12
  __int64 v27; // rax
  __int64 v28; // rdx
  char v29; // al
  __m128 *v30; // rcx
  unsigned int v31; // r14d
  char v32; // bl
  _QWORD *v33; // rax
  __int16 v34; // bx
  unsigned int v35; // r14d
  __int64 v36; // rdi
  __int64 v37; // rsi
  __int64 v38; // rbx
  __int64 v39; // rax
  _BYTE *v40; // rsi
  __int64 v41; // rbx
  __int64 v42; // r14
  __int64 v43; // r12
  unsigned __int64 v44; // rbx
  __int64 v45; // rax
  unsigned __int64 v46; // rcx
  __int64 v47; // rax
  _QWORD *v48; // rax
  __int64 **v49; // rax
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // rax
  __int64 v53; // rdi
  _QWORD *v54; // rax
  int v55; // r9d
  double v56; // xmm4_8
  double v57; // xmm5_8
  __int64 v58; // r13
  unsigned int v59; // ecx
  _QWORD *v60; // rax
  __int64 v61; // rax
  _QWORD *v62; // rsi
  _QWORD **v63; // rdi
  int v64; // r8d
  __int32 v65; // eax
  _QWORD *v66; // rbx
  unsigned int v67; // r15d
  unsigned __int64 v68; // rcx
  unsigned int v69; // r12d
  __int64 v70; // r14
  _QWORD *j; // rax
  __int64 **v72; // rdx
  unsigned __int64 *v73; // rcx
  unsigned __int64 v74; // rdx
  double v75; // xmm4_8
  double v76; // xmm5_8
  __int64 v77; // r13
  __int64 v78; // rdx
  __int64 v79; // rbx
  __int64 v80; // r15
  __int64 v81; // r12
  unsigned __int64 *v82; // rcx
  unsigned __int64 v83; // rdx
  double v84; // xmm4_8
  double v85; // xmm5_8
  __int64 v86; // rsi
  __int64 v87; // r8
  __int64 v88; // rax
  __int64 v89; // r12
  unsigned int v90; // eax
  unsigned int v91; // r15d
  unsigned int v92; // r12d
  _QWORD *v93; // rcx
  __int64 v94; // rdx
  __int32 v95; // r14d
  __int64 v96; // rax
  char v97; // al
  __int64 v98; // rdx
  __m128i *v99; // rcx
  __int64 v100; // r15
  __int64 *v101; // r14
  _QWORD *v102; // rax
  __int64 v103; // r12
  __int64 v104; // rsi
  __int64 v105; // rax
  __int64 *v106; // rax
  __int64 *v107; // rax
  int v108; // r8d
  __int64 *v109; // r10
  __int64 *v110; // rcx
  __int64 *v111; // rax
  __int64 v112; // rdx
  __int64 *v113; // rax
  __int64 v114; // r12
  __int64 v115; // r12
  unsigned int v116; // r12d
  __m128 *v117; // rdx
  char v118; // al
  unsigned int v119; // r14d
  char v120; // bl
  _QWORD *v121; // rax
  __int16 v122; // bx
  unsigned int v123; // r14d
  __int64 v124; // rdi
  __int64 v125; // r8
  __int64 v126; // rbx
  __int64 v127; // rcx
  __int64 v128; // rax
  _BYTE *v129; // rsi
  __int64 v130; // r9
  unsigned int v131; // ebx
  unsigned __int64 v132; // rsi
  __int64 v133; // rax
  __int64 v134; // rdx
  __int64 v135; // rax
  __int64 *v136; // rax
  __int64 v137; // r14
  __int64 v138; // rsi
  __int64 v139; // rax
  __int64 v140; // [rsp-180h] [rbp-180h]
  __int64 v141; // [rsp-180h] [rbp-180h]
  __int64 v142; // [rsp-178h] [rbp-178h]
  __int64 v143; // [rsp-178h] [rbp-178h]
  unsigned int v144; // [rsp-170h] [rbp-170h]
  unsigned int v145; // [rsp-168h] [rbp-168h]
  unsigned __int64 v146; // [rsp-168h] [rbp-168h]
  __int64 v147; // [rsp-160h] [rbp-160h]
  int v148; // [rsp-150h] [rbp-150h]
  __int64 v149; // [rsp-148h] [rbp-148h]
  _QWORD *v150; // [rsp-148h] [rbp-148h]
  __int64 v151; // [rsp-148h] [rbp-148h]
  __int64 v152; // [rsp-148h] [rbp-148h]
  unsigned int v153; // [rsp-148h] [rbp-148h]
  unsigned __int64 v154; // [rsp-148h] [rbp-148h]
  __int64 v155; // [rsp-140h] [rbp-140h]
  unsigned int v156; // [rsp-140h] [rbp-140h]
  __int64 *v157; // [rsp-138h] [rbp-138h]
  __int64 v158; // [rsp-138h] [rbp-138h]
  __int64 v159; // [rsp-130h] [rbp-130h]
  __int64 v160; // [rsp-130h] [rbp-130h]
  __int64 v161; // [rsp-128h] [rbp-128h]
  __int64 v162; // [rsp-128h] [rbp-128h]
  _QWORD *v163; // [rsp-128h] [rbp-128h]
  __int64 v164; // [rsp-128h] [rbp-128h]
  __int64 v165; // [rsp-120h] [rbp-120h]
  __int64 v166; // [rsp-120h] [rbp-120h]
  __int64 v167[2]; // [rsp-118h] [rbp-118h] BYREF
  _BYTE *v168; // [rsp-108h] [rbp-108h] BYREF
  _BYTE *v169; // [rsp-100h] [rbp-100h]
  _BYTE *v170; // [rsp-F8h] [rbp-F8h]
  __m128i v171; // [rsp-E8h] [rbp-E8h] BYREF
  __int64 v172; // [rsp-D8h] [rbp-D8h]
  __m128i v173; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v174; // [rsp-B8h] [rbp-B8h]
  __m128 v175; // [rsp-A8h] [rbp-A8h] BYREF
  __int64 v176; // [rsp-98h] [rbp-98h]
  __m128 v177; // [rsp-88h] [rbp-88h] BYREF
  _QWORD v178[15]; // [rsp-78h] [rbp-78h] BYREF

  if ( *(_DWORD *)(*(_QWORD *)a1 + 8LL) >> 8 != 3 )
    return 0;
  v10 = a1;
  if ( (*(_BYTE *)(a1 + 23) & 0x10) != 0 )
    return 0;
  for ( i = *(_QWORD *)(a1 + 8); i; i = *(_QWORD *)(i + 8) )
  {
    v15 = sub_1648700(i);
    v16 = *((_BYTE *)v15 + 16);
    if ( v16 > 0x17u )
    {
      if ( v16 != 56 )
        return 0;
    }
    else if ( v16 != 5 || *((_WORD *)v15 + 9) != 32 )
    {
      return 0;
    }
    if ( !(unsigned __int8)sub_185B5B0((__int64)v15, a2, v13, v14) )
      return 0;
  }
  v18 = *(_DWORD *)(v10 + 32);
  v157 = *(__int64 **)(v10 - 24);
  v19 = *v157;
  v168 = 0;
  v165 = v19;
  v20 = *(_QWORD *)(v10 + 40);
  v21 = (unsigned int)(1 << (v18 >> 15)) >> 1;
  v169 = 0;
  v170 = 0;
  v147 = v20;
  if ( !v21 )
    v21 = sub_15A9FE0(a2, *(_QWORD *)v10);
  v22 = v165;
  v23 = *(_BYTE *)(v165 + 8);
  if ( v23 != 13 )
  {
    if ( ((v23 - 14) & 0xFD) != 0 )
      goto LABEL_40;
    v114 = *(_QWORD *)(v165 + 32);
    v156 = v114;
    if ( (unsigned int)v114 <= 0x10 )
    {
      sub_185D6E0((__int64)&v168, (unsigned int)v114);
      v137 = *(_QWORD *)(v165 + 24);
      v146 = sub_12BE0A0(a2, v137);
      v22 = v137;
      v144 = sub_15A9FE0(a2, v137);
      v154 = 8 * sub_12BE0A0(a2, v137);
      if ( !(_DWORD)v114 )
        goto LABEL_40;
    }
    else
    {
      if ( (unsigned __int8)sub_1648D00(v10, 16) )
        goto LABEL_18;
      sub_185D6E0((__int64)&v168, (unsigned int)v114);
      v115 = *(_QWORD *)(v165 + 24);
      v146 = sub_12BE0A0(a2, v115);
      v144 = sub_15A9FE0(a2, v115);
      v154 = 8 * sub_12BE0A0(a2, v115);
    }
    v143 = v21;
    v141 = v147 + 8;
    v116 = 0;
    do
    {
      v164 = sub_15A0A60((__int64)v157, v116);
      v133 = *(_QWORD *)(v165 + 24);
      LOWORD(v176) = 265;
      v175.m128_i32[0] = v116;
      v160 = v133;
      v171.m128i_i64[0] = (__int64)sub_1649960(v10);
      v171.m128i_i64[1] = v134;
      v173.m128i_i64[0] = (__int64)&v171;
      LOWORD(v174) = 773;
      v173.m128i_i64[1] = (__int64)".";
      v118 = v176;
      if ( (_BYTE)v176 )
      {
        if ( (_BYTE)v176 == 1 )
        {
          a4 = _mm_loadu_si128(&v173);
          v177 = (__m128)a4;
          v178[0] = v174;
        }
        else
        {
          v117 = (__m128 *)v175.m128_u64[0];
          if ( BYTE1(v176) != 1 )
          {
            v117 = &v175;
            v118 = 2;
          }
          v177.m128_u64[1] = (unsigned __int64)v117;
          LOBYTE(v178[0]) = 2;
          v177.m128_u64[0] = (unsigned __int64)&v173;
          BYTE1(v178[0]) = v118;
        }
      }
      else
      {
        LOWORD(v178[0]) = 256;
      }
      v119 = *(_DWORD *)(*(_QWORD *)v10 + 8LL);
      v120 = *(_BYTE *)(v10 + 33) >> 2;
      v121 = sub_1648A60(88, 1u);
      v122 = v120 & 7;
      v123 = v119 >> 8;
      v124 = (__int64)v121;
      if ( v121 )
      {
        v125 = v164;
        v163 = v121;
        sub_15E5070((__int64)v121, v160, 0, 7, v125, (__int64)&v177, v122, v123, 0);
        v124 = (__int64)v163;
      }
      v167[0] = v124;
      *(_BYTE *)(v124 + 80) = *(_BYTE *)(v10 + 80) & 2 | *(_BYTE *)(v124 + 80) & 0xFD;
      sub_15E6480(v124, v10);
      v126 = v167[0];
      sub_1631BE0(v141, v167[0]);
      v127 = *(_QWORD *)(v147 + 8);
      v128 = *(_QWORD *)(v126 + 56);
      *(_QWORD *)(v126 + 64) = v141;
      v127 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v126 + 56) = v127 | v128 & 7;
      *(_QWORD *)(v127 + 8) = v126 + 56;
      *(_QWORD *)(v147 + 8) = *(_QWORD *)(v147 + 8) & 7LL | (v126 + 56);
      v129 = v169;
      if ( v169 == v170 )
      {
        sub_17C7180((__int64)&v168, v169, v167);
        v130 = v167[0];
      }
      else
      {
        v130 = v167[0];
        if ( v169 )
        {
          *(_QWORD *)v169 = v167[0];
          v129 = v169;
          v130 = v167[0];
        }
        v169 = v129 + 8;
      }
      v131 = v116;
      v132 = -(__int64)(v143 | (v116 * v146)) & (v143 | (v116 * v146));
      if ( (unsigned int)v132 > v144 )
      {
        sub_15E4CC0(v130, v132);
        v130 = v167[0];
      }
      v22 = v130;
      ++v116;
      sub_185ADB0(v10, v130, v154 * v131, v154, v156);
    }
    while ( v156 != v116 );
LABEL_40:
    v24 = v168;
    result = 0;
    if ( v169 != v168 )
    {
      v48 = (_QWORD *)sub_16498A0(v10);
      v49 = (__int64 **)sub_1643350(v48);
      v52 = sub_15A06D0(v49, v22, v50, v51);
      v53 = *(_QWORD *)(v10 + 8);
      v158 = v52;
      if ( v53 )
      {
        v159 = v10;
        do
        {
          v54 = sub_1648700(v53);
          v58 = (__int64)v54;
          v59 = *((_DWORD *)v54 + 5) & 0xFFFFFFF;
          if ( (*((_BYTE *)v54 + 23) & 0x40) != 0 )
            v60 = (_QWORD *)*(v54 - 1);
          else
            v60 = &v54[-3 * v59];
          v61 = v60[6];
          v62 = *(_QWORD **)(v61 + 24);
          if ( *(_DWORD *)(v61 + 32) > 0x40u )
            v62 = (_QWORD *)*v62;
          v63 = (_QWORD **)v168;
          v64 = (int)v62;
          v65 = 0;
          if ( (unsigned int)v62 < (unsigned __int64)((v169 - v168) >> 3) )
          {
            v65 = (int)v62;
            v63 = (_QWORD **)&v168[8 * (unsigned int)v62];
          }
          v66 = *v63;
          if ( v59 > 3 )
          {
            v162 = v66[3];
            if ( *(_BYTE *)(v58 + 16) == 5 )
            {
              v177.m128_u64[0] = (unsigned __int64)v178;
              v178[0] = v158;
              v177.m128_u64[1] = 0x800000001LL;
              v67 = *(_DWORD *)(v58 + 20) & 0xFFFFFFF;
              if ( v67 == 3 )
              {
                v72 = (__int64 **)v178;
                v68 = 1;
              }
              else
              {
                v68 = 1;
                v69 = 3;
                v70 = *(_QWORD *)(v58 + 24 * (3LL - v67));
                for ( j = v178; ; j = (_QWORD *)v177.m128_u64[0] )
                {
                  j[v68] = v70;
                  ++v69;
                  v68 = (unsigned int)++v177.m128_i32[2];
                  if ( v69 == v67 )
                    break;
                  v70 = *(_QWORD *)(v58 + 24 * (v69 - (unsigned __int64)(*(_DWORD *)(v58 + 20) & 0xFFFFFFF)));
                  if ( v177.m128_i32[3] <= (unsigned int)v68 )
                  {
                    sub_16CD150((__int64)&v177, v178, 0, 8, v64, v55);
                    v68 = v177.m128_u32[2];
                  }
                }
                v72 = (__int64 **)v177.m128_u64[0];
              }
              v175.m128_i8[4] = 0;
              v66 = (_QWORD *)sub_15A2E80(v162, (__int64)v66, v72, v68, 0, (__int64)&v175, 0);
              if ( (_QWORD *)v177.m128_u64[0] != v178 )
                _libc_free(v177.m128_u64[0]);
            }
            else
            {
              v177.m128_u64[0] = (unsigned __int64)v178;
              v178[0] = v158;
              v177.m128_u64[1] = 0x800000001LL;
              v91 = *(_DWORD *)(v58 + 20) & 0xFFFFFFF;
              if ( v91 != 3 )
              {
                v92 = 3;
                v93 = v178;
                v94 = 1;
                v95 = v65;
                v96 = *(_QWORD *)(v58 + 24 * (3LL - v91));
                while ( 1 )
                {
                  v93[v94] = v96;
                  ++v92;
                  v94 = (unsigned int)++v177.m128_i32[2];
                  if ( v92 == v91 )
                    break;
                  v96 = *(_QWORD *)(v58 + 24 * (v92 - (unsigned __int64)(*(_DWORD *)(v58 + 20) & 0xFFFFFFF)));
                  if ( v177.m128_i32[3] <= (unsigned int)v94 )
                  {
                    v152 = *(_QWORD *)(v58 + 24 * (v92 - (unsigned __int64)(*(_DWORD *)(v58 + 20) & 0xFFFFFFF)));
                    sub_16CD150((__int64)&v177, v178, 0, 8, v64, v55);
                    v94 = v177.m128_u32[2];
                    v96 = v152;
                  }
                  v93 = (_QWORD *)v177.m128_u64[0];
                }
                v65 = v95;
              }
              v173.m128i_i32[0] = v65;
              LOWORD(v174) = 265;
              v167[0] = (__int64)sub_1649960(v58);
              v171.m128i_i64[0] = (__int64)v167;
              v171.m128i_i64[1] = (__int64)".";
              v97 = v174;
              v167[1] = v98;
              LOWORD(v172) = 773;
              if ( (_BYTE)v174 )
              {
                if ( (_BYTE)v174 == 1 )
                {
                  a5 = _mm_loadu_si128(&v171);
                  v175 = (__m128)a5;
                  v176 = v172;
                }
                else
                {
                  v99 = (__m128i *)v173.m128i_i64[0];
                  if ( BYTE1(v174) != 1 )
                  {
                    v99 = &v173;
                    v97 = 2;
                  }
                  v175.m128_u64[1] = (unsigned __int64)v99;
                  v175.m128_u64[0] = (unsigned __int64)&v171;
                  LOBYTE(v176) = 2;
                  BYTE1(v176) = v97;
                }
              }
              else
              {
                LOWORD(v176) = 256;
              }
              v100 = v177.m128_u32[2];
              v101 = (__int64 *)v177.m128_u64[0];
              if ( !v162 )
              {
                v135 = *v66;
                if ( *(_BYTE *)(*v66 + 8LL) == 16 )
                  v135 = **(_QWORD **)(v135 + 16);
                v162 = *(_QWORD *)(v135 + 24);
              }
              v153 = v177.m128_i32[2] + 1;
              v102 = sub_1648A60(72, v177.m128_i32[2] + 1);
              v103 = (__int64)v102;
              if ( v102 )
              {
                v104 = (__int64)&v102[-3 * v153];
                v105 = *v66;
                if ( *(_BYTE *)(*v66 + 8LL) == 16 )
                  v105 = **(_QWORD **)(v105 + 16);
                v148 = *(_DWORD *)(v105 + 8) >> 8;
                v106 = (__int64 *)sub_15F9F50(v162, (__int64)v101, v100);
                v107 = (__int64 *)sub_1646BA0(v106, v148);
                v108 = v153;
                v109 = v107;
                if ( *(_BYTE *)(*v66 + 8LL) == 16 )
                {
                  v136 = sub_16463B0(v107, *(_QWORD *)(*v66 + 32LL));
                  v108 = v153;
                  v109 = v136;
                }
                else
                {
                  v110 = &v101[v100];
                  if ( v101 != v110 )
                  {
                    v111 = v101;
                    while ( 1 )
                    {
                      v112 = *(_QWORD *)*v111;
                      if ( *(_BYTE *)(v112 + 8) == 16 )
                        break;
                      if ( v110 == ++v111 )
                        goto LABEL_112;
                    }
                    v113 = sub_16463B0(v109, *(_QWORD *)(v112 + 32));
                    v108 = v153;
                    v109 = v113;
                  }
                }
LABEL_112:
                sub_15F1EA0(v103, (__int64)v109, 32, v104, v108, v58);
                *(_QWORD *)(v103 + 56) = v162;
                *(_QWORD *)(v103 + 64) = sub_15F9F50(v162, (__int64)v101, v100);
                sub_15F9CE0(v103, (__int64)v66, v101, v100, (__int64)&v175);
              }
              if ( (_QWORD *)v177.m128_u64[0] != v178 )
                _libc_free(v177.m128_u64[0]);
              v66 = (_QWORD *)v103;
            }
          }
          sub_164D160(v58, (__int64)v66, a3, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6, v56, v57, a9, a10);
          if ( *(_BYTE *)(v58 + 16) == 56 )
            sub_15F20C0((_QWORD *)v58);
          else
            sub_159D850(v58);
          v53 = *(_QWORD *)(v159 + 8);
        }
        while ( v53 );
        v10 = v159;
      }
      sub_1631C10(v147 + 8, v10);
      v73 = *(unsigned __int64 **)(v10 + 64);
      v74 = *(_QWORD *)(v10 + 56) & 0xFFFFFFFFFFFFFFF8LL;
      *v73 = v74 | *v73 & 7;
      *(_QWORD *)(v74 + 8) = v73;
      *(_QWORD *)(v10 + 56) &= 7uLL;
      *(_QWORD *)(v10 + 64) = 0;
      sub_15E5530(v10);
      sub_159D9E0(v10);
      sub_164BE60(v10, a3, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6, v75, v76, a9, a10);
      *(_DWORD *)(v10 + 20) = *(_DWORD *)(v10 + 20) & 0xF0000000 | 1;
      sub_1648B90(v10);
      v24 = v168;
      v77 = (v169 - v168) >> 3;
      v78 = v77;
      if ( (_DWORD)v77 )
      {
        v79 = 0;
        v80 = 0;
        do
        {
          while ( 1 )
          {
            v81 = *(_QWORD *)&v24[8 * v79];
            if ( !*(_QWORD *)(v81 + 8) )
              break;
            if ( (unsigned int)v77 == ++v79 )
              goto LABEL_71;
          }
          sub_1631C10(v147 + 8, *(_QWORD *)&v24[8 * v79]);
          v82 = *(unsigned __int64 **)(v81 + 64);
          v83 = *(_QWORD *)(v81 + 56) & 0xFFFFFFFFFFFFFFF8LL;
          *v82 = v83 | *v82 & 7;
          *(_QWORD *)(v83 + 8) = v82;
          *(_QWORD *)(v81 + 56) &= 7uLL;
          *(_QWORD *)(v81 + 64) = 0;
          sub_15E5530(v81);
          sub_159D9E0(v81);
          sub_164BE60(v81, a3, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6, v84, v85, a9, a10);
          *(_DWORD *)(v81 + 20) = *(_DWORD *)(v81 + 20) & 0xF0000000 | 1;
          sub_1648B90(v81);
          if ( (_DWORD)v80 == (_DWORD)v79 )
            v80 = (unsigned int)(v80 + 1);
          ++v79;
          v24 = v168;
        }
        while ( (unsigned int)v77 != v79 );
LABEL_71:
        v78 = (v169 - v24) >> 3;
      }
      else
      {
        v80 = 0;
      }
      result = 0;
      if ( v78 != v80 )
        result = *(_QWORD *)&v24[8 * v80];
    }
    goto LABEL_19;
  }
  if ( v21 <= (unsigned int)sub_15A9FE0(a2, v165) )
  {
    v145 = *(_DWORD *)(v165 + 12);
    sub_185D6E0((__int64)&v168, v145);
    v22 = v165;
    v155 = sub_15A9930(a2, v165);
    if ( v145 )
    {
      v161 = a2;
      v25 = 0;
      v142 = v147 + 8;
      v140 = v21;
LABEL_23:
      v26 = sub_15A0A60((__int64)v157, v25);
      v27 = *(_QWORD *)(*(_QWORD *)(v165 + 16) + 8 * v25);
      LOWORD(v176) = 265;
      v175.m128_i32[0] = v25;
      v149 = v27;
      v171.m128i_i64[0] = (__int64)sub_1649960(v10);
      LOWORD(v174) = 773;
      v171.m128i_i64[1] = v28;
      v173.m128i_i64[0] = (__int64)&v171;
      v173.m128i_i64[1] = (__int64)".";
      v29 = v176;
      if ( (_BYTE)v176 )
      {
        if ( (_BYTE)v176 == 1 )
        {
          a3 = (__m128)_mm_loadu_si128(&v173);
          v177 = a3;
          v178[0] = v174;
        }
        else
        {
          v30 = (__m128 *)v175.m128_u64[0];
          if ( BYTE1(v176) != 1 )
          {
            v30 = &v175;
            v29 = 2;
          }
          v177.m128_u64[1] = (unsigned __int64)v30;
          v177.m128_u64[0] = (unsigned __int64)&v173;
          LOBYTE(v178[0]) = 2;
          BYTE1(v178[0]) = v29;
        }
      }
      else
      {
        LOWORD(v178[0]) = 256;
      }
      v31 = *(_DWORD *)(*(_QWORD *)v10 + 8LL);
      v32 = *(_BYTE *)(v10 + 33) >> 2;
      v33 = sub_1648A60(88, 1u);
      v34 = v32 & 7;
      v35 = v31 >> 8;
      v36 = (__int64)v33;
      if ( v33 )
      {
        v37 = v149;
        v150 = v33;
        sub_15E5070((__int64)v33, v37, 0, 7, v26, (__int64)&v177, v34, v35, 0);
        v36 = (__int64)v150;
      }
      v167[0] = v36;
      *(_BYTE *)(v36 + 80) = *(_BYTE *)(v10 + 80) & 2 | *(_BYTE *)(v36 + 80) & 0xFD;
      sub_15E6480(v36, v10);
      v38 = v167[0];
      sub_1631BE0(v142, v167[0]);
      v39 = *(_QWORD *)(v147 + 8);
      *(_QWORD *)(v38 + 64) = v142;
      *(_QWORD *)(v38 + 56) = v39 & 0xFFFFFFFFFFFFFFF8LL | *(_QWORD *)(v38 + 56) & 7LL;
      *(_QWORD *)((v39 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v38 + 56;
      *(_QWORD *)(v147 + 8) = *(_QWORD *)(v147 + 8) & 7LL | (v38 + 56);
      v40 = v169;
      if ( v169 == v170 )
      {
        sub_17C7180((__int64)&v168, v169, v167);
      }
      else
      {
        if ( v169 )
        {
          *(_QWORD *)v169 = v167[0];
          v40 = v169;
        }
        v169 = v40 + 8;
      }
      v41 = (*(_QWORD *)(v155 + 8 * v25 + 16) | v140) & -(*(_QWORD *)(v155 + 8 * v25 + 16) | v140);
      if ( (unsigned int)sub_15A9FE0(v161, *(_QWORD *)(*(_QWORD *)(v165 + 16) + 8 * v25)) >= (unsigned int)v41 )
      {
        v89 = v167[0];
        v90 = sub_15A9FE0(v161, *(_QWORD *)(*(_QWORD *)(v165 + 16) + 8 * v25));
        sub_15E4CC0(v89, v90);
      }
      else
      {
        sub_15E4CC0(v167[0], v41);
      }
      v42 = 1;
      v43 = *(_QWORD *)(v167[0] + 24);
      v44 = (unsigned int)sub_15A9FE0(v161, v43);
      while ( 2 )
      {
        switch ( *(_BYTE *)(v43 + 8) )
        {
          case 0:
          case 8:
          case 0xA:
          case 0xC:
          case 0x10:
            v88 = *(_QWORD *)(v43 + 32);
            v43 = *(_QWORD *)(v43 + 24);
            v42 *= v88;
            continue;
          case 1:
            v45 = 16;
            goto LABEL_39;
          case 2:
            v45 = 32;
            goto LABEL_39;
          case 3:
          case 9:
            v45 = 64;
            goto LABEL_39;
          case 4:
            v45 = 80;
            goto LABEL_39;
          case 5:
          case 6:
            v45 = 128;
            goto LABEL_39;
          case 7:
            v45 = 8 * (unsigned int)sub_15A9520(v161, 0);
            goto LABEL_39;
          case 0xB:
            v45 = *(_DWORD *)(v43 + 8) >> 8;
            goto LABEL_39;
          case 0xD:
            v45 = 8LL * *(_QWORD *)sub_15A9930(v161, v43);
            goto LABEL_39;
          case 0xE:
            v151 = *(_QWORD *)(v43 + 24);
            sub_15A9FE0(v161, v151);
            v86 = v151;
            v87 = 1;
            while ( 2 )
            {
              switch ( *(_BYTE *)(v86 + 8) )
              {
                case 0:
                case 8:
                case 0xA:
                case 0xC:
                case 0x10:
                  v139 = *(_QWORD *)(v86 + 32);
                  v86 = *(_QWORD *)(v86 + 24);
                  v87 *= v139;
                  continue;
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 9:
                case 0xB:
                  goto LABEL_147;
                case 7:
                  sub_15A9520(v161, 0);
                  goto LABEL_147;
                case 0xD:
                  JUMPOUT(0x185FB07);
                case 0xE:
                  v138 = *(_QWORD *)(v86 + 24);
                  sub_15A9FE0(v161, v138);
                  sub_127FA20(v161, v138);
                  JUMPOUT(0x185FAEB);
                case 0xF:
                  sub_15A9520(v161, *(_DWORD *)(v86 + 8) >> 8);
LABEL_147:
                  JUMPOUT(0x185FA0E);
              }
            }
          case 0xF:
            v45 = 8 * (unsigned int)sub_15A9520(v161, *(_DWORD *)(v43 + 8) >> 8);
LABEL_39:
            v22 = v167[0];
            v46 = 8 * v44 * ((v44 + ((unsigned __int64)(v45 * v42 + 7) >> 3) - 1) / v44);
            v47 = *(_QWORD *)(v155 + 8 * v25++ + 16);
            sub_185ADB0(v10, v167[0], 8 * v47, v46, v145);
            if ( v145 == v25 )
              goto LABEL_40;
            goto LABEL_23;
        }
      }
    }
    goto LABEL_40;
  }
LABEL_18:
  v24 = v168;
  result = 0;
LABEL_19:
  if ( v24 )
  {
    v166 = result;
    j_j___libc_free_0(v24, v170 - v24);
    return v166;
  }
  return result;
}
