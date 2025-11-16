// Function: sub_29287A0
// Address: 0x29287a0
//
__int64 __fastcall sub_29287A0(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // r13
  __m128i v7; // rax
  unsigned __int64 v8; // rcx
  unsigned __int64 v9; // rsi
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // r12
  unsigned __int8 *v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rdx
  _DWORD *v19; // rax
  unsigned __int8 *v20; // rsi
  _QWORD *v21; // rdi
  __int16 v22; // ax
  __int64 v23; // r13
  unsigned __int64 v24; // rdx
  __int64 v25; // r8
  __int64 v26; // r10
  __int64 v27; // rsi
  __int64 v28; // rdi
  unsigned __int64 v29; // r9
  int v30; // r13d
  __int64 *v31; // rdi
  __int64 v32; // rdx
  __int64 v33; // rdx
  __int64 v34; // r12
  __int64 v35; // rdi
  int v36; // eax
  bool v37; // cl
  __int64 v38; // rsi
  __int64 v39; // rdi
  int v40; // esi
  __int64 **v41; // rax
  unsigned __int64 v42; // rdx
  __int64 v43; // rdx
  __int64 v44; // r9
  unsigned __int64 v45; // rax
  unsigned __int16 v46; // cx
  unsigned __int64 v47; // rax
  unsigned __int64 v48; // rax
  _BYTE *v49; // rdi
  __int64 v50; // r8
  __int64 v51; // r12
  int v52; // eax
  bool v53; // r8
  __int16 v54; // bx
  __int64 v55; // r9
  unsigned __int64 v56; // rax
  unsigned __int16 v57; // cx
  __int64 v58; // rax
  __int64 v59; // r9
  __int64 v60; // rdi
  unsigned int v61; // ebx
  bool v62; // r8
  unsigned __int16 v63; // cx
  __int64 v64; // rbx
  unsigned __int8 *v65; // rax
  __int64 v66; // rdi
  unsigned int v67; // ebx
  __int64 v69; // rax
  unsigned __int64 v70; // r13
  unsigned __int64 v71; // rax
  __m128i v72; // rax
  __int64 v73; // rbx
  __int64 v74; // r12
  __int64 v75; // rsi
  unsigned __int8 v76; // dl
  unsigned int v77; // eax
  unsigned int v78; // ecx
  __int64 v79; // rax
  __int64 v80; // rdi
  __int64 v81; // r12
  unsigned __int64 v82; // r13
  __int64 v83; // rax
  __int64 v84; // r8
  unsigned __int8 *v85; // rax
  const void *v86; // r12
  __int64 v87; // rax
  __int64 v88; // rax
  unsigned __int8 *v89; // r13
  __int64 *v90; // rax
  __int64 *v91; // rax
  __int64 v92; // rax
  __int64 v93; // rbx
  __int64 *v94; // rax
  unsigned __int8 *v95; // rbx
  __int64 v96; // rdx
  __int64 v97; // rcx
  __int64 v98; // r8
  __int64 v99; // r9
  __int64 v100; // r9
  unsigned __int64 v101; // rax
  unsigned __int16 v102; // cx
  __int64 v103; // rax
  __int64 v104; // r9
  __int64 v105; // rdx
  unsigned __int64 v106; // rax
  unsigned __int16 v107; // cx
  unsigned __int64 v108; // rax
  unsigned __int64 v109; // rax
  __int64 v110; // rdi
  __int64 v111; // r8
  unsigned __int64 v112; // rax
  __int64 v113; // rsi
  __int64 v114; // rdi
  __int64 v115; // rax
  __int64 v116; // rdx
  __int64 v117; // rbx
  __int64 v118; // r12
  unsigned __int8 *v119; // rax
  unsigned __int8 *v120; // r15
  __int64 v121; // rax
  __int64 v122; // rdx
  __m128i v123; // kr00_16
  __int64 *v124; // rax
  unsigned __int8 *v125; // r15
  char v126; // al
  __int64 v127; // rax
  __int64 v128; // rax
  __m128i *v129; // rsi
  __int64 *v130; // rbx
  __int64 v131; // r12
  __int64 *v132; // rax
  __int64 *v133; // rax
  __int64 v134; // rax
  __int64 v135; // rbx
  __int64 *v136; // rax
  int v137; // eax
  bool v138; // cl
  unsigned __int64 v139; // rax
  __int64 v140; // rsi
  __int64 v141; // rcx
  int v142; // esi
  __int64 **v143; // rax
  unsigned __int64 v144; // rdx
  char v145; // si
  __int64 v146; // rdx
  __int64 v147; // r12
  __int64 v148; // r12
  char v149; // di
  __int64 v150; // rax
  __int64 v151; // rdx
  __int64 v152; // r13
  __int64 v153; // rdx
  unsigned __int64 v154; // r12
  __int64 v155; // rax
  __int64 v156; // [rsp+0h] [rbp-130h]
  unsigned int v157; // [rsp+8h] [rbp-128h]
  __int64 v158; // [rsp+8h] [rbp-128h]
  __int64 v159; // [rsp+10h] [rbp-120h]
  _QWORD **v160; // [rsp+20h] [rbp-110h]
  int v161; // [rsp+20h] [rbp-110h]
  int v162; // [rsp+20h] [rbp-110h]
  __int64 v163; // [rsp+28h] [rbp-108h]
  int v164; // [rsp+28h] [rbp-108h]
  int v165; // [rsp+28h] [rbp-108h]
  unsigned __int8 v166; // [rsp+38h] [rbp-F8h]
  __int64 v167; // [rsp+40h] [rbp-F0h]
  __int64 *v168; // [rsp+40h] [rbp-F0h]
  __int64 v169; // [rsp+40h] [rbp-F0h]
  unsigned int v170; // [rsp+48h] [rbp-E8h]
  char v171; // [rsp+4Fh] [rbp-E1h]
  char v172; // [rsp+50h] [rbp-E0h]
  __int64 v173; // [rsp+50h] [rbp-E0h]
  __int64 v174; // [rsp+58h] [rbp-D8h]
  unsigned __int8 *v175; // [rsp+58h] [rbp-D8h]
  __int64 v176; // [rsp+60h] [rbp-D0h]
  __int64 v177; // [rsp+60h] [rbp-D0h]
  __int64 *v178; // [rsp+60h] [rbp-D0h]
  unsigned __int8 v179; // [rsp+68h] [rbp-C8h]
  unsigned __int8 *v180; // [rsp+80h] [rbp-B0h] BYREF
  unsigned int v181; // [rsp+88h] [rbp-A8h]
  const void *v182; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v183; // [rsp+98h] [rbp-98h]
  __int64 v184; // [rsp+A0h] [rbp-90h] BYREF
  __int64 v185; // [rsp+A8h] [rbp-88h]
  __int64 v186; // [rsp+B0h] [rbp-80h]
  __int64 v187; // [rsp+B8h] [rbp-78h]
  __m128i v188; // [rsp+C0h] [rbp-70h] BYREF
  _QWORD v189[2]; // [rsp+D0h] [rbp-60h] BYREF
  __int16 v190; // [rsp+E0h] [rbp-50h]

  v2 = a1;
  sub_B91FC0(&v184, a2);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v174 = *(_QWORD *)(a2 - 8);
  else
    v174 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v6 = *(_QWORD *)(a1 + 32);
  v7.m128i_i64[1] = *(_QWORD *)(a1 + 40);
  v171 = -1;
  v176 = *(_QWORD *)(a1 + 144);
  _BitScanReverse64(&v8, 1LL << *(_WORD *)(v6 + 2));
  v9 = 0x8000000000000000LL >> ((unsigned __int8)v8 ^ 0x3Fu);
  v10 = -(__int64)(v9 | (*(_QWORD *)(a1 + 112) - v7.m128i_i64[1]));
  if ( (v10 & (v9 | (*(_QWORD *)(a1 + 112) - v7.m128i_i64[1]))) != 0 )
  {
    _BitScanReverse64((unsigned __int64 *)&v7, v10 & (v9 | (*(_QWORD *)(a1 + 112) - v7.m128i_i64[1])));
    v10 = 63;
    v171 = 63 - (v7.m128i_i8[0] ^ 0x3F);
  }
  v172 = *(_BYTE *)(a1 + 136);
  if ( !v172 )
  {
    v89 = (unsigned __int8 *)sub_291C360((__int64 *)a1, a1 + 176, *(_QWORD *)(*(_QWORD *)(a1 + 152) + 8LL));
    v168 = (__int64 *)(a2 + 72);
    if ( v176 == v174 )
    {
      if ( (*(_BYTE *)(a2 + 7) & 0x20) != 0 )
      {
        v113 = 38;
        v114 = sub_B91C10(a2, 38);
        if ( !v114 || (v115 = sub_AE94B0(v114), v177 = v116, v117 = v115, v116 == v115) )
        {
          v126 = *(_BYTE *)(a2 + 7) & 0x20;
        }
        else
        {
          v173 = v2;
          do
          {
            v118 = *(_QWORD *)(v117 + 24);
            v119 = sub_BD3990(*(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), v113);
            v113 = v118;
            v120 = v119;
            sub_B58E30(&v188, v118);
            v121 = v188.m128i_i64[1];
            v123 = v188;
            v180 = (unsigned __int8 *)(*(_OWORD *)&v123 >> 64);
            v122 = (__int64)v123;
            v182 = (const void *)v188.m128i_i64[0];
            if ( v188.m128i_i64[1] != v188.m128i_i64[0] )
            {
              while ( 1 )
              {
                while ( 1 )
                {
                  v124 = (__int64 *)(v122 & 0xFFFFFFFFFFFFFFF8LL);
                  if ( (v122 & 4) == 0 )
                    break;
                  v113 = *v124;
                  if ( v120 == *(unsigned __int8 **)(*v124 + 136) )
                    goto LABEL_197;
                  v122 = (unsigned __int64)(v124 + 1) | 4;
                  v121 = v122;
                  if ( v188.m128i_i64[1] == v122 )
                    goto LABEL_143;
                }
                if ( v120 == (unsigned __int8 *)v124[17] )
                  break;
                v121 = (__int64)(v124 + 18);
                v122 = v121;
                if ( v188.m128i_i64[1] == v121 )
                  goto LABEL_143;
              }
LABEL_197:
              v121 = v122;
            }
LABEL_143:
            if ( v121 != v188.m128i_i64[1]
              || (v125 = (unsigned __int8 *)sub_B595C0(v118),
                  v125 == sub_BD3990(*(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), v113)) )
            {
              v113 = (__int64)sub_BD3990(*(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), v113);
              sub_B59720(v118, v113, v89);
            }
            v117 = *(_QWORD *)(v117 + 8);
          }
          while ( v117 != v177 );
          v2 = v173;
          v126 = *(_BYTE *)(a2 + 7) & 0x20;
        }
        if ( v126 )
        {
          v127 = sub_B91C10(a2, 38);
          if ( v127 )
          {
            v128 = *(_QWORD *)(v127 + 8);
            v129 = (__m128i *)(v128 & 0xFFFFFFFFFFFFFFF8LL);
            if ( (v128 & 4) == 0 )
              v129 = 0;
            sub_B967C0(&v188, v129);
            v130 = (__int64 *)v188.m128i_i64[0];
            v178 = (__int64 *)(v188.m128i_i64[0] + 8LL * v188.m128i_u32[2]);
            if ( (__int64 *)v188.m128i_i64[0] != v178 )
            {
              do
              {
                v131 = *v130;
                v180 = sub_BD3990(*(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), (__int64)v129);
                sub_B129C0(&v182, v131);
                v129 = (__m128i *)&v180;
                if ( sub_F9EA20((__int64 *)&v182, (__int64 *)&v180)
                  || (v175 = sub_B13320(v131),
                      v175 == sub_BD3990(
                                *(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)),
                                (__int64)&v180)) )
                {
                  v129 = (__m128i *)sub_BD3990(
                                      *(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)),
                                      (__int64)&v180);
                  sub_B13360(v131, (unsigned __int8 *)v129, v89, 0);
                }
                ++v130;
              }
              while ( v178 != v130 );
              v178 = (__int64 *)v188.m128i_i64[0];
            }
            if ( v178 != v189 )
              _libc_free((unsigned __int64)v178);
          }
        }
      }
      sub_AC2B30(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF), (__int64)v89);
      v132 = (__int64 *)sub_BD5C60(a2);
      *(_QWORD *)(a2 + 72) = sub_A7B980(v168, v132, 1, 86);
      v133 = (__int64 *)sub_BD5C60(a2);
      v134 = sub_A77A40(v133, v171);
      v188.m128i_i32[0] = 0;
      v135 = v134;
      v136 = (__int64 *)sub_BD5C60(a2);
      *(_QWORD *)(a2 + 72) = sub_A7B660(v168, v136, &v188, 1, v135);
    }
    else
    {
      sub_AC2B30(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), (__int64)v89);
      v90 = (__int64 *)sub_BD5C60(a2);
      *(_QWORD *)(a2 + 72) = sub_A7B980(v168, v90, 2, 86);
      v91 = (__int64 *)sub_BD5C60(a2);
      v92 = sub_A77A40(v91, v171);
      v188.m128i_i32[0] = 1;
      v93 = v92;
      v94 = (__int64 *)sub_BD5C60(a2);
      *(_QWORD *)(a2 + 72) = sub_A7B660(v168, v94, &v188, 1, v93);
    }
    v95 = *(unsigned __int8 **)(v2 + 152);
    if ( sub_F50EE0(v95, 0) )
    {
      v147 = *(_QWORD *)(v2 + 16);
      v188 = (__m128i)4uLL;
      v148 = v147 + 216;
      v189[0] = v95;
      LOBYTE(v96) = v95 + 4096 != 0;
      if ( ((v95 != 0) & (unsigned __int8)v96) != 0 && v95 != (unsigned __int8 *)-8192LL )
        sub_BD73F0((__int64)&v188);
      sub_D6B260(v148, v188.m128i_i8, v96, v97, v98, v99);
      if ( v189[0] != -4096 && v189[0] != 0 && v189[0] != -8192 )
        sub_BD60C0(&v188);
    }
    goto LABEL_125;
  }
  if ( *(_QWORD *)(a1 + 72) || *(_QWORD *)(a1 + 64) )
    goto LABEL_7;
  if ( *(_QWORD *)(a1 + 96) <= v7.m128i_i64[1] && *(_QWORD *)(a1 + 104) >= *(_QWORD *)(a1 + 48) )
  {
    v86 = *(const void **)(a1 + 128);
    v182 = (const void *)sub_9C6480(*(_QWORD *)a1, *(_QWORD *)(v6 + 72));
    v183 = v7.m128i_i64[1];
    if ( v86 != v182 )
    {
      v6 = *(_QWORD *)(a1 + 32);
      goto LABEL_116;
    }
    v152 = *(_QWORD *)a1;
    v169 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 72LL);
    v188.m128i_i64[0] = sub_9208B0(*(_QWORD *)a1, v169);
    v188.m128i_i64[1] = v153;
    v154 = (v188.m128i_i64[0] + 7) & 0xFFFFFFFFFFFFFFF8LL;
    v179 = v153;
    v7.m128i_i64[0] = sub_9208B0(v152, v169);
    v4 = v179;
    v188 = v7;
    if ( v7.m128i_i64[0] != v154 )
    {
      v155 = *(_QWORD *)(a1 + 32);
LABEL_210:
      v6 = v155;
      goto LABEL_116;
    }
    v6 = *(_QWORD *)(a1 + 32);
    v155 = v6;
    if ( v188.m128i_i8[8] != v179 )
      goto LABEL_210;
    if ( (unsigned __int8)sub_2918CB0(*(_QWORD *)(v6 + 72)) )
    {
LABEL_7:
      v172 = 0;
      goto LABEL_8;
    }
  }
LABEL_116:
  if ( *(_QWORD *)(a1 + 24) == v6 )
  {
    v87 = *(_QWORD *)(a1 + 120);
    if ( v87 != *(_QWORD *)(a1 + 104) )
    {
      v88 = sub_AD64C0(
              *(_QWORD *)(*(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) + 8LL),
              v87 - *(_QWORD *)(a1 + 112),
              0);
      sub_AC2B30(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), v88);
    }
LABEL_125:
    LODWORD(v51) = 0;
    return (unsigned int)v51;
  }
LABEL_8:
  v11 = *(_QWORD *)(a1 + 16);
  v188 = (__m128i)4uLL;
  v12 = v11 + 216;
  v189[0] = a2;
  if ( a2 != -8192 && a2 != -4096 )
    sub_BD73F0((__int64)&v188);
  sub_D6B260(v12, v188.m128i_i8, v7.m128i_i64[1], v10, v4, v5);
  if ( v189[0] != 0 && v189[0] != -4096 && v189[0] != -8192 )
    sub_BD60C0(&v188);
  v13 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  if ( v176 == v174 )
    v14 = 1 - v13;
  else
    v14 = -v13;
  v15 = *(_QWORD *)(a2 + 32 * v14);
  v16 = sub_BD4CB0((unsigned __int8 *)v15, (void (__fastcall *)(__int64, unsigned __int8 *))nullsub_96, (__int64)&v182);
  if ( *v16 == 60 )
  {
    v188.m128i_i64[0] = (__int64)v16;
    sub_2928360(*(_QWORD *)(a1 + 16) + 40LL, v188.m128i_i64);
  }
  v17 = *(_QWORD *)(v15 + 8);
  v163 = v17;
  v18 = v17;
  if ( (unsigned int)*(unsigned __int8 *)(v17 + 8) - 17 <= 1 )
    v18 = **(_QWORD **)(v17 + 16);
  v19 = sub_AE2980(*(_QWORD *)a1, *(_DWORD *)(v18 + 8) >> 8);
  v20 = (unsigned __int8 *)(*(_QWORD *)(a1 + 112) - *(_QWORD *)(a1 + 96));
  v181 = v19[3];
  if ( v181 > 0x40 )
    sub_C43690((__int64)&v180, (__int64)v20, 0);
  else
    v180 = v20;
  v21 = (_QWORD *)(a2 + 72);
  if ( v176 == v174 )
    v22 = sub_A74840(v21, 1);
  else
    v22 = sub_A74840(v21, 0);
  v23 = 1;
  if ( HIBYTE(v22) )
    v23 = 1LL << v22;
  sub_C44AB0((__int64)&v188, (__int64)&v180, 0x40u);
  if ( v188.m128i_i32[2] > 0x40u )
  {
    v24 = (v23 | *(_QWORD *)v188.m128i_i64[0]) & -(v23 | *(_QWORD *)v188.m128i_i64[0]);
    if ( v24 )
    {
      _BitScanReverse64(&v24, v24);
      v166 = 63 - (v24 ^ 0x3F);
    }
    else
    {
      v166 = -1;
    }
    j_j___libc_free_0_0(v188.m128i_u64[0]);
LABEL_30:
    if ( !v172 )
      goto LABEL_31;
LABEL_89:
    v72.m128i_i64[0] = (__int64)sub_BD5D20(v15);
    v190 = 773;
    v188 = v72;
    v189[0] = ".";
    LODWORD(v183) = v181;
    if ( v181 > 0x40 )
      sub_C43780((__int64)&v182, (const void **)&v180);
    else
      v182 = v180;
    v73 = sub_291C070(v2 + 176, v15, (__int64)&v182, v163, v188.m128i_i64);
    if ( (unsigned int)v183 > 0x40 && v182 )
      j_j___libc_free_0_0((unsigned __int64)v182);
    v74 = sub_291C360((__int64 *)v2, v2 + 176, *(_QWORD *)(*(_QWORD *)(v2 + 152) + 8LL));
    v75 = sub_AD64C0(
            *(_QWORD *)(*(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) + 8LL),
            *(_QWORD *)(v2 + 120) - *(_QWORD *)(v2 + 112),
            0);
    if ( v176 == v174 )
    {
      v149 = v171;
      v171 = v166;
      v150 = v73;
      v73 = v74;
      v166 = v149;
      v74 = v150;
    }
    v76 = sub_2919180(a2);
    v77 = 256;
    LOBYTE(v77) = v171;
    v78 = v166;
    BYTE1(v78) = 1;
    v79 = sub_B343C0(v2 + 176, 0xEEu, v73, v78, v74, v77, v75, v76, 0, 0, 0, 0);
    v80 = v184;
    v81 = v79;
    if ( v184 )
    {
      v82 = *(_QWORD *)(v2 + 112) - *(_QWORD *)(v2 + 96);
      v83 = sub_E00740(v184);
      v84 = v185;
      v80 = v83;
    }
    else
    {
      v84 = v185;
      if ( !v185 && !v186 && !v187 )
      {
LABEL_101:
        v188.m128i_i32[2] = sub_AE43F0(*(_QWORD *)v2, *(_QWORD *)(v73 + 8));
        if ( v188.m128i_i32[2] > 0x40u )
          sub_C43690((__int64)&v188, 0, 0);
        else
          v188.m128i_i64[0] = 0;
        if ( v176 == v174 )
        {
          sub_29228E0(
            *(_QWORD *)(v2 + 24),
            *(_BYTE *)(v2 + 137),
            8LL * *(_QWORD *)(v2 + 112),
            8LL * *(_QWORD *)(v2 + 128),
            a2,
            v81,
            v73,
            0);
        }
        else
        {
          v85 = sub_BD45C0((unsigned __int8 *)v73, *(_QWORD *)v2, (__int64)&v188, 1, 0, 0, 0, 0);
          if ( *v85 == 60 )
          {
            v151 = v188.m128i_i64[0];
            if ( v188.m128i_i32[2] > 0x40u )
              v151 = *(_QWORD *)v188.m128i_i64[0];
            sub_29228E0((__int64)v85, *(_BYTE *)(v2 + 137), 8 * v151, 8LL * *(_QWORD *)(v2 + 128), a2, v81, v73, 0);
          }
        }
        if ( v188.m128i_i32[2] > 0x40u && v188.m128i_i64[0] )
          j_j___libc_free_0_0(v188.m128i_u64[0]);
        LODWORD(v51) = 0;
        goto LABEL_77;
      }
      v82 = *(_QWORD *)(v2 + 112) - *(_QWORD *)(v2 + 96);
    }
    v188.m128i_i64[0] = v80;
    if ( v84 )
      v84 = sub_E00750(v84, v82);
    v188.m128i_i64[1] = v84;
    v189[0] = v186;
    v189[1] = v187;
    sub_B9A100(v81, v188.m128i_i64);
    goto LABEL_101;
  }
  v166 = -1;
  v70 = (v23 | v188.m128i_i64[0]) & -(v23 | v188.m128i_i64[0]);
  if ( !v70 )
    goto LABEL_30;
  _BitScanReverse64(&v71, v70);
  v166 = 63 - (v71 ^ 0x3F);
  if ( v172 )
    goto LABEL_89;
LABEL_31:
  v25 = *(_QWORD *)(v2 + 112);
  v26 = *(_QWORD *)(v2 + 40);
  v27 = *(_QWORD *)(v2 + 120);
  if ( v25 == v26 )
    v172 = *(_QWORD *)(v2 + 48) == v27;
  v28 = *(_QWORD *)(v2 + 72);
  if ( v28 )
  {
    v29 = *(_QWORD *)(v2 + 88);
    v170 = (v25 - v26) / v29;
    v157 = (v27 - v26) / v29;
    v30 = v157 - v170;
    v160 = *(_QWORD ***)(v2 + 64);
    if ( !v160 )
      goto LABEL_35;
  }
  else
  {
    v30 = 0;
    v170 = 0;
    v157 = 0;
    v160 = *(_QWORD ***)(v2 + 64);
    if ( !v160 )
      goto LABEL_109;
  }
  v69 = sub_BCD140(*v160, 8 * ((int)v27 - (int)v25));
  v28 = *(_QWORD *)(v2 + 72);
  v160 = (_QWORD **)v69;
  if ( !v28 )
  {
    if ( *(_QWORD *)(v2 + 64) && !v172 )
    {
      v159 = v69;
      goto LABEL_38;
    }
    goto LABEL_109;
  }
LABEL_35:
  if ( v172 )
  {
LABEL_109:
    v159 = *(_QWORD *)(v2 + 56);
    goto LABEL_38;
  }
  v31 = *(__int64 **)(v28 + 24);
  v159 = (__int64)v31;
  if ( v30 != 1 )
    v159 = sub_BCDA70(v31, v30);
LABEL_38:
  v188.m128i_i64[0] = (__int64)sub_BD5D20(v15);
  v189[0] = ".";
  v190 = 773;
  v188.m128i_i64[1] = v32;
  LODWORD(v183) = v181;
  if ( v181 > 0x40 )
    sub_C43780((__int64)&v182, (const void **)&v180);
  else
    v182 = v180;
  v167 = sub_291C070(v2 + 176, v15, (__int64)&v182, v163, v188.m128i_i64);
  if ( (unsigned int)v183 > 0x40 && v182 )
    j_j___libc_free_0_0((unsigned __int64)v182);
  v33 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v34 = *(_QWORD *)(a2 + 32 * (3 - v33));
  v35 = v34 + 24;
  if ( v176 != v174 )
  {
    if ( *(_DWORD *)(v34 + 32) <= 0x40u )
    {
      v37 = *(_QWORD *)(v34 + 24) == 0;
    }
    else
    {
      v156 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
      v164 = *(_DWORD *)(v34 + 32);
      v36 = sub_C444A0(v35);
      v33 = v156;
      v37 = v164 == v36;
    }
    if ( v37 )
      goto LABEL_54;
    v38 = *(_QWORD *)(*(_QWORD *)(v2 + 32) + 8LL);
    v39 = v38;
    if ( (unsigned int)*(unsigned __int8 *)(v38 + 8) - 17 <= 1 )
      v39 = **(_QWORD **)(v38 + 16);
    v40 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32 * (1 - v33)) + 8LL) + 8LL) >> 8;
    if ( v40 == *(_DWORD *)(v39 + 8) >> 8 )
    {
LABEL_54:
      v43 = *(_QWORD *)(v2 + 32);
      if ( !*(_QWORD *)(v2 + 72) )
        goto LABEL_51;
    }
    else
    {
      v41 = (__int64 **)sub_BCE3C0(*(__int64 **)(v2 + 248), v40);
      v42 = *(_QWORD *)(v2 + 32);
      v190 = 257;
      LODWORD(v43) = sub_291AC80((__int64 *)(v2 + 176), 0x32u, v42, v41, (__int64)&v188, 0, (int)v182, 0);
      if ( !*(_QWORD *)(v2 + 72) )
      {
LABEL_51:
        if ( *(_QWORD *)(v2 + 64) && !v172 )
        {
          v44 = *(_QWORD *)(v2 + 32);
          _BitScanReverse64(&v45, 1LL << *(_WORD *)(v44 + 2));
          LOBYTE(v46) = 63 - (v45 ^ 0x3F);
          HIBYTE(v46) = 1;
          v47 = sub_291B4B0((__int64 *)(v2 + 176), *(_QWORD *)(v44 + 72), v44, v46, "load");
          v48 = sub_291C8F0(*(_QWORD *)v2, (unsigned int **)(v2 + 176), v47, *(_QWORD *)(v2 + 64));
          v49 = *(_BYTE **)v2;
          v50 = *(_QWORD *)(v2 + 112) - *(_QWORD *)(v2 + 40);
          v190 = 259;
          v188.m128i_i64[0] = (__int64)"extract";
          v51 = sub_291AEB0(v49, v2 + 176, v48, (__int64)v160, v50, &v188);
LABEL_127:
          if ( *(_QWORD *)(v2 + 64) && v172 != 1 && v176 == v174 )
          {
            v105 = *(_QWORD *)(v2 + 32);
            _BitScanReverse64(&v106, 1LL << *(_WORD *)(v105 + 2));
            LOBYTE(v107) = 63 - (v106 ^ 0x3F);
            HIBYTE(v107) = 1;
            v108 = sub_291B4B0((__int64 *)(v2 + 176), *(_QWORD *)(v105 + 72), v105, v107, "oldload");
            v109 = sub_291C8F0(*(_QWORD *)v2, (unsigned int **)(v2 + 176), v108, *(_QWORD *)(v2 + 64));
            v110 = *(_QWORD *)v2;
            v111 = *(_QWORD *)(v2 + 112) - *(_QWORD *)(v2 + 40);
            v190 = 259;
            v188.m128i_i64[0] = (__int64)"insert";
            v112 = sub_291CC20(v110, v2 + 176, v109, v51, v111, &v188);
            v51 = sub_291C8F0(*(_QWORD *)v2, (unsigned int **)(v2 + 176), v112, *(_QWORD *)(v2 + 56));
          }
          goto LABEL_65;
        }
LABEL_56:
        v34 = *(_QWORD *)(a2 + 32 * (3LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
        v35 = v34 + 24;
        goto LABEL_57;
      }
    }
    if ( !v172 )
    {
      v100 = *(_QWORD *)(v2 + 32);
      _BitScanReverse64(&v101, 1LL << *(_WORD *)(v100 + 2));
      LOBYTE(v102) = 63 - (v101 ^ 0x3F);
      HIBYTE(v102) = 1;
      v103 = sub_291B4B0((__int64 *)(v2 + 176), *(_QWORD *)(v100 + 72), v100, v102, "load");
      v190 = 259;
      v188.m128i_i64[0] = (__int64)"vec";
      v51 = sub_2918880((__int64 *)(v2 + 176), v103, v170, v157, &v188, v104);
      goto LABEL_127;
    }
    goto LABEL_56;
  }
  if ( *(_DWORD *)(v34 + 32) <= 0x40u )
  {
    v138 = *(_QWORD *)(v34 + 24) == 0;
  }
  else
  {
    v158 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
    v162 = *(_DWORD *)(v34 + 32);
    v137 = sub_C444A0(v35);
    v33 = v158;
    v35 = v34 + 24;
    v138 = v162 == v137;
  }
  v139 = *(_QWORD *)(v2 + 32);
  if ( !v138 )
  {
    v140 = *(_QWORD *)(v139 + 8);
    v141 = (unsigned int)*(unsigned __int8 *)(v140 + 8) - 17 > 1 ? *(_QWORD *)(v139 + 8) : **(_QWORD **)(v140 + 16);
    v142 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a2 - 32 * v33) + 8LL) + 8LL) >> 8;
    if ( v142 != *(_DWORD *)(v141 + 8) >> 8 )
    {
      v143 = (__int64 **)sub_BCE3C0(*(__int64 **)(v2 + 248), v142);
      v144 = *(_QWORD *)(v2 + 32);
      v190 = 257;
      v139 = sub_291AC80((__int64 *)(v2 + 176), 0x32u, v144, v143, (__int64)&v188, 0, (int)v182, 0);
      v34 = *(_QWORD *)(a2 + 32 * (3LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
      v35 = v34 + 24;
    }
  }
  v145 = v166;
  v166 = v171;
  LODWORD(v43) = v167;
  v171 = v145;
  v167 = v139;
LABEL_57:
  v188.m128i_i64[0] = (__int64)"copyload";
  v190 = 259;
  if ( *(_DWORD *)(v34 + 32) <= 0x40u )
  {
    v53 = *(_QWORD *)(v34 + 24) == 0;
  }
  else
  {
    v161 = *(_DWORD *)(v34 + 32);
    v165 = v43;
    v52 = sub_C444A0(v35);
    LODWORD(v43) = v165;
    v53 = v161 == v52;
  }
  LOBYTE(v54) = v171;
  HIBYTE(v54) = 1;
  v51 = sub_A82CA0((unsigned int **)(v2 + 176), v159, v43, v54, !v53, (__int64)&v188);
  v188.m128i_i64[0] = 0x190000000ALL;
  sub_B47C00(v51, a2, v188.m128i_i32, 2);
  if ( v184 || v185 || v186 || v187 )
  {
    sub_E00EB0(&v188, &v184, *(_QWORD *)(v2 + 112) - *(_QWORD *)(v2 + 96), *(_QWORD *)(v51 + 8), *(_QWORD *)v2);
    sub_B9A100(v51, v188.m128i_i64);
  }
  if ( !*(_QWORD *)(v2 + 72) )
    goto LABEL_127;
  if ( v172 != 1 && v176 == v174 )
  {
    v55 = *(_QWORD *)(v2 + 32);
    _BitScanReverse64(&v56, 1LL << *(_WORD *)(v55 + 2));
    LOBYTE(v57) = 63 - (v56 ^ 0x3F);
    HIBYTE(v57) = 1;
    v58 = sub_291B4B0((__int64 *)(v2 + 176), *(_QWORD *)(v55 + 72), v55, v57, "oldload");
    v190 = 259;
    v188.m128i_i64[0] = (__int64)"vec";
    v51 = sub_2918170(v2 + 176, v58, v51, v170, &v188, v59);
  }
LABEL_65:
  v60 = *(_QWORD *)(a2 + 32 * (3LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  v61 = *(_DWORD *)(v60 + 32);
  if ( v61 <= 0x40 )
    v62 = *(_QWORD *)(v60 + 24) == 0;
  else
    v62 = v61 == (unsigned int)sub_C444A0(v60 + 24);
  LOBYTE(v63) = v166;
  HIBYTE(v63) = 1;
  v64 = sub_2463EC0((__int64 *)(v2 + 176), v51, v167, v63, !v62);
  v188.m128i_i64[0] = 0x190000000ALL;
  sub_B47C00(v64, a2, v188.m128i_i32, 2);
  if ( v184 || v185 || v186 || v187 )
  {
    sub_E00EB0(&v188, &v184, *(_QWORD *)(v2 + 112) - *(_QWORD *)(v2 + 96), *(_QWORD *)(v51 + 8), *(_QWORD *)v2);
    sub_B9A100(v64, v188.m128i_i64);
  }
  v188.m128i_i32[2] = sub_AE43F0(*(_QWORD *)v2, *(_QWORD *)(v167 + 8));
  if ( v188.m128i_i32[2] > 0x40u )
  {
    sub_C43690((__int64)&v188, 0, 0);
    if ( v176 != v174 )
    {
LABEL_71:
      v65 = sub_BD45C0((unsigned __int8 *)v167, *(_QWORD *)v2, (__int64)&v188, 1, 0, 0, 0, 0);
      if ( *v65 == 60 )
      {
        v146 = v188.m128i_i64[0];
        if ( v188.m128i_i32[2] > 0x40u )
          v146 = *(_QWORD *)v188.m128i_i64[0];
        sub_29228E0((__int64)v65, *(_BYTE *)(v2 + 137), 8 * v146, 8LL * *(_QWORD *)(v2 + 128), a2, v64, v167, v51);
      }
      goto LABEL_72;
    }
  }
  else
  {
    v188.m128i_i64[0] = 0;
    if ( v176 != v174 )
      goto LABEL_71;
  }
  sub_29228E0(
    *(_QWORD *)(v2 + 24),
    *(_BYTE *)(v2 + 137),
    8LL * *(_QWORD *)(v2 + 112),
    8LL * *(_QWORD *)(v2 + 128),
    a2,
    v64,
    v167,
    v51);
LABEL_72:
  v66 = *(_QWORD *)(a2 + 32 * (3LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  v67 = *(_DWORD *)(v66 + 32);
  if ( v67 <= 0x40 )
    LOBYTE(v51) = *(_QWORD *)(v66 + 24) == 0;
  else
    LOBYTE(v51) = v67 == (unsigned int)sub_C444A0(v66 + 24);
  if ( v188.m128i_i32[2] > 0x40u && v188.m128i_i64[0] )
    j_j___libc_free_0_0(v188.m128i_u64[0]);
LABEL_77:
  if ( v181 > 0x40 && v180 )
    j_j___libc_free_0_0((unsigned __int64)v180);
  return (unsigned int)v51;
}
