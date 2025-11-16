// Function: sub_33AEC60
// Address: 0x33aec60
//
void __fastcall sub_33AEC60(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int16 v6; // ax
  int v7; // edx
  bool v8; // zf
  __int64 v9; // rax
  __int64 v10; // rsi
  __m128i v11; // rax
  __int64 v12; // r15
  int v13; // eax
  unsigned __int16 *v14; // rax
  __int64 v15; // rsi
  __int64 v16; // r10
  unsigned __int16 v17; // r9
  __int64 v18; // r8
  __int64 v19; // rdx
  __int64 v20; // rax
  _QWORD *v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rax
  __int64 v31; // r13
  int v32; // eax
  unsigned int *v33; // rax
  const __m128i *v34; // roff
  __m128i v35; // xmm3
  __int64 v36; // rax
  __m128i v37; // xmm4
  __m128i v38; // xmm5
  __int64 v39; // rax
  __int64 v40; // rdi
  __int64 v41; // rax
  __int64 v42; // rsi
  __int64 v43; // rax
  __int64 v44; // r8
  __int64 v45; // rdx
  __int64 v46; // r15
  __int64 v47; // rdx
  __int64 v48; // r14
  __int64 *m128i_i64; // rdx
  __int64 v50; // rax
  __int64 v51; // rsi
  __int64 v52; // rax
  __int64 v53; // r9
  __int64 v54; // r8
  __int64 v55; // rdx
  __int64 v56; // r15
  __int64 v57; // rdx
  __int64 v58; // r14
  __int64 *v59; // rdx
  __int64 v60; // rax
  int v61; // edx
  __int64 v62; // rdi
  int v63; // esi
  __int64 v64; // rax
  __int64 v65; // r8
  __int64 v66; // r9
  __int64 v67; // rdx
  __int64 v68; // r15
  __int64 v69; // rdx
  __int64 v70; // r14
  __int64 *v71; // rdx
  int v72; // esi
  __int64 v73; // rdi
  __int64 v74; // rax
  __int64 v75; // r8
  __int64 v76; // rdx
  __int64 v77; // r15
  __int64 v78; // rdx
  __int64 v79; // r14
  __int64 *v80; // rdx
  unsigned int v81; // r14d
  unsigned int v82; // eax
  const __m128i *v83; // rcx
  const __m128i *v84; // r15
  __int64 v85; // rdx
  __int64 v86; // r8
  __int64 v87; // r9
  __int64 v88; // rdx
  __m128i *v89; // rax
  __int64 v90; // rax
  __int64 v91; // rdx
  unsigned __int64 v92; // r10
  __int64 *v93; // rdi
  int v94; // eax
  __int64 v95; // r9
  __int64 v96; // rax
  __int64 v97; // r8
  unsigned __int64 v98; // rdx
  __int64 *v99; // rax
  __int64 v100; // rax
  __int64 *v101; // rax
  __int64 v102; // rdi
  __int64 v103; // rax
  unsigned __int64 v104; // rdx
  int v105; // r9d
  __int32 v106; // ecx
  int v107; // r8d
  __int64 v108; // rax
  __int64 v109; // r15
  _QWORD *v110; // rax
  __int64 v111; // rdi
  int v112; // eax
  int v113; // edx
  int v114; // r9d
  __int64 v115; // r14
  __m128i v116; // xmm0
  _QWORD *v117; // rax
  const __m128i *v118; // rax
  __m128i v119; // xmm6
  __m128i v120; // xmm7
  unsigned int v121; // r15d
  __int64 *v122; // rax
  __int64 v123; // r8
  __int64 v124; // rax
  __int64 v125; // rdx
  __int64 v126; // r9
  unsigned __int64 v127; // rdx
  __int64 v128; // rax
  __int128 v129; // [rsp-20h] [rbp-1330h]
  __int64 v130; // [rsp-10h] [rbp-1320h]
  __int64 v131; // [rsp-10h] [rbp-1320h]
  __int128 v132; // [rsp-10h] [rbp-1320h]
  __int128 v133; // [rsp-10h] [rbp-1320h]
  __int64 v134; // [rsp-8h] [rbp-1318h]
  __int64 v135; // [rsp-8h] [rbp-1318h]
  bool v136; // [rsp+21h] [rbp-12EFh]
  bool v137; // [rsp+22h] [rbp-12EEh]
  char v138; // [rsp+23h] [rbp-12EDh]
  int v139; // [rsp+24h] [rbp-12ECh]
  unsigned __int16 v140; // [rsp+28h] [rbp-12E8h]
  int v141; // [rsp+30h] [rbp-12E0h]
  unsigned int v142; // [rsp+30h] [rbp-12E0h]
  __int64 v143; // [rsp+30h] [rbp-12E0h]
  int v144; // [rsp+38h] [rbp-12D8h]
  __int64 v145; // [rsp+38h] [rbp-12D8h]
  __m128i v146; // [rsp+40h] [rbp-12D0h] BYREF
  unsigned __int64 v147; // [rsp+50h] [rbp-12C0h]
  __int64 *v148; // [rsp+58h] [rbp-12B8h]
  __m128i v149; // [rsp+60h] [rbp-12B0h]
  __int64 v150; // [rsp+70h] [rbp-12A0h]
  __int64 v151; // [rsp+78h] [rbp-1298h]
  __int64 v152; // [rsp+80h] [rbp-1290h]
  __int64 v153; // [rsp+88h] [rbp-1288h]
  __int64 v154; // [rsp+90h] [rbp-1280h] BYREF
  int v155; // [rsp+98h] [rbp-1278h]
  __m128i v156; // [rsp+A0h] [rbp-1270h] BYREF
  __int64 v157; // [rsp+B0h] [rbp-1260h]
  __int64 v158; // [rsp+C0h] [rbp-1250h] BYREF
  unsigned __int64 v159; // [rsp+C8h] [rbp-1248h]
  __int64 v160; // [rsp+D0h] [rbp-1240h]
  int v161; // [rsp+D8h] [rbp-1238h]
  __int64 *v162; // [rsp+E0h] [rbp-1230h] BYREF
  __int64 v163; // [rsp+E8h] [rbp-1228h]
  __int64 v164; // [rsp+F0h] [rbp-1220h] BYREF
  int v165; // [rsp+F8h] [rbp-1218h]
  __m128i *v166; // [rsp+120h] [rbp-11F0h] BYREF
  __int64 v167; // [rsp+128h] [rbp-11E8h]
  __m128i v168; // [rsp+130h] [rbp-11E0h] BYREF
  __m128i v169; // [rsp+140h] [rbp-11D0h]
  __m128i v170; // [rsp+150h] [rbp-11C0h]
  __int64 v171; // [rsp+1B0h] [rbp-1160h] BYREF
  __int64 v172; // [rsp+1B8h] [rbp-1158h]
  __int64 v173; // [rsp+1C0h] [rbp-1150h]
  unsigned __int64 v174; // [rsp+1C8h] [rbp-1148h]
  __int64 v175; // [rsp+1D0h] [rbp-1140h]
  __int64 v176; // [rsp+1D8h] [rbp-1138h]
  __int64 v177; // [rsp+1E0h] [rbp-1130h]
  unsigned __int64 v178; // [rsp+1E8h] [rbp-1128h]
  __int64 v179; // [rsp+1F0h] [rbp-1120h]
  __int64 v180; // [rsp+1F8h] [rbp-1118h]
  __int64 v181; // [rsp+200h] [rbp-1110h]
  __int64 v182; // [rsp+208h] [rbp-1108h] BYREF
  int v183; // [rsp+210h] [rbp-1100h]
  __int64 v184; // [rsp+218h] [rbp-10F8h]
  _BYTE *v185; // [rsp+220h] [rbp-10F0h]
  __int64 v186; // [rsp+228h] [rbp-10E8h]
  _BYTE v187[1792]; // [rsp+230h] [rbp-10E0h] BYREF
  _BYTE *v188; // [rsp+930h] [rbp-9E0h]
  __int64 v189; // [rsp+938h] [rbp-9D8h]
  _BYTE v190[512]; // [rsp+940h] [rbp-9D0h] BYREF
  _BYTE *v191; // [rsp+B40h] [rbp-7D0h]
  __int64 v192; // [rsp+B48h] [rbp-7C8h]
  _BYTE v193[1792]; // [rsp+B50h] [rbp-7C0h] BYREF
  _BYTE *v194; // [rsp+1250h] [rbp-C0h]
  __int64 v195; // [rsp+1258h] [rbp-B8h]
  _BYTE v196[64]; // [rsp+1260h] [rbp-B0h] BYREF
  __int64 v197; // [rsp+12A0h] [rbp-70h]
  __int64 v198; // [rsp+12A8h] [rbp-68h]
  int v199; // [rsp+12B0h] [rbp-60h]
  char v200; // [rsp+12D0h] [rbp-40h]

  v6 = *(_WORD *)(a2 + 2);
  v7 = *(_DWORD *)(a1 + 848);
  v154 = 0;
  LOWORD(v147) = (v6 >> 2) & 0x3FF;
  v137 = (_WORD)v147 == 13;
  LOBYTE(v6) = *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL);
  v155 = v7;
  v8 = (_BYTE)v6 == 7;
  v138 = v6;
  v9 = *(_QWORD *)a1;
  v136 = !v8;
  if ( *(_QWORD *)a1 )
  {
    v148 = &v154;
    if ( &v154 != (__int64 *)(v9 + 48) )
    {
      v10 = *(_QWORD *)(v9 + 48);
      v154 = v10;
      if ( v10 )
        sub_B96E90((__int64)&v154, v10, 1);
    }
  }
  else
  {
    v148 = &v154;
  }
  v11.m128i_i64[0] = sub_338B750(a1, *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))));
  v146 = v11;
  v12 = v11.m128i_i64[0];
  v13 = *(_DWORD *)(v11.m128i_i64[0] + 24);
  if ( v13 == 35 || v13 == 11 )
  {
    v20 = *(_QWORD *)(v12 + 96);
    v21 = *(_QWORD **)(v20 + 24);
    if ( *(_DWORD *)(v20 + 32) > 0x40u )
      v21 = (_QWORD *)*v21;
    v152 = sub_3400D50(*(_QWORD *)(a1 + 864), v21, v148, 1);
    v12 = v152;
    v146.m128i_i64[0] = v152;
    v153 = v22;
    v146.m128i_i64[1] = (unsigned int)v22 | v146.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  }
  else if ( (unsigned int)(v13 - 13) <= 1 || (unsigned int)(v13 - 37) <= 1 )
  {
    v14 = *(unsigned __int16 **)(v12 + 48);
    v15 = *(_QWORD *)(v12 + 80);
    v16 = *(_QWORD *)(a1 + 864);
    v17 = *v14;
    v18 = *((_QWORD *)v14 + 1);
    v171 = v15;
    if ( v15 )
    {
      v140 = v17;
      v141 = v18;
      v144 = v16;
      sub_B96E90((__int64)&v171, v15, 1);
      v17 = v140;
      LODWORD(v18) = v141;
      LODWORD(v16) = v144;
    }
    LODWORD(v172) = *(_DWORD *)(v12 + 72);
    v150 = sub_33ED290(v16, *(_QWORD *)(v12 + 96), (unsigned int)&v171, v17, v18, 0, 1, 0);
    v12 = v150;
    v146.m128i_i64[0] = v150;
    v151 = v19;
    v146.m128i_i64[1] = (unsigned int)v19 | v146.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    if ( v171 )
      sub_B91220((__int64)&v171, v171);
  }
  v23 = *(_QWORD *)(sub_338B750(a1, *(_QWORD *)(a2 + 32 * (3LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)))) + 96);
  if ( *(_DWORD *)(v23 + 32) <= 0x40u )
    v24 = *(_QWORD *)(v23 + 24);
  else
    v24 = **(_QWORD **)(v23 + 24);
  v139 = v24;
  if ( (_WORD)v147 == 13 )
  {
    v128 = sub_BCB120(*(_QWORD **)(*(_QWORD *)(a1 + 864) + 64LL));
    LODWORD(v24) = 0;
    v25 = v128;
  }
  else
  {
    v25 = *(_QWORD *)(a2 + 8);
  }
  v26 = *(_QWORD *)(a1 + 864);
  v142 = v24;
  v174 = 0xFFFFFFFF00000020LL;
  v181 = v26;
  v185 = v187;
  v186 = 0x2000000000LL;
  v189 = 0x2000000000LL;
  v192 = 0x2000000000LL;
  v188 = v190;
  v194 = v196;
  v145 = v25;
  v191 = v193;
  v195 = 0x400000000LL;
  v171 = 0;
  v172 = 0;
  v173 = 0;
  v175 = 0;
  v176 = 0;
  v177 = 0;
  v178 = 0;
  v179 = 0;
  v180 = 0;
  v182 = 0;
  v183 = 0;
  v184 = 0;
  v197 = 0;
  v198 = 0;
  v199 = 0;
  v200 = 0;
  v166 = *(__m128i **)(a2 + 72);
  v27 = sub_A74610(&v166);
  *((_QWORD *)&v129 + 1) = v146.m128i_i64[1];
  v146.m128i_i64[0] = v12;
  *(_QWORD *)&v129 = v12;
  sub_33AC4A0(a1, (__int64)&v171, a2, 4u, v142, v145, v129, v27, 1);
  sub_3384280(&v156, a1, (__int64)&v171, a3, v28, v29);
  v30 = v157;
  if ( *(_DWORD *)(v157 + 24) == 309 )
    v30 = **(_QWORD **)(v157 + 40);
  if ( v138 != 7 && *(_DWORD *)(v30 + 24) == 50 )
    v30 = **(_QWORD **)(v30 + 40);
  v31 = **(_QWORD **)(v30 + 40);
  v32 = *(_DWORD *)(v31 + 64);
  if ( v32
    && (v33 = (unsigned int *)(*(_QWORD *)(v31 + 40) + 40LL * (unsigned int)(v32 - 1)),
        v143 = *(_QWORD *)v33,
        *(_WORD *)(*(_QWORD *)(*(_QWORD *)v33 + 48LL) + 16LL * v33[2]) == 262) )
  {
    v166 = &v168;
    v167 = 0x800000000LL;
    v34 = *(const __m128i **)(v31 + 40);
    v35 = _mm_loadu_si128(v34);
    LODWORD(v167) = 1;
    v168 = v35;
    v36 = 40LL * *(unsigned int *)(v31 + 64);
    v37 = _mm_loadu_si128((const __m128i *)((char *)v34 + v36 - 40));
    LODWORD(v167) = 2;
    v169 = v37;
    v38 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v31 + 40) + v36 - 80));
    LODWORD(v167) = 3;
    v170 = v38;
  }
  else
  {
    v143 = 0;
    v166 = &v168;
    v167 = 0x800000000LL;
    v118 = *(const __m128i **)(v31 + 40);
    v119 = _mm_loadu_si128(v118);
    LODWORD(v167) = 1;
    v168 = v119;
    v120 = _mm_loadu_si128((const __m128i *)((char *)v118 + 40 * *(unsigned int *)(v31 + 64) - 40));
    LODWORD(v167) = 2;
    v169 = v120;
  }
  v39 = sub_338B750(a1, *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  v40 = *(_QWORD *)(a1 + 864);
  v41 = *(_QWORD *)(v39 + 96);
  if ( *(_DWORD *)(v41 + 32) <= 0x40u )
    v42 = *(_QWORD *)(v41 + 24);
  else
    v42 = **(_QWORD **)(v41 + 24);
  v43 = sub_3400BD0(v40, v42, (_DWORD)v148, 8, 0, 1, 0);
  v46 = v45;
  v47 = (unsigned int)v167;
  v48 = v43;
  if ( (unsigned __int64)(unsigned int)v167 + 1 > HIDWORD(v167) )
  {
    sub_C8D5F0((__int64)&v166, &v168, (unsigned int)v167 + 1LL, 0x10u, v44, v130);
    v47 = (unsigned int)v167;
  }
  m128i_i64 = v166[v47].m128i_i64;
  *m128i_i64 = v48;
  m128i_i64[1] = v46;
  LODWORD(m128i_i64) = *(_DWORD *)(a2 + 4);
  LODWORD(v167) = v167 + 1;
  v50 = *(_QWORD *)(sub_338B750(a1, *(_QWORD *)(a2 + 32 * (1LL - ((unsigned int)m128i_i64 & 0x7FFFFFF)))) + 96);
  if ( *(_DWORD *)(v50 + 32) <= 0x40u )
    v51 = *(_QWORD *)(v50 + 24);
  else
    v51 = **(_QWORD **)(v50 + 24);
  v52 = sub_3400BD0(*(_QWORD *)(a1 + 864), v51, (_DWORD)v148, 7, 0, 1, 0);
  v54 = v134;
  v56 = v55;
  v57 = (unsigned int)v167;
  v58 = v52;
  if ( (unsigned __int64)(unsigned int)v167 + 1 > HIDWORD(v167) )
  {
    sub_C8D5F0((__int64)&v166, &v168, (unsigned int)v167 + 1LL, 0x10u, v134, v53);
    v57 = (unsigned int)v167;
  }
  v59 = v166[v57].m128i_i64;
  *v59 = v58;
  v59[1] = v56;
  LODWORD(v167) = v167 + 1;
  v60 = (unsigned int)v167;
  if ( (unsigned __int64)(unsigned int)v167 + 1 > HIDWORD(v167) )
  {
    sub_C8D5F0((__int64)&v166, &v168, (unsigned int)v167 + 1LL, 0x10u, v54, v53);
    v60 = (unsigned int)v167;
  }
  v61 = (int)v148;
  v166[v60] = _mm_load_si128(&v146);
  v62 = *(_QWORD *)(a1 + 864);
  LODWORD(v167) = v167 + 1;
  v63 = (v143 == 0) + *(_DWORD *)(v31 + 64) - 4;
  if ( (_WORD)v147 == 13 )
    v63 = v139;
  v64 = sub_3400BD0(v62, v63, v61, 7, 0, 1, 0);
  v68 = v67;
  v69 = (unsigned int)v167;
  v70 = v64;
  if ( (unsigned __int64)(unsigned int)v167 + 1 > HIDWORD(v167) )
  {
    sub_C8D5F0((__int64)&v166, &v168, (unsigned int)v167 + 1LL, 0x10u, v65, v66);
    v69 = (unsigned int)v167;
  }
  v71 = v166[v69].m128i_i64;
  v72 = (unsigned __int16)v147;
  *v71 = v70;
  v71[1] = v68;
  v73 = *(_QWORD *)(a1 + 864);
  LODWORD(v167) = v167 + 1;
  v74 = sub_3400BD0(v73, v72, (_DWORD)v148, 7, 0, 1, 0);
  v77 = v76;
  v78 = (unsigned int)v167;
  v79 = v74;
  if ( (unsigned __int64)(unsigned int)v167 + 1 > HIDWORD(v167) )
  {
    sub_C8D5F0((__int64)&v166, &v168, (unsigned int)v167 + 1LL, 0x10u, v75, v131);
    v78 = (unsigned int)v167;
  }
  v80 = v166[v78].m128i_i64;
  *v80 = v79;
  v80[1] = v77;
  v81 = v139 + 4;
  v82 = v167 + 1;
  LODWORD(v167) = v167 + 1;
  if ( (_WORD)v147 == 13 && v139 )
  {
    v121 = 4;
    do
    {
      v123 = sub_338B750(a1, *(_QWORD *)(a2 + 32 * (v121 - (unsigned __int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF))));
      v124 = (unsigned int)v167;
      v126 = v125;
      v127 = (unsigned int)v167 + 1LL;
      if ( v127 > HIDWORD(v167) )
      {
        v146.m128i_i64[0] = v123;
        v146.m128i_i64[1] = v126;
        sub_C8D5F0((__int64)&v166, &v168, v127, 0x10u, v123, v126);
        v124 = (unsigned int)v167;
        v126 = v146.m128i_i64[1];
        v123 = v146.m128i_i64[0];
      }
      v122 = v166[v124].m128i_i64;
      ++v121;
      *v122 = v123;
      v122[1] = v126;
      v82 = v167 + 1;
      LODWORD(v167) = v167 + 1;
    }
    while ( v121 != v81 );
  }
  v83 = *(const __m128i **)(v31 + 40);
  v84 = v83 + 5;
  v85 = 5LL * *(unsigned int *)(v31 + 64);
  v86 = (__int64)&v83[-2] + v85 * 8 - 8;
  if ( v143 )
    v86 = (__int64)&v83[-5].m128i_i64[v85];
  v87 = 0xCCCCCCCCCCCCCCCDLL * ((v86 - (__int64)v84) >> 3);
  v88 = v82;
  if ( v87 + (unsigned __int64)v82 > HIDWORD(v167) )
  {
    v147 = 0xCCCCCCCCCCCCCCCDLL * ((v86 - (__int64)v84) >> 3);
    v146.m128i_i64[0] = v86;
    sub_C8D5F0((__int64)&v166, &v168, v87 + v82, 0x10u, v86, v87);
    v88 = (unsigned int)v167;
    LODWORD(v87) = v147;
    v86 = v146.m128i_i64[0];
  }
  v89 = &v166[v88];
  if ( v84 != (const __m128i *)v86 )
  {
    do
    {
      if ( v89 )
        *v89 = _mm_loadu_si128(v84);
      v84 = (const __m128i *)((char *)v84 + 40);
      ++v89;
    }
    while ( (const __m128i *)v86 != v84 );
    LODWORD(v88) = v167;
  }
  LODWORD(v167) = v88 + v87;
  sub_33AEA10((unsigned __int8 *)a2, v81, (__int64)&v166, a1);
  if ( v137 && v136 )
  {
    v90 = *(_QWORD *)(a1 + 864);
    v91 = *(_QWORD *)(a2 + 8);
    v92 = *(_QWORD *)(v90 + 16);
    v163 = 0x300000000LL;
    v162 = &v164;
    v93 = *(__int64 **)(v90 + 40);
    v147 = v92;
    v146.m128i_i64[0] = v91;
    v94 = sub_2E79000(v93);
    v158 = 0;
    LOBYTE(v159) = 0;
    sub_34B8C80(v147, v94, v146.m128i_i32[0], (unsigned int)&v162, 0, 0, __PAIR128__(v159, 0));
    v96 = (unsigned int)v163;
    v97 = v135;
    v98 = (unsigned int)v163 + 1LL;
    if ( v98 > HIDWORD(v163) )
    {
      sub_C8D5F0((__int64)&v162, &v164, v98, 0x10u, v135, v95);
      v96 = (unsigned int)v163;
    }
    v99 = &v162[2 * v96];
    *v99 = 1;
    v99[1] = 0;
    LODWORD(v163) = v163 + 1;
    v100 = (unsigned int)v163;
    if ( (unsigned __int64)(unsigned int)v163 + 1 > HIDWORD(v163) )
    {
      sub_C8D5F0((__int64)&v162, &v164, (unsigned int)v163 + 1LL, 0x10u, v97, v95);
      v100 = (unsigned int)v163;
    }
    v101 = &v162[2 * v100];
    *v101 = 262;
    v101[1] = 0;
    v102 = *(_QWORD *)(a1 + 864);
    LODWORD(v163) = v163 + 1;
    v103 = sub_33E5830(v102, v162);
    v106 = v103;
    v107 = v104;
    if ( v162 != &v164 )
    {
      v147 = v104;
      v146.m128i_i64[0] = v103;
      _libc_free((unsigned __int64)v162);
      v107 = v147;
      v106 = v146.m128i_i32[0];
    }
    *((_QWORD *)&v132 + 1) = (unsigned int)v167;
    *(_QWORD *)&v132 = v166;
    v108 = sub_3411630(*(_QWORD *)(a1 + 864), 394, (_DWORD)v148, v106, v107, v105, v132);
    v162 = (__int64 *)a2;
    v109 = v108;
    v110 = sub_337DC20(a1 + 8, (__int64 *)&v162);
    *v110 = v109;
    *((_DWORD *)v110 + 2) = 0;
    v111 = *(_QWORD *)(a1 + 864);
    v158 = v31;
    LODWORD(v159) = 0;
    v160 = v31;
    v161 = 1;
    v162 = (__int64 *)v109;
    LODWORD(v163) = 1;
    v164 = v109;
    v165 = 2;
    sub_3417D60(v111, &v158, &v162, 2);
  }
  else
  {
    v112 = sub_33E5110(*(_QWORD *)(a1 + 864), 1, 0, 262, 0);
    *((_QWORD *)&v133 + 1) = (unsigned int)v167;
    *(_QWORD *)&v133 = v166;
    v115 = sub_3411630(*(_QWORD *)(a1 + 864), 394, (_DWORD)v148, v112, v113, v114, v133);
    if ( v138 != 7 )
    {
      v116 = _mm_load_si128(&v156);
      v162 = (__int64 *)a2;
      v146 = v116;
      v117 = sub_337DC20(a1 + 8, (__int64 *)&v162);
      v149 = _mm_load_si128(&v146);
      *v117 = v149.m128i_i64[0];
      *((_DWORD *)v117 + 2) = v149.m128i_i32[2];
    }
    sub_34158F0(*(_QWORD *)(a1 + 864), v31, v115);
  }
  sub_33EBEB0(*(_QWORD *)(a1 + 864), v31);
  *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 960) + 8LL) + 48LL) + 40LL) = 1;
  if ( v166 != &v168 )
    _libc_free((unsigned __int64)v166);
  if ( v194 != v196 )
    _libc_free((unsigned __int64)v194);
  if ( v191 != v193 )
    _libc_free((unsigned __int64)v191);
  if ( v188 != v190 )
    _libc_free((unsigned __int64)v188);
  if ( v185 != v187 )
    _libc_free((unsigned __int64)v185);
  if ( v182 )
    sub_B91220((__int64)&v182, v182);
  if ( v178 )
    j_j___libc_free_0(v178);
  if ( v154 )
    sub_B91220((__int64)v148, v154);
}
