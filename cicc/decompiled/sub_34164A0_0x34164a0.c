// Function: sub_34164A0
// Address: 0x34164a0
//
__int64 __fastcall sub_34164A0(_QWORD *a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v8; // rcx
  __int64 *v9; // rax
  unsigned __int16 *v10; // rax
  __int64 v11; // rax
  char *v12; // r14
  char v13; // r8
  __int64 *v14; // r13
  char v15; // r15
  int v16; // eax
  unsigned __int64 v17; // rax
  size_t v18; // rax
  __int64 v19; // rax
  bool v22; // al
  __int64 v23; // rcx
  _QWORD *v24; // rax
  _BYTE *v25; // rdx
  _QWORD *i; // rdx
  __int64 v27; // r9
  __int64 v28; // rbx
  __int64 v29; // r14
  __int64 v30; // rax
  unsigned int v31; // r15d
  __int64 v32; // rdi
  __int64 v33; // rax
  const __m128i *v34; // r15
  const __m128i *v35; // r13
  const __m128i *v36; // rbx
  __m128i *v37; // rsi
  unsigned __int16 *v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rax
  __int64 v41; // rax
  __m128i v42; // xmm1
  __int64 *v43; // rdx
  __int64 *v44; // rax
  __int64 *k; // rdx
  __int64 v46; // r15
  __int64 v47; // rax
  __int64 v48; // r8
  unsigned __int64 v49; // r14
  __int64 v50; // r9
  _BYTE *v51; // r13
  unsigned __int8 v52; // r10
  __int64 v53; // r12
  __int64 v54; // rbx
  __int64 v55; // rax
  __int64 v56; // rdi
  __int64 v57; // rcx
  __int64 *v58; // rax
  __m128i *v59; // rsi
  _QWORD *v60; // rax
  __int64 v61; // rdx
  __int64 v62; // rsi
  unsigned __int16 *v63; // rax
  __int64 v64; // rdx
  __int64 v65; // rax
  __int64 v66; // r13
  __int64 v67; // r15
  __int64 (__fastcall *v68)(__int64, __int64, unsigned int); // r14
  __int64 v69; // rax
  int v70; // edx
  unsigned __int16 v71; // ax
  __int64 v72; // rax
  __int64 v73; // rcx
  __int64 v74; // rdx
  __int64 v75; // r15
  unsigned __int64 v76; // rax
  __int64 v77; // rdx
  unsigned __int32 v78; // r14d
  const __m128i *v79; // rdx
  void (***v80)(); // rdi
  void (*v81)(); // rax
  __int64 v82; // r15
  __m128i *v83; // r8
  __int64 v84; // r9
  __int64 *v85; // r14
  unsigned int v86; // r13d
  __m128i *v87; // rdi
  const __m128i *v88; // roff
  __m128i v89; // xmm2
  __int8 v90; // al
  __int64 v91; // rax
  __int64 v92; // rdx
  __int64 v93; // rax
  __m128i **v94; // rax
  __int64 v95; // rax
  __int64 v96; // rax
  __int64 v97; // rsi
  __m128i v98; // xmm6
  __int64 v99; // rax
  __m128i v100; // xmm0
  __int64 v101; // r13
  __int64 (__fastcall *v102)(__int64, __int64, __int64 *, _QWORD, _QWORD); // r14
  __int64 v103; // rax
  unsigned __int32 v104; // eax
  __int64 v105; // rdx
  __int64 v106; // r9
  unsigned __int8 *v107; // r13
  __int64 v108; // rdx
  __int64 v109; // r14
  __int64 v110; // rcx
  __int64 v111; // r8
  __int64 v112; // r9
  __int64 v113; // rax
  __m128i *v114; // rsi
  unsigned __int16 *v115; // rax
  __int64 v116; // rdx
  __int64 v117; // rax
  __int64 v118; // r15
  __int64 v119; // rdi
  unsigned __int8 v120; // r15
  __int64 v121; // r10
  __int64 *v122; // r15
  __int64 j; // r8
  __int64 v124; // rax
  __int64 v125; // rdx
  unsigned __int64 v126; // r9
  unsigned int v127; // eax
  __int64 v128; // rdx
  unsigned int v129; // r15d
  __int64 v130; // r12
  __int64 v131; // rcx
  __int64 v132; // rdx
  __int64 v133; // rbx
  _QWORD *v134; // rax
  char v135; // al
  char v136; // r13
  __int64 v137; // rax
  char v138; // dl
  int v139; // eax
  __int64 *v140; // rax
  __int64 *v141; // r13
  __int64 *v142; // rbx
  __int64 v143; // r14
  __int64 v144; // rax
  unsigned __int64 v145; // rdx
  __int64 v146; // [rsp+8h] [rbp-13C8h]
  __int64 v147; // [rsp+8h] [rbp-13C8h]
  __int64 v148; // [rsp+10h] [rbp-13C0h]
  __int64 v149; // [rsp+10h] [rbp-13C0h]
  __int64 v150; // [rsp+18h] [rbp-13B8h]
  __int16 v151; // [rsp+22h] [rbp-13AEh]
  __int64 v152; // [rsp+30h] [rbp-13A0h]
  __int64 v153; // [rsp+30h] [rbp-13A0h]
  __int64 v154; // [rsp+30h] [rbp-13A0h]
  __int64 v155; // [rsp+40h] [rbp-1390h]
  __int64 v156; // [rsp+40h] [rbp-1390h]
  __int64 v157; // [rsp+40h] [rbp-1390h]
  unsigned __int32 v158; // [rsp+58h] [rbp-1378h]
  __int64 v159; // [rsp+60h] [rbp-1370h]
  __int64 v160; // [rsp+68h] [rbp-1368h]
  unsigned int v161; // [rsp+70h] [rbp-1360h]
  unsigned __int8 v162; // [rsp+70h] [rbp-1360h]
  unsigned __int8 v163; // [rsp+70h] [rbp-1360h]
  __int64 v164; // [rsp+80h] [rbp-1350h]
  __int64 v165; // [rsp+88h] [rbp-1348h]
  char *s; // [rsp+90h] [rbp-1340h]
  __int64 *v167; // [rsp+98h] [rbp-1338h]
  __int64 v168; // [rsp+98h] [rbp-1338h]
  __int64 *v169; // [rsp+98h] [rbp-1338h]
  __m128i v170; // [rsp+A0h] [rbp-1330h] BYREF
  __int64 v171; // [rsp+B0h] [rbp-1320h]
  unsigned int v172; // [rsp+B8h] [rbp-1318h]
  char v173; // [rsp+BCh] [rbp-1314h]
  unsigned __int8 v174; // [rsp+BDh] [rbp-1313h]
  __int16 v175; // [rsp+BEh] [rbp-1312h]
  __int64 v176; // [rsp+C0h] [rbp-1310h]
  __int64 v177; // [rsp+C8h] [rbp-1308h]
  unsigned __int8 *v178; // [rsp+D0h] [rbp-1300h]
  __int64 v179; // [rsp+D8h] [rbp-12F8h]
  __int64 v180; // [rsp+E0h] [rbp-12F0h]
  __int64 v181; // [rsp+E8h] [rbp-12E8h]
  __int64 v182; // [rsp+F0h] [rbp-12E0h]
  __int64 v183; // [rsp+F8h] [rbp-12D8h]
  __m128i v184; // [rsp+100h] [rbp-12D0h]
  __int64 v185; // [rsp+118h] [rbp-12B8h]
  unsigned __int64 v186; // [rsp+128h] [rbp-12A8h]
  __int128 v187; // [rsp+130h] [rbp-12A0h] BYREF
  __int64 v188; // [rsp+140h] [rbp-1290h] BYREF
  int v189; // [rsp+148h] [rbp-1288h]
  unsigned __int64 v190; // [rsp+150h] [rbp-1280h] BYREF
  __m128i *v191; // [rsp+158h] [rbp-1278h]
  const __m128i *v192; // [rsp+160h] [rbp-1270h]
  __int128 v193; // [rsp+170h] [rbp-1260h]
  __int64 v194; // [rsp+180h] [rbp-1250h]
  _BYTE *v195; // [rsp+190h] [rbp-1240h] BYREF
  __int64 v196; // [rsp+198h] [rbp-1238h]
  _BYTE v197[16]; // [rsp+1A0h] [rbp-1230h] BYREF
  __m128i v198; // [rsp+1B0h] [rbp-1220h] BYREF
  __int64 v199; // [rsp+1C0h] [rbp-1210h]
  __int64 v200; // [rsp+1C8h] [rbp-1208h]
  __m128i v201; // [rsp+1D0h] [rbp-1200h] BYREF
  _QWORD v202[8]; // [rsp+1E0h] [rbp-11F0h] BYREF
  __int64 *v203; // [rsp+220h] [rbp-11B0h] BYREF
  __int64 v204; // [rsp+228h] [rbp-11A8h]
  _BYTE v205[64]; // [rsp+230h] [rbp-11A0h] BYREF
  __m128i v206; // [rsp+270h] [rbp-1160h] BYREF
  __m128i v207; // [rsp+280h] [rbp-1150h] BYREF
  __m128i v208; // [rsp+290h] [rbp-1140h] BYREF
  __int64 v209; // [rsp+2A0h] [rbp-1130h]
  unsigned __int64 v210; // [rsp+2A8h] [rbp-1128h] BYREF
  __m128i *v211; // [rsp+2B0h] [rbp-1120h]
  const __m128i *v212; // [rsp+2B8h] [rbp-1118h]
  __int64 v213; // [rsp+2C0h] [rbp-1110h]
  __int64 v214; // [rsp+2C8h] [rbp-1108h] BYREF
  int v215; // [rsp+2D0h] [rbp-1100h]
  __int64 v216; // [rsp+2D8h] [rbp-10F8h]
  _BYTE *v217; // [rsp+2E0h] [rbp-10F0h]
  __int64 v218; // [rsp+2E8h] [rbp-10E8h]
  _BYTE v219[1792]; // [rsp+2F0h] [rbp-10E0h] BYREF
  _BYTE *v220; // [rsp+9F0h] [rbp-9E0h]
  __int64 v221; // [rsp+9F8h] [rbp-9D8h]
  _BYTE v222[512]; // [rsp+A00h] [rbp-9D0h] BYREF
  _BYTE *v223; // [rsp+C00h] [rbp-7D0h]
  __int64 v224; // [rsp+C08h] [rbp-7C8h]
  _BYTE v225[1792]; // [rsp+C10h] [rbp-7C0h] BYREF
  _BYTE *v226; // [rsp+1310h] [rbp-C0h]
  __int64 v227; // [rsp+1318h] [rbp-B8h]
  _BYTE v228[64]; // [rsp+1320h] [rbp-B0h] BYREF
  __int64 v229; // [rsp+1360h] [rbp-70h]
  __int64 v230; // [rsp+1368h] [rbp-68h]
  int v231; // [rsp+1370h] [rbp-60h]
  char v232; // [rsp+1390h] [rbp-40h]

  v6 = (__int64)a1;
  v8 = a3;
  v185 = a5;
  v174 = BYTE4(a5);
  v9 = (__int64 *)a1[8];
  v171 = a3;
  v167 = v9;
  v10 = *(unsigned __int16 **)(a3 + 48);
  v172 = a5;
  LODWORD(a3) = *v10;
  *((_QWORD *)&v187 + 1) = *((_QWORD *)v10 + 1);
  v11 = a1[2];
  LOWORD(v187) = a3;
  v160 = a2;
  s = *(char **)(v11 + 8LL * a2 + 525288);
  LOBYTE(a6) = s == 0 || a2 == 0;
  if ( (_BYTE)a6 )
    return 0;
  v161 = *(_DWORD *)(v8 + 68);
  if ( (_WORD)a3 )
  {
    if ( (unsigned __int16)(a3 - 17) <= 0xD3u )
      goto LABEL_4;
LABEL_12:
    v159 = 0;
    v170.m128i_i64[0] = (__int64)&v206;
    goto LABEL_13;
  }
  v170.m128i_i32[0] = a3;
  v22 = sub_30070B0((__int64)&v187);
  LOWORD(a3) = v170.m128i_i16[0];
  if ( !v22 )
    goto LABEL_12;
LABEL_4:
  v12 = (char *)&v203;
  v13 = 0;
  v170.m128i_i64[0] = (__int64)&v206;
  LOWORD(v203) = 256;
  while ( 1 )
  {
    v14 = (__int64 *)a1[3];
    v15 = v13;
    if ( (_WORD)a3 )
    {
      v16 = (unsigned __int16)a3;
      LOBYTE(a3) = (unsigned __int16)(a3 - 176) <= 0x34u;
      LODWORD(v17) = word_4456340[v16 - 1];
    }
    else
    {
      v17 = sub_3007240((__int64)&v187);
      v186 = v17;
      a3 = HIDWORD(v17);
    }
    v206.m128i_i8[4] = a3;
    v206.m128i_i32[0] = v17;
    v18 = strlen(s);
    v19 = sub_97F930(*v14, s, v18, v170.m128i_i64[0], v15);
    if ( v19 )
      break;
    if ( (char *)&v203 + 2 == ++v12 )
      return 0;
    v13 = *v12;
    LOWORD(a3) = v187;
  }
  v159 = v19;
LABEL_13:
  v23 = v161;
  v24 = v197;
  v25 = v197;
  v195 = v197;
  v196 = 0x200000000LL;
  if ( v161 )
  {
    if ( v161 > 2uLL )
    {
      sub_C8D5F0((__int64)&v195, v197, v161, 8u, a5, a6);
      v25 = v195;
      v24 = &v195[8 * (unsigned int)v196];
    }
    v23 = v161;
    for ( i = &v25[8 * v161]; i != v24; ++v24 )
    {
      if ( v24 )
        *v24 = 0;
    }
    LODWORD(v196) = v161;
  }
  v165 = 0;
  v158 = 0;
  v27 = *(_QWORD *)(v171 + 56);
  if ( v27 )
  {
    v155 = a4;
    v28 = *(_QWORD *)(v171 + 56);
    while ( 1 )
    {
      v29 = *(_QWORD *)(v28 + 16);
      if ( *(_DWORD *)(v29 + 24) != 299 )
        goto LABEL_32;
      if ( (*(_BYTE *)(v29 + 33) & 4) != 0 )
        goto LABEL_32;
      v23 = *(unsigned __int16 *)(v29 + 32);
      if ( (v23 & 0x380) != 0 )
        goto LABEL_32;
      v30 = *(_QWORD *)(v29 + 40);
      v31 = *(_DWORD *)(v30 + 48);
      if ( (v174 & (v31 == v172)) != 0 )
        goto LABEL_32;
      v32 = *(_QWORD *)(v29 + 112);
      if ( (*(_BYTE *)(v32 + 37) & 0xF) != 0 )
        goto LABEL_32;
      v23 &= 8u;
      v152 = *(_QWORD *)(v30 + 40);
      if ( (_DWORD)v23 )
        goto LABEL_32;
      if ( (unsigned int)sub_2EAC1E0(v32) )
        goto LABEL_32;
      v23 = v165;
      if ( v165 )
      {
        v33 = *(_QWORD *)(v29 + 40);
        if ( v165 != *(_QWORD *)v33 )
          goto LABEL_32;
        v23 = v158;
        if ( *(_DWORD *)(v33 + 8) != v158 )
          goto LABEL_32;
      }
      v150 = v31;
      v115 = (unsigned __int16 *)(*(_QWORD *)(v152 + 48) + 16LL * v31);
      v116 = *v115;
      v117 = *((_QWORD *)v115 + 1);
      v206.m128i_i16[0] = v116;
      v206.m128i_i64[1] = v117;
      v118 = sub_3007410(v170.m128i_i64[0], v167, v116, v152, a5, v27);
      v119 = sub_2E79000(*(__int64 **)(v6 + 40));
      if ( (unsigned int)*(unsigned __int8 *)(v118 + 8) - 17 <= 1 )
        v118 = **(_QWORD **)(v118 + 16);
      v120 = sub_AE5020(v119, v118);
      if ( (unsigned __int8)sub_2EAC4F0(*(_QWORD *)(v29 + 112)) < v120 )
        goto LABEL_32;
      v206.m128i_i64[0] = 0;
      v207.m128i_i8[12] = 1;
      v121 = v171;
      v201.m128i_i64[0] = (__int64)v202;
      v201.m128i_i64[1] = 0x800000000LL;
      v204 = 0x800000000LL;
      v203 = (__int64 *)v205;
      v206.m128i_i64[1] = (__int64)&v208;
      v207.m128i_i64[0] = 16;
      v207.m128i_i32[2] = 0;
      v122 = *(__int64 **)(v29 + 40);
      for ( j = (__int64)&v122[5 * *(unsigned int *)(v29 + 64)]; (__int64 *)j != v122; v122 += 5 )
      {
        v124 = *v122;
        if ( v121 != *v122 )
        {
          v125 = v201.m128i_u32[2];
          v126 = v201.m128i_u32[2] + 1LL;
          if ( v126 > v201.m128i_u32[3] )
          {
            v147 = v121;
            v149 = j;
            v154 = *v122;
            sub_C8D5F0((__int64)&v201, v202, v201.m128i_u32[2] + 1LL, 8u, j, v126);
            v125 = v201.m128i_u32[2];
            v121 = v147;
            j = v149;
            v124 = v154;
          }
          *(_QWORD *)(v201.m128i_i64[0] + 8 * v125) = v124;
          ++v201.m128i_i32[2];
        }
      }
      v127 = sub_33CA560();
      LODWORD(v128) = v201.m128i_i32[2];
      v129 = v127;
      if ( !v201.m128i_i32[2] )
      {
LABEL_151:
        v135 = sub_3285B00(v171, v170.m128i_i64[0], (__int64)&v203, v129, 0, v27);
        v23 = v207.m128i_u8[12];
        v136 = v135 ^ 1;
        goto LABEL_152;
      }
      v153 = v29;
      v148 = v28;
      v173 = 0;
      v146 = v6;
      v130 = v170.m128i_i64[0];
      while ( 1 )
      {
        v131 = (unsigned int)v128;
        v132 = (unsigned int)(v128 - 1);
        v133 = *(_QWORD *)(v201.m128i_i64[0] + 8 * v131 - 8);
        v201.m128i_i32[2] = v132;
        if ( !v207.m128i_i8[12] )
          goto LABEL_160;
        v134 = (_QWORD *)v206.m128i_i64[1];
        v131 = v207.m128i_u32[1];
        v132 = v206.m128i_i64[1] + 8LL * v207.m128i_u32[1];
        if ( v206.m128i_i64[1] != v132 )
        {
          while ( v133 != *v134 )
          {
            if ( (_QWORD *)v132 == ++v134 )
              goto LABEL_171;
          }
          goto LABEL_148;
        }
LABEL_171:
        if ( v207.m128i_i32[1] < (unsigned __int32)v207.m128i_i32[0] )
        {
          ++v207.m128i_i32[1];
          *(_QWORD *)v132 = v133;
          v23 = v207.m128i_u8[12];
          ++v206.m128i_i64[0];
        }
        else
        {
LABEL_160:
          sub_C8CC70(v130, v133, v132, v131, a5, v27);
          v23 = v207.m128i_u8[12];
          if ( !v138 )
            goto LABEL_148;
        }
        if ( v129 && v129 <= v207.m128i_i32[1] - v207.m128i_i32[2] )
          break;
        if ( v171 == v133 )
          break;
        v139 = *(_DWORD *)(v133 + 24);
        if ( v139 == 315 )
          break;
        if ( v139 != 316 )
        {
          v140 = *(__int64 **)(v133 + 40);
          v141 = &v140[5 * *(unsigned int *)(v133 + 64)];
          v128 = v201.m128i_u32[2];
          if ( v140 != v141 )
          {
            v142 = *(__int64 **)(v133 + 40);
            do
            {
              a5 = v128 + 1;
              v143 = *v142;
              if ( v128 + 1 > (unsigned __int64)v201.m128i_u32[3] )
              {
                sub_C8D5F0((__int64)&v201, v202, v128 + 1, 8u, a5, v27);
                v128 = v201.m128i_u32[2];
              }
              v142 += 5;
              *(_QWORD *)(v201.m128i_i64[0] + 8 * v128) = v143;
              v128 = (unsigned int)++v201.m128i_i32[2];
            }
            while ( v141 != v142 );
          }
          goto LABEL_149;
        }
        v144 = (unsigned int)v204;
        v145 = (unsigned int)v204 + 1LL;
        if ( v145 > HIDWORD(v204) )
        {
          sub_C8D5F0((__int64)&v203, v205, v145, 8u, a5, v27);
          v144 = (unsigned int)v204;
        }
        v203[v144] = v133;
        LODWORD(v204) = v204 + 1;
LABEL_148:
        LODWORD(v128) = v201.m128i_i32[2];
LABEL_149:
        if ( !(_DWORD)v128 )
        {
          v29 = v153;
          v28 = v148;
          v6 = v146;
          goto LABEL_151;
        }
      }
      v29 = v153;
      v28 = v148;
      v136 = v173;
      v6 = v146;
LABEL_152:
      if ( !(_BYTE)v23 )
        _libc_free(v206.m128i_u64[1]);
      if ( v203 != (__int64 *)v205 )
        _libc_free((unsigned __int64)v203);
      if ( (_QWORD *)v201.m128i_i64[0] != v202 )
        _libc_free(v201.m128i_u64[0]);
      if ( v136 )
      {
        *(_QWORD *)&v195[8 * v150] = v29;
        v137 = *(_QWORD *)(v29 + 40);
        v23 = *(_QWORD *)v137;
        v165 = *(_QWORD *)v137;
        v158 = *(_DWORD *)(v137 + 8);
      }
LABEL_32:
      v28 = *(_QWORD *)(v28 + 32);
      if ( !v28 )
      {
        a4 = v155;
        break;
      }
    }
  }
  v190 = 0;
  v191 = 0;
  v192 = 0;
  v34 = *(const __m128i **)(v171 + 40);
  v35 = (const __m128i *)((char *)v34 + 40 * *(unsigned int *)(v171 + 64));
  if ( v34 != v35 )
  {
    v156 = a4;
    v36 = *(const __m128i **)(v171 + 40);
    do
    {
      while ( 1 )
      {
        v38 = (unsigned __int16 *)(*(_QWORD *)(v36->m128i_i64[0] + 48) + 16LL * v36->m128i_u32[2]);
        v39 = *v38;
        v40 = *((_QWORD *)v38 + 1);
        LOWORD(v203) = v39;
        v204 = v40;
        v41 = sub_3007410((__int64)&v203, v167, v39, v23, a5, v27);
        v42 = _mm_loadu_si128(v36);
        v37 = v191;
        v206.m128i_i64[0] = 0;
        v207.m128i_i64[1] = v41;
        v184 = v42;
        v208 = 0u;
        v206.m128i_i64[1] = v42.m128i_i64[0];
        v207.m128i_i32[0] = v42.m128i_i32[2];
        if ( v191 != v192 )
          break;
        v36 = (const __m128i *)((char *)v36 + 40);
        sub_332CDC0(&v190, v191, (const __m128i *)v170.m128i_i64[0]);
        if ( v35 == v36 )
          goto LABEL_41;
      }
      if ( v191 )
      {
        *v191 = _mm_load_si128(&v206);
        v37[1] = _mm_load_si128(&v207);
        v37[2] = _mm_load_si128(&v208);
        v37 = v191;
      }
      v36 = (const __m128i *)((char *)v36 + 40);
      v191 = v37 + 3;
    }
    while ( v35 != v36 );
LABEL_41:
    a4 = v156;
  }
  v203 = (__int64 *)v205;
  v204 = 0x200000000LL;
  if ( v161 )
  {
    v43 = (__int64 *)v205;
    v44 = (__int64 *)v205;
    if ( v161 > 2uLL )
    {
      sub_C8D5F0((__int64)&v203, v205, v161, 0x10u, a5, v27);
      v43 = v203;
      v44 = &v203[2 * (unsigned int)v204];
    }
    for ( k = &v43[2 * v161]; k != v44; v44 += 2 )
    {
      if ( v44 )
      {
        *v44 = 0;
        *((_DWORD *)v44 + 2) = 0;
      }
    }
    LODWORD(v204) = v161;
  }
  v46 = 0;
  v47 = sub_BCE3C0(v167, 0);
  v49 = (unsigned __int64)v195;
  v50 = v47;
  v51 = &v195[8 * (unsigned int)v196];
  if ( v51 != v195 )
  {
    v164 = v6;
    v52 = v174;
    v157 = a4;
    v53 = v171;
    v54 = v47;
    while ( 1 )
    {
      while ( v52 && v172 == v46 )
      {
LABEL_57:
        v49 += 8LL;
        ++v46;
        if ( v51 == (_BYTE *)v49 )
          goto LABEL_63;
      }
      if ( *(_QWORD *)v49 )
      {
        v55 = *(_QWORD *)(*(_QWORD *)v49 + 40LL);
        v56 = *(_QWORD *)(v55 + 80);
        v57 = *(_QWORD *)(v55 + 88);
      }
      else
      {
        v162 = v52;
        v60 = sub_33EDFE0(
                v164,
                *(unsigned __int16 *)(*(_QWORD *)(v53 + 48) + 16LL * (unsigned int)v46),
                *(_QWORD *)(*(_QWORD *)(v53 + 48) + 16LL * (unsigned int)v46 + 8),
                1,
                v48,
                v50);
        v52 = v162;
        v57 = v61;
        v56 = (__int64)v60;
      }
      v183 = v57;
      v58 = &v203[2 * v46];
      v182 = v56;
      *v58 = v56;
      *((_DWORD *)v58 + 2) = v183;
      v59 = v191;
      v181 = v57;
      v206.m128i_i64[0] = 0;
      v208 = 0u;
      v207.m128i_i64[1] = v54;
      v180 = v56;
      v206.m128i_i64[1] = v56;
      v207.m128i_i32[0] = v57;
      if ( v191 != v192 )
      {
        if ( v191 )
        {
          *v191 = _mm_load_si128(&v206);
          v59[1] = _mm_load_si128(&v207);
          v59[2] = _mm_load_si128(&v208);
          v59 = v191;
        }
        v191 = v59 + 3;
        goto LABEL_57;
      }
      v49 += 8LL;
      ++v46;
      v163 = v52;
      sub_332CDC0(&v190, v191, (const __m128i *)v170.m128i_i64[0]);
      v52 = v163;
      if ( v51 == (_BYTE *)v49 )
      {
LABEL_63:
        v6 = v164;
        a4 = v157;
        break;
      }
    }
  }
  v62 = *(_QWORD *)(v171 + 80);
  v188 = v62;
  if ( v62 )
    sub_B96E90((__int64)&v188, v62, 1);
  v189 = *(_DWORD *)(v171 + 72);
  if ( v159 && *(_BYTE *)(v159 + 40) )
  {
    v101 = *(_QWORD *)(v6 + 16);
    v102 = *(__int64 (__fastcall **)(__int64, __int64, __int64 *, _QWORD, _QWORD))(*(_QWORD *)v101 + 528LL);
    v103 = sub_2E79000(*(__int64 **)(v6 + 40));
    v104 = v102(v101, v103, v167, (unsigned int)v187, *((_QWORD *)&v187 + 1));
    v201.m128i_i64[1] = v105;
    v201.m128i_i32[0] = v104;
    v107 = sub_3401740(v6, 1, (__int64)&v188, v104, v105, v106, v187);
    v109 = v108;
    v113 = sub_3007410((__int64)&v201, v167, v108, v110, v111, v112);
    v179 = v109;
    v114 = v191;
    v207.m128i_i64[1] = v113;
    v206.m128i_i64[0] = 0;
    v207.m128i_i32[0] = v109;
    v208 = 0u;
    v178 = v107;
    v206.m128i_i64[1] = (__int64)v107;
    if ( v191 == v192 )
    {
      sub_332CDC0(&v190, v191, (const __m128i *)v170.m128i_i64[0]);
    }
    else
    {
      if ( v191 )
      {
        *v191 = _mm_load_si128(&v206);
        v114[1] = _mm_load_si128(&v207);
        v114[2] = _mm_load_si128(&v208);
        v114 = v191;
      }
      v191 = v114 + 3;
    }
  }
  if ( !v174 )
  {
    v66 = sub_BCB120(v167);
    if ( v165 )
      goto LABEL_70;
LABEL_124:
    v158 = 0;
    v165 = v6 + 288;
    goto LABEL_70;
  }
  v63 = (unsigned __int16 *)(*(_QWORD *)(v171 + 48) + 16LL * v172);
  v64 = *v63;
  v65 = *((_QWORD *)v63 + 1);
  v206.m128i_i16[0] = v64;
  v206.m128i_i64[1] = v65;
  v66 = sub_3007410(v170.m128i_i64[0], v167, v64, v171, v48, v50);
  if ( !v165 )
    goto LABEL_124;
LABEL_70:
  v67 = *(_QWORD *)(v6 + 16);
  v68 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v67 + 32LL);
  v69 = sub_2E79000(*(__int64 **)(v6 + 40));
  if ( v68 == sub_2D42F30 )
  {
    v70 = sub_AE2980(v69, 0)[1];
    v71 = 2;
    if ( v70 != 1 )
    {
      v71 = 3;
      if ( v70 != 2 )
      {
        v71 = 4;
        if ( v70 != 4 )
        {
          v71 = 5;
          if ( v70 != 8 )
          {
            v71 = 6;
            if ( v70 != 16 )
            {
              v71 = 7;
              if ( v70 != 32 )
              {
                v71 = 8;
                if ( v70 != 64 )
                  v71 = 9 * (v70 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v71 = v68(v67, v69, 0);
  }
  if ( v159 )
    s = *(char **)(v159 + 16);
  v72 = sub_33EED90(v6, s, v71, 0);
  v206 = 0u;
  v73 = v72;
  v232 = 0;
  v75 = v74;
  v207.m128i_i64[1] = 0xFFFFFFFF00000020LL;
  v217 = v219;
  v218 = 0x2000000000LL;
  v221 = 0x2000000000LL;
  v224 = 0x2000000000LL;
  v226 = v228;
  v227 = 0x400000000LL;
  v76 = v188;
  v220 = v222;
  v207.m128i_i64[0] = 0;
  v208 = 0u;
  v209 = 0;
  v210 = 0;
  v211 = 0;
  v212 = 0;
  v213 = v6;
  v215 = 0;
  v216 = 0;
  v223 = v225;
  v229 = 0;
  v230 = 0;
  v231 = 0;
  v214 = v188;
  if ( v188 )
  {
    v168 = v73;
    sub_B96E90((__int64)&v214, v188, 1);
    v76 = v210;
    v73 = v168;
  }
  v206.m128i_i64[0] = v165;
  v215 = v189;
  v77 = *(_QWORD *)(v6 + 16);
  v206.m128i_i32[2] = v158;
  v78 = *(_DWORD *)(v77 + 4 * v160 + 531128);
  v176 = v73;
  v177 = v75;
  v208.m128i_i64[1] = v73;
  LODWORD(v209) = v75;
  v207.m128i_i64[0] = v66;
  v208.m128i_i32[0] = v78;
  v210 = v190;
  LODWORD(v77) = -1431655765 * ((__int64)((__int64)v191->m128i_i64 - v190) >> 4);
  v211 = v191;
  v190 = 0;
  v191 = 0;
  v207.m128i_i32[3] = v77;
  v79 = v192;
  v192 = 0;
  v212 = v79;
  if ( v76 )
    j_j___libc_free_0(v76);
  v80 = *(void (****)())(v213 + 16);
  v81 = **v80;
  if ( v81 != nullsub_1688 )
    ((void (__fastcall *)(void (***)(), _QWORD, _QWORD, unsigned __int64 *))v81)(
      v80,
      *(_QWORD *)(v213 + 40),
      v78,
      &v210);
  v82 = 0;
  sub_3377410((__int64)&v198, *(_WORD **)(v6 + 16), v170.m128i_i64[0]);
  v85 = v203;
  HIWORD(v86) = v151;
  v169 = &v203[2 * (unsigned int)v204];
  if ( v169 != v203 )
  {
    do
    {
      if ( v174 && v82 == v172 )
      {
        v99 = *(unsigned int *)(a4 + 8);
        v100 = _mm_load_si128(&v198);
        if ( v99 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
        {
          v170 = v100;
          sub_C8D5F0(a4, (const void *)(a4 + 16), v99 + 1, 0x10u, (__int64)v83, v84);
          v99 = *(unsigned int *)(a4 + 8);
          v100 = _mm_load_si128(&v170);
        }
        *(__m128i *)(*(_QWORD *)a4 + 16 * v99) = v100;
        ++*(_DWORD *)(a4 + 8);
      }
      else
      {
        v193 = 0u;
        v95 = *(_QWORD *)&v195[8 * v82];
        BYTE4(v194) = 0;
        LODWORD(v194) = 0;
        if ( v95 )
        {
          v170.m128i_i64[0] = v95;
          sub_34161C0(v6, v95, 0, v199, v200);
          v87 = &v201;
          v88 = *(const __m128i **)(v170.m128i_i64[0] + 112);
          v89 = _mm_loadu_si128(v88);
          v90 = v88[1].m128i_i8[4];
          LODWORD(v194) = v88[1].m128i_i32[0];
          BYTE4(v194) = v90;
          v193 = (__int128)v89;
        }
        else
        {
          v96 = *v85;
          v97 = *(_QWORD *)(v6 + 40);
          v170.m128i_i64[0] = (__int64)&v201;
          sub_2EAC300((__int64)&v201, v97, *(_DWORD *)(v96 + 96), 0);
          v98 = _mm_load_si128(&v201);
          v87 = (__m128i *)v170.m128i_i64[0];
          LODWORD(v194) = v202[0];
          v193 = (__int128)v98;
          BYTE4(v194) = BYTE4(v202[0]);
        }
        v201 = 0u;
        v202[0] = 0;
        v175 = (unsigned __int8)v175;
        v202[1] = 0;
        v91 = *(_QWORD *)(v171 + 48) + 16LL * (unsigned int)v82;
        LOWORD(v86) = *(_WORD *)v91;
        v83 = sub_33F1F00(
                (__int64 *)v6,
                v86,
                *(_QWORD *)(v91 + 8),
                (__int64)&v188,
                v199,
                v200,
                *v85,
                v85[1],
                v193,
                v194,
                (unsigned __int8)v175,
                0,
                (__int64)v87,
                0);
        v93 = *(unsigned int *)(a4 + 8);
        v84 = v92;
        if ( v93 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
        {
          v170.m128i_i64[0] = (__int64)v83;
          v170.m128i_i64[1] = v92;
          sub_C8D5F0(a4, (const void *)(a4 + 16), v93 + 1, 0x10u, (__int64)v83, v92);
          v93 = *(unsigned int *)(a4 + 8);
          v84 = v170.m128i_i64[1];
          v83 = (__m128i *)v170.m128i_i64[0];
        }
        v94 = (__m128i **)(*(_QWORD *)a4 + 16 * v93);
        *v94 = v83;
        v94[1] = (__m128i *)v84;
        ++*(_DWORD *)(a4 + 8);
      }
      ++v82;
      v85 += 2;
    }
    while ( v169 != v85 );
  }
  if ( v226 != v228 )
    _libc_free((unsigned __int64)v226);
  if ( v223 != v225 )
    _libc_free((unsigned __int64)v223);
  if ( v220 != v222 )
    _libc_free((unsigned __int64)v220);
  if ( v217 != v219 )
    _libc_free((unsigned __int64)v217);
  if ( v214 )
    sub_B91220((__int64)&v214, v214);
  if ( v210 )
    j_j___libc_free_0(v210);
  if ( v188 )
    sub_B91220((__int64)&v188, v188);
  if ( v203 != (__int64 *)v205 )
    _libc_free((unsigned __int64)v203);
  if ( v190 )
    j_j___libc_free_0(v190);
  if ( v195 != v197 )
    _libc_free((unsigned __int64)v195);
  return 1;
}
