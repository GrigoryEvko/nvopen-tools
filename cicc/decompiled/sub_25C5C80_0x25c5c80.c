// Function: sub_25C5C80
// Address: 0x25c5c80
//
__int64 __fastcall sub_25C5C80(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // eax
  unsigned __int64 i; // rcx
  __int64 v5; // r8
  unsigned __int64 v6; // r9
  _BYTE *v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rax
  unsigned __int64 v10; // r12
  unsigned __int64 v11; // rdx
  __int64 v12; // rax
  const __m128i *v13; // r13
  __m128i *v14; // rax
  __int64 *v15; // rdi
  __int64 *v16; // r13
  __int64 v17; // rdx
  __int64 v18; // rdx
  unsigned __int64 v19; // r12
  __int64 v20; // r15
  char v21; // r14
  unsigned __int8 *v22; // rbx
  int v23; // edx
  __int64 v24; // rax
  _QWORD *v25; // rdi
  int v26; // esi
  unsigned int v27; // edx
  __int64 *v28; // r12
  __int64 v29; // r8
  __int64 v30; // r12
  __int64 v31; // rdx
  char v32; // si
  __int64 v33; // r8
  unsigned int v34; // edi
  __int64 *v35; // rax
  __int64 v36; // r9
  __int64 *v37; // r15
  _QWORD *v38; // r14
  __int64 v39; // rdi
  bool v40; // r15
  __int64 v41; // r14
  unsigned __int64 v42; // r12
  unsigned __int64 v43; // rdi
  __int64 v44; // r12
  __int8 *v45; // r14
  __int64 v46; // rax
  unsigned __int64 v47; // rbx
  __int64 v48; // rdx
  __int64 v49; // rdx
  unsigned __int64 v50; // r15
  unsigned __int32 v51; // ecx
  __int64 v52; // rax
  __int64 v53; // r12
  __int64 v54; // r14
  unsigned int v55; // eax
  unsigned int v56; // eax
  __int64 v57; // rbx
  unsigned __int64 v58; // rsi
  unsigned __int64 v59; // rbx
  unsigned __int64 v60; // r12
  __int8 v61; // bl
  __m128i *v62; // rax
  const __m128i *v63; // rdx
  __m128i v64; // xmm1
  signed __int64 v65; // r14
  unsigned int v66; // esi
  int v67; // ecx
  int v68; // ecx
  int v69; // eax
  unsigned int v70; // edi
  __int64 v71; // r9
  unsigned __int64 v72; // r10
  int v73; // r14d
  unsigned int v74; // r8d
  unsigned int v75; // eax
  int v76; // edx
  unsigned int v77; // esi
  _QWORD *v78; // rdx
  unsigned __int64 v79; // rdi
  __int64 v80; // rax
  __int64 v81; // r14
  const void **v82; // r15
  unsigned __int64 v83; // rbx
  __int64 v84; // r13
  __int64 v85; // r14
  _QWORD *v86; // rax
  __int64 v87; // rax
  __int64 v88; // rax
  _QWORD *v89; // rbx
  _QWORD *v90; // r12
  _QWORD *v91; // r13
  __int64 v92; // rsi
  _BYTE *v93; // rbx
  __int64 v94; // rdx
  _QWORD *v95; // r12
  __int64 v96; // rsi
  _QWORD *v97; // r14
  unsigned int v99; // edx
  unsigned int v100; // edi
  unsigned int v101; // r8d
  unsigned __int64 v102; // rsi
  _QWORD *v103; // rax
  char *v104; // r13
  __int64 v105; // rax
  char v106; // si
  __int64 v107; // rax
  unsigned int v108; // edx
  unsigned __int64 v109; // rax
  __int64 v110; // rdi
  __int64 v111; // rdx
  unsigned __int64 v112; // rax
  __int64 v113; // rbx
  int v114; // eax
  __int64 v115; // rdx
  __int64 v116; // rcx
  __int64 v117; // r8
  __int64 v118; // r9
  __int64 v119; // rcx
  __int64 v120; // r8
  __int64 v121; // r9
  __int64 v122; // rdx
  __int64 v123; // r8
  __int64 v124; // r9
  __int64 v125; // rcx
  __int64 v126; // rax
  __int64 v127; // rdx
  unsigned __int64 v128; // rax
  bool v129; // zf
  int v130; // r10d
  int v131; // ecx
  unsigned __int64 v132; // r9
  int v133; // r10d
  int v134; // esi
  __int64 v135; // rcx
  int v136; // r11d
  unsigned int v137; // edx
  __int64 *v138; // rdi
  __int64 *v139; // rax
  __int64 v140; // r8
  __int64 v141; // rdi
  unsigned int v142; // edx
  __int64 *v143; // rcx
  __int64 v144; // rdi
  int v145; // ecx
  __int64 v146; // rdx
  __int64 v147; // rsi
  int v148; // eax
  __int64 v149; // rax
  bool v150; // al
  unsigned __int8 *v151; // rax
  unsigned __int64 v152; // rdx
  __int64 v153; // rax
  __int64 v154; // r12
  __int64 v155; // rax
  unsigned __int64 v156; // rax
  __int64 v157; // rdx
  unsigned __int64 v158; // r13
  int v159; // eax
  int v160; // eax
  int v161; // edi
  __int64 v162; // rax
  unsigned int v163; // r12d
  bool v164; // al
  int v165; // ecx
  int v166; // r9d
  __int64 v167; // rax
  int v168; // eax
  __int64 *v169; // [rsp+0h] [rbp-2580h]
  unsigned __int64 v170; // [rsp+18h] [rbp-2568h]
  bool v171; // [rsp+28h] [rbp-2558h]
  int v172; // [rsp+30h] [rbp-2550h]
  unsigned int v173; // [rsp+40h] [rbp-2540h]
  __int64 v176; // [rsp+58h] [rbp-2528h]
  __int64 v177; // [rsp+68h] [rbp-2518h]
  unsigned __int8 v178; // [rsp+70h] [rbp-2510h]
  __int64 v179; // [rsp+88h] [rbp-24F8h] BYREF
  __int64 *v180; // [rsp+90h] [rbp-24F0h] BYREF
  __int64 *v181; // [rsp+98h] [rbp-24E8h]
  __int64 *v182; // [rsp+A0h] [rbp-24E0h] BYREF
  _QWORD *v183; // [rsp+A8h] [rbp-24D8h]
  __int64 v184; // [rsp+B0h] [rbp-24D0h]
  unsigned int v185; // [rsp+B8h] [rbp-24C8h]
  const void **v186; // [rsp+C0h] [rbp-24C0h] BYREF
  __int64 v187; // [rsp+C8h] [rbp-24B8h]
  char v188; // [rsp+D0h] [rbp-24B0h] BYREF
  __int64 v189[54]; // [rsp+110h] [rbp-2470h] BYREF
  __int64 v190; // [rsp+2C0h] [rbp-22C0h] BYREF
  unsigned __int64 v191; // [rsp+2C8h] [rbp-22B8h]
  _DWORD v192[3]; // [rsp+2D0h] [rbp-22B0h] BYREF
  char v193; // [rsp+2DCh] [rbp-22A4h]
  __int64 v194; // [rsp+2E0h] [rbp-22A0h] BYREF
  __int64 *v195; // [rsp+320h] [rbp-2260h]
  __int64 v196; // [rsp+328h] [rbp-2258h]
  __int64 v197; // [rsp+330h] [rbp-2250h] BYREF
  int v198; // [rsp+338h] [rbp-2248h]
  __int64 v199; // [rsp+340h] [rbp-2240h]
  int v200; // [rsp+348h] [rbp-2238h]
  __int64 v201; // [rsp+350h] [rbp-2230h]
  __m128i v202; // [rsp+470h] [rbp-2110h] BYREF
  unsigned __int64 v203; // [rsp+480h] [rbp-2100h] BYREF
  unsigned int v204; // [rsp+488h] [rbp-20F8h] BYREF
  char v205; // [rsp+48Ch] [rbp-20F4h]
  char v206[64]; // [rsp+490h] [rbp-20F0h] BYREF
  _BYTE *v207; // [rsp+4D0h] [rbp-20B0h] BYREF
  __int64 v208; // [rsp+4D8h] [rbp-20A8h]
  _BYTE v209[320]; // [rsp+4E0h] [rbp-20A0h] BYREF
  unsigned __int64 v210; // [rsp+620h] [rbp-1F60h] BYREF
  __m128i v211; // [rsp+628h] [rbp-1F58h] BYREF
  unsigned int v212; // [rsp+638h] [rbp-1F48h] BYREF
  char v213; // [rsp+63Ch] [rbp-1F44h]
  char v214[64]; // [rsp+640h] [rbp-1F40h] BYREF
  _BYTE *v215; // [rsp+680h] [rbp-1F00h] BYREF
  __int64 v216; // [rsp+688h] [rbp-1EF8h]
  _BYTE v217[320]; // [rsp+690h] [rbp-1EF0h] BYREF
  unsigned __int64 v218; // [rsp+7D0h] [rbp-1DB0h] BYREF
  unsigned __int64 v219; // [rsp+7D8h] [rbp-1DA8h]
  _BYTE v220[80]; // [rsp+7E0h] [rbp-1DA0h] BYREF
  char *v221; // [rsp+830h] [rbp-1D50h] BYREF
  unsigned int v222; // [rsp+838h] [rbp-1D48h]
  char v223; // [rsp+840h] [rbp-1D40h] BYREF
  char v224[8]; // [rsp+980h] [rbp-1C00h] BYREF
  unsigned __int64 v225; // [rsp+988h] [rbp-1BF8h]
  char v226; // [rsp+99Ch] [rbp-1BE4h]
  char *v227; // [rsp+9E0h] [rbp-1BA0h] BYREF
  unsigned int v228; // [rsp+9E8h] [rbp-1B98h]
  char v229; // [rsp+9F0h] [rbp-1B90h] BYREF
  __int16 v230; // [rsp+B30h] [rbp-1A50h]
  __int64 v231; // [rsp+B38h] [rbp-1A48h] BYREF
  __int64 v232; // [rsp+B40h] [rbp-1A40h]
  _QWORD *v233; // [rsp+B48h] [rbp-1A38h] BYREF
  unsigned int v234; // [rsp+B50h] [rbp-1A30h]
  _BYTE v235[56]; // [rsp+2548h] [rbp-38h] BYREF

  v177 = *(_QWORD *)(a2 + 40) + 312LL;
  v2 = *(_QWORD *)(a1 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v2 + 8) - 17 <= 1 )
    v2 = **(_QWORD **)(v2 + 16);
  v3 = sub_AE2980(v177, *(_DWORD *)(v2 + 8) >> 8)[3];
  v230 = 0;
  v231 = 0;
  v173 = v3;
  v7 = &v233;
  v232 = 1;
  do
  {
    *(_QWORD *)v7 = -4096;
    v7 += 416;
  }
  while ( v7 != v235 );
  v8 = *(_QWORD *)(a2 + 80);
  v9 = v8 - 24;
  if ( !v8 )
    v9 = 0;
  v176 = v9;
  v218 = (unsigned __int64)v220;
  v219 = 0x400000000LL;
  v10 = *(_QWORD *)(a1 + 16);
  if ( v10 )
  {
    v11 = 4;
    v12 = 0;
    for ( i = (unsigned __int64)v220; ; i = v218 )
    {
      v5 = v12 + 1;
      v210 = v10;
      v13 = (const __m128i *)&v210;
      v211 = 0;
      v211.m128i_i8[8] = 1;
      if ( v12 + 1 > v11 )
      {
        if ( i > (unsigned __int64)&v210 || (unsigned __int64)&v210 >= i + 24 * v12 )
        {
          sub_C8D5F0((__int64)&v218, v220, v5, 0x18u, v5, v6);
          i = v218;
          v12 = (unsigned int)v219;
        }
        else
        {
          v104 = (char *)&v210 - i;
          sub_C8D5F0((__int64)&v218, v220, v5, 0x18u, v5, v6);
          i = v218;
          v12 = (unsigned int)v219;
          v13 = (const __m128i *)&v104[v218];
        }
      }
      v14 = (__m128i *)(i + 24 * v12);
      *v14 = _mm_loadu_si128(v13);
      v14[1].m128i_i64[0] = v13[1].m128i_i64[0];
      v12 = (unsigned int)(v219 + 1);
      LODWORD(v219) = v219 + 1;
      v10 = *(_QWORD *)(v10 + 8);
      if ( !v10 )
        break;
      v11 = HIDWORD(v219);
    }
    v15 = (__int64 *)v218;
    if ( (_DWORD)v12 )
    {
      v16 = (__int64 *)&v210;
      while ( 1 )
      {
        v17 = (unsigned int)v12;
        v12 = (unsigned int)(v12 - 1);
        v18 = (__int64)&v15[3 * v17 - 3];
        v19 = *(_QWORD *)v18;
        v20 = *(_QWORD *)(v18 + 8);
        v21 = *(_BYTE *)(v18 + 16);
        LODWORD(v219) = v12;
        v22 = *(unsigned __int8 **)(v19 + 24);
        v23 = *v22;
        if ( (unsigned __int8)v23 <= 0x1Cu )
        {
          if ( (_BYTE)v23 != 5 || *((_WORD *)v22 + 1) != 34 )
            goto LABEL_21;
        }
        else if ( (_BYTE)v23 != 63 )
        {
          switch ( (_BYTE)v23 )
          {
            case '>':
              if ( !sub_B46500(*(unsigned __int8 **)(v19 + 24)) && (v22[2] & 1) == 0 )
              {
                if ( (v22[7] & 0x40) != 0 )
                {
                  if ( v19 != *((_QWORD *)v22 - 1) + 32LL )
                    goto LABEL_21;
LABEL_171:
                  v210 = (unsigned __int64)&v211.m128i_u64[1];
                  v211.m128i_i64[0] = 0x200000000LL;
                  v105 = sub_B46690((__int64)v22);
                  sub_25BFA70((__int64)&v202, v177, v105, v20, v21);
                  if ( v206[0] )
                  {
                    sub_AC1580((__int64)v16, (__int64)&v202);
                    if ( v206[0] )
                    {
                      v206[0] = 0;
                      if ( v204 > 0x40 && v203 )
                        j_j___libc_free_0_0(v203);
                      if ( v202.m128i_i32[2] > 0x40u && v202.m128i_i64[0] )
                        j_j___libc_free_0_0(v202.m128i_u64[0]);
                    }
                  }
                  v202.m128i_i8[0] = 0;
                  v202.m128i_i64[1] = (__int64)&v204;
                  v203 = 0x200000000LL;
                  if ( v211.m128i_i32[0] )
                    sub_25C2C90((__int64)&v202.m128i_i64[1], v16);
LABEL_174:
                  sub_25C0430((__int64)v16);
                  goto LABEL_22;
                }
                if ( (unsigned __int8 *)v19 == &v22[-32 * (*((_DWORD *)v22 + 1) & 0x7FFFFFF) + 32] )
                  goto LABEL_171;
              }
LABEL_21:
              v202.m128i_i8[0] = 3;
              v202.m128i_i64[1] = (__int64)&v204;
              v203 = 0x200000000LL;
              goto LABEL_22;
            case '=':
              if ( !sub_B46500(*(unsigned __int8 **)(v19 + 24)) && (v22[2] & 1) == 0 )
              {
                v80 = sub_B46690((__int64)v22);
                sub_25BFA70((__int64)v16, v177, v80, v20, v21);
                if ( v214[0] )
                {
                  v202.m128i_i8[0] = 2;
                  sub_A6E600((__int64)&v202.m128i_i64[1], (unsigned __int64)v16, 1);
                  if ( v214[0] )
                  {
                    v214[0] = 0;
                    if ( v212 > 0x40 && v211.m128i_i64[1] )
                      j_j___libc_free_0_0(v211.m128i_u64[1]);
                    if ( v211.m128i_i32[0] > 0x40u && v210 )
                      j_j___libc_free_0_0(v210);
                  }
                  goto LABEL_22;
                }
              }
              goto LABEL_21;
            case 'U':
              v145 = *((_DWORD *)v22 + 1);
              v146 = *((_QWORD *)v22 - 4);
              v147 = v145 & 0x7FFFFFF;
              if ( v146 )
              {
                if ( !*(_BYTE *)v146
                  && *(_QWORD *)(v146 + 24) == *((_QWORD *)v22 + 10)
                  && (*(_BYTE *)(v146 + 33) & 0x20) != 0
                  && ((*(_DWORD *)(v146 + 36) - 243) & 0xFFFFFFFD) == 0 )
                {
                  v162 = *(_QWORD *)&v22[32 * (3 - v147)];
                  v163 = *(_DWORD *)(v162 + 32);
                  if ( v163 <= 0x40 )
                    v164 = *(_QWORD *)(v162 + 24) == 0;
                  else
                    v164 = v163 == (unsigned int)sub_C444A0(v162 + 24);
                  if ( !v164 )
                    goto LABEL_21;
                  v210 = (unsigned __int64)&v211.m128i_u64[1];
                  v211.m128i_i64[0] = 0x200000000LL;
                  sub_25BED70(
                    (__int64)&v202,
                    *(_QWORD *)&v22[32 * (2LL - (*((_DWORD *)v22 + 1) & 0x7FFFFFF))],
                    v20,
                    v21);
                  if ( v206[0] )
                  {
                    sub_AC1580((__int64)v16, (__int64)&v202);
                    if ( v206[0] )
                    {
                      v206[0] = 0;
                      if ( v204 > 0x40 && v203 )
                        j_j___libc_free_0_0(v203);
                      if ( v202.m128i_i32[2] > 0x40u && v202.m128i_i64[0] )
                        j_j___libc_free_0_0(v202.m128i_u64[0]);
                    }
                  }
                  goto LABEL_369;
                }
                if ( !*(_BYTE *)v146
                  && *(_QWORD *)(v146 + 24) == *((_QWORD *)v22 + 10)
                  && (*(_BYTE *)(v146 + 33) & 0x20) != 0 )
                {
                  v148 = *(_DWORD *)(v146 + 36);
                  if ( v148 == 238 || (unsigned int)(v148 - 240) <= 1 )
                  {
                    v149 = *(_QWORD *)&v22[32 * (3 - v147)];
                    if ( *(_DWORD *)(v149 + 32) <= 0x40u )
                    {
                      v150 = *(_QWORD *)(v149 + 24) == 0;
                    }
                    else
                    {
                      v172 = *(_DWORD *)(v149 + 32);
                      v150 = v172 == (unsigned int)sub_C444A0(v149 + 24);
                    }
                    if ( !v150 )
                      goto LABEL_21;
                    if ( (v22[7] & 0x40) != 0 )
                      v151 = (unsigned __int8 *)*((_QWORD *)v22 - 1);
                    else
                      v151 = &v22[-32 * v147];
                    if ( (unsigned __int8 *)v19 != v151 )
                    {
                      if ( (unsigned __int8 *)v19 == v151 + 32 )
                      {
                        sub_25BED70((__int64)v16, *(_QWORD *)&v22[32 * (2 - v147)], v20, v21);
                        if ( v214[0] )
                        {
                          v202.m128i_i8[0] = 2;
                          sub_A6E600((__int64)&v202.m128i_i64[1], (unsigned __int64)v16, 1);
                          if ( v214[0] )
                            sub_9963D0((__int64)v16);
                          goto LABEL_22;
                        }
                      }
                      goto LABEL_21;
                    }
                    v210 = (unsigned __int64)&v211.m128i_u64[1];
                    v211.m128i_i64[0] = 0x200000000LL;
                    sub_25BED70(
                      (__int64)&v202,
                      *(_QWORD *)&v22[32 * (2LL - (*((_DWORD *)v22 + 1) & 0x7FFFFFF))],
                      v20,
                      v21);
                    if ( v206[0] )
                    {
                      sub_AC1580((__int64)v16, (__int64)&v202);
                      if ( v206[0] )
                        sub_9963D0((__int64)&v202);
                    }
LABEL_369:
                    v202.m128i_i8[0] = 0;
                    v202.m128i_i64[1] = (__int64)&v204;
                    v203 = 0x200000000LL;
                    if ( !v211.m128i_i32[0] )
                      goto LABEL_174;
                    sub_25C2990((__int64)&v202.m128i_i64[1], (__int64)v16);
                    sub_25C0430((__int64)v16);
LABEL_22:
                    v189[0] = (__int64)v22;
                    v24 = *((_QWORD *)v22 + 5);
                    v190 = v24;
                    if ( (v232 & 1) != 0 )
                    {
                      v25 = &v233;
                      v26 = 15;
                      goto LABEL_24;
                    }
                    v66 = v234;
                    v25 = v233;
                    if ( !v234 )
                    {
                      v99 = v232;
                      ++v231;
                      v210 = 0;
                      v100 = ((unsigned int)v232 >> 1) + 1;
                      goto LABEL_151;
                    }
                    v26 = v234 - 1;
LABEL_24:
                    v27 = v26 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
                    v28 = &v25[52 * v27];
                    v29 = *v28;
                    if ( v24 == *v28 )
                    {
LABEL_25:
                      v30 = (__int64)(v28 + 1);
                      v31 = (__int64)v22;
                      goto LABEL_26;
                    }
                    v132 = 0;
                    v133 = 1;
                    while ( v29 != -4096 )
                    {
                      if ( v29 == -8192 && !v132 )
                        v132 = (unsigned __int64)v28;
                      v27 = v26 & (v133 + v27);
                      v28 = &v25[52 * v27];
                      v29 = *v28;
                      if ( v24 == *v28 )
                        goto LABEL_25;
                      ++v133;
                    }
                    v99 = v232;
                    v101 = 48;
                    v66 = 16;
                    if ( !v132 )
                      v132 = (unsigned __int64)v28;
                    ++v231;
                    v210 = v132;
                    v100 = ((unsigned int)v232 >> 1) + 1;
                    if ( (v232 & 1) == 0 )
                    {
                      v66 = v234;
LABEL_151:
                      v101 = 3 * v66;
                    }
                    if ( 4 * v100 >= v101 )
                    {
                      v66 *= 2;
                    }
                    else if ( v66 - HIDWORD(v232) - v100 > v66 >> 3 )
                    {
LABEL_154:
                      v102 = v210;
                      LODWORD(v232) = (2 * (v99 >> 1) + 2) | v99 & 1;
                      if ( *(_QWORD *)v210 != -4096 )
                        --HIDWORD(v232);
                      v30 = v210 + 8;
                      *(_QWORD *)v210 = v24;
                      memset((void *)(v102 + 8), 0, 0x198u);
                      *(_BYTE *)(v102 + 16) = 1;
                      v103 = (_QWORD *)(v102 + 24);
                      do
                      {
                        if ( v103 )
                          *v103 = -4096;
                        v103 += 12;
                      }
                      while ( v103 != (_QWORD *)(v102 + 408) );
                      *(_WORD *)(v102 + 408) = 0;
                      v31 = v189[0];
LABEL_26:
                      v32 = *(_BYTE *)(v30 + 8) & 1;
                      if ( v32 )
                      {
                        v33 = v30 + 16;
                        v34 = ((unsigned __int8)((unsigned int)v31 >> 4)
                             ^ (unsigned __int8)((unsigned int)v31 >> 9))
                            & 3;
                        v35 = (__int64 *)(v30
                                        + 16
                                        + 96LL
                                        * (((unsigned __int8)((unsigned int)v31 >> 4)
                                          ^ (unsigned __int8)((unsigned int)v31 >> 9))
                                         & 3));
                        v36 = *v35;
                        v37 = v35;
                        if ( v31 == *v35 )
                          goto LABEL_28;
                        v69 = 3;
LABEL_228:
                        v130 = 1;
                        while ( v36 != -4096 )
                        {
                          v34 = v69 & (v130 + v34);
                          v37 = (__int64 *)(v33 + 96LL * v34);
                          v36 = *v37;
                          if ( v31 == *v37 )
                            goto LABEL_231;
                          ++v130;
                        }
                        v37 = 0;
LABEL_231:
                        if ( v32 )
                        {
                          v68 = 3;
                          v70 = ((unsigned __int8)((unsigned int)v31 >> 4)
                               ^ (unsigned __int8)((unsigned int)v31 >> 9))
                              & 3;
                          v35 = (__int64 *)(v33
                                          + 96LL
                                          * (((unsigned __int8)((unsigned int)v31 >> 4)
                                            ^ (unsigned __int8)((unsigned int)v31 >> 9))
                                           & 3));
                          v71 = *v35;
LABEL_85:
                          if ( v31 == v71 )
                          {
LABEL_28:
                            v38 = v35 + 1;
                            v39 = (__int64)(v35 + 2);
                            if ( !v37 )
                              goto LABEL_96;
                          }
                          else
                          {
                            v72 = 0;
                            v73 = 1;
                            while ( v71 != -4096 )
                            {
                              if ( v71 == -8192 && !v72 )
                                v72 = (unsigned __int64)v35;
                              v70 = v68 & (v73 + v70);
                              v35 = (__int64 *)(v33 + 96LL * v70);
                              v71 = *v35;
                              if ( v31 == *v35 )
                                goto LABEL_28;
                              ++v73;
                            }
                            v74 = 4;
                            if ( !v72 )
                              v72 = (unsigned __int64)v35;
                            v210 = v72;
                            v75 = *(_DWORD *)(v30 + 8);
                            ++*(_QWORD *)v30;
                            v76 = (v75 >> 1) + 1;
                            if ( v32 )
                            {
                              v77 = 8;
                              if ( (unsigned int)(4 * v76) < 0xC )
                                goto LABEL_92;
LABEL_221:
                              sub_25C3910(v30, v77);
                              sub_25BC970(v30, v189, v16);
                              v75 = *(_DWORD *)(v30 + 8);
                              goto LABEL_93;
                            }
                            v74 = *(_DWORD *)(v30 + 24);
LABEL_220:
                            v77 = 2 * v74;
                            if ( 3 * v74 <= 4 * v76 )
                              goto LABEL_221;
LABEL_92:
                            if ( v74 - *(_DWORD *)(v30 + 12) - v76 <= v74 >> 3 )
                            {
                              v77 = v74;
                              goto LABEL_221;
                            }
LABEL_93:
                            *(_DWORD *)(v30 + 8) = (2 * (v75 >> 1) + 2) | v75 & 1;
                            v78 = (_QWORD *)v210;
                            if ( *(_QWORD *)v210 != -4096 )
                              --*(_DWORD *)(v30 + 12);
                            v38 = v78 + 1;
                            *v78 = v189[0];
                            memset(v78 + 1, 0, 0x58u);
                            v39 = (__int64)(v78 + 2);
                            v78[2] = v78 + 4;
                            v78[3] = 0x200000000LL;
                            if ( !v37 )
                            {
LABEL_96:
                              *(_BYTE *)v38 = v202.m128i_i8[0];
                              sub_25C2C90(v39, &v202.m128i_i64[1]);
                              v40 = 0;
                              *(_BYTE *)(v30 + 401) |= *(_BYTE *)v38 == 3;
                              if ( *(_BYTE *)v38 <= 1u )
                                v40 = *((_DWORD *)v38 + 4) != 0;
                              *(_BYTE *)(v30 + 400) |= v40;
                              goto LABEL_30;
                            }
                          }
                          LOBYTE(v210) = 3;
                          v211.m128i_i64[0] = (__int64)&v212;
                          v211.m128i_i64[1] = 0x200000000LL;
                          *(_BYTE *)v38 = 3;
                          sub_25C2C90(v39, v211.m128i_i64);
                          v40 = 0;
                          sub_25C0430((__int64)&v211);
                          *(_BYTE *)(v30 + 401) = 1;
LABEL_30:
                          v41 = v202.m128i_i64[1];
                          v42 = v202.m128i_i64[1] + 32LL * (unsigned int)v203;
                          if ( v202.m128i_i64[1] != v42 )
                          {
                            do
                            {
                              v42 -= 32LL;
                              if ( *(_DWORD *)(v42 + 24) > 0x40u )
                              {
                                v43 = *(_QWORD *)(v42 + 16);
                                if ( v43 )
                                  j_j___libc_free_0_0(v43);
                              }
                              if ( *(_DWORD *)(v42 + 8) > 0x40u && *(_QWORD *)v42 )
                                j_j___libc_free_0_0(*(_QWORD *)v42);
                            }
                            while ( v41 != v42 );
                            v42 = v202.m128i_u64[1];
                          }
                          if ( (unsigned int *)v42 != &v204 )
                            _libc_free(v42);
                          LOBYTE(v230) = v40 | v230;
                          if ( v40 && *((_QWORD *)v22 + 5) != v176 )
                            HIBYTE(v230) = 1;
                          LODWORD(v12) = v219;
                          goto LABEL_45;
                        }
                        v131 = *(_DWORD *)(v30 + 24);
                        if ( v131 )
                        {
                          v68 = v131 - 1;
LABEL_84:
                          v70 = v68 & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
                          v35 = (__int64 *)(v33 + 96LL * v70);
                          v71 = *v35;
                          goto LABEL_85;
                        }
                      }
                      else
                      {
                        v67 = *(_DWORD *)(v30 + 24);
                        if ( v67 )
                        {
                          v68 = v67 - 1;
                          v33 = *(_QWORD *)(v30 + 16);
                          v69 = v68;
                          v34 = v68 & (((unsigned int)v31 >> 4) ^ ((unsigned int)v31 >> 9));
                          v37 = (__int64 *)(v33 + 96LL * v34);
                          v36 = *v37;
                          if ( v31 != *v37 )
                            goto LABEL_228;
                          goto LABEL_84;
                        }
                        v37 = 0;
                      }
                      v210 = 0;
                      v75 = *(_DWORD *)(v30 + 8);
                      v74 = 0;
                      ++*(_QWORD *)v30;
                      v76 = (v75 >> 1) + 1;
                      goto LABEL_220;
                    }
                    sub_25C44B0((__int64)&v231, v66);
                    sub_25BC8A0((__int64)&v231, &v190, v16);
                    v24 = v190;
                    v99 = v232;
                    goto LABEL_154;
                  }
                }
              }
              break;
            default:
              v152 = (unsigned int)(v23 - 34);
              if ( (unsigned __int8)v152 > 0x33u )
                goto LABEL_21;
              v153 = 0x8000000000041LL;
              if ( !_bittest64(&v153, v152) )
                goto LABEL_21;
              v145 = *((_DWORD *)v22 + 1);
              break;
          }
          if ( v19 >= (unsigned __int64)&v22[-32 * (v145 & 0x7FFFFFF)]
            && v19 < (unsigned __int64)sub_24E54B0(*(unsigned __int8 **)(v19 + 24))
            && !(unsigned __int8)sub_B49B80(
                                   (__int64)v22,
                                   (__int64)(v19 - (_QWORD)&v22[-32 * (*((_DWORD *)v22 + 1) & 0x7FFFFFF)]) >> 5,
                                   81) )
          {
            v154 = (__int64)(v19 - (_QWORD)&v22[-32 * (*((_DWORD *)v22 + 1) & 0x7FFFFFF)]) >> 5;
            if ( (unsigned __int8)sub_B49B80((__int64)v22, v154, 98) )
            {
              v171 = v21;
              if ( v21 )
              {
                if ( sub_CF49B0(v22, v154, 78) || sub_CF49B0(v22, v154, 50) )
                  v171 = sub_B49EE0(v22, v154) != 0;
                v190 = (__int64)v192;
                v191 = 0x200000000LL;
                v210 = *((_QWORD *)v22 + 9);
                v155 = sub_A747F0(v16, (int)v154 + 1, 98);
                if ( !v155 )
                  v155 = sub_B49640((__int64)v22, v154, 98);
                v179 = v155;
                v156 = sub_A72AC0(&v179);
                sub_A6E600((__int64)v16, v156, v157);
                v170 = v210 + 32LL * v211.m128i_u32[0];
                if ( v210 != v170 )
                {
                  v169 = v16;
                  v158 = v210;
                  do
                  {
                    LODWORD(v187) = *(_DWORD *)(v158 + 24);
                    if ( (unsigned int)v187 <= 0x40 )
                      v186 = *(const void ***)(v158 + 16);
                    else
                      sub_C43780((__int64)&v186, (const void **)(v158 + 16));
                    sub_C46A40((__int64)&v186, v20);
                    v159 = v187;
                    LODWORD(v187) = 0;
                    LODWORD(v189[1]) = v159;
                    v189[0] = (__int64)v186;
                    LODWORD(v181) = *(_DWORD *)(v158 + 8);
                    if ( (unsigned int)v181 > 0x40 )
                      sub_C43780((__int64)&v180, (const void **)v158);
                    else
                      v180 = *(__int64 **)v158;
                    v158 += 32LL;
                    sub_C46A40((__int64)&v180, v20);
                    v160 = (int)v181;
                    LODWORD(v181) = 0;
                    LODWORD(v183) = v160;
                    v182 = v180;
                    sub_AADC30((__int64)&v202, (__int64)&v182, v189);
                    sub_AC1580((__int64)&v190, (__int64)&v202);
                    sub_969240((__int64 *)&v203);
                    sub_969240(v202.m128i_i64);
                    sub_969240((__int64 *)&v182);
                    sub_969240((__int64 *)&v180);
                    sub_969240(v189);
                    sub_969240((__int64 *)&v186);
                  }
                  while ( v170 != v158 );
                  v16 = v169;
                }
                v202.m128i_i8[0] = v171;
                v202.m128i_i64[1] = (__int64)&v204;
                v203 = 0x200000000LL;
                if ( (_DWORD)v191 )
                  sub_25C2990((__int64)&v202.m128i_i64[1], (__int64)&v190);
                sub_25C0430((__int64)v16);
                sub_25C0430((__int64)&v190);
                goto LABEL_22;
              }
            }
          }
          goto LABEL_21;
        }
        v202 = 0;
        if ( v21 )
          break;
        v20 = 0;
LABEL_72:
        v60 = *((_QWORD *)v22 + 2);
        if ( !v60 )
          goto LABEL_46;
        v61 = v21;
        while ( 1 )
        {
          v202.m128i_i64[0] = v20;
          i = HIDWORD(v219);
          v6 = v12 + 1;
          v63 = (const __m128i *)v16;
          v202.m128i_i8[8] = v61;
          v64 = _mm_loadu_si128(&v202);
          v210 = v60;
          v211 = v64;
          if ( v12 + 1 > (unsigned __int64)HIDWORD(v219) )
          {
            if ( v15 > v16 || v16 >= &v15[3 * v12] )
            {
              sub_C8D5F0((__int64)&v218, v220, v6, 0x18u, v5, v6);
              v15 = (__int64 *)v218;
              v12 = (unsigned int)v219;
              v63 = (const __m128i *)v16;
            }
            else
            {
              v65 = (char *)v16 - (char *)v15;
              sub_C8D5F0((__int64)&v218, v220, v6, 0x18u, v5, v6);
              v15 = (__int64 *)v218;
              v12 = (unsigned int)v219;
              v63 = (const __m128i *)(v218 + v65);
            }
          }
          v62 = (__m128i *)&v15[3 * v12];
          *v62 = _mm_loadu_si128(v63);
          v62[1].m128i_i64[0] = v63[1].m128i_i64[0];
          v12 = (unsigned int)(v219 + 1);
          LODWORD(v219) = v219 + 1;
          v60 = *(_QWORD *)(v60 + 8);
          if ( !v60 )
            break;
          v15 = (__int64 *)v218;
        }
LABEL_45:
        v15 = (__int64 *)v218;
LABEL_46:
        if ( !(_DWORD)v12 )
          goto LABEL_47;
      }
      v211.m128i_i32[0] = v173;
      if ( v173 > 0x40 )
        sub_C43690((__int64)v16, 0, 0);
      else
        v210 = 0;
      if ( (unsigned __int8)sub_BB6360((__int64)v22, v177, (__int64)v16, 0, 0) )
      {
        if ( v211.m128i_i32[0] <= 0x40u )
        {
          if ( v211.m128i_i32[0] )
          {
            i = (unsigned int)(64 - v211.m128i_i32[0]);
            v20 += (__int64)(v210 << (64 - v211.m128i_i8[0])) >> (64 - v211.m128i_i8[0]);
          }
          v12 = (unsigned int)v219;
          v15 = (__int64 *)v218;
          v21 = 1;
          goto LABEL_72;
        }
        v79 = v210;
        v20 += *(_QWORD *)v210;
        v21 = 1;
        if ( !v210 )
          goto LABEL_106;
      }
      else
      {
        v21 = 0;
        v20 = 0;
        if ( v211.m128i_i32[0] <= 0x40u )
          goto LABEL_106;
        v79 = v210;
        if ( !v210 )
          goto LABEL_106;
      }
      j_j___libc_free_0_0(v79);
LABEL_106:
      v12 = (unsigned int)v219;
      v15 = (__int64 *)v218;
      goto LABEL_72;
    }
LABEL_47:
    if ( v15 != (__int64 *)v220 )
      _libc_free((unsigned __int64)v15);
  }
  v178 = v230;
  if ( (_BYTE)v230 )
  {
    v182 = 0;
    v183 = 0;
    v184 = 0;
    v44 = *(_QWORD *)(a2 + 80);
    v185 = 0;
    if ( v44 )
      v44 -= 24;
    v180 = &v231;
    v181 = (__int64 *)&v182;
    v186 = (const void **)&v188;
    v187 = 0x200000000LL;
    if ( !HIBYTE(v230) )
    {
LABEL_53:
      sub_25C5120((__int64)&v218, (__int64 *)&v180, v44, i, v5, v6);
      sub_25C2C90((__int64)&v186, (__int64 *)&v218);
      sub_25C0430((__int64)&v218);
      if ( (_DWORD)v187 )
        goto LABEL_54;
      v178 = 0;
LABEL_124:
      sub_25C0430((__int64)&v186);
      v88 = v185;
      if ( v185 )
      {
        v89 = v183;
        v90 = &v183[11 * v185];
        do
        {
          if ( *v89 != -8192 && *v89 != -4096 )
            sub_25C0430((__int64)(v89 + 1));
          v89 += 11;
        }
        while ( v90 != v89 );
        v88 = v185;
      }
      sub_C7D6A0((__int64)v183, 88 * v88, 8);
      goto LABEL_132;
    }
    v106 = v232 & 1;
    if ( (v232 & 1) != 0 )
    {
      i = (unsigned __int64)&v233;
      v5 = 15;
    }
    else
    {
      v107 = v234;
      i = (unsigned __int64)v233;
      if ( !v234 )
        goto LABEL_387;
      v5 = v234 - 1;
    }
    v108 = v5 & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
    v109 = i + 416LL * v108;
    v110 = *(_QWORD *)v109;
    if ( v44 == *(_QWORD *)v109 )
    {
LABEL_183:
      v111 = 6656;
      if ( !v106 )
        v111 = 416LL * v234;
      if ( v109 != i + v111 && *(_BYTE *)(v109 + 409) )
        goto LABEL_53;
      v192[0] = 8;
      memset(v189, 0, sizeof(v189));
      LODWORD(v189[2]) = 8;
      v189[1] = (__int64)&v189[4];
      v189[12] = (__int64)&v189[14];
      v191 = (unsigned __int64)&v194;
      v195 = &v197;
      BYTE4(v189[3]) = 1;
      HIDWORD(v189[13]) = 8;
      v192[2] = 0;
      v193 = 1;
      v196 = 0x800000000LL;
      v192[1] = 1;
      v194 = v44;
      v190 = 1;
      v112 = *(_QWORD *)(v44 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v112 == v44 + 48 )
        goto LABEL_330;
      if ( !v112 )
        BUG();
      v113 = v112 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v112 - 24) - 30 > 0xA )
      {
LABEL_330:
        v114 = 0;
        v115 = 0;
        v113 = 0;
      }
      else
      {
        v114 = sub_B46E30(v113);
        v115 = v113;
      }
      v199 = v113;
      v198 = v114;
      v197 = v115;
      v200 = 0;
      v201 = v44;
      LODWORD(v196) = 1;
      sub_CE27D0((__int64)&v190);
      sub_CE35F0((__int64)&v210, (__int64)v189);
      sub_CE35F0((__int64)&v202, (__int64)&v190);
      sub_CE35F0((__int64)&v218, (__int64)&v202);
      sub_CE35F0((__int64)v224, (__int64)&v210);
      if ( v207 != v209 )
        _libc_free((unsigned __int64)v207);
      if ( !v205 )
        _libc_free(v202.m128i_u64[1]);
      if ( v215 != v217 )
        _libc_free((unsigned __int64)v215);
      if ( !v213 )
        _libc_free(v211.m128i_u64[0]);
      if ( v195 != &v197 )
        _libc_free((unsigned __int64)v195);
      if ( !v193 )
        _libc_free(v191);
      if ( (__int64 *)v189[12] != &v189[14] )
        _libc_free(v189[12]);
      if ( !BYTE4(v189[3]) )
        _libc_free(v189[1]);
      sub_C8CD80((__int64)&v202, (__int64)v206, (__int64)&v218, v116, v117, v118);
      v207 = v209;
      v208 = 0x800000000LL;
      if ( v222 )
        sub_25C58C0((__int64)&v207, (__int64 *)&v221, v222, v119, v120, v121);
      sub_C8CD80((__int64)&v210, (__int64)v214, (__int64)v224, v119, v120, v121);
      v125 = v228;
      v215 = v217;
      v216 = 0x800000000LL;
      if ( v228 )
      {
        sub_25C58C0((__int64)&v215, (__int64 *)&v227, v122, v228, v123, v124);
        v125 = (unsigned int)v216;
      }
LABEL_211:
      v126 = (unsigned int)v208;
      while ( 1 )
      {
        v127 = 40 * v126;
        if ( v126 == v125 )
        {
          v123 = (__int64)v215;
          if ( v207 == &v207[v127] )
          {
LABEL_259:
            if ( v215 != v217 )
              _libc_free((unsigned __int64)v215);
            if ( !v213 )
              _libc_free(v211.m128i_u64[0]);
            if ( v207 != v209 )
              _libc_free((unsigned __int64)v207);
            if ( !v205 )
              _libc_free(v202.m128i_u64[1]);
            if ( v227 != &v229 )
              _libc_free((unsigned __int64)v227);
            if ( !v226 )
              _libc_free(v225);
            if ( v221 != &v223 )
              _libc_free((unsigned __int64)v221);
            if ( !v220[12] )
              _libc_free(v219);
            if ( v185 )
            {
              v142 = (v185 - 1) & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
              v143 = &v183[11 * v142];
              v144 = *v143;
              if ( v44 == *v143 )
              {
LABEL_277:
                if ( v143 != &v183[11 * v185] )
                {
                  sub_25C2990((__int64)&v186, (__int64)(v143 + 1));
LABEL_54:
                  if ( (unsigned __int8)sub_B2D670(a1, 98) )
                  {
                    v45 = &v211.m128i_i8[8];
                    v218 = sub_B2D8E0(a1, 98);
                    v46 = sub_A72AC0((__int64 *)&v218);
                    v210 = (unsigned __int64)&v211.m128i_u64[1];
                    v47 = v46;
                    v49 = v46 + 32 * v48;
                    v211.m128i_i64[0] = 0x200000000LL;
                    if ( v46 == v49 )
                    {
                      v57 = 0;
                    }
                    else
                    {
                      v50 = v46;
                      v51 = 0;
                      v52 = 0;
                      v53 = v49;
                      while ( 1 )
                      {
                        v54 = (__int64)&v45[32 * v52];
                        if ( v54 )
                        {
                          v55 = *(_DWORD *)(v47 + 8);
                          *(_DWORD *)(v54 + 8) = v55;
                          if ( v55 > 0x40 )
                            sub_C43780(v54, (const void **)v47);
                          else
                            *(_QWORD *)v54 = *(_QWORD *)v47;
                          v56 = *(_DWORD *)(v47 + 24);
                          *(_DWORD *)(v54 + 24) = v56;
                          if ( v56 > 0x40 )
                            sub_C43780(v54 + 16, (const void **)(v47 + 16));
                          else
                            *(_QWORD *)(v54 + 16) = *(_QWORD *)(v47 + 16);
                          v51 = v211.m128i_i32[0];
                        }
                        v50 += 32LL;
                        v211.m128i_i32[0] = ++v51;
                        v52 = v51;
                        v57 = v51;
                        if ( v53 == v50 )
                          break;
                        v58 = v51 + 1LL;
                        v45 = (__int8 *)v210;
                        v47 = v50;
                        if ( v58 > v211.m128i_u32[1] )
                        {
                          if ( v210 > v50 || v210 + 32LL * v51 <= v50 )
                          {
                            v47 = v50;
                            sub_9D5330((__int64)&v210, v58);
                            v52 = v211.m128i_u32[0];
                            v45 = (__int8 *)v210;
                            v51 = v211.m128i_i32[0];
                          }
                          else
                          {
                            v59 = v50 - v210;
                            sub_9D5330((__int64)&v210, v58);
                            v45 = (__int8 *)v210;
                            v52 = v211.m128i_u32[0];
                            v47 = v210 + v59;
                            v51 = v211.m128i_i32[0];
                          }
                        }
                      }
                    }
                    if ( (unsigned int)v187 == v57 )
                    {
                      v81 = v210;
                      v82 = v186;
                      v83 = v210 + 32 * v57;
                      if ( v210 != v83 )
                      {
                        do
                        {
                          if ( *(_DWORD *)(v81 + 8) <= 0x40u )
                          {
                            if ( *(const void **)v81 != *v82 )
                              goto LABEL_122;
                          }
                          else if ( !sub_C43C50(v81, v82) )
                          {
                            goto LABEL_122;
                          }
                          if ( *(_DWORD *)(v81 + 24) <= 0x40u )
                          {
                            if ( *(const void **)(v81 + 16) != v82[2] )
                              goto LABEL_122;
                          }
                          else if ( !sub_C43C50(v81 + 16, v82 + 2) )
                          {
                            goto LABEL_122;
                          }
                          v81 += 32;
                          v82 += 4;
                        }
                        while ( v83 != v81 );
                      }
                      sub_25C0430((__int64)&v210);
                      v178 = 0;
                      goto LABEL_124;
                    }
LABEL_122:
                    sub_AC10C0(&v218, (unsigned int *)&v186, (__int64 *)&v210);
                    sub_25C2C90((__int64)&v186, (__int64 *)&v218);
                    sub_25C0430((__int64)&v218);
                    sub_25C0430((__int64)&v210);
                  }
                  v84 = (unsigned int)v187;
                  v85 = (__int64)v186;
                  v86 = (_QWORD *)sub_BD5C60(a1);
                  v87 = sub_A78DB0(v86, 98, v85, v84);
                  sub_B2D460(a1, v87);
                  goto LABEL_124;
                }
              }
              else
              {
                v165 = 1;
                while ( v144 != -4096 )
                {
                  v166 = v165 + 1;
                  v142 = (v185 - 1) & (v165 + v142);
                  v143 = &v183[11 * v142];
                  v144 = *v143;
                  if ( v44 == *v143 )
                    goto LABEL_277;
                  v165 = v166;
                }
              }
            }
            v178 = 0;
            goto LABEL_124;
          }
          v125 = (__int64)v215;
          v128 = (unsigned __int64)v207;
          while ( *(_QWORD *)(v128 + 32) == *(_QWORD *)(v125 + 32)
               && *(_DWORD *)(v128 + 24) == *(_DWORD *)(v125 + 24)
               && *(_DWORD *)(v128 + 8) == *(_DWORD *)(v125 + 8) )
          {
            v128 += 40LL;
            v125 += 40;
            if ( &v207[v127] == (_BYTE *)v128 )
              goto LABEL_259;
          }
        }
        v179 = *(_QWORD *)&v207[v127 - 8];
        sub_25C5120((__int64)&v190, (__int64 *)&v180, v179, v125, v123, v124);
        if ( (_DWORD)v191 )
          break;
LABEL_217:
        sub_25C0430((__int64)&v190);
        v129 = (_DWORD)v208 == 1;
        v126 = (unsigned int)(v208 - 1);
        LODWORD(v208) = v208 - 1;
        if ( !v129 )
        {
          sub_CE27D0((__int64)&v202);
          v125 = (unsigned int)v216;
          goto LABEL_211;
        }
        v125 = (unsigned int)v216;
      }
      v134 = v185;
      if ( v185 )
      {
        v135 = v179;
        v136 = 1;
        v137 = (v185 - 1) & (((unsigned int)v179 >> 9) ^ ((unsigned int)v179 >> 4));
        v138 = &v183[11 * v137];
        v139 = 0;
        v140 = *v138;
        if ( v179 == *v138 )
        {
LABEL_254:
          v141 = (__int64)(v138 + 1);
LABEL_255:
          sub_25C2990(v141, (__int64)&v190);
          goto LABEL_217;
        }
        while ( v140 != -4096 )
        {
          if ( v140 == -8192 && !v139 )
            v139 = v138;
          v137 = (v185 - 1) & (v136 + v137);
          v138 = &v183[11 * v137];
          v140 = *v138;
          if ( v179 == *v138 )
            goto LABEL_254;
          ++v136;
        }
        if ( !v139 )
          v139 = v138;
        v182 = (__int64 *)((char *)v182 + 1);
        v161 = v184 + 1;
        v189[0] = (__int64)v139;
        if ( 4 * ((int)v184 + 1) < 3 * v185 )
        {
          if ( v185 - HIDWORD(v184) - v161 > v185 >> 3 )
          {
LABEL_341:
            LODWORD(v184) = v161;
            if ( *v139 != -4096 )
              --HIDWORD(v184);
            *v139 = v135;
            v141 = (__int64)(v139 + 1);
            v139[1] = (__int64)(v139 + 3);
            v139[2] = 0x200000000LL;
            goto LABEL_255;
          }
LABEL_346:
          sub_25C5A30((__int64)&v182, v134);
          sub_25C2800((__int64)&v182, &v179, v189);
          v135 = v179;
          v161 = v184 + 1;
          v139 = (__int64 *)v189[0];
          goto LABEL_341;
        }
      }
      else
      {
        v182 = (__int64 *)((char *)v182 + 1);
        v189[0] = 0;
      }
      v134 = 2 * v185;
      goto LABEL_346;
    }
    v168 = 1;
    while ( v110 != -4096 )
    {
      v6 = (unsigned int)(v168 + 1);
      v108 = v5 & (v168 + v108);
      v109 = i + 416LL * v108;
      v110 = *(_QWORD *)v109;
      if ( v44 == *(_QWORD *)v109 )
        goto LABEL_183;
      v168 = v6;
    }
    if ( v106 )
    {
      v167 = 6656;
      goto LABEL_388;
    }
    v107 = v234;
LABEL_387:
    v167 = 416 * v107;
LABEL_388:
    v109 = i + v167;
    goto LABEL_183;
  }
LABEL_132:
  if ( (v232 & 1) != 0 )
  {
    v93 = v235;
    v91 = &v233;
    goto LABEL_135;
  }
  v91 = v233;
  v92 = 52LL * v234;
  if ( !v234 )
    goto LABEL_178;
  v93 = &v233[v92];
  if ( &v233[v92] == v233 )
    goto LABEL_178;
  do
  {
LABEL_135:
    if ( *v91 != -4096 && *v91 != -8192 )
    {
      if ( (v91[2] & 1) != 0 )
      {
        v95 = v91 + 3;
        v97 = v91 + 51;
      }
      else
      {
        v94 = *((unsigned int *)v91 + 8);
        v95 = (_QWORD *)v91[3];
        v96 = 12 * v94;
        if ( !(_DWORD)v94 )
          goto LABEL_149;
        v97 = &v95[v96];
        if ( v95 == &v95[v96] )
          goto LABEL_149;
      }
      do
      {
        if ( *v95 != -8192 && *v95 != -4096 )
          sub_25C0430((__int64)(v95 + 2));
        v95 += 12;
      }
      while ( v97 != v95 );
      if ( (v91[2] & 1) == 0 )
      {
        v95 = (_QWORD *)v91[3];
        v96 = 12LL * *((unsigned int *)v91 + 8);
LABEL_149:
        sub_C7D6A0((__int64)v95, v96 * 8, 8);
      }
    }
    v91 += 52;
  }
  while ( v93 != (_BYTE *)v91 );
  if ( (v232 & 1) == 0 )
  {
    v91 = v233;
    v92 = 52LL * v234;
LABEL_178:
    sub_C7D6A0((__int64)v91, v92 * 8, 8);
  }
  return v178;
}
