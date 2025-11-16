// Function: sub_37929A0
// Address: 0x37929a0
//
unsigned __int8 *__fastcall sub_37929A0(__int64 *a1, __int64 a2, __int64 a3, __m128i a4)
{
  __int64 v5; // rbx
  __int64 v6; // r9
  __int64 (__fastcall *v7)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v8; // rax
  unsigned __int16 v9; // si
  __int64 v10; // r8
  __int64 v11; // rax
  unsigned __int16 v12; // r14
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // r15
  __m128i v16; // xmm4
  __m128i v17; // xmm1
  __m128i v18; // xmm2
  __int16 v19; // ax
  unsigned __int64 v20; // r13
  __int64 v21; // rdx
  __int64 v22; // rdx
  int v23; // edx
  __int64 v24; // r15
  unsigned __int8 *v25; // r14
  char v27; // al
  bool v28; // al
  __int64 v29; // rdx
  __int64 v30; // r9
  __m128i v31; // xmm6
  unsigned __int64 v32; // r8
  __int8 v33; // bl
  char v34; // r15
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  __int64 v37; // rax
  __int16 v38; // dx
  __m128i v39; // rax
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // rcx
  __int16 v46; // r12
  __int64 v47; // rsi
  __int64 v48; // r9
  __int64 v49; // rdx
  __int64 v50; // rdi
  __int64 v51; // rax
  unsigned int v52; // edx
  __int64 v53; // rax
  __int64 v54; // rax
  unsigned __int8 *v55; // rax
  unsigned __int64 v56; // rdi
  __int64 v57; // rdx
  __m128i v58; // xmm7
  __int64 v59; // rdx
  unsigned int v60; // eax
  __int64 v61; // rdx
  int v62; // r9d
  int v63; // r9d
  __int64 v64; // r14
  const __m128i *v65; // rax
  __int64 v66; // rcx
  __int64 v67; // r8
  __int64 v68; // r9
  __int64 *v69; // r13
  int v70; // r15d
  __int64 v71; // r12
  char v72; // al
  int v73; // eax
  __m128i *v74; // rax
  __int64 v75; // rdx
  __int64 v76; // r8
  __int64 v77; // rdx
  unsigned __int64 v78; // r9
  __m128i **v79; // rdx
  __int64 v80; // rdx
  unsigned __int64 v81; // r8
  unsigned __int64 v82; // rcx
  __m128i **v83; // rdx
  __int64 v84; // rcx
  __int64 v85; // r14
  __int64 v86; // rdi
  __int64 v87; // rdx
  unsigned __int64 v88; // rdx
  unsigned __int16 *v89; // rax
  int v90; // edx
  __int64 v91; // rax
  __int64 v92; // rdx
  bool v93; // al
  unsigned int v94; // r13d
  __int64 v95; // r12
  unsigned int *v96; // r14
  __int64 v97; // rax
  __int16 v98; // dx
  __int64 v99; // rax
  signed int v100; // r15d
  unsigned int v101; // r14d
  unsigned int *v102; // rdx
  __int64 v103; // r12
  unsigned int *v104; // r15
  __int64 v105; // r14
  unsigned int v106; // r13d
  __int64 v107; // rax
  __int16 v108; // cx
  __int64 v109; // rax
  _BYTE *v110; // rax
  __int64 v111; // rdx
  __int64 v112; // r14
  unsigned int v113; // r12d
  __int64 v114; // rcx
  _BYTE *v115; // rax
  int v116; // eax
  _BYTE *v117; // rax
  unsigned int *v118; // rdx
  __int64 v119; // rax
  __int16 v120; // cx
  __int64 v121; // rax
  __int64 v122; // rdx
  unsigned __int64 v123; // rax
  __int64 v124; // r8
  __int64 v125; // r9
  __int64 v126; // rdx
  unsigned __int64 v127; // rax
  unsigned __int64 v128; // rtt
  __int64 v129; // r13
  _QWORD *v130; // rdx
  _QWORD *v131; // rsi
  _QWORD *i; // rcx
  __int64 v133; // rcx
  unsigned int j; // r13d
  __int64 v135; // rdx
  __int64 v136; // rsi
  _BYTE *v137; // rdx
  _QWORD *v138; // rdi
  _QWORD *v139; // r15
  __int64 v140; // rdx
  __int64 v141; // r14
  __int64 v142; // rax
  __int64 v143; // rax
  unsigned __int8 *v144; // rax
  __int64 v145; // rdx
  __int64 v146; // rdi
  unsigned __int8 *v147; // rdx
  unsigned __int64 v148; // rax
  unsigned int v149; // r13d
  __m128i v150; // rax
  __int64 v151; // r9
  unsigned int v152; // r15d
  __int64 v153; // r14
  unsigned __int64 v154; // rax
  __int64 v155; // rdx
  __int64 v156; // rcx
  __int64 v157; // r8
  __int64 v158; // r9
  __int64 v159; // rax
  __int32 v160; // ecx
  __int64 v161; // rdi
  __int64 v162; // rdx
  __int64 v163; // r9
  __int64 v164; // rdx
  __int64 v165; // rax
  __int64 v166; // rsi
  _BYTE *v167; // rax
  __int64 v168; // r14
  __int128 v169; // [rsp-10h] [rbp-660h]
  __int128 v170; // [rsp-10h] [rbp-660h]
  __int128 v171; // [rsp-10h] [rbp-660h]
  __int128 v172; // [rsp-10h] [rbp-660h]
  unsigned __int16 v173; // [rsp+0h] [rbp-650h]
  __m128i *v174; // [rsp+0h] [rbp-650h]
  __int64 v175; // [rsp+0h] [rbp-650h]
  __int64 v176; // [rsp+30h] [rbp-620h]
  unsigned __int64 v177; // [rsp+30h] [rbp-620h]
  unsigned __int8 v178; // [rsp+38h] [rbp-618h]
  int v179; // [rsp+38h] [rbp-618h]
  int v180; // [rsp+40h] [rbp-610h]
  unsigned __int64 v181; // [rsp+50h] [rbp-600h]
  __int64 *v182; // [rsp+50h] [rbp-600h]
  unsigned int v183; // [rsp+50h] [rbp-600h]
  char v184; // [rsp+5Fh] [rbp-5F1h]
  unsigned __int64 v185; // [rsp+60h] [rbp-5F0h]
  int v186; // [rsp+68h] [rbp-5E8h]
  int v187; // [rsp+68h] [rbp-5E8h]
  __int64 v189; // [rsp+70h] [rbp-5E0h]
  unsigned __int64 v190; // [rsp+70h] [rbp-5E0h]
  __m128i *v191; // [rsp+70h] [rbp-5E0h]
  __int16 v192; // [rsp+78h] [rbp-5D8h]
  __int64 v193; // [rsp+80h] [rbp-5D0h]
  unsigned __int64 v194; // [rsp+90h] [rbp-5C0h]
  unsigned int v195; // [rsp+90h] [rbp-5C0h]
  unsigned __int64 v196; // [rsp+98h] [rbp-5B8h]
  int v197; // [rsp+98h] [rbp-5B8h]
  __m128i v198; // [rsp+A0h] [rbp-5B0h] BYREF
  __int64 v199; // [rsp+B0h] [rbp-5A0h]
  __int64 v200; // [rsp+B8h] [rbp-598h]
  _QWORD *v201; // [rsp+C0h] [rbp-590h]
  __int64 v202; // [rsp+C8h] [rbp-588h]
  unsigned __int8 *v203; // [rsp+D0h] [rbp-580h]
  __int64 v204; // [rsp+D8h] [rbp-578h]
  unsigned __int8 *v205; // [rsp+E0h] [rbp-570h]
  __int64 v206; // [rsp+E8h] [rbp-568h]
  __int64 v207; // [rsp+F0h] [rbp-560h]
  __int64 v208; // [rsp+F8h] [rbp-558h]
  __m128i v209; // [rsp+100h] [rbp-550h]
  __m128i v210; // [rsp+110h] [rbp-540h] BYREF
  __int64 v211; // [rsp+120h] [rbp-530h]
  __int64 v212; // [rsp+138h] [rbp-518h] BYREF
  unsigned int v213; // [rsp+140h] [rbp-510h] BYREF
  __int64 v214; // [rsp+148h] [rbp-508h]
  unsigned __int16 v215; // [rsp+150h] [rbp-500h] BYREF
  __int64 v216; // [rsp+158h] [rbp-4F8h]
  __int64 v217; // [rsp+160h] [rbp-4F0h] BYREF
  int v218; // [rsp+168h] [rbp-4E8h]
  __m128i v219; // [rsp+170h] [rbp-4E0h] BYREF
  unsigned __int64 v220; // [rsp+180h] [rbp-4D0h]
  __int64 v221; // [rsp+188h] [rbp-4C8h]
  __m128i v222; // [rsp+190h] [rbp-4C0h] BYREF
  __m128i v223; // [rsp+1A0h] [rbp-4B0h] BYREF
  unsigned __int64 v224; // [rsp+1B0h] [rbp-4A0h]
  __int64 v225; // [rsp+1B8h] [rbp-498h]
  unsigned __int64 v226; // [rsp+1C0h] [rbp-490h]
  __int64 v227; // [rsp+1C8h] [rbp-488h]
  __int64 v228; // [rsp+1D0h] [rbp-480h]
  __int64 v229; // [rsp+1D8h] [rbp-478h]
  __int64 v230; // [rsp+1E0h] [rbp-470h]
  __int64 v231; // [rsp+1E8h] [rbp-468h]
  __int64 v232; // [rsp+1F0h] [rbp-460h] BYREF
  int v233; // [rsp+1F8h] [rbp-458h]
  __m128i v234; // [rsp+200h] [rbp-450h] BYREF
  __int64 v235; // [rsp+210h] [rbp-440h]
  __int128 v236; // [rsp+220h] [rbp-430h] BYREF
  __int64 v237; // [rsp+230h] [rbp-420h]
  _OWORD v238[2]; // [rsp+240h] [rbp-410h] BYREF
  __int64 *v239; // [rsp+260h] [rbp-3F0h] BYREF
  __int64 v240; // [rsp+268h] [rbp-3E8h]
  _BYTE v241[128]; // [rsp+270h] [rbp-3E0h] BYREF
  unsigned int *v242; // [rsp+2F0h] [rbp-360h] BYREF
  __int64 v243; // [rsp+2F8h] [rbp-358h]
  _BYTE v244[256]; // [rsp+300h] [rbp-350h] BYREF
  _BYTE *v245; // [rsp+400h] [rbp-250h] BYREF
  __int64 v246; // [rsp+408h] [rbp-248h]
  _BYTE v247[256]; // [rsp+410h] [rbp-240h] BYREF
  __m128i v248; // [rsp+510h] [rbp-140h] BYREF
  _QWORD v249[38]; // [rsp+520h] [rbp-130h] BYREF

  v5 = (__int64)a1;
  v6 = *a1;
  v7 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v8 = *(__int16 **)(a3 + 48);
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  v11 = a1[1];
  if ( v7 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v248, v6, *(_QWORD *)(v11 + 64), v9, v10);
    LOWORD(v213) = v248.m128i_i16[4];
    v214 = v249[0];
  }
  else
  {
    v213 = v7(v6, *(_QWORD *)(v11 + 64), v9, v10);
    v214 = v57;
  }
  v12 = *(_WORD *)(a3 + 96);
  v13 = *(_QWORD *)(a3 + 104);
  v14 = *(_QWORD *)(a3 + 80);
  v215 = v12;
  v216 = v13;
  v217 = v14;
  if ( v14 )
  {
    sub_B96E90((__int64)&v217, v14, 1);
    v12 = v215;
  }
  v15 = *(_QWORD *)(a3 + 112);
  v218 = *(_DWORD *)(a3 + 72);
  v16 = _mm_loadu_si128((const __m128i *)(v15 + 56));
  v17 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a3 + 40));
  v18 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a3 + 40) + 40LL));
  v19 = *(_WORD *)(v15 + 32);
  v238[0] = _mm_loadu_si128((const __m128i *)(v15 + 40));
  v192 = v19;
  v219 = v18;
  v238[1] = v16;
  if ( v12 )
  {
    if ( v12 == 1 || (unsigned __int16)(v12 - 504) <= 7u )
      goto LABEL_135;
    v20 = *(_QWORD *)&byte_444C4A0[16 * v12 - 16];
    v198.m128i_i8[0] = byte_444C4A0[16 * v12 - 8];
  }
  else
  {
    v230 = sub_3007260((__int64)&v215);
    v20 = v230;
    v231 = v21;
    v198.m128i_i8[0] = v21;
  }
  if ( (_WORD)v213 )
  {
    if ( (_WORD)v213 != 1 && (unsigned __int16)(v213 - 504) > 7u )
    {
      v196 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v213 - 16];
      v184 = byte_444C4A0[16 * (unsigned __int16)v213 - 8];
      goto LABEL_9;
    }
LABEL_135:
    BUG();
  }
  v228 = sub_3007260((__int64)&v213);
  v229 = v22;
  v196 = v228;
  v184 = v22;
LABEL_9:
  v23 = v196 - v20;
  if ( (*(_BYTE *)(v15 + 37) & 0xF) != 0 || (*(_BYTE *)(a3 + 32) & 8) != 0 )
    goto LABEL_11;
  if ( !v12 )
  {
    v28 = sub_3007100((__int64)&v215);
    v23 = v196 - v20;
    if ( !v28 )
      goto LABEL_19;
LABEL_11:
    LODWORD(v24) = 0;
    goto LABEL_12;
  }
  if ( (unsigned __int16)(v12 - 176) <= 0x34u )
    goto LABEL_11;
LABEL_19:
  v187 = v23;
  v27 = sub_2EAC4F0(v15);
  v23 = v187;
  v24 = 1LL << v27;
LABEL_12:
  v186 = v23;
  sub_3776670(&v234, a1[1], *a1, (unsigned int)v20, v213, v214, v24, v23);
  if ( !(_BYTE)v235 )
  {
    v25 = 0;
    goto LABEL_14;
  }
  v239 = (__int64 *)v241;
  v240 = 0x800000000LL;
  v220 = sub_2D5B750((unsigned __int16 *)&v234);
  v221 = v29;
  v30 = (unsigned __int8)v29;
  v181 = v220;
  if ( v198.m128i_i8[0] && !(_BYTE)v29 || v20 > v220 )
  {
    v31 = _mm_load_si128(&v234);
    v180 = v24;
    v32 = v220;
    v249[0] = v235;
    v33 = v198.m128i_i8[0];
    v248 = v31;
    while ( 1 )
    {
      v20 -= v32;
      if ( v32 )
      {
        v34 = v30;
        v33 = v30;
        if ( v20 < v32 )
        {
          sub_3776670(&v210, a1[1], *a1, (unsigned int)v20, v213, v214, v180, v186);
          v58 = _mm_loadu_si128(&v210);
          v249[0] = v211;
          v248 = v58;
          if ( !(_BYTE)v211 )
          {
            v25 = 0;
            goto LABEL_42;
          }
          v245 = (_BYTE *)sub_2D5B750((unsigned __int16 *)&v248);
          v32 = (unsigned __int64)v245;
          v246 = v59;
          v30 = (unsigned __int8)v59;
        }
      }
      else
      {
        v34 = v33;
        if ( v33 )
        {
          v34 = v30;
          if ( !(_BYTE)v30 )
            v34 = v33;
        }
      }
      v35 = (unsigned int)v240;
      a4 = _mm_load_si128(&v248);
      v36 = (unsigned int)v240 + 1LL;
      if ( v36 > HIDWORD(v240) )
      {
        v177 = v32;
        v178 = v30;
        v198 = a4;
        sub_C8D5F0((__int64)&v239, v241, v36, 0x10u, v32, v30);
        v35 = (unsigned int)v240;
        v32 = v177;
        v30 = v178;
        a4 = _mm_load_si128(&v198);
      }
      *(__m128i *)&v239[2 * v35] = a4;
      LODWORD(v240) = v240 + 1;
      if ( !v34 && (_BYTE)v30 || v20 <= v32 )
      {
        v5 = (__int64)a1;
        break;
      }
    }
  }
  v37 = *(_QWORD *)(a3 + 112);
  LOBYTE(v38) = *(_BYTE *)(v37 + 34);
  HIBYTE(v38) = 1;
  v39.m128i_i64[0] = (__int64)sub_33F1F00(
                                *(__int64 **)(v5 + 8),
                                v234.m128i_u32[0],
                                v234.m128i_i64[1],
                                (__int64)&v217,
                                v17.m128i_i64[0],
                                v17.m128i_i64[1],
                                v219.m128i_i64[0],
                                v219.m128i_i64[1],
                                *(_OWORD *)v37,
                                *(_QWORD *)(v37 + 16),
                                v38,
                                v192,
                                (__int64)v238,
                                0);
  v198 = v39;
  sub_3050D50(a2, v39.m128i_i64[0], 1, v40, v41, v42);
  v45 = (unsigned int)v240;
  if ( (_DWORD)v240 )
  {
    v64 = v198.m128i_i64[0];
    v242 = (unsigned int *)v244;
    v243 = 0x1000000000LL;
    sub_3050D50((__int64)&v242, v198.m128i_i64[0], v198.m128i_i64[1], (unsigned int)v240, v43, v44);
    v65 = *(const __m128i **)(a3 + 112);
    v212 = 0;
    v236 = (__int128)_mm_loadu_si128(v65);
    v237 = v65[1].m128i_i64[0];
    sub_3777490(v5, v64, v234.m128i_u32[0], v234.m128i_i64[1], (__int64)&v236, (unsigned int *)&v219, a4, &v212);
    v69 = v239;
    v182 = &v239[2 * (unsigned int)v240];
    if ( v182 != v239 )
    {
      v176 = a3;
      v70 = v173;
      v71 = a2;
      do
      {
        v85 = *v69;
        v198.m128i_i64[0] = v69[1];
        v86 = *(_QWORD *)(v176 + 112);
        if ( v212 )
        {
          v189 = v212;
          v87 = v189 | (1LL << sub_2EAC4F0(v86));
          v72 = -1;
          v88 = -v87 & v87;
          if ( v88 )
          {
            _BitScanReverse64(&v88, v88);
            v72 = 63 - (v88 ^ 0x3F);
          }
        }
        else
        {
          v72 = *(_BYTE *)(v86 + 34);
        }
        LOBYTE(v70) = v72;
        v73 = v70;
        BYTE1(v73) = 1;
        v70 = v73;
        v74 = sub_33F1F00(
                *(__int64 **)(v5 + 8),
                v85,
                v198.m128i_i64[0],
                (__int64)&v217,
                v17.m128i_i64[0],
                v17.m128i_i64[1],
                v219.m128i_i64[0],
                v219.m128i_i64[1],
                v236,
                v237,
                v73,
                v192,
                (__int64)v238,
                0);
        v76 = v75;
        v77 = (unsigned int)v243;
        v78 = (unsigned int)v243 + 1LL;
        if ( v78 > HIDWORD(v243) )
        {
          v175 = v76;
          v191 = v74;
          sub_C8D5F0((__int64)&v242, v244, (unsigned int)v243 + 1LL, 0x10u, v76, v78);
          v77 = (unsigned int)v243;
          v76 = v175;
          v74 = v191;
        }
        v79 = (__m128i **)&v242[4 * v77];
        v79[1] = (__m128i *)v76;
        *v79 = v74;
        v80 = *(unsigned int *)(v71 + 8);
        v81 = v194 & 0xFFFFFFFF00000000LL | 1;
        v82 = *(unsigned int *)(v71 + 12);
        LODWORD(v243) = v243 + 1;
        v194 = v81;
        if ( v80 + 1 > v82 )
        {
          v174 = v74;
          v190 = v81;
          sub_C8D5F0(v71, (const void *)(v71 + 16), v80 + 1, 0x10u, v81, v80 + 1);
          v80 = *(unsigned int *)(v71 + 8);
          v74 = v174;
          v81 = v190;
        }
        v83 = (__m128i **)(*(_QWORD *)v71 + 16 * v80);
        v83[1] = (__m128i *)v81;
        v69 += 2;
        v84 = v198.m128i_i64[0];
        *v83 = v74;
        ++*(_DWORD *)(v71 + 8);
        sub_3777490(v5, (__int64)v74, v85, v84, (__int64)&v236, (unsigned int *)&v219, a4, &v212);
      }
      while ( v182 != v69 );
    }
    v195 = v243;
    v89 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v242 + 48LL) + 16LL * v242[2]);
    v90 = *v89;
    v91 = *((_QWORD *)v89 + 1);
    v248.m128i_i16[0] = v90;
    v248.m128i_i64[1] = v91;
    if ( (_WORD)v90 )
    {
      v92 = (unsigned int)(v90 - 17);
      v93 = (unsigned __int16)v92 <= 0xD3u;
    }
    else
    {
      v93 = sub_30070B0((__int64)&v248);
    }
    if ( !v93 )
    {
      v25 = sub_3775FE0(*(_QWORD **)(v5 + 8), v213, v214, (__int64 *)&v242, 0, v195, a4);
      goto LABEL_76;
    }
    v245 = v247;
    v94 = v195;
    v246 = 0x1000000000LL;
    sub_3376470((__int64)&v245, v195, v92, v66, v67, v68);
    v95 = (__int64)v242;
    v183 = v195 - 1;
    v96 = &v242[4 * (v195 - 1)];
    v97 = *(_QWORD *)(*(_QWORD *)v96 + 48LL) + 16LL * v96[2];
    v98 = *(_WORD *)v97;
    v99 = *(_QWORD *)(v97 + 8);
    v222.m128i_i16[0] = v98;
    v222.m128i_i64[1] = v99;
    if ( v98 )
    {
      if ( (unsigned __int16)(v98 - 17) > 0xD3u )
        goto LABEL_80;
      v100 = v195 - 1;
    }
    else
    {
      v100 = v195 - 1;
      if ( !sub_30070B0((__int64)&v222) )
      {
LABEL_80:
        v101 = v195 - 1;
        v100 = v195 - 2;
        v102 = (unsigned int *)(v95 + 16LL * (int)(v195 - 2));
        v103 = 16LL * (int)(v195 - 2);
        if ( (int)(v195 - 2) >= 0 )
        {
          v198.m128i_i64[0] = v5;
          v104 = v102;
          v105 = (int)(v195 - 2);
          do
          {
            v106 = v105;
            v103 = 16 * v105;
            v107 = *(_QWORD *)(*(_QWORD *)v104 + 48LL) + 16LL * v104[2];
            v108 = *(_WORD *)v107;
            v109 = *(_QWORD *)(v107 + 8);
            v222.m128i_i16[0] = v108;
            v222.m128i_i64[1] = v109;
            if ( v108 )
            {
              if ( (unsigned __int16)(v108 - 17) <= 0xD3u )
                goto LABEL_86;
            }
            else if ( sub_30070B0((__int64)&v222) )
            {
LABEL_86:
              v100 = v105;
              v5 = v198.m128i_i64[0];
              v101 = v105 + 1;
              goto LABEL_87;
            }
            --v105;
            v104 -= 4;
          }
          while ( (int)v105 >= 0 );
          v101 = v106;
          v5 = v198.m128i_i64[0];
          v100 = v106 - 1;
          v103 = 16LL * (int)(v106 - 1);
        }
LABEL_87:
        v205 = sub_3775FE0(*(_QWORD **)(v5 + 8), v222.m128i_u32[0], v222.m128i_i64[1], (__int64 *)&v242, v101, v195, a4);
        v94 = v195 - 1;
        v110 = &v245[16 * v183];
        v206 = v111;
        *(_QWORD *)v110 = v205;
        v112 = (__int64)v242;
        *((_DWORD *)v110 + 2) = v206;
        v96 = (unsigned int *)(v103 + v112);
      }
    }
    v113 = v94 - 1;
    v114 = (int)(v94 - 1);
    v115 = &v245[16 * v114];
    *(_QWORD *)v115 = *(_QWORD *)v96;
    *((_DWORD *)v115 + 2) = v96[2];
    v116 = v100 - 1;
    if ( v100 - 1 >= 0 )
    {
      v193 = 16LL * v116;
      v185 = 16 * (v100 - (unsigned __int64)(unsigned int)v116) - 32;
      do
      {
        v118 = &v242[(unsigned __int64)v193 / 4];
        v119 = *(_QWORD *)(*(_QWORD *)&v242[(unsigned __int64)v193 / 4] + 48LL)
             + 16LL * v242[(unsigned __int64)v193 / 4 + 2];
        v120 = *(_WORD *)v119;
        v121 = *(_QWORD *)(v119 + 8);
        v223.m128i_i16[0] = v120;
        v223.m128i_i64[1] = v121;
        if ( v222.m128i_i16[0] != v120 || v222.m128i_i64[1] != v121 && !v120 )
        {
          v224 = sub_2D5B750((unsigned __int16 *)&v222);
          v225 = v122;
          v123 = sub_2D5B750((unsigned __int16 *)&v223);
          v227 = v126;
          v226 = v123;
          v128 = v123;
          v127 = v123 / v224;
          v248.m128i_i64[0] = (__int64)v249;
          v248.m128i_i64[1] = 0x1000000000LL;
          v198.m128i_i32[0] = v128 / v224;
          v129 = (unsigned int)v127;
          if ( (_DWORD)v127 )
          {
            v130 = v249;
            v131 = v249;
            if ( (unsigned int)v127 > 0x10uLL )
            {
              v179 = v127;
              sub_C8D5F0((__int64)&v248, v249, (unsigned int)v127, 0x10u, v124, v125);
              v131 = (_QWORD *)v248.m128i_i64[0];
              LODWORD(v127) = v179;
              v130 = (_QWORD *)(v248.m128i_i64[0] + 16LL * v248.m128i_u32[2]);
            }
            for ( i = &v131[2 * v129]; i != v130; v130 += 2 )
            {
              if ( v130 )
              {
                *v130 = 0;
                *((_DWORD *)v130 + 2) = 0;
              }
            }
            v248.m128i_i32[2] = v127;
          }
          v133 = 0;
          for ( j = v195 - v113; v195 != v113; v133 += 16 )
          {
            v135 = v113;
            v136 = v248.m128i_i64[0];
            ++v113;
            v137 = &v245[16 * v135];
            *(_QWORD *)(v248.m128i_i64[0] + v133) = *(_QWORD *)v137;
            *(_DWORD *)(v136 + v133 + 8) = *((_DWORD *)v137 + 2);
          }
          if ( j != (_DWORD)v127 )
          {
            do
            {
              v138 = *(_QWORD **)(v5 + 8);
              v232 = 0;
              v233 = 0;
              v139 = sub_33F17F0(v138, 51, (__int64)&v232, v222.m128i_u32[0], v222.m128i_i64[1]);
              v141 = v140;
              if ( v232 )
                sub_B91220((__int64)&v232, v232);
              v142 = j;
              v202 = v141;
              ++j;
              v143 = v248.m128i_i64[0] + 16 * v142;
              v201 = v139;
              *(_QWORD *)v143 = v139;
              *(_DWORD *)(v143 + 8) = v202;
            }
            while ( v198.m128i_i32[0] != j );
          }
          *((_QWORD *)&v170 + 1) = v248.m128i_u32[2];
          *(_QWORD *)&v170 = v248.m128i_i64[0];
          v144 = sub_33FC220(
                   *(_QWORD **)(v5 + 8),
                   159,
                   (__int64)&v217,
                   v223.m128i_u32[0],
                   v223.m128i_i64[1],
                   v125,
                   v170);
          v146 = v145;
          v147 = v144;
          v148 = (unsigned __int64)v245;
          v204 = v146;
          v203 = v147;
          *(_QWORD *)&v245[16 * v183] = v147;
          *(_DWORD *)(v148 + 16LL * v183 + 8) = v204;
          v222 = _mm_load_si128(&v223);
          if ( (_QWORD *)v248.m128i_i64[0] != v249 )
            _libc_free(v248.m128i_u64[0]);
          v113 = v195 - 1;
          v118 = &v242[(unsigned __int64)v193 / 4];
        }
        --v113;
        v193 -= 16;
        v114 = (int)v113;
        v117 = &v245[16 * v113];
        *(_QWORD *)v117 = *(_QWORD *)v118;
        *((_DWORD *)v117 + 2) = v118[2];
      }
      while ( v185 != v193 );
    }
    v198.m128i_i64[0] = v114;
    v149 = v113;
    v150.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v222);
    v152 = v195 - v113;
    v153 = v195 - v113;
    v248 = v150;
    if ( v196 == v153 * v150.m128i_i64[0] && v184 == v248.m128i_i8[8] )
    {
      *((_QWORD *)&v172 + 1) = v152;
      *(_QWORD *)&v172 = &v245[16 * v198.m128i_i64[0]];
      v25 = sub_33FC220(*(_QWORD **)(v5 + 8), 159, (__int64)&v217, v213, v214, v151, v172);
    }
    else
    {
      v154 = sub_2D5B750((unsigned __int16 *)&v222);
      v155 = v196 % v154;
      v248.m128i_i64[0] = (__int64)v249;
      v156 = v196 / v154;
      v197 = v196 / v154;
      v198.m128i_i64[0] = v156;
      v248.m128i_i64[1] = 0x1000000000LL;
      sub_3376470((__int64)&v248, (unsigned int)v156, v155, v156, v157, v158);
      v159 = sub_3288990(*(_QWORD *)(v5 + 8), v222.m128i_u32[0], v222.m128i_i64[1]);
      v160 = v198.m128i_i32[0];
      v161 = v159;
      v163 = v162;
      if ( v152 )
      {
        v164 = 0;
        do
        {
          v165 = v149;
          v166 = v248.m128i_i64[0];
          ++v149;
          v167 = &v245[16 * v165];
          *(_QWORD *)(v248.m128i_i64[0] + v164) = *(_QWORD *)v167;
          *(_DWORD *)(v166 + v164 + 8) = *((_DWORD *)v167 + 2);
          v164 += 16;
        }
        while ( v195 != v149 );
      }
      if ( v160 != v152 )
      {
        while ( 1 )
        {
          v168 = v248.m128i_i64[0] + 16 * v153;
          v200 = v163;
          ++v152;
          v199 = v161;
          *(_QWORD *)v168 = v161;
          *(_DWORD *)(v168 + 8) = v200;
          if ( v197 == v152 )
            break;
          v153 = v152;
        }
      }
      *((_QWORD *)&v171 + 1) = v248.m128i_u32[2];
      *(_QWORD *)&v171 = v248.m128i_i64[0];
      v25 = sub_33FC220(*(_QWORD **)(v5 + 8), 159, (__int64)&v217, v213, v214, v163, v171);
      if ( (_QWORD *)v248.m128i_i64[0] != v249 )
        _libc_free(v248.m128i_u64[0]);
    }
    if ( v245 != v247 )
      _libc_free((unsigned __int64)v245);
LABEL_76:
    v56 = (unsigned __int64)v242;
    if ( v242 == (unsigned int *)v244 )
      goto LABEL_42;
LABEL_41:
    _libc_free(v56);
    goto LABEL_42;
  }
  v46 = v234.m128i_i16[0];
  if ( v234.m128i_i16[0] )
  {
    if ( (unsigned __int16)(v234.m128i_i16[0] - 17) <= 0xD3u )
      goto LABEL_36;
LABEL_60:
    v60 = sub_327FCF0(*(__int64 **)(*(_QWORD *)(v5 + 8) + 64LL), v234.m128i_u32[0], v234.m128i_i64[1], v196 / v181, 0);
    sub_33FAF80(*(_QWORD *)(v5 + 8), 167, (__int64)&v217, v60, v61, v62, a4);
    v25 = sub_33FAF80(*(_QWORD *)(v5 + 8), 234, (__int64)&v217, v213, v214, v63, a4);
    goto LABEL_42;
  }
  if ( !sub_30070B0((__int64)&v234) )
    goto LABEL_60;
LABEL_36:
  if ( !(_BYTE)v235 || v46 != (_WORD)v213 || !v46 && v214 != v234.m128i_i64[1] )
  {
    v248.m128i_i64[0] = (__int64)v249;
    v248.m128i_i64[1] = 0x1000000000LL;
    sub_3376470((__int64)&v248, (unsigned int)(v196 / v181), v196 % v181, v45, v43, v44);
    v47 = sub_3288990(*(_QWORD *)(v5 + 8), v234.m128i_u32[0], v234.m128i_i64[1]);
    v50 = v49;
    v51 = v248.m128i_i64[0];
    v209 = _mm_load_si128(&v198);
    *(_QWORD *)v248.m128i_i64[0] = v198.m128i_i64[0];
    *(_DWORD *)(v51 + 8) = v209.m128i_i32[2];
    v52 = 1;
    if ( (unsigned int)(v196 / v181) != 1 )
    {
      do
      {
        v53 = v52;
        v208 = v50;
        ++v52;
        v54 = v248.m128i_i64[0] + 16 * v53;
        v207 = v47;
        *(_QWORD *)v54 = v47;
        *(_DWORD *)(v54 + 8) = v208;
      }
      while ( (unsigned int)(v196 / v181) != v52 );
    }
    *((_QWORD *)&v169 + 1) = v248.m128i_u32[2];
    *(_QWORD *)&v169 = v248.m128i_i64[0];
    v55 = sub_33FC220(*(_QWORD **)(v5 + 8), 159, (__int64)&v217, v213, v214, v48, v169);
    v56 = v248.m128i_i64[0];
    v25 = v55;
    if ( (_QWORD *)v248.m128i_i64[0] == v249 )
      goto LABEL_42;
    goto LABEL_41;
  }
  v25 = (unsigned __int8 *)v198.m128i_i64[0];
LABEL_42:
  if ( v239 != (__int64 *)v241 )
    _libc_free((unsigned __int64)v239);
LABEL_14:
  if ( v217 )
    sub_B91220((__int64)&v217, v217);
  return v25;
}
