// Function: sub_346E5E0
// Address: 0x346e5e0
//
__m128i *__fastcall sub_346E5E0(__int64 a1, __int64 a2, __m128i *a3, __m128i a4)
{
  unsigned int v4; // r12d
  __int64 v7; // rsi
  unsigned __int64 *v8; // rax
  unsigned int v9; // ecx
  unsigned __int64 v10; // rbx
  __int64 v11; // rax
  unsigned __int16 v12; // r14
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned __int16 v15; // r14
  __int64 v16; // rax
  unsigned int v17; // r14d
  __int64 v18; // rsi
  __int64 v19; // rdx
  char v20; // al
  _QWORD *v21; // rax
  unsigned __int64 v22; // rdx
  __int64 (__fastcall *v23)(__int64, __int64, __int64, __int64); // r14
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 (__fastcall *v26)(__int64, __int64, unsigned int); // rax
  int v27; // eax
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rdx
  __int64 v32; // rsi
  unsigned int v33; // eax
  unsigned __int64 v34; // r15
  __int128 v35; // rax
  __int64 v36; // r9
  unsigned __int8 *v37; // rax
  unsigned __int64 v38; // rdx
  unsigned __int8 *v39; // rax
  __int64 *v40; // rsi
  unsigned __int64 v41; // rdx
  __int64 v42; // rax
  unsigned __int8 v43; // al
  unsigned int v44; // edx
  __int64 v45; // r9
  unsigned __int8 *v46; // rax
  unsigned int v47; // edx
  __int64 v48; // rcx
  __int64 v49; // r8
  int v50; // r9d
  unsigned __int8 *v51; // rax
  __int64 v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // r8
  int v55; // r9d
  unsigned int v56; // edx
  int v57; // r9d
  unsigned int v58; // edx
  unsigned int v59; // eax
  int v60; // r9d
  unsigned int v61; // edx
  unsigned int v62; // eax
  __int64 v63; // r9
  unsigned int v64; // edx
  __m128i *v65; // r12
  __int64 v67; // rdx
  __int64 v68; // rcx
  __int64 v69; // r8
  unsigned __int16 v70; // ax
  __int64 v71; // rdx
  bool v72; // al
  __int64 v73; // r8
  unsigned __int16 v74; // ax
  int v75; // eax
  unsigned int v76; // r15d
  __int64 v77; // rdx
  __int128 v78; // rax
  __int64 v79; // r9
  __int64 v80; // r15
  unsigned int v81; // eax
  unsigned int v82; // edx
  __int64 v83; // r9
  unsigned int v84; // edx
  unsigned int v85; // edx
  __int64 v86; // r8
  __m128i *v87; // r10
  unsigned int v88; // ecx
  __int64 v89; // rdx
  __int64 v90; // r11
  __int16 v91; // ax
  __int64 v92; // rdx
  unsigned int v93; // esi
  __int64 v94; // rax
  __int64 *v95; // rsi
  unsigned int v96; // edx
  __int64 v97; // rdi
  __int64 v98; // rax
  __m128i v99; // xmm3
  __int64 v100; // rdx
  __int64 v101; // r8
  unsigned __int8 v102; // al
  unsigned int v103; // edx
  __m128i v104; // xmm4
  unsigned __int16 *v105; // rax
  __int64 v106; // rsi
  __int64 v107; // rdx
  unsigned __int8 v108; // al
  __m128i *v109; // rax
  __int64 v110; // rdx
  __int64 v111; // r15
  __int64 v112; // rcx
  __int64 v113; // r8
  __int64 v114; // rdx
  __int64 v115; // rbx
  int v116; // esi
  __int64 v117; // rax
  int v118; // r9d
  __int64 v119; // r8
  __int64 v120; // rcx
  __int64 v121; // rdx
  int v122; // esi
  int v123; // eax
  int v124; // r9d
  __int64 v125; // r11
  __int64 v126; // r8
  unsigned int v127; // ecx
  unsigned int v128; // edx
  unsigned int v129; // edx
  unsigned __int8 *v130; // rax
  __int64 *v131; // rsi
  __int64 v132; // rdx
  unsigned int v133; // edx
  bool v134; // al
  unsigned int v135; // edx
  __int64 v136; // rdx
  __int64 v137; // rdx
  __int128 v138; // [rsp+0h] [rbp-350h]
  __int128 v139; // [rsp+0h] [rbp-350h]
  __int128 v140; // [rsp+0h] [rbp-350h]
  __int64 v141; // [rsp+8h] [rbp-348h]
  __int16 v142; // [rsp+12h] [rbp-33Eh]
  __int64 v143; // [rsp+18h] [rbp-338h]
  __int16 v144; // [rsp+22h] [rbp-32Eh]
  __int64 v145; // [rsp+28h] [rbp-328h]
  unsigned int v146; // [rsp+30h] [rbp-320h]
  __m128i *v147; // [rsp+38h] [rbp-318h]
  __int64 v148; // [rsp+38h] [rbp-318h]
  unsigned int v149; // [rsp+40h] [rbp-310h]
  unsigned int v150; // [rsp+44h] [rbp-30Ch]
  unsigned int v151; // [rsp+48h] [rbp-308h]
  __m128i *v152; // [rsp+50h] [rbp-300h]
  __int64 v153; // [rsp+60h] [rbp-2F0h]
  __int64 v154; // [rsp+68h] [rbp-2E8h]
  __int64 v155; // [rsp+70h] [rbp-2E0h]
  unsigned int v156; // [rsp+78h] [rbp-2D8h]
  unsigned __int64 v157; // [rsp+80h] [rbp-2D0h]
  __int64 v158; // [rsp+88h] [rbp-2C8h]
  __int64 v159; // [rsp+90h] [rbp-2C0h]
  unsigned __int64 v160; // [rsp+98h] [rbp-2B8h]
  __int128 v162; // [rsp+B0h] [rbp-2A0h]
  unsigned __int64 v163; // [rsp+B0h] [rbp-2A0h]
  __int16 v164; // [rsp+C2h] [rbp-28Eh]
  __int16 v165; // [rsp+CAh] [rbp-286h]
  unsigned __int64 v166; // [rsp+D0h] [rbp-280h]
  __int128 v167; // [rsp+D0h] [rbp-280h]
  __int16 v168; // [rsp+D2h] [rbp-27Eh]
  __int64 v169; // [rsp+E0h] [rbp-270h]
  unsigned __int64 v170; // [rsp+E0h] [rbp-270h]
  __int64 v171; // [rsp+E8h] [rbp-268h]
  unsigned __int64 v172; // [rsp+E8h] [rbp-268h]
  unsigned __int64 v173; // [rsp+E8h] [rbp-268h]
  __m128i *v174; // [rsp+F0h] [rbp-260h]
  __int128 v175; // [rsp+F0h] [rbp-260h]
  unsigned __int64 v176; // [rsp+100h] [rbp-250h]
  unsigned __int64 v177; // [rsp+108h] [rbp-248h]
  int v178; // [rsp+110h] [rbp-240h]
  unsigned __int16 v179; // [rsp+116h] [rbp-23Ah]
  unsigned __int64 v180; // [rsp+118h] [rbp-238h]
  unsigned int v181; // [rsp+118h] [rbp-238h]
  unsigned int v182; // [rsp+118h] [rbp-238h]
  unsigned int v183; // [rsp+120h] [rbp-230h]
  unsigned __int64 v184; // [rsp+120h] [rbp-230h]
  unsigned int v185; // [rsp+120h] [rbp-230h]
  unsigned int v186; // [rsp+120h] [rbp-230h]
  __m128i *v187; // [rsp+120h] [rbp-230h]
  __int64 v188; // [rsp+128h] [rbp-228h]
  __int64 v189; // [rsp+128h] [rbp-228h]
  unsigned __int64 v190; // [rsp+128h] [rbp-228h]
  __int64 v191; // [rsp+128h] [rbp-228h]
  __m128i v192; // [rsp+130h] [rbp-220h]
  __m128i v193; // [rsp+140h] [rbp-210h]
  unsigned __int64 v194; // [rsp+150h] [rbp-200h]
  __int64 v195; // [rsp+150h] [rbp-200h]
  unsigned int v196; // [rsp+150h] [rbp-200h]
  __int64 v197; // [rsp+150h] [rbp-200h]
  __int64 v198; // [rsp+158h] [rbp-1F8h]
  __int128 v199; // [rsp+160h] [rbp-1F0h]
  __int64 v200; // [rsp+160h] [rbp-1F0h]
  unsigned __int8 *v201; // [rsp+170h] [rbp-1E0h]
  unsigned __int8 *v202; // [rsp+1A0h] [rbp-1B0h]
  unsigned __int8 *v203; // [rsp+1D0h] [rbp-180h]
  __int64 v204; // [rsp+1F0h] [rbp-160h] BYREF
  int v205; // [rsp+1F8h] [rbp-158h]
  unsigned int v206; // [rsp+200h] [rbp-150h] BYREF
  unsigned __int64 v207; // [rsp+208h] [rbp-148h]
  __int64 v208; // [rsp+210h] [rbp-140h] BYREF
  __int64 v209; // [rsp+218h] [rbp-138h]
  unsigned __int16 v210; // [rsp+220h] [rbp-130h] BYREF
  __int64 v211; // [rsp+228h] [rbp-128h]
  unsigned __int64 v212; // [rsp+230h] [rbp-120h] BYREF
  unsigned int v213; // [rsp+238h] [rbp-118h]
  unsigned __int64 v214; // [rsp+240h] [rbp-110h]
  __int64 v215; // [rsp+248h] [rbp-108h]
  __int64 v216; // [rsp+250h] [rbp-100h]
  __int64 v217; // [rsp+258h] [rbp-F8h]
  __m128i v218; // [rsp+260h] [rbp-F0h] BYREF
  __int64 v219; // [rsp+270h] [rbp-E0h]
  __int128 v220; // [rsp+280h] [rbp-D0h] BYREF
  __int64 v221; // [rsp+290h] [rbp-C0h]
  __m128i v222; // [rsp+2A0h] [rbp-B0h] BYREF
  __int64 v223; // [rsp+2B0h] [rbp-A0h]
  __m128i v224; // [rsp+2C0h] [rbp-90h] BYREF
  __int64 v225; // [rsp+2D0h] [rbp-80h]
  __int128 v226; // [rsp+2E0h] [rbp-70h]
  __int64 v227; // [rsp+2F0h] [rbp-60h]
  __int64 v228; // [rsp+300h] [rbp-50h] BYREF
  __int64 v229; // [rsp+308h] [rbp-48h]
  __int64 v230; // [rsp+310h] [rbp-40h]
  __int64 v231; // [rsp+318h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 80);
  v204 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v204, v7, 1);
  v205 = *(_DWORD *)(a2 + 72);
  v8 = *(unsigned __int64 **)(a2 + 40);
  v160 = v8[5];
  v9 = *((_DWORD *)v8 + 12);
  v157 = *v8;
  v194 = v8[10];
  v10 = *((unsigned int *)v8 + 22);
  v192 = _mm_loadu_si128((const __m128i *)v8);
  v193 = _mm_loadu_si128((const __m128i *)(v8 + 5));
  v155 = *((unsigned int *)v8 + 2);
  v11 = *(_QWORD *)(*v8 + 48) + 16 * v155;
  v12 = *(_WORD *)v11;
  v13 = *(_QWORD *)(v11 + 8);
  LOWORD(v206) = v12;
  v207 = v13;
  if ( v12 )
  {
    if ( (unsigned __int16)(v12 - 17) <= 0xD3u )
    {
      v12 = word_4456580[v12 - 1];
      v13 = 0;
    }
  }
  else
  {
    v185 = v9;
    v200 = v13;
    v72 = sub_30070B0((__int64)&v206);
    v13 = v200;
    v9 = v185;
    if ( v72 )
    {
      v74 = sub_3009970((__int64)&v206, v7, v200, v185, v73);
      v9 = v185;
      v12 = v74;
    }
  }
  v209 = v13;
  v154 = v9;
  v14 = *(_QWORD *)(v160 + 48) + 16LL * v9;
  LOWORD(v208) = v12;
  v15 = *(_WORD *)v14;
  v16 = *(_QWORD *)(v14 + 8);
  v210 = v15;
  v158 = v16;
  v211 = v16;
  if ( v15 )
  {
    if ( (unsigned __int16)(v15 - 17) <= 0xD3u )
    {
      v158 = 0;
      v15 = word_4456580[v15 - 1];
    }
  }
  else if ( sub_30070B0((__int64)&v210) )
  {
    v70 = sub_3009970((__int64)&v210, v7, v67, v68, v69);
    v158 = v71;
    v15 = v70;
  }
  v156 = v15;
  if ( (_WORD)v206 )
  {
    if ( (unsigned __int16)(v206 - 176) > 0x34u )
      goto LABEL_11;
LABEL_42:
    sub_C64ED0("Cannot expand masked_compress for scalable vectors.", 1u);
  }
  if ( sub_3007100((__int64)&v206) )
    goto LABEL_42;
LABEL_11:
  v17 = sub_33CD850((__int64)a3, v206, v207, 0);
  if ( (_WORD)v206 )
  {
    if ( (_WORD)v206 == 1 || (unsigned __int16)(v206 - 504) <= 7u )
      BUG();
    v18 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v206 - 16];
    v20 = byte_444C4A0[16 * (unsigned __int16)v206 - 8];
  }
  else
  {
    v216 = sub_3007260((__int64)&v206);
    v18 = v216;
    v217 = v19;
    v20 = v19;
  }
  LOBYTE(v215) = v20;
  v214 = (unsigned __int64)(v18 + 7) >> 3;
  v21 = sub_33EDE90((__int64)a3, v214, v215, v17);
  v177 = v22;
  v176 = (unsigned __int64)v21;
  v159 = (__int64)v21;
  sub_2EAC300((__int64)&v218, a3[2].m128i_i64[1], *((_DWORD *)v21 + 24), 0);
  v23 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)a1 + 72LL);
  v24 = sub_2E79000((__int64 *)a3[2].m128i_i64[1]);
  v25 = v24;
  if ( v23 == sub_2FE4D20 )
  {
    v26 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)a1 + 32LL);
    if ( v26 == sub_2D42F30 )
    {
      v27 = sub_AE2980(v25, 0)[1];
      v179 = 2;
      if ( v27 != 1 )
      {
        v179 = 3;
        if ( v27 != 2 )
        {
          v179 = 4;
          if ( v27 != 4 )
          {
            v179 = 5;
            if ( v27 != 8 )
            {
              v179 = 6;
              if ( v27 != 16 )
              {
                v179 = 7;
                if ( v27 != 32 )
                {
                  v179 = 8;
                  if ( v27 != 64 )
                    v179 = 9 * (v27 == 128);
                }
              }
            }
          }
        }
      }
    }
    else
    {
      v179 = v26(a1, v25, 0);
    }
  }
  else
  {
    v179 = ((__int64 (__fastcall *)(__int64, __int64))v23)(a1, v24);
  }
  v174 = a3 + 18;
  *(_QWORD *)&v199 = sub_3400BD0((__int64)a3, 0, (__int64)&v204, v179, 0, 0, a4, 0);
  *((_QWORD *)&v199 + 1) = v31;
  v178 = *(_DWORD *)(v194 + 24);
  if ( v178 == 51 )
  {
    v213 = 1;
    v212 = 0;
    v32 = (__int64)&v212;
    if ( !sub_33D1410(v194, (__int64)&v212, v141, v28, v29) )
    {
      v149 = 0;
      v183 = 0;
      v147 = 0;
      goto LABEL_26;
    }
    v183 = 0;
    goto LABEL_74;
  }
  v228 = 0;
  v104 = _mm_loadu_si128(&v218);
  v229 = 0;
  v230 = 0;
  v231 = 0;
  v105 = (unsigned __int16 *)(*(_QWORD *)(v194 + 48) + 16LL * (unsigned int)v10);
  v227 = v219;
  v106 = *v105;
  v107 = *((_QWORD *)v105 + 1);
  v226 = (__int128)v104;
  v108 = sub_33CC4A0((__int64)a3, v106, v107, v28, v29, v30);
  v109 = sub_33F4560(
           a3,
           (unsigned __int64)v174,
           0,
           (__int64)&v204,
           v194,
           v10,
           v176,
           v177,
           v226,
           v227,
           v108,
           0,
           (__int64)&v228);
  v181 = v110;
  v111 = (__int64)v109;
  v174 = v109;
  v183 = v110;
  v213 = 1;
  v212 = 0;
  if ( sub_33D1410(v194, (__int64)&v212, v110, v112, v113) )
  {
LABEL_74:
    v32 = (__int64)&v212;
    v147 = (__m128i *)sub_34007B0((__int64)a3, (__int64)&v212, (__int64)&v204, v208, v209, 0, a4, 0);
    v149 = v135;
    goto LABEL_26;
  }
  v196 = sub_327FF20((unsigned __int16 *)&v208, (__int64)&v212);
  v115 = v114;
  if ( v210 )
  {
    v116 = word_4456340[v210 - 1];
    if ( (unsigned __int16)(v210 - 176) > 0x34u )
      LOWORD(v117) = sub_2D43050(2, v116);
    else
      LOWORD(v117) = sub_2D43AD0(2, v116);
    v119 = 0;
  }
  else
  {
    v117 = sub_3009490(&v210, 2u, 0);
    v169 = v117;
    v119 = v136;
  }
  v120 = v169;
  LOWORD(v120) = v117;
  sub_33FAF80((__int64)a3, 216, (__int64)&v204, v120, v119, v118, a4);
  if ( v210 )
  {
    v188 = v121;
    v122 = word_4456340[v210 - 1];
    if ( (unsigned __int16)(v210 - 176) > 0x34u )
      LOWORD(v123) = sub_2D43050(v196, v122);
    else
      LOWORD(v123) = sub_2D43AD0(v196, v122);
    v125 = v188;
    v126 = 0;
  }
  else
  {
    v191 = v121;
    v123 = sub_3009490(&v210, v196, v115);
    v125 = v191;
    v168 = HIWORD(v123);
    v126 = v137;
  }
  HIWORD(v127) = v168;
  LOWORD(v127) = v123;
  v189 = v125;
  sub_33FAF80((__int64)a3, 214, (__int64)&v204, v127, v126, v124, a4);
  v190 = v128 | v189 & 0xFFFFFFFF00000000LL;
  v203 = sub_33FAF80((__int64)a3, 382, (__int64)&v204, v196, v115, 0, a4);
  *((_QWORD *)&v140 + 1) = v129 | v190 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v140 = v203;
  v130 = sub_3466750(a1, a3, v176, v177, v206, v207, a4, v140);
  v131 = (__int64 *)a3[2].m128i_i64[1];
  v228 = 0;
  v197 = (__int64)v130;
  v198 = v132;
  v229 = 0;
  v230 = 0;
  v231 = 0;
  sub_2EAC3A0((__int64)&v220, v131);
  v32 = (unsigned int)v208;
  v143 = v181;
  v183 = 1;
  v147 = sub_33F1F00(
           a3->m128i_i64,
           (unsigned int)v208,
           v209,
           (__int64)&v204,
           v111,
           v181,
           v197,
           v198,
           v220,
           v221,
           0,
           0,
           (__int64)&v228,
           0);
  v149 = v133;
  v174 = v147;
LABEL_26:
  if ( (_WORD)v206 )
  {
    if ( (unsigned __int16)(v206 - 176) > 0x34u )
    {
LABEL_28:
      v33 = word_4456340[(unsigned __int16)v206 - 1];
      goto LABEL_29;
    }
  }
  else if ( !sub_3007100((__int64)&v206) )
  {
    goto LABEL_54;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( (_WORD)v206 )
  {
    if ( (unsigned __int16)(v206 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
    goto LABEL_28;
  }
LABEL_54:
  v33 = sub_3007130((__int64)&v206, v32);
LABEL_29:
  if ( v33 )
  {
    v153 = v33;
    v195 = 0;
    v34 = v183;
    v150 = v33 - 1;
    while ( 1 )
    {
      *(_QWORD *)&v35 = sub_3400EE0((__int64)a3, v195, (__int64)&v204, 0, a4);
      v162 = v35;
      v192.m128i_i64[1] = v155 | v192.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      v37 = sub_3406EB0(
              a3,
              0x9Eu,
              (__int64)&v204,
              (unsigned int)v208,
              v209,
              v36,
              __PAIR128__(v192.m128i_u64[1], v157),
              v35);
      v180 = v38;
      v184 = (unsigned __int64)v37;
      v39 = sub_3466750(a1, a3, v159, v177, v206, v207, a4, v199);
      v40 = (__int64 *)a3[2].m128i_i64[1];
      v228 = 0;
      v170 = (unsigned __int64)v39;
      v166 = v41;
      v229 = 0;
      v230 = 0;
      v231 = 0;
      sub_2EAC3A0((__int64)&v222, v40);
      a4 = _mm_loadu_si128(&v222);
      v227 = v223;
      v226 = (__int128)a4;
      v42 = *(_QWORD *)(v184 + 48) + 16LL * (unsigned int)v180;
      LOWORD(v4) = *(_WORD *)v42;
      v43 = sub_33CC4A0((__int64)a3, v4, *(_QWORD *)(v42 + 8), v184, v170, v166);
      v152 = sub_33F4560(
               a3,
               (unsigned __int64)v174,
               v34,
               (__int64)&v204,
               v184,
               v180,
               v170,
               v166,
               v226,
               v227,
               v43,
               0,
               (__int64)&v228);
      v34 = v44;
      v174 = v152;
      v151 = v44;
      v193.m128i_i64[1] = v154 | v193.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      v46 = sub_3406EB0(a3, 0x9Eu, (__int64)&v204, v156, v158, v45, __PAIR128__(v193.m128i_u64[1], v160), v162);
      v51 = sub_33FB960((__int64)a3, (__int64)v46, v47, a4, v48, v49, v50);
      v171 = v52;
      sub_33FB960((__int64)a3, (__int64)v51, v52, a4, v53, v54, v55);
      v172 = v56 | v171 & 0xFFFFFFFF00000000LL;
      sub_33FAF80((__int64)a3, 216, (__int64)&v204, 2, 0, v57, a4);
      HIWORD(v59) = v164;
      LOWORD(v59) = v179;
      v173 = v58 | v172 & 0xFFFFFFFF00000000LL;
      v202 = sub_33FAF80((__int64)a3, 214, (__int64)&v204, v59, 0, v60, a4);
      HIWORD(v62) = v165;
      LOWORD(v62) = v179;
      *((_QWORD *)&v138 + 1) = v61 | v173 & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&v138 = v202;
      *(_QWORD *)&v199 = sub_3406EB0(a3, 0x38u, (__int64)&v204, v62, 0, v63, v199, v138);
      *((_QWORD *)&v199 + 1) = v64 | *((_QWORD *)&v199 + 1) & 0xFFFFFFFF00000000LL;
      if ( v178 != 51 && v150 == (_DWORD)v195 )
        break;
LABEL_33:
      if ( ++v195 == v153 )
      {
        v183 = v34;
        goto LABEL_35;
      }
    }
    if ( (_WORD)v206 )
    {
      if ( (unsigned __int16)(v206 - 176) > 0x34u )
      {
LABEL_49:
        v75 = word_4456340[(unsigned __int16)v206 - 1];
LABEL_50:
        HIWORD(v76) = v142;
        *(_QWORD *)&v175 = sub_3400BD0((__int64)a3, (unsigned int)(v75 - 1), (__int64)&v204, v179, 0, 0, a4, 0);
        *((_QWORD *)&v175 + 1) = v77;
        LOWORD(v76) = 2;
        *(_QWORD *)&v78 = sub_33ED040(a3, 0xAu);
        v80 = sub_340F900(a3, 0xD0u, (__int64)&v204, v76, 0, v79, v199, v175, v78);
        HIWORD(v81) = v144;
        LOWORD(v81) = v179;
        v146 = v82;
        *(_QWORD *)&v199 = sub_3406EB0(a3, 0xB6u, (__int64)&v204, v81, 0, v83, v199, v175);
        *((_QWORD *)&v199 + 1) = v84 | *((_QWORD *)&v199 + 1) & 0xFFFFFFFF00000000LL;
        v201 = sub_3466750(a1, a3, v159, v177, v206, v207, a4, v199);
        v86 = v209;
        v87 = v147;
        v163 = v85 | v166 & 0xFFFFFFFF00000000LL;
        v88 = v208;
        v89 = *(_QWORD *)(v80 + 48) + 16LL * v146;
        *(_QWORD *)&v167 = v184;
        *((_QWORD *)&v167 + 1) = v180;
        v90 = v149;
        v91 = *(_WORD *)v89;
        v92 = *(_QWORD *)(v89 + 8);
        LOWORD(v228) = v91;
        v229 = v92;
        if ( v91 )
        {
          v93 = ((unsigned __int16)(v91 - 17) < 0xD4u) + 205;
        }
        else
        {
          v148 = v209;
          v182 = v208;
          v187 = v87;
          v134 = sub_30070B0((__int64)&v228);
          v86 = v148;
          v88 = v182;
          v87 = v187;
          v90 = v149;
          v93 = 205 - (!v134 - 1);
        }
        *((_QWORD *)&v139 + 1) = v90;
        *(_QWORD *)&v139 = v87;
        v94 = sub_340EC60(a3, v93, (__int64)&v204, v88, v86, 0x2000, v80, v146, v167, v139);
        v95 = (__int64 *)a3[2].m128i_i64[1];
        v147 = (__m128i *)v94;
        v149 = v96;
        v186 = v96;
        v228 = 0;
        v229 = 0;
        v230 = 0;
        v231 = 0;
        sub_2EAC3A0((__int64)&v224, v95);
        v97 = v145;
        v98 = v147[3].m128i_i64[0] + 16LL * v186;
        v99 = _mm_loadu_si128(&v224);
        LOWORD(v97) = *(_WORD *)v98;
        v227 = v225;
        v100 = *(_QWORD *)(v98 + 8);
        v226 = (__int128)v99;
        v145 = v97;
        v102 = sub_33CC4A0((__int64)a3, (unsigned int)v97, v100, (__int64)v147, v101, v186);
        v174 = sub_33F4560(
                 a3,
                 (unsigned __int64)v152,
                 v151,
                 (__int64)&v204,
                 (unsigned __int64)v147,
                 v186,
                 (unsigned __int64)v201,
                 v163,
                 v226,
                 v227,
                 v102,
                 0,
                 (__int64)&v228);
        v34 = v103;
        goto LABEL_33;
      }
    }
    else if ( !sub_3007100((__int64)&v206) )
    {
      goto LABEL_68;
    }
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( (_WORD)v206 )
    {
      if ( (unsigned __int16)(v206 - 176) <= 0x34u )
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
      goto LABEL_49;
    }
LABEL_68:
    v75 = sub_3007130((__int64)&v206, 56);
    goto LABEL_50;
  }
LABEL_35:
  v228 = 0;
  v229 = 0;
  v230 = 0;
  v231 = 0;
  v65 = sub_33F1F00(
          a3->m128i_i64,
          v206,
          v207,
          (__int64)&v204,
          (__int64)v174,
          v183 | v143 & 0xFFFFFFFF00000000LL,
          v159,
          v177,
          *(_OWORD *)&v218,
          v219,
          0,
          0,
          (__int64)&v228,
          0);
  if ( v213 > 0x40 && v212 )
    j_j___libc_free_0_0(v212);
  if ( v204 )
    sub_B91220((__int64)&v204, v204);
  return v65;
}
