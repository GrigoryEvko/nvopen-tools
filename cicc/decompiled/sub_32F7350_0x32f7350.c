// Function: sub_32F7350
// Address: 0x32f7350
//
__int64 __fastcall sub_32F7350(
        _QWORD *a1,
        __int64 **a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        char a6,
        char a7,
        char a8)
{
  __int64 v8; // rbx
  __int64 v10; // r13
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned int v14; // r13d
  int v15; // r9d
  __int64 v16; // r14
  unsigned int v17; // r12d
  char v18; // r13
  __int64 v19; // rax
  int v20; // ebx
  __int64 v21; // rdx
  __int64 v22; // r13
  __int16 v24; // cx
  __int64 v25; // r11
  __int64 v26; // r10
  __m128i *v27; // rsi
  __int64 v28; // rdx
  int v29; // r14d
  int v30; // eax
  int v31; // edx
  int v32; // ebx
  __int64 v33; // rbx
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // r14
  unsigned int v37; // eax
  unsigned int v38; // ecx
  __int64 v39; // rax
  unsigned __int64 v40; // rax
  int v41; // eax
  __int64 v42; // r13
  unsigned int v43; // eax
  __int64 v44; // rax
  unsigned int v45; // edx
  unsigned int v46; // ebx
  __int64 v47; // r13
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  __m128i v51; // xmm3
  unsigned __int16 v52; // cx
  unsigned int v53; // r14d
  __int64 v54; // r13
  unsigned int v55; // ebx
  __int64 *v56; // r12
  int v57; // eax
  __int64 v58; // r8
  __int64 v59; // r9
  int v60; // ebx
  __int64 v61; // r14
  __int64 v62; // rax
  __int64 v63; // rdx
  __int64 v64; // rdi
  unsigned __int64 v65; // r12
  __int64 v66; // rbx
  __int64 v67; // rax
  __int64 v68; // rax
  unsigned __int64 v69; // r12
  unsigned __int64 v70; // rdx
  __int64 *v71; // rax
  unsigned __int32 v72; // eax
  __int64 v73; // r14
  __int64 v74; // r13
  __int64 v75; // r12
  int v76; // esi
  __int64 v77; // rax
  unsigned int v78; // edx
  __int64 v79; // r12
  int v80; // edx
  int v81; // r14d
  __int64 v82; // rbx
  __int64 (__fastcall *v83)(__int64, __int64, unsigned int, __int64); // r10
  __int64 v84; // rax
  unsigned __int16 v85; // si
  __int64 v86; // r8
  __m128i v87; // rax
  unsigned int v88; // eax
  int v89; // eax
  int v90; // edx
  char v91; // al
  int v92; // r10d
  char v93; // dl
  __int64 v94; // rcx
  __int64 v95; // rbx
  const __m128i *v96; // roff
  int v97; // eax
  __int64 *v98; // rsi
  __int64 *v99; // rax
  __int64 v100; // rbx
  int v101; // edx
  int v102; // r12d
  __int64 v103; // rbx
  __int64 *v104; // rax
  __int64 v105; // rsi
  __int64 v106; // r8
  __int64 v107; // r9
  __int64 v108; // rcx
  unsigned int v109; // edx
  __int64 v110; // rbx
  __int64 v111; // r13
  unsigned int v112; // r12d
  __int64 v113; // rdx
  unsigned __int16 v114; // ax
  __int64 v115; // rdx
  __int64 v116; // rax
  __int64 v117; // rsi
  __int64 v118; // rdx
  __int64 v119; // rdx
  __int64 v120; // rax
  __int64 v121; // rax
  __int64 v122; // rdx
  __int64 v123; // rdx
  __int8 v124; // al
  unsigned int v125; // eax
  int v126; // eax
  __int64 v127; // rcx
  __int64 v128; // rsi
  int v129; // edx
  __int64 v130; // rbx
  __int64 v131; // r13
  unsigned __int16 v132; // ax
  __int64 v133; // rax
  __int64 v134; // rdx
  unsigned int v135; // eax
  unsigned int v136; // eax
  unsigned int v137; // edx
  __int64 v138; // rbx
  __int64 v139; // r12
  unsigned __int64 v140; // rdx
  __int64 v141; // rax
  unsigned __int64 v142; // r13
  unsigned __int64 v143; // rdx
  __int64 *v144; // rax
  unsigned __int32 v145; // eax
  __int64 v146; // rax
  __int64 v147; // r12
  __int64 v148; // rcx
  __int64 v149; // rax
  __int64 v150; // r13
  __int64 v151; // r14
  unsigned __int16 *v152; // rsi
  __int64 v153; // rdx
  __int64 v154; // r8
  unsigned int v155; // esi
  int v156; // edi
  __int64 *v157; // rax
  __int64 v158; // rsi
  __int64 v159; // r10
  __int64 v160; // r11
  __int64 v161; // r8
  __int64 v162; // r9
  unsigned int v163; // edx
  int v164; // edi
  bool v165; // al
  unsigned int v166; // eax
  unsigned __int16 *v167; // r14
  unsigned int v168; // edx
  bool v169; // al
  __int16 v170; // ax
  bool v171; // al
  __int64 v172; // rdx
  __int64 v173; // r8
  __int64 v174; // rdx
  void *v175; // rax
  __int64 *v176; // rsi
  unsigned int v177; // eax
  __m128i v178; // rax
  int v179; // edx
  bool v180; // al
  char v181; // al
  int v182; // r10d
  char v183; // dl
  const __m128i *v184; // roff
  int v185; // eax
  __int64 *v186; // rcx
  __int64 *v187; // rax
  int v188; // edx
  __int64 v189; // rax
  int v190; // eax
  __int64 v191; // rax
  __int64 v192; // rdx
  int v193; // eax
  unsigned int v194; // edx
  __int128 v195; // [rsp-20h] [rbp-280h]
  __int128 v196; // [rsp-10h] [rbp-270h]
  __int64 v197; // [rsp-10h] [rbp-270h]
  __int128 v198; // [rsp-10h] [rbp-270h]
  __int128 v199; // [rsp-10h] [rbp-270h]
  __int64 v200; // [rsp-8h] [rbp-268h]
  __int64 v201; // [rsp+0h] [rbp-260h]
  __int64 v202; // [rsp+0h] [rbp-260h]
  __int64 v203; // [rsp+0h] [rbp-260h]
  __int64 v204; // [rsp+0h] [rbp-260h]
  __int64 v205; // [rsp+8h] [rbp-258h]
  int v206; // [rsp+10h] [rbp-250h]
  __int64 v207; // [rsp+10h] [rbp-250h]
  unsigned int v208; // [rsp+10h] [rbp-250h]
  unsigned int v209; // [rsp+10h] [rbp-250h]
  __int64 v210; // [rsp+18h] [rbp-248h]
  int v211; // [rsp+20h] [rbp-240h]
  __int64 v212; // [rsp+20h] [rbp-240h]
  __int64 v213; // [rsp+20h] [rbp-240h]
  int v214; // [rsp+28h] [rbp-238h]
  __int64 v215; // [rsp+28h] [rbp-238h]
  __int64 v216; // [rsp+28h] [rbp-238h]
  __int64 v217; // [rsp+28h] [rbp-238h]
  __int64 v218; // [rsp+28h] [rbp-238h]
  __int64 v219; // [rsp+28h] [rbp-238h]
  unsigned int v220; // [rsp+28h] [rbp-238h]
  __int16 v221; // [rsp+2Ah] [rbp-236h]
  int v222; // [rsp+30h] [rbp-230h]
  int v223; // [rsp+30h] [rbp-230h]
  __int16 v224; // [rsp+32h] [rbp-22Eh]
  __int64 v225; // [rsp+38h] [rbp-228h]
  int v226; // [rsp+38h] [rbp-228h]
  char v227; // [rsp+38h] [rbp-228h]
  int v228; // [rsp+40h] [rbp-220h]
  int v229; // [rsp+40h] [rbp-220h]
  char v230; // [rsp+40h] [rbp-220h]
  unsigned __int64 v231; // [rsp+40h] [rbp-220h]
  __int64 v232; // [rsp+40h] [rbp-220h]
  int v233; // [rsp+40h] [rbp-220h]
  __int64 v234; // [rsp+40h] [rbp-220h]
  __int64 v235; // [rsp+40h] [rbp-220h]
  __int64 v236; // [rsp+40h] [rbp-220h]
  char v237; // [rsp+50h] [rbp-210h]
  unsigned int v238; // [rsp+5Ch] [rbp-204h]
  __int64 v239; // [rsp+60h] [rbp-200h]
  __int64 v240; // [rsp+60h] [rbp-200h]
  char v241; // [rsp+60h] [rbp-200h]
  __int64 v243; // [rsp+68h] [rbp-1F8h]
  int v244; // [rsp+70h] [rbp-1F0h]
  __int64 v245; // [rsp+70h] [rbp-1F0h]
  int v246; // [rsp+78h] [rbp-1E8h]
  __int16 v248; // [rsp+88h] [rbp-1D8h]
  unsigned __int64 v249; // [rsp+88h] [rbp-1D8h]
  __m128i v250; // [rsp+A0h] [rbp-1C0h] BYREF
  __m128i v251; // [rsp+B0h] [rbp-1B0h] BYREF
  __int64 v252; // [rsp+C0h] [rbp-1A0h] BYREF
  __int64 v253; // [rsp+C8h] [rbp-198h]
  __int64 v254; // [rsp+D0h] [rbp-190h] BYREF
  int v255; // [rsp+D8h] [rbp-188h]
  unsigned __int64 v256; // [rsp+E0h] [rbp-180h] BYREF
  char v257; // [rsp+E8h] [rbp-178h]
  __int64 v258; // [rsp+F0h] [rbp-170h] BYREF
  char v259; // [rsp+F8h] [rbp-168h]
  __int64 v260; // [rsp+100h] [rbp-160h] BYREF
  int v261; // [rsp+108h] [rbp-158h]
  __int64 v262; // [rsp+110h] [rbp-150h]
  __int64 v263; // [rsp+118h] [rbp-148h]
  unsigned __int64 v264; // [rsp+120h] [rbp-140h] BYREF
  __int64 v265; // [rsp+128h] [rbp-138h]
  __int64 v266; // [rsp+130h] [rbp-130h]
  __int64 v267; // [rsp+138h] [rbp-128h]
  unsigned __int64 v268; // [rsp+140h] [rbp-120h] BYREF
  __int64 v269; // [rsp+148h] [rbp-118h]
  unsigned __int64 v270; // [rsp+150h] [rbp-110h] BYREF
  __int64 v271; // [rsp+158h] [rbp-108h]
  __m128i v272; // [rsp+160h] [rbp-100h] BYREF
  __int64 v273; // [rsp+170h] [rbp-F0h]
  __m128i v274; // [rsp+180h] [rbp-E0h] BYREF
  __m128i v275; // [rsp+190h] [rbp-D0h]
  __m128i v276; // [rsp+1A0h] [rbp-C0h] BYREF
  __m128i v277; // [rsp+1B0h] [rbp-B0h] BYREF

  v8 = a5;
  v252 = a3;
  v253 = a4;
  if ( a5 <= 1 )
  {
    LODWORD(v22) = 0;
    return (unsigned int)v22;
  }
  v10 = **a2;
  v11 = *(_QWORD *)(v10 + 80);
  v254 = v11;
  if ( v11 )
    sub_B96E90((__int64)&v254, v11, 1);
  v255 = *(_DWORD *)(v10 + 72);
  if ( (_WORD)v252 )
  {
    if ( (_WORD)v252 == 1 || (unsigned __int16)(v252 - 504) <= 7u )
LABEL_236:
      BUG();
    v12 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v252 - 16];
    LOBYTE(v13) = byte_444C4A0[16 * (unsigned __int16)v252 - 8];
  }
  else
  {
    v12 = sub_3007260((__int64)&v252);
    v262 = v12;
    v263 = v13;
  }
  v257 = v13;
  v276.m128i_i8[8] = v13;
  v256 = (v12 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  v276.m128i_i64[0] = v8 * v256;
  v14 = sub_CA1930(&v276);
  if ( !(_WORD)v252 )
  {
    if ( !sub_30070B0((__int64)&v252) )
      goto LABEL_8;
    if ( !sub_3007100((__int64)&v252) )
      goto LABEL_55;
    goto LABEL_60;
  }
  if ( (unsigned __int16)(v252 - 17) <= 0xD3u )
  {
    if ( (unsigned __int16)(v252 - 176) > 0x34u )
    {
LABEL_59:
      v15 = word_4456340[(unsigned __int16)v252 - 1];
      goto LABEL_9;
    }
LABEL_60:
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( (_WORD)v252 )
    {
      if ( (unsigned __int16)(v252 - 176) <= 0x34u )
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
      goto LABEL_59;
    }
LABEL_55:
    v15 = sub_3007130((__int64)&v252, v11);
    goto LABEL_9;
  }
LABEL_8:
  v15 = 1;
LABEL_9:
  v274 = 0u;
  v16 = 0;
  v17 = v14;
  v248 = 0;
  v18 = 0;
  v19 = v8;
  v275 = 0u;
  do
  {
    v20 = v16;
    v21 = *(_QWORD *)((*a2)[2 * v16] + 112);
    if ( v18 )
    {
      if ( *(_WORD *)(v21 + 32) != v248 )
      {
        LODWORD(v22) = 0;
        goto LABEL_13;
      }
      v27 = &v274;
      v225 = v19;
      v229 = v15;
      v276 = _mm_loadu_si128((const __m128i *)(v21 + 40));
      v277 = _mm_loadu_si128((const __m128i *)(v21 + 56));
      sub_E00020(&v250, (__int64)&v274, (__int64)&v276);
      v51 = _mm_loadu_si128(&v251);
      v19 = v225;
      v15 = v229;
      v274 = _mm_loadu_si128(&v250);
      v275 = v51;
    }
    else
    {
      v24 = *(_WORD *)(v21 + 32);
      v25 = *(_QWORD *)(v21 + 48);
      v18 = 1;
      v26 = *(_QWORD *)(v21 + 56);
      v27 = *(__m128i **)(v21 + 64);
      v28 = *(_QWORD *)(v21 + 40);
      v248 = v24;
      v274.m128i_i64[1] = v25;
      v274.m128i_i64[0] = v28;
      v275.m128i_i64[0] = v26;
      v275.m128i_i64[1] = (__int64)v27;
    }
    ++v16;
  }
  while ( v19 != v16 );
  LODWORD(v22) = v17;
  v29 = v20;
  v238 = v20 + 1;
  if ( a7 )
  {
    v52 = v252;
    v53 = (v20 + 1) * v15;
    if ( (_WORD)v252 )
    {
      if ( (unsigned __int16)(v252 - 17) <= 0xD3u )
      {
        v54 = 0;
        v52 = word_4456580[(unsigned __int16)v252 - 1];
        goto LABEL_65;
      }
    }
    else
    {
      v171 = sub_30070B0((__int64)&v252);
      v52 = 0;
      if ( v171 )
      {
        v52 = sub_3009970((__int64)&v252, (__int64)v27, v172, 0, v173);
        v54 = v174;
        goto LABEL_65;
      }
    }
    v54 = v253;
LABEL_65:
    v55 = v52;
    v56 = *(__int64 **)(*a1 + 64LL);
    LOWORD(v57) = sub_2D43050(v52, v53);
    v226 = 0;
    if ( !(_WORD)v57 )
    {
      v57 = sub_3009400(v56, v55, v54, v53, 0);
      v224 = HIWORD(v57);
      v226 = v179;
    }
    HIWORD(v60) = v224;
    LOWORD(v60) = v57;
    v223 = v60;
    if ( a6 )
    {
      v61 = 0;
      v276.m128i_i64[0] = (__int64)&v277;
      v276.m128i_i64[1] = 0x800000000LL;
      do
      {
        v62 = *(_QWORD *)((*a2)[v61] + 40);
        v63 = *(unsigned int *)(v62 + 48);
        v64 = *(_QWORD *)(v62 + 40);
        v65 = *(_QWORD *)(v62 + 48);
        v66 = v64;
        v67 = *(_QWORD *)(v64 + 48) + 16 * v63;
        if ( (_WORD)v252 != *(_WORD *)v67 || !(_WORD)v252 && v253 != *(_QWORD *)(v67 + 8) )
        {
          v108 = sub_33CF5B0(v64, v65);
          v110 = 16LL * v109;
          v111 = v108;
          v231 = v109 | v65 & 0xFFFFFFFF00000000LL;
          v112 = v109;
          v113 = v110 + *(_QWORD *)(v108 + 48);
          v114 = *(_WORD *)v113;
          v115 = *(_QWORD *)(v113 + 8);
          LOWORD(v264) = v114;
          v265 = v115;
          if ( v114 )
          {
            if ( v114 == 1 || (unsigned __int16)(v114 - 504) <= 7u )
              goto LABEL_236;
            v119 = *(_QWORD *)&byte_444C4A0[16 * v114 - 16];
            LODWORD(v120) = (unsigned __int8)byte_444C4A0[16 * v114 - 8];
          }
          else
          {
            v215 = v108;
            v116 = sub_3007260((__int64)&v264);
            v108 = v215;
            v117 = v116;
            v120 = v118;
            v266 = v117;
            v119 = v117;
            v267 = v120;
            LODWORD(v120) = (unsigned __int8)v120;
          }
          if ( v256 != v119 || (_BYTE)v120 != v257 )
          {
            LOBYTE(v120) = *(_DWORD *)(v108 + 24) == 35 || *(_DWORD *)(v108 + 24) == 11;
            if ( !(_BYTE)v120 )
            {
              LODWORD(v22) = v120;
              if ( (__m128i *)v276.m128i_i64[0] != &v277 )
                _libc_free(v276.m128i_u64[0]);
              goto LABEL_13;
            }
            if ( (_WORD)v252 )
            {
              if ( (_WORD)v252 == 1 || (unsigned __int16)(v252 - 504) <= 7u )
                goto LABEL_236;
              v189 = 16LL * ((unsigned __int16)v252 - 1);
              v123 = *(_QWORD *)&byte_444C4A0[v189];
              v124 = byte_444C4A0[v189 + 8];
            }
            else
            {
              v216 = v108;
              v121 = sub_3007260((__int64)&v252);
              v108 = v216;
              v268 = v121;
              v269 = v122;
              v123 = v121;
              v124 = v269;
            }
            v201 = v108;
            v272.m128i_i64[0] = v123;
            v272.m128i_i8[8] = v124;
            v125 = sub_CA1930(&v272);
            v126 = sub_327FC40(*(_QWORD **)(*a1 + 64LL), v125);
            v127 = v201;
            v206 = v126;
            v128 = *(_QWORD *)(v201 + 80);
            v211 = v129;
            v217 = *a1;
            v260 = v128;
            if ( v128 )
            {
              sub_B96E90((__int64)&v260, v128, 1);
              v127 = v201;
            }
            v130 = *(_QWORD *)(v127 + 48) + v110;
            v261 = *(_DWORD *)(v127 + 72);
            v131 = *(_QWORD *)(v127 + 96) + 24LL;
            v132 = *(_WORD *)v130;
            v272.m128i_i64[1] = *(_QWORD *)(v130 + 8);
            v272.m128i_i16[0] = v132;
            if ( v132 )
            {
              if ( v132 == 1 || (unsigned __int16)(v132 - 504) <= 7u )
                goto LABEL_236;
              v134 = 16LL * (v132 - 1);
              v133 = *(_QWORD *)&byte_444C4A0[v134];
              LOBYTE(v134) = byte_444C4A0[v134 + 8];
            }
            else
            {
              v133 = sub_3007260((__int64)&v272);
              v270 = v133;
              v271 = v134;
            }
            v259 = v134;
            v258 = v133;
            v135 = sub_CA1930(&v258);
            sub_C44AB0((__int64)&v264, v131, v135);
            v136 = sub_CA1930(&v256);
            sub_C44AB0((__int64)&v272, (__int64)&v264, v136);
            v111 = sub_34007B0(v217, (unsigned int)&v272, (unsigned int)&v260, v206, v211, 0, 0);
            v112 = v137;
            if ( v272.m128i_i32[2] > 0x40u && v272.m128i_i64[0] )
              j_j___libc_free_0_0(v272.m128i_u64[0]);
            if ( (unsigned int)v265 > 0x40 && v264 )
              j_j___libc_free_0_0(v264);
            if ( v260 )
              sub_B91220((__int64)&v260, v260);
          }
          v65 = v112 | v231 & 0xFFFFFFFF00000000LL;
          v66 = sub_33FB890(*a1, (unsigned int)v252, v253, v111, v65);
          v63 = (unsigned int)v63;
        }
        v68 = v276.m128i_u32[2];
        v69 = v63 | v65 & 0xFFFFFFFF00000000LL;
        v70 = v276.m128i_u32[2] + 1LL;
        if ( v70 > v276.m128i_u32[3] )
        {
          sub_C8D5F0((__int64)&v276, &v277, v70, 0x10u, v58, v59);
          v68 = v276.m128i_u32[2];
        }
        v71 = (__int64 *)(v276.m128i_i64[0] + 16 * v68);
        v61 += 2;
        *v71 = v66;
        v71[1] = v69;
        v72 = ++v276.m128i_i32[2];
      }
      while ( 2LL * v238 != v61 );
      v73 = v72;
      v74 = v276.m128i_i64[0];
      v75 = *a1;
      if ( (_WORD)v252 )
      {
        v76 = (unsigned __int16)(v252 - 17) < 0xD4u ? 159 : 156;
        goto LABEL_76;
      }
      goto LABEL_231;
    }
    v138 = 0;
    v276.m128i_i64[0] = (__int64)&v277;
    v276.m128i_i64[1] = 0x800000000LL;
    while ( 1 )
    {
      v146 = *(_QWORD *)((*a2)[2 * v138] + 40);
      v147 = sub_33CF5B0(*(_QWORD *)(v146 + 40), *(_QWORD *)(v146 + 48));
      v148 = v147;
      v149 = (unsigned int)v153;
      v150 = v153;
      v151 = 16LL * (unsigned int)v153;
      v152 = (unsigned __int16 *)(v151 + *(_QWORD *)(v147 + 48));
      LODWORD(v153) = *v152;
      v154 = *((_QWORD *)v152 + 1);
      v155 = (unsigned __int16)v252;
      if ( (_WORD)v252 == (_WORD)v153 )
      {
        if ( (_WORD)v252 )
          goto LABEL_119;
        if ( v253 == v154 )
          goto LABEL_119;
        v164 = *(_DWORD *)(v147 + 24);
        if ( v164 != 158 && v164 != 161 )
          goto LABEL_119;
      }
      else
      {
        v156 = *(_DWORD *)(v147 + 24);
        if ( v156 != 161 && v156 != 158 )
          goto LABEL_119;
        if ( (_WORD)v252 )
        {
          if ( (unsigned __int16)(v252 - 17) > 0xD3u )
            goto LABEL_127;
          v232 = 0;
          v155 = (unsigned __int16)word_4456580[(unsigned __int16)v252 - 1];
          goto LABEL_128;
        }
      }
      v203 = v154;
      v208 = v153;
      v220 = (unsigned __int16)v252;
      v165 = sub_30070B0((__int64)&v252);
      v155 = v220;
      v148 = v147;
      LODWORD(v153) = v208;
      v154 = v203;
      if ( !v165 )
      {
LABEL_127:
        v232 = v253;
        goto LABEL_128;
      }
      v166 = sub_3009970((__int64)&v252, v220, v208, v147, v203);
      v148 = v147;
      v232 = v153;
      v155 = v166;
      v167 = (unsigned __int16 *)(*(_QWORD *)(v147 + 48) + v151);
      LODWORD(v153) = *v167;
      v154 = *((_QWORD *)v167 + 1);
LABEL_128:
      v272.m128i_i16[0] = v153;
      v272.m128i_i64[1] = v154;
      if ( (_WORD)v153 )
      {
        if ( (unsigned __int16)(v153 - 17) <= 0xD3u )
        {
          v154 = 0;
          LOWORD(v153) = word_4456580[(unsigned __int16)v153 - 1];
        }
      }
      else
      {
        v204 = v154;
        v209 = v153;
        v213 = v148;
        v169 = sub_30070B0((__int64)&v272);
        v148 = v213;
        LOWORD(v153) = v209;
        v154 = v204;
        if ( v169 )
        {
          v170 = sub_3009970((__int64)&v272, v155, v209, v213, v204);
          v148 = v213;
          v154 = v153;
          LOWORD(v153) = v170;
        }
      }
      if ( (_WORD)v153 == (_WORD)v155 && ((_WORD)v153 || v154 == v232) )
      {
        if ( (_WORD)v252 )
        {
          if ( (unsigned __int16)(v252 - 17) > 0xD3u )
            goto LABEL_194;
        }
        else
        {
          v235 = v148;
          v180 = sub_30070B0((__int64)&v252);
          v148 = v235;
          if ( !v180 )
          {
LABEL_194:
            v233 = 158;
LABEL_136:
            v157 = *(__int64 **)(v148 + 40);
            v158 = *(_QWORD *)(v148 + 80);
            v159 = *v157;
            v160 = v157[1];
            v161 = v157[5];
            v162 = v157[6];
            v272.m128i_i64[0] = v158;
            v218 = *a1;
            if ( v158 )
            {
              v202 = v161;
              v205 = v162;
              v207 = v159;
              v210 = v160;
              v212 = v148;
              sub_B96E90((__int64)&v272, v158, 1);
              v161 = v202;
              v162 = v205;
              v159 = v207;
              v160 = v210;
              v148 = v212;
            }
            *((_QWORD *)&v198 + 1) = v162;
            *(_QWORD *)&v198 = v161;
            *((_QWORD *)&v195 + 1) = v160;
            *(_QWORD *)&v195 = v159;
            v272.m128i_i32[2] = *(_DWORD *)(v148 + 72);
            v148 = sub_3406EB0(v218, v233, (unsigned int)&v272, v252, v253, v162, v195, v198);
            v149 = v163;
            if ( v272.m128i_i64[0] )
            {
              v219 = v163;
              v234 = v148;
              sub_B91220((__int64)&v272, v272.m128i_i64[0]);
              v149 = v219;
              v148 = v234;
            }
            goto LABEL_119;
          }
        }
        v233 = 161;
        if ( *(_DWORD *)(v148 + 24) != 158 )
          goto LABEL_136;
        *((_QWORD *)&v199 + 1) = v150;
        *(_QWORD *)&v199 = v147;
        v148 = sub_33FAF80(*a1, 156, (unsigned int)&v254, v252, v253, v59, v199);
        v149 = v194;
      }
      else
      {
        v148 = sub_33FB890(*a1, (unsigned int)v252, v253, v147, v150);
        v149 = v168;
      }
LABEL_119:
      v139 = v148;
      v140 = v149 | v150 & 0xFFFFFFFF00000000LL;
      v141 = v276.m128i_u32[2];
      v142 = v140;
      v143 = v276.m128i_u32[2] + 1LL;
      if ( v143 > v276.m128i_u32[3] )
      {
        sub_C8D5F0((__int64)&v276, &v277, v143, 0x10u, v154, v59);
        v141 = v276.m128i_u32[2];
      }
      v144 = (__int64 *)(v276.m128i_i64[0] + 16 * v141);
      ++v138;
      *v144 = v139;
      v144[1] = v142;
      v145 = ++v276.m128i_i32[2];
      if ( v238 <= (unsigned int)v138 )
      {
        v73 = v145;
        v74 = v276.m128i_i64[0];
        v75 = *a1;
        if ( (_WORD)v252 )
        {
          v76 = (unsigned __int16)(v252 - 17) < 0xD4u ? 159 : 156;
          goto LABEL_76;
        }
LABEL_231:
        v76 = !sub_30070B0((__int64)&v252) ? 156 : 159;
LABEL_76:
        *((_QWORD *)&v196 + 1) = v73;
        *(_QWORD *)&v196 = v74;
        v77 = sub_33FC220(v75, v76, (unsigned int)&v254, v223, v226, v59, v196);
        v49 = v197;
        v46 = v78;
        v47 = v77;
        v50 = v200;
        if ( (__m128i *)v276.m128i_i64[0] != &v277 )
          _libc_free(v276.m128i_u64[0]);
LABEL_78:
        v79 = **a2;
        v243 = sub_3263720(a1, a2, (__int64 *)v238, v48, v49, v50);
        v81 = v80;
        v230 = sub_325E090((__int64)*a2, *((unsigned int *)a2 + 2));
        if ( a8 )
        {
          v82 = 16LL * v46;
          v83 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)a1[1] + 592LL);
          v84 = v82 + *(_QWORD *)(v47 + 48);
          v85 = *(_WORD *)v84;
          v86 = *(_QWORD *)(v84 + 8);
          if ( v83 == sub_2D56A50 )
          {
            sub_2FE6CC0((__int64)&v276, a1[1], *(_QWORD *)(*a1 + 64LL), v85, v86);
            v272.m128i_i16[0] = v276.m128i_i16[4];
            v272.m128i_i64[1] = v277.m128i_i64[0];
          }
          else
          {
            v272.m128i_i32[0] = v83(a1[1], *(_QWORD *)(*a1 + 64LL), v85, v86);
            v272.m128i_i64[1] = v192;
          }
          v87.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v272);
          v276 = v87;
          v88 = sub_CA1930(&v276);
          v239 = *a1;
          sub_C44AB0((__int64)&v276, *(_QWORD *)(v47 + 96) + 24LL, v88);
          v89 = sub_34007B0(v239, (unsigned int)&v276, (unsigned int)&v254, v272.m128i_i32[0], v272.m128i_i32[2], 0, 0);
          v246 = v90;
          v244 = v89;
          if ( v276.m128i_i32[2] > 0x40u && v276.m128i_i64[0] )
            j_j___libc_free_0_0(v276.m128i_u64[0]);
          v240 = *a1;
          v91 = sub_2EAC4F0(*(_QWORD *)(v79 + 112));
          v92 = v240;
          v93 = v91;
          v94 = *(unsigned __int16 *)(v82 + *(_QWORD *)(v47 + 48));
          v95 = *(_QWORD *)(v82 + *(_QWORD *)(v47 + 48) + 8);
          if ( v230 )
          {
            v96 = *(const __m128i **)(v79 + 112);
            v276 = _mm_loadu_si128(v96);
            v277.m128i_i64[0] = v96[1].m128i_i64[0];
          }
          else
          {
            v227 = v91;
            v236 = v94;
            v190 = sub_2EAC1E0(*(_QWORD *)(v79 + 112));
            v277.m128i_i8[4] = 0;
            v93 = v227;
            v277.m128i_i32[0] = v190;
            v94 = v236;
            v92 = v240;
            v276 = 0u;
          }
          v97 = *(_DWORD *)(v79 + 24);
          v98 = *(__int64 **)(v79 + 40);
          if ( v97 > 365 )
          {
            if ( v97 <= 467 )
            {
              if ( v97 <= 464 )
                goto LABEL_90;
              goto LABEL_163;
            }
            if ( v97 != 497 )
              goto LABEL_90;
          }
          else if ( v97 <= 363 )
          {
            if ( v97 != 339 && (v97 & 0xFFFFFFBF) != 0x12B )
            {
LABEL_90:
              v99 = v98 + 5;
              goto LABEL_91;
            }
LABEL_163:
            v99 = v98 + 10;
LABEL_91:
            v100 = sub_33F5040(
                     v92,
                     v243,
                     v81,
                     (unsigned int)&v254,
                     v244,
                     v246,
                     *v99,
                     v99[1],
                     *(_OWORD *)&v276,
                     v277.m128i_i64[0],
                     v94,
                     v95,
                     v93,
                     v248,
                     (__int64)&v274);
            v102 = v101;
LABEL_92:
            v249 = v100;
            v103 = 0;
            do
            {
              v104 = &(*a2)[2 * v103++];
              v105 = *v104;
              LODWORD(v265) = v102;
              v264 = v249;
              sub_32EB790((__int64)a1, v105, (__int64 *)&v264, 1, 1);
            }
            while ( v238 > (unsigned int)v103 );
            if ( *(_DWORD *)(v243 + 24) != 328 )
            {
              v264 = v243;
              sub_32B3B20((__int64)(a1 + 71), (__int64 *)&v264);
              if ( *(int *)(v243 + 88) < 0 )
              {
                *(_DWORD *)(v243 + 88) = *((_DWORD *)a1 + 12);
                v191 = *((unsigned int *)a1 + 12);
                if ( v191 + 1 > (unsigned __int64)*((unsigned int *)a1 + 13) )
                {
                  sub_C8D5F0((__int64)(a1 + 5), a1 + 7, v191 + 1, 8u, v106, v107);
                  v191 = *((unsigned int *)a1 + 12);
                }
                *(_QWORD *)(a1[5] + 8 * v191) = v243;
                ++*((_DWORD *)a1 + 12);
              }
            }
            LODWORD(v22) = 1;
            goto LABEL_13;
          }
          v99 = v98 + 15;
          goto LABEL_91;
        }
        v245 = *a1;
        v181 = sub_2EAC4F0(*(_QWORD *)(v79 + 112));
        v182 = v245;
        v183 = v181;
        if ( v230 )
        {
          v184 = *(const __m128i **)(v79 + 112);
          v272 = _mm_loadu_si128(v184);
          v273 = v184[1].m128i_i64[0];
        }
        else
        {
          v241 = v181;
          v193 = sub_2EAC1E0(*(_QWORD *)(v79 + 112));
          BYTE4(v273) = 0;
          v183 = v241;
          LODWORD(v273) = v193;
          v182 = v245;
          v272 = 0u;
        }
        v185 = *(_DWORD *)(v79 + 24);
        v186 = *(__int64 **)(v79 + 40);
        if ( v185 > 365 )
        {
          if ( v185 <= 467 )
          {
            if ( v185 <= 464 )
              goto LABEL_202;
            goto LABEL_208;
          }
          if ( v185 != 497 )
            goto LABEL_202;
        }
        else if ( v185 <= 363 )
        {
          if ( v185 != 339 && (v185 & 0xFFFFFFBF) != 0x12B )
          {
LABEL_202:
            v187 = v186 + 5;
LABEL_203:
            v100 = sub_33F4560(
                     v182,
                     v243,
                     v81,
                     (unsigned int)&v254,
                     v47,
                     v46,
                     *v187,
                     v187[1],
                     *(_OWORD *)&v272,
                     v273,
                     v183,
                     v248,
                     (__int64)&v274);
            v102 = v188;
            goto LABEL_92;
          }
LABEL_208:
          v187 = v186 + 10;
          goto LABEL_203;
        }
        v187 = v186 + 15;
        goto LABEL_203;
      }
    }
  }
  switch ( v17 )
  {
    case 1u:
      LOWORD(v30) = 2;
      v31 = 0;
      break;
    case 2u:
      LOWORD(v30) = 3;
      v31 = 0;
      break;
    case 4u:
      LOWORD(v30) = 4;
      v31 = 0;
      break;
    case 8u:
      LOWORD(v30) = 5;
      v31 = 0;
      break;
    case 0x10u:
      LOWORD(v30) = 6;
      v31 = 0;
      break;
    case 0x20u:
      LOWORD(v30) = 7;
      v31 = 0;
      break;
    case 0x40u:
      LOWORD(v30) = 8;
      v31 = 0;
      break;
    case 0x80u:
      v214 = 9;
      v222 = 0;
      LODWORD(v269) = 128;
      goto LABEL_218;
    default:
      v30 = sub_3007020(*(_QWORD **)(*a1 + 64LL), v17);
      v221 = HIWORD(v30);
      break;
  }
  HIWORD(v32) = v221;
  v222 = v31;
  LODWORD(v269) = v17;
  LOWORD(v32) = v30;
  v214 = v32;
  if ( v17 > 0x40 )
  {
LABEL_218:
    sub_C43690((__int64)&v268, 0, 0);
    goto LABEL_33;
  }
  v268 = 0;
LABEL_33:
  v33 = 0;
  v228 = v29;
  v237 = *(_BYTE *)sub_2E79000(*(__int64 **)(*a1 + 40LL));
  while ( 1 )
  {
    v34 = (unsigned int)(v228 - v33);
    if ( v237 )
      v34 = v33;
    v35 = *(_QWORD *)((*a2)[2 * v34] + 40);
    v36 = sub_33CF5B0(*(_QWORD *)(v35 + 40), *(_QWORD *)(v35 + 48));
    v37 = sub_CA1930(&v256);
    v38 = v37;
    if ( (unsigned int)v269 > 0x40 )
    {
      sub_C47690((__int64 *)&v268, v37);
    }
    else
    {
      v39 = 0;
      if ( v38 != (_DWORD)v269 )
        v39 = v268 << v38;
      v40 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v269) & v39;
      if ( !(_DWORD)v269 )
        v40 = 0;
      v268 = v40;
    }
    v41 = *(_DWORD *)(v36 + 24);
    LOBYTE(v22) = v41 == 11 || v41 == 35;
    if ( (_BYTE)v22 )
    {
      v42 = *(_QWORD *)(v36 + 96);
      v43 = sub_CA1930(&v256);
      v22 = v42 + 24;
      sub_C44AB0((__int64)&v272, v22, v43);
      sub_C44AB0((__int64)&v276, (__int64)&v272, v17);
      if ( (unsigned int)v269 > 0x40 )
        sub_C43BD0(&v268, v276.m128i_i64);
      else
        v268 |= v276.m128i_i64[0];
      if ( v276.m128i_i32[2] > 0x40u && v276.m128i_i64[0] )
        j_j___libc_free_0_0(v276.m128i_u64[0]);
      if ( v272.m128i_i32[2] > 0x40u && v272.m128i_i64[0] )
        j_j___libc_free_0_0(v272.m128i_u64[0]);
      goto LABEL_51;
    }
    if ( v41 != 12 && v41 != 36 )
      break;
    v175 = sub_C33340();
    v176 = (__int64 *)(*(_QWORD *)(v36 + 96) + 24LL);
    if ( (void *)*v176 == v175 )
      sub_C3E660((__int64)&v270, (__int64)v176);
    else
      sub_C3A850((__int64)&v270, v176);
    v177 = sub_CA1930(&v256);
    sub_C44AB0((__int64)&v272, (__int64)&v270, v177);
    sub_C44AB0((__int64)&v276, (__int64)&v272, v17);
    if ( (unsigned int)v269 > 0x40 )
      sub_C43BD0(&v268, v276.m128i_i64);
    else
      v268 |= v276.m128i_i64[0];
    if ( v276.m128i_i32[2] > 0x40u && v276.m128i_i64[0] )
      j_j___libc_free_0_0(v276.m128i_u64[0]);
    if ( v272.m128i_i32[2] > 0x40u && v272.m128i_i64[0] )
      j_j___libc_free_0_0(v272.m128i_u64[0]);
    if ( (unsigned int)v271 > 0x40 && v270 )
      j_j___libc_free_0_0(v270);
    v178.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v252);
    v276 = v178;
    if ( v178.m128i_i64[0] != v256 || v276.m128i_i8[8] != v257 )
      goto LABEL_154;
LABEL_51:
    v33 = (unsigned int)(v33 + 1);
    if ( v238 <= (unsigned int)v33 )
    {
      v44 = sub_34007B0(*a1, (unsigned int)&v268, (unsigned int)&v254, v214, v222, 0, 0);
      v46 = v45;
      v47 = v44;
      sub_969240((__int64 *)&v268);
      goto LABEL_78;
    }
  }
  if ( !(unsigned __int8)sub_33CA6D0(v36) && !(unsigned __int8)sub_33CA720(v36) )
    BUG();
LABEL_154:
  if ( (unsigned int)v269 > 0x40 && v268 )
    j_j___libc_free_0_0(v268);
LABEL_13:
  if ( v254 )
    sub_B91220((__int64)&v254, v254);
  return (unsigned int)v22;
}
