// Function: sub_3489D70
// Address: 0x3489d70
//
__int64 __fastcall sub_3489D70(__int64 a1, __int64 a2, __int64 a3, char a4, char a5, __int64 a6, __m128i a7)
{
  __int64 v10; // rsi
  unsigned __int16 *v11; // rax
  int v12; // r13d
  __int64 v13; // r15
  __int64 *v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rsi
  int v17; // eax
  __int16 v18; // r13
  __int64 v19; // rdx
  __int64 v20; // rdx
  unsigned __int16 v21; // r13
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdx
  const __m128i *v25; // rax
  __m128i v26; // xmm0
  __int64 v27; // rbx
  unsigned __int32 v28; // ebx
  int v29; // eax
  unsigned __int64 v30; // rax
  __m128i *v31; // rax
  __int64 v32; // rdx
  char v33; // bl
  __int64 v34; // r9
  int v35; // eax
  __int128 v36; // kr00_16
  __int64 v37; // r8
  __int64 v38; // r9
  unsigned __int8 *v39; // rbx
  unsigned __int8 *v40; // r10
  __int64 v41; // rdx
  __int64 v42; // rax
  __int64 v43; // rcx
  __int64 v44; // r13
  __int64 (__fastcall *v45)(__int64, __int64, __int64, _QWORD, __int64); // rbx
  __int64 v46; // rax
  unsigned int v47; // eax
  __int64 v48; // rdx
  __int64 v49; // rbx
  __int64 v50; // rdx
  __int128 v51; // rax
  __int64 v52; // r9
  __int64 v53; // r12
  unsigned int v54; // edx
  __int64 v55; // rcx
  __int64 v56; // rdx
  unsigned int v57; // r8d
  __int16 v58; // ax
  __int64 v59; // rdx
  __int64 v60; // r13
  __int64 v61; // rbx
  unsigned int v62; // esi
  __int64 v63; // r12
  __int64 v65; // rdx
  __int64 v66; // rcx
  __int64 v67; // r8
  __int64 v68; // rdx
  __int64 v69; // rcx
  __int64 v70; // r8
  __int64 v71; // rdx
  __int64 v72; // rcx
  __int64 v73; // r8
  __int64 v74; // rdx
  __int16 *v75; // rax
  unsigned __int16 v76; // r15
  __int64 v77; // rdx
  __int64 *v78; // rdi
  __int64 v79; // rax
  __int64 v80; // rsi
  int v81; // eax
  __int16 v82; // r12
  __int64 v83; // rdx
  __int64 v84; // rdx
  __int64 v85; // rax
  unsigned int v86; // r12d
  __int64 v87; // rcx
  __m128i *v88; // rax
  __int64 v89; // r9
  char v90; // r13
  int v91; // eax
  __int64 v92; // rdx
  __int64 v93; // rdx
  unsigned __int64 v94; // r9
  __int64 *v95; // rax
  __int64 v96; // r10
  unsigned __int64 v97; // r11
  unsigned __int8 *v98; // r12
  unsigned __int8 *v99; // rax
  __int64 v100; // rdx
  unsigned __int8 *v101; // rbx
  __int64 (__fastcall *v102)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v103; // rdx
  unsigned int v104; // edx
  __int64 v105; // rdx
  __int64 v106; // rcx
  __int64 v107; // r8
  bool v108; // al
  __int64 v109; // rcx
  __int64 v110; // r8
  unsigned __int64 v111; // rdi
  unsigned __int64 v112; // rdx
  __int64 v113; // rdx
  __int64 v114; // r8
  __int64 v115; // rax
  __int64 v116; // rcx
  __int64 v117; // rdx
  __int64 v118; // r8
  __int64 v119; // r9
  bool v120; // al
  unsigned int v121; // edx
  __int64 v122; // r9
  unsigned int v123; // edx
  unsigned int v124; // edx
  __int64 v125; // r9
  unsigned int v126; // edx
  __int64 v127; // r9
  unsigned int v128; // edx
  __int64 v129; // r9
  unsigned int v130; // edx
  __int64 v131; // rdx
  __int64 v132; // r8
  __int64 v133; // r9
  __int64 v134; // rdx
  __int64 v135; // rcx
  __int64 v136; // r8
  __int64 v137; // r9
  __int64 v138; // r9
  __int64 v139; // r8
  __int64 v140; // r9
  __int64 v141; // rbx
  __int64 v142; // rdx
  __int64 v143; // r13
  __int64 v144; // r9
  __int64 v145; // rdx
  __int64 v146; // r8
  __int64 v147; // r9
  __int64 v148; // rdx
  __int64 v149; // rdx
  __int64 v150; // rax
  __int64 v151; // rdx
  __int64 v152; // rdx
  void *v153; // rax
  int v154; // r9d
  unsigned int v155; // edx
  unsigned int v156; // r13d
  void *v157; // rax
  unsigned int v158; // edx
  unsigned int v159; // r13d
  __int128 v160; // rax
  __int64 v161; // r9
  unsigned int v162; // edx
  unsigned int v163; // edx
  __int128 v164; // [rsp-40h] [rbp-720h]
  __int128 v165; // [rsp-20h] [rbp-700h]
  __int128 v166; // [rsp-20h] [rbp-700h]
  __int128 v167; // [rsp-20h] [rbp-700h]
  __int128 v168; // [rsp-20h] [rbp-700h]
  __int128 v169; // [rsp-20h] [rbp-700h]
  __int128 v170; // [rsp-20h] [rbp-700h]
  __int128 v171; // [rsp-10h] [rbp-6F0h]
  __int128 v172; // [rsp-10h] [rbp-6F0h]
  __int128 v173; // [rsp-10h] [rbp-6F0h]
  __int128 v174; // [rsp-10h] [rbp-6F0h]
  __int64 v175; // [rsp+8h] [rbp-6D8h]
  unsigned __int8 *v176; // [rsp+10h] [rbp-6D0h]
  __int64 v177; // [rsp+18h] [rbp-6C8h]
  unsigned __int8 *v178; // [rsp+20h] [rbp-6C0h]
  __int64 v179; // [rsp+28h] [rbp-6B8h]
  __int64 v180; // [rsp+30h] [rbp-6B0h]
  unsigned __int8 *v181; // [rsp+30h] [rbp-6B0h]
  __int128 v182; // [rsp+30h] [rbp-6B0h]
  unsigned __int8 *v183; // [rsp+30h] [rbp-6B0h]
  unsigned __int64 v184; // [rsp+38h] [rbp-6A8h]
  __int128 v185; // [rsp+50h] [rbp-690h]
  unsigned int v186; // [rsp+50h] [rbp-690h]
  unsigned __int8 *v187; // [rsp+50h] [rbp-690h]
  __int64 v188; // [rsp+68h] [rbp-678h]
  __int64 v189; // [rsp+68h] [rbp-678h]
  void *v190; // [rsp+68h] [rbp-678h]
  void *v191; // [rsp+68h] [rbp-678h]
  __int128 v192; // [rsp+70h] [rbp-670h] BYREF
  unsigned __int64 *v193; // [rsp+80h] [rbp-660h]
  __int64 *v194; // [rsp+88h] [rbp-658h]
  __m128i si128; // [rsp+90h] [rbp-650h]
  __int128 v196; // [rsp+A0h] [rbp-640h]
  unsigned __int8 *v197; // [rsp+B0h] [rbp-630h]
  __int64 v198; // [rsp+B8h] [rbp-628h]
  unsigned __int8 *v199; // [rsp+C0h] [rbp-620h]
  __int64 v200; // [rsp+C8h] [rbp-618h]
  unsigned __int8 *v201; // [rsp+D0h] [rbp-610h]
  __int64 v202; // [rsp+D8h] [rbp-608h]
  unsigned __int8 *v203; // [rsp+E0h] [rbp-600h]
  __int64 v204; // [rsp+E8h] [rbp-5F8h]
  unsigned __int8 *v205; // [rsp+F0h] [rbp-5F0h]
  __int64 v206; // [rsp+F8h] [rbp-5E8h]
  __int64 v207; // [rsp+100h] [rbp-5E0h]
  __int64 v208; // [rsp+108h] [rbp-5D8h]
  __int64 v209; // [rsp+110h] [rbp-5D0h]
  __int64 v210; // [rsp+118h] [rbp-5C8h]
  __int64 v211; // [rsp+120h] [rbp-5C0h]
  __int64 v212; // [rsp+128h] [rbp-5B8h]
  __int64 v213; // [rsp+130h] [rbp-5B0h]
  __int64 v214; // [rsp+138h] [rbp-5A8h]
  __int64 v215; // [rsp+140h] [rbp-5A0h]
  __int64 v216; // [rsp+148h] [rbp-598h]
  unsigned __int8 *v217; // [rsp+150h] [rbp-590h]
  __int64 v218; // [rsp+158h] [rbp-588h]
  char v219; // [rsp+168h] [rbp-578h] BYREF
  char v220; // [rsp+16Ch] [rbp-574h] BYREF
  char v221; // [rsp+17Ah] [rbp-566h] BYREF
  char v222; // [rsp+17Bh] [rbp-565h] BYREF
  int v223; // [rsp+17Ch] [rbp-564h] BYREF
  __int64 v224; // [rsp+180h] [rbp-560h] BYREF
  int v225; // [rsp+188h] [rbp-558h]
  __int64 v226; // [rsp+190h] [rbp-550h] BYREF
  __int64 v227; // [rsp+198h] [rbp-548h]
  __int16 v228; // [rsp+1A0h] [rbp-540h] BYREF
  __int64 v229; // [rsp+1A8h] [rbp-538h]
  __int64 v230; // [rsp+1B0h] [rbp-530h] BYREF
  __int64 v231; // [rsp+1B8h] [rbp-528h]
  __int16 v232; // [rsp+1C0h] [rbp-520h] BYREF
  __int64 v233; // [rsp+1C8h] [rbp-518h]
  int v234; // [rsp+1D0h] [rbp-510h] BYREF
  unsigned __int64 v235; // [rsp+1D8h] [rbp-508h]
  __int64 v236; // [rsp+1E0h] [rbp-500h]
  __int64 v237; // [rsp+1E8h] [rbp-4F8h]
  __int64 v238; // [rsp+1F0h] [rbp-4F0h] BYREF
  __int64 v239; // [rsp+1F8h] [rbp-4E8h]
  int v240; // [rsp+200h] [rbp-4E0h] BYREF
  __int64 v241; // [rsp+208h] [rbp-4D8h]
  const __m128i *v242; // [rsp+210h] [rbp-4D0h] BYREF
  __int64 v243; // [rsp+218h] [rbp-4C8h]
  __int64 (__fastcall *v244)(unsigned __int64 *, const __m128i **, int); // [rsp+220h] [rbp-4C0h]
  __int64 (__fastcall *v245)(__int64 *, __int64, __m128i); // [rsp+228h] [rbp-4B8h]
  unsigned __int16 *v246; // [rsp+230h] [rbp-4B0h] BYREF
  __int64 v247; // [rsp+238h] [rbp-4A8h]
  void (__fastcall *v248)(unsigned __int16 **, unsigned __int16 **, __int64); // [rsp+240h] [rbp-4A0h]
  __int64 *v249; // [rsp+248h] [rbp-498h]
  int *v250; // [rsp+250h] [rbp-490h]
  int *v251; // [rsp+258h] [rbp-488h]
  char *v252; // [rsp+260h] [rbp-480h]
  char *v253; // [rsp+268h] [rbp-478h]
  const __m128i *v254; // [rsp+270h] [rbp-470h] BYREF
  __int64 v255; // [rsp+278h] [rbp-468h]
  __int64 (__fastcall *v256)(unsigned __int64 *, const __m128i **, int); // [rsp+280h] [rbp-460h] BYREF
  __int64 (__fastcall *v257)(__int64 *, __int64, __m128i); // [rsp+288h] [rbp-458h]
  __int64 *v258; // [rsp+380h] [rbp-360h] BYREF
  __int64 v259; // [rsp+388h] [rbp-358h]
  _QWORD v260[32]; // [rsp+390h] [rbp-350h] BYREF
  __int64 *v261; // [rsp+490h] [rbp-250h] BYREF
  __int64 v262; // [rsp+498h] [rbp-248h]
  _BYTE v263[256]; // [rsp+4A0h] [rbp-240h] BYREF
  unsigned __int64 v264; // [rsp+5A0h] [rbp-140h] BYREF
  __int64 v265; // [rsp+5A8h] [rbp-138h]
  unsigned __int64 v266; // [rsp+5B0h] [rbp-130h] BYREF
  unsigned int v267; // [rsp+5B8h] [rbp-128h]

  v10 = *(_QWORD *)(a2 + 80);
  *(_QWORD *)&v196 = a6;
  v220 = a4;
  v219 = a5;
  v224 = v10;
  if ( v10 )
    sub_B96E90((__int64)&v224, v10, 1);
  v225 = *(_DWORD *)(a2 + 72);
  v11 = *(unsigned __int16 **)(a2 + 48);
  v12 = *v11;
  v13 = *((_QWORD *)v11 + 1);
  LOWORD(v226) = v12;
  v227 = v13;
  if ( (_WORD)v12 )
  {
    if ( (unsigned __int16)(v12 - 17) <= 0xD3u )
    {
      v13 = 0;
      LOWORD(v12) = word_4456580[v12 - 1];
    }
  }
  else
  {
    si128.m128i_i64[0] = (__int64)&v226;
    if ( sub_30070B0((__int64)&v226) )
    {
      LOWORD(v12) = sub_3009970((__int64)&v226, v10, v71, v72, v73);
      v13 = v74;
    }
  }
  v14 = *(__int64 **)(a3 + 40);
  v228 = v12;
  v229 = v13;
  v15 = sub_2E79000(v14);
  v16 = (unsigned int)v226;
  v17 = sub_2FE6750(a1, (unsigned int)v226, v227, v15);
  LODWORD(v230) = v17;
  v18 = v17;
  v231 = v19;
  if ( (_WORD)v17 )
  {
    if ( (unsigned __int16)(v17 - 17) > 0xD3u )
    {
LABEL_8:
      v20 = v231;
      goto LABEL_9;
    }
    v20 = 0;
    v18 = word_4456580[(unsigned __int16)v17 - 1];
  }
  else
  {
    if ( !sub_30070B0((__int64)&v230) )
      goto LABEL_8;
    v18 = sub_3009970((__int64)&v230, v16, v68, v69, v70);
  }
LABEL_9:
  v232 = v18;
  v21 = v226;
  v233 = v20;
  if ( (_WORD)v226 )
  {
    if ( (unsigned __int16)(v226 - 17) > 0xD3u )
    {
LABEL_11:
      v22 = v227;
      goto LABEL_12;
    }
    v22 = 0;
    v21 = word_4456580[(unsigned __int16)v226 - 1];
  }
  else
  {
    if ( !sub_30070B0((__int64)&v226) )
      goto LABEL_11;
    v21 = sub_3009970((__int64)&v226, v16, v65, v66, v67);
  }
LABEL_12:
  LOWORD(v264) = v21;
  v265 = v22;
  if ( v21 )
  {
    if ( v21 == 1 || (unsigned __int16)(v21 - 504) <= 7u )
      BUG();
    v23 = *(_QWORD *)&byte_444C4A0[16 * v21 - 16];
  }
  else
  {
    v23 = sub_3007260((__int64)&v264);
    v236 = v23;
    v237 = v24;
  }
  v223 = v23;
  LOWORD(v234) = 0;
  v235 = 0;
  if ( !(_WORD)v226 )
    goto LABEL_44;
  if ( !*(_QWORD *)(a1 + 8LL * (unsigned __int16)v226 + 112) )
  {
    if ( (unsigned __int16)(v226 - 17) <= 0xD3u || *(_BYTE *)(a1 + (unsigned __int16)v226 + 524896) != 1 )
      goto LABEL_44;
    v102 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)a1 + 592LL);
    if ( v102 == sub_2D56A50 )
    {
      v16 = a1;
      sub_2FE6CC0((__int64)&v264, a1, *(_QWORD *)(a3 + 64), v226, v227);
      LOWORD(v234) = v265;
      v235 = v266;
    }
    else
    {
      v16 = *(_QWORD *)(a3 + 64);
      v234 = v102(a1, v16, v226, v227);
      v235 = v112;
    }
    if ( (v264 = sub_2D5B750((unsigned __int16 *)&v234),
          v265 = v103,
          sub_CA1930(&v264) < (unsigned __int64)(unsigned int)(2 * v223))
      || (v104 = 1, (_WORD)v234 != 1)
      && (!(_WORD)v234 || (v104 = (unsigned __int16)v234, !*(_QWORD *)(a1 + 8LL * (unsigned __int16)v234 + 112)))
      || *(_BYTE *)(a1 + 500LL * v104 + 6472) )
    {
LABEL_44:
      v63 = 0;
      goto LABEL_45;
    }
  }
  if ( (*(_BYTE *)(a2 + 28) & 4) == 0 )
  {
    v25 = *(const __m128i **)(a2 + 40);
    v26 = _mm_loadu_si128(v25);
    v180 = v25->m128i_i64[0];
    v27 = v25[2].m128i_i64[1];
    v192 = (__int128)v26;
    v188 = v27;
    v28 = v25[3].m128i_u32[0];
    sub_33DD090((__int64)&v264, a3, v26.m128i_i64[0], v26.m128i_i64[1], 0);
    v29 = v265;
    if ( (unsigned int)v265 > 0x40 )
    {
      v240 = sub_C44500((__int64)&v264);
      if ( v267 <= 0x40 || (v111 = v266) == 0 )
      {
LABEL_109:
        if ( v264 )
          j_j___libc_free_0_0(v264);
        goto LABEL_22;
      }
    }
    else
    {
      if ( (_DWORD)v265 )
      {
        v29 = 64;
        if ( v264 << (64 - (unsigned __int8)v265) != -1 )
        {
          _BitScanReverse64(&v30, ~(v264 << (64 - (unsigned __int8)v265)));
          v29 = v30 ^ 0x3F;
        }
      }
      v240 = v29;
      if ( v267 <= 0x40 || (v111 = v266) == 0 )
      {
LABEL_22:
        v221 = 0;
        v258 = v260;
        v254 = (const __m128i *)&v256;
        v194 = (__int64 *)v263;
        v261 = (__int64 *)v263;
        v222 = 0;
        LOBYTE(v238) = 0;
        v255 = 0x1000000000LL;
        v259 = 0x1000000000LL;
        si128.m128i_i64[0] = (__int64)&v261;
        v262 = 0x1000000000LL;
        v193 = &v266;
        v264 = (unsigned __int64)&v266;
        v265 = 0x1000000000LL;
        v244 = 0;
        v31 = (__m128i *)sub_22077B0(0x68u);
        if ( v31 )
        {
          v32 = si128.m128i_i64[0];
          v31->m128i_i64[1] = (__int64)&v232;
          v31[1].m128i_i64[0] = (__int64)&v228;
          v31[1].m128i_i64[1] = (__int64)&v240;
          v31[2].m128i_i64[1] = (__int64)&v223;
          v31[3].m128i_i64[0] = (__int64)&v221;
          v31[3].m128i_i64[1] = (__int64)&v222;
          v31->m128i_i64[0] = a3;
          v31[2].m128i_i64[0] = (__int64)&v224;
          v31[4].m128i_i64[0] = (__int64)&v238;
          v31[4].m128i_i64[1] = (__int64)&v254;
          v31[5].m128i_i64[0] = v32;
          v31[5].m128i_i64[1] = (__int64)&v264;
          v31[6].m128i_i64[0] = (__int64)&v258;
        }
        v242 = v31;
        v245 = sub_3442310;
        v244 = sub_343FAE0;
        v179 = v28;
        v248 = 0;
        sub_343FAE0((unsigned __int64 *)&v246, &v242, 2);
        v249 = (__int64 *)v245;
        v248 = (void (__fastcall *)(unsigned __int16 **, unsigned __int16 **, __int64))v244;
        v33 = sub_33CA8D0((_QWORD *)v188, v28, (__int64)&v246, 0, 0);
        if ( v248 )
          v248(&v246, &v246, 3);
        if ( v244 )
          v244((unsigned __int64 *)&v242, &v242, 3);
        if ( !v33 )
          goto LABEL_92;
        v176 = 0;
        v175 = 0;
        v35 = *(_DWORD *)(v188 + 24);
        if ( v35 == 156 )
        {
          *((_QWORD *)&v173 + 1) = (unsigned int)v255;
          *(_QWORD *)&v173 = v254;
          si128.m128i_i64[0] = (__int64)sub_33FC220((_QWORD *)a3, 156, (__int64)&v224, v230, v231, v34, v173);
          si128.m128i_i64[1] = v124;
          *((_QWORD *)&v168 + 1) = (unsigned int)v262;
          *(_QWORD *)&v168 = v261;
          v178 = sub_33FC220((_QWORD *)a3, 156, (__int64)&v224, v226, v227, v125, v168);
          v177 = v126;
          *((_QWORD *)&v174 + 1) = (unsigned int)v265;
          *(_QWORD *)&v174 = v264;
          v176 = sub_33FC220((_QWORD *)a3, 156, (__int64)&v224, v226, v227, v127, v174);
          v175 = v128;
          *((_QWORD *)&v169 + 1) = (unsigned int)v259;
          *(_QWORD *)&v169 = v258;
          v36 = (__int128)si128;
          *(_QWORD *)&v185 = sub_33FC220((_QWORD *)a3, 156, (__int64)&v224, v230, v231, v129, v169);
          *((_QWORD *)&v185 + 1) = v130;
        }
        else if ( v35 == 168 )
        {
          v215 = sub_3288900(a3, v230, v231, (int)&v224, v254->m128i_i64[0], v254->m128i_i64[1]);
          v216 = v148;
          si128.m128i_i64[0] = v215;
          si128.m128i_i64[1] = (unsigned int)v148;
          v213 = sub_3288900(a3, v226, v227, (int)&v224, *v261, v261[1]);
          v214 = v149;
          v178 = (unsigned __int8 *)v213;
          v177 = (unsigned int)v149;
          v150 = sub_3288900(a3, v226, v227, (int)&v224, *(_QWORD *)v264, *(_QWORD *)(v264 + 8));
          v212 = v151;
          v211 = v150;
          v176 = (unsigned __int8 *)v150;
          v175 = (unsigned int)v151;
          v209 = sub_3288900(a3, v230, v231, (int)&v224, *v258, v258[1]);
          *(_QWORD *)&v185 = v209;
          v210 = v152;
          *((_QWORD *)&v185 + 1) = (unsigned int)v152;
          v36 = (__int128)si128;
        }
        else
        {
          v177 = *((unsigned int *)v261 + 2);
          v178 = (unsigned __int8 *)*v261;
          *(_QWORD *)&v185 = *v258;
          *((_QWORD *)&v185 + 1) = *((unsigned int *)v258 + 2);
          v36 = __PAIR128__(v254->m128i_u32[2], v254->m128i_i64[0]);
        }
        si128 = _mm_load_si128((const __m128i *)&v192);
        if ( v222 )
        {
          v180 = (__int64)sub_3406EB0((_QWORD *)a3, 0xC0u, (__int64)&v224, (unsigned int)v226, v227, v34, v192, v36);
          v207 = v180;
          v208 = v117;
          si128.m128i_i64[1] = (unsigned int)v117 | si128.m128i_i64[1] & 0xFFFFFFFF00000000LL;
          sub_3489D20(v196, v180, (unsigned int)v117, 0xFFFFFFFF00000000LL, v118, v119);
        }
        v246 = (unsigned __int16 *)&v226;
        v250 = &v234;
        v251 = &v223;
        v252 = &v220;
        v247 = a1;
        v248 = (void (__fastcall *)(unsigned __int16 **, unsigned __int16 **, __int64))a3;
        v249 = &v224;
        v253 = &v219;
        v205 = sub_3445150(&v246, v180, si128.m128i_i64[1], (__int64)v178, v177, v34, v26);
        v39 = v205;
        v40 = v205;
        si128.m128i_i64[0] = (__int64)v205;
        v206 = v41;
        si128.m128i_i64[1] = (unsigned int)v41 | si128.m128i_i64[1] & 0xFFFFFFFF00000000LL;
        if ( !v205 )
        {
LABEL_92:
          v63 = 0;
LABEL_93:
          if ( (unsigned __int64 *)v264 != v193 )
            _libc_free(v264);
          if ( v261 != v194 )
            _libc_free((unsigned __int64)v261);
          if ( v258 != v260 )
            _libc_free((unsigned __int64)v258);
          if ( v254 != (const __m128i *)&v256 )
            _libc_free((unsigned __int64)v254);
          goto LABEL_45;
        }
        v42 = *(unsigned int *)(v196 + 8);
        if ( v42 + 1 > (unsigned __int64)*(unsigned int *)(v196 + 12) )
        {
          sub_C8D5F0(v196, (const void *)(v196 + 16), v42 + 1, 8u, v37, v38);
          v40 = v39;
          v42 = *(unsigned int *)(v196 + 8);
        }
        v43 = v196;
        *(_QWORD *)(*(_QWORD *)v196 + 8 * v42) = v39;
        ++*(_DWORD *)(v43 + 8);
        if ( !v221 )
        {
LABEL_38:
          if ( (_BYTE)v238 )
          {
            si128.m128i_i64[0] = (__int64)v40;
            v197 = sub_3406EB0(
                     (_QWORD *)a3,
                     0xC0u,
                     (__int64)&v224,
                     (unsigned int)v226,
                     v227,
                     v38,
                     __PAIR128__(si128.m128i_u64[1], (unsigned __int64)v40),
                     v185);
            si128.m128i_i64[0] = (__int64)v197;
            v198 = v131;
            v187 = v197;
            si128.m128i_i64[1] = (unsigned int)v131 | si128.m128i_i64[1] & 0xFFFFFFFF00000000LL;
            sub_3489D20(v196, (__int64)v197, (unsigned int)v131, 0xFFFFFFFF00000000LL, v132, v133);
            v40 = v187;
          }
          v181 = v40;
          v44 = *(_QWORD *)(a3 + 64);
          v45 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, __int64))(*(_QWORD *)a1 + 528LL);
          v46 = sub_2E79000(*(__int64 **)(a3 + 40));
          v47 = v45(a1, v46, v44, (unsigned int)v226, v227);
          v49 = v48;
          v186 = v47;
          *(_QWORD *)&v196 = sub_3400BD0(a3, 1, (__int64)&v224, (unsigned int)v226, v227, 0, v26, 0);
          *((_QWORD *)&v196 + 1) = v50;
          *(_QWORD *)&v51 = sub_33ED040((_QWORD *)a3, 0x11u);
          *((_QWORD *)&v164 + 1) = v179;
          *(_QWORD *)&v164 = v188;
          v53 = sub_340F900((_QWORD *)a3, 0xD0u, (__int64)&v224, v186, v49, v52, v164, v196, v51);
          v55 = v54;
          v56 = *(_QWORD *)(v53 + 48) + 16LL * v54;
          v57 = v226;
          v58 = *(_WORD *)v56;
          v59 = *(_QWORD *)(v56 + 8);
          si128.m128i_i64[0] = (__int64)v181;
          v60 = v55;
          v61 = v227;
          LOWORD(v242) = v58;
          v243 = v59;
          if ( v58 )
          {
            v62 = ((unsigned __int16)(v58 - 17) < 0xD4u) + 205;
          }
          else
          {
            *(_QWORD *)&v196 = v226;
            v120 = sub_30070B0((__int64)&v242);
            v57 = v196;
            v62 = 205 - (!v120 - 1);
          }
          v63 = sub_340EC60((_QWORD *)a3, v62, (__int64)&v224, v57, v61, 0, v53, v60, v192, *(_OWORD *)&si128);
          goto LABEL_93;
        }
        *(_QWORD *)&v182 = sub_3406EB0(
                             (_QWORD *)a3,
                             0x39u,
                             (__int64)&v224,
                             (unsigned int)v226,
                             v227,
                             v38,
                             v192,
                             *(_OWORD *)&si128);
        *((_QWORD *)&v182 + 1) = v134;
        sub_3489D20(v196, v182, v134, v135, v136, v137);
        if ( (_WORD)v226 )
        {
          if ( (unsigned __int16)(v226 - 17) <= 0xD3u )
          {
LABEL_123:
            v203 = sub_3445150(&v246, v182, *((__int64 *)&v182 + 1), (__int64)v176, v175, v138, v26);
            v141 = (__int64)v203;
            v204 = v142;
            v142 = (unsigned int)v142;
            v184 = (unsigned int)v142 | *((_QWORD *)&v182 + 1) & 0xFFFFFFFF00000000LL;
LABEL_124:
            v143 = v196;
            sub_3489D20(v196, v141, v142, 0xFFFFFFFF00000000LL, v139, v140);
            *((_QWORD *)&v170 + 1) = v184;
            *(_QWORD *)&v170 = v141;
            v199 = sub_3406EB0(
                     (_QWORD *)a3,
                     0x38u,
                     (__int64)&v224,
                     (unsigned int)v226,
                     v227,
                     v144,
                     v170,
                     *(_OWORD *)&si128);
            si128.m128i_i64[0] = (__int64)v199;
            v200 = v145;
            v183 = v199;
            si128.m128i_i64[1] = (unsigned int)v145 | si128.m128i_i64[1] & 0xFFFFFFFF00000000LL;
            sub_3489D20(v143, (__int64)v199, (unsigned int)v145, 0xFFFFFFFF00000000LL, v146, v147);
            v40 = v183;
            goto LABEL_38;
          }
        }
        else if ( sub_30070B0((__int64)&v226) )
        {
          goto LABEL_123;
        }
        *(_QWORD *)&v160 = sub_3400BD0(a3, 1, (__int64)&v224, (unsigned int)v230, v231, 0, v26, 0);
        v201 = sub_3406EB0((_QWORD *)a3, 0xC0u, (__int64)&v224, (unsigned int)v226, v227, v161, v182, v160);
        v141 = (__int64)v201;
        v202 = v142;
        v142 = (unsigned int)v142;
        v184 = (unsigned int)v142 | *((_QWORD *)&v182 + 1) & 0xFFFFFFFF00000000LL;
        goto LABEL_124;
      }
    }
    j_j___libc_free_0_0(v111);
    if ( (unsigned int)v265 <= 0x40 )
      goto LABEL_22;
    goto LABEL_109;
  }
  v75 = *(__int16 **)(a2 + 48);
  v76 = *v75;
  v77 = *((_QWORD *)v75 + 1);
  LOWORD(v238) = v76;
  v239 = v77;
  if ( v76 )
  {
    if ( (unsigned __int16)(v76 - 17) <= 0xD3u )
    {
      v76 = word_4456580[v76 - 1];
      v77 = 0;
    }
  }
  else
  {
    v194 = (__int64 *)v77;
    si128.m128i_i64[0] = (__int64)&v238;
    v108 = sub_30070B0((__int64)&v238);
    v77 = (__int64)v194;
    if ( v108 )
      v76 = sub_3009970((__int64)&v238, v16, (__int64)v194, v109, v110);
  }
  v78 = *(__int64 **)(a3 + 40);
  v241 = v77;
  LOWORD(v240) = v76;
  v79 = sub_2E79000(v78);
  v80 = (unsigned int)v238;
  v81 = sub_2FE6750(a1, (unsigned int)v238, v239, v79);
  LODWORD(v242) = v81;
  v82 = v81;
  v243 = v83;
  if ( (_WORD)v81 )
  {
    if ( (unsigned __int16)(v81 - 17) <= 0xD3u )
    {
      v82 = word_4456580[(unsigned __int16)v81 - 1];
      v84 = 0;
      goto LABEL_65;
    }
  }
  else if ( sub_30070B0((__int64)&v242) )
  {
    v82 = sub_3009970((__int64)&v242, v80, v105, v106, v107);
    goto LABEL_65;
  }
  v84 = v243;
LABEL_65:
  LOWORD(v246) = v82;
  v261 = (__int64 *)v263;
  v194 = (__int64 *)v263;
  v262 = 0x1000000000LL;
  v265 = 0x1000000000LL;
  v85 = *(_QWORD *)(a2 + 40);
  v247 = v84;
  v222 = 0;
  v264 = (unsigned __int64)&v266;
  v86 = *(_DWORD *)(v85 + 48);
  v193 = &v266;
  v87 = *(_QWORD *)(v85 + 40);
  v256 = 0;
  v189 = v87;
  v88 = (__m128i *)sub_22077B0(0x38u);
  if ( v88 )
  {
    v88->m128i_i64[1] = (__int64)&v261;
    v88[2].m128i_i64[0] = (__int64)&v246;
    v88->m128i_i64[0] = (__int64)&v222;
    v88[1].m128i_i64[0] = a3;
    v88[1].m128i_i64[1] = (__int64)&v224;
    v88[2].m128i_i64[1] = (__int64)&v264;
    v88[3].m128i_i64[0] = (__int64)&v240;
  }
  v254 = v88;
  v257 = sub_34415F0;
  v260[0] = 0;
  v256 = sub_343FC20;
  sub_343FC20((unsigned __int64 *)&v258, &v254, 2);
  v260[1] = v257;
  v260[0] = v256;
  v90 = sub_33CA8D0((_QWORD *)v189, v86, (__int64)&v258, 0, 0);
  if ( v260[0] )
    ((void (__fastcall *)(__int64 **, __int64 **, __int64))v260[0])(&v258, &v258, 3);
  if ( v256 )
    v256((unsigned __int64 *)&v254, &v254, 3);
  if ( v90 )
  {
    v192 = 0u;
    si128 = 0u;
    v91 = *(_DWORD *)(v189 + 24);
    if ( v91 == 156 )
    {
      *((_QWORD *)&v172 + 1) = (unsigned int)v262;
      *(_QWORD *)&v172 = v261;
      *(_QWORD *)&v192 = sub_33FC220((_QWORD *)a3, 156, (__int64)&v224, (__int64)v242, v243, v89, v172);
      *((_QWORD *)&v192 + 1) = v121 | *((_QWORD *)&v192 + 1) & 0xFFFFFFFF00000000LL;
      *((_QWORD *)&v167 + 1) = (unsigned int)v265;
      *(_QWORD *)&v167 = v264;
      si128.m128i_i64[0] = (__int64)sub_33FC220((_QWORD *)a3, 156, (__int64)&v224, v238, v239, v122, v167);
      v94 = v123 | si128.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      si128.m128i_i64[1] = v94;
    }
    else if ( v91 == 168 )
    {
      if ( *(_DWORD *)(*v261 + 24) == 51 )
      {
        v258 = 0;
        LODWORD(v259) = 0;
        v153 = sub_33F17F0((_QWORD *)a3, 51, (__int64)&v258, (unsigned int)v242, v243);
        v156 = v163;
        if ( v258 )
        {
          v191 = v153;
          sub_B91220((__int64)&v258, (__int64)v258);
          v153 = v191;
        }
      }
      else
      {
        v153 = sub_33FAF80(a3, 168, (__int64)&v224, (__int64)v242, v243, v89, a7);
        v156 = v155;
      }
      *(_QWORD *)&v192 = v153;
      *((_QWORD *)&v192 + 1) = v156 | *((_QWORD *)&v192 + 1) & 0xFFFFFFFF00000000LL;
      if ( *(_DWORD *)(*(_QWORD *)v264 + 24LL) == 51 )
      {
        v258 = 0;
        LODWORD(v259) = 0;
        v157 = sub_33F17F0((_QWORD *)a3, 51, (__int64)&v258, v238, v239);
        v159 = v162;
        if ( v258 )
        {
          v190 = v157;
          sub_B91220((__int64)&v258, (__int64)v258);
          v157 = v190;
        }
      }
      else
      {
        v157 = sub_33FAF80(a3, 168, (__int64)&v224, v238, v239, v154, a7);
        v159 = v158;
      }
      si128.m128i_i64[0] = (__int64)v157;
      si128.m128i_i64[1] = v159 | si128.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    }
    else
    {
      v92 = *((unsigned int *)v261 + 2);
      *(_QWORD *)&v192 = *v261;
      *((_QWORD *)&v192 + 1) = v92 | *((_QWORD *)&v192 + 1) & 0xFFFFFFFF00000000LL;
      v93 = *(unsigned int *)(v264 + 8);
      si128.m128i_i64[0] = *(_QWORD *)v264;
      v94 = v93 | si128.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      si128.m128i_i64[1] = v94;
    }
    v95 = *(__int64 **)(a2 + 40);
    v96 = *v95;
    v97 = v95[1];
    if ( v222 )
    {
      v171 = v192;
      *(_QWORD *)&v166 = *v95;
      *((_QWORD *)&v192 + 1) = v95[1];
      *((_QWORD *)&v166 + 1) = *((_QWORD *)&v192 + 1);
      *(_QWORD *)&v192 = v96;
      v217 = sub_3405C90((_QWORD *)a3, 0xC0u, (__int64)&v224, (unsigned int)v238, v239, 4, a7, v166, v171);
      v98 = v217;
      v218 = v113;
      v97 = (unsigned int)v113 | *((_QWORD *)&v192 + 1) & 0xFFFFFFFF00000000LL;
      v115 = *(unsigned int *)(v196 + 8);
      if ( v115 + 1 > (unsigned __int64)*(unsigned int *)(v196 + 12) )
      {
        *((_QWORD *)&v192 + 1) = (unsigned int)v113 | *((_QWORD *)&v192 + 1) & 0xFFFFFFFF00000000LL;
        *(_QWORD *)&v192 = v217;
        sub_C8D5F0(v196, (const void *)(v196 + 16), v115 + 1, 8u, v114, v94);
        v115 = *(unsigned int *)(v196 + 8);
        v97 = *((_QWORD *)&v192 + 1);
      }
      v116 = v196;
      *(_QWORD *)(*(_QWORD *)v196 + 8 * v115) = v217;
      ++*(_DWORD *)(v116 + 8);
    }
    else
    {
      v98 = (unsigned __int8 *)*v95;
    }
    *((_QWORD *)&v165 + 1) = v97;
    *(_QWORD *)&v165 = v98;
    v99 = sub_3406EB0((_QWORD *)a3, 0x3Au, (__int64)&v224, (unsigned int)v238, v239, v94, v165, *(_OWORD *)&si128);
    *(_QWORD *)&v192 = v100;
    v101 = v99;
  }
  else
  {
    v101 = 0;
  }
  if ( (unsigned __int64 *)v264 != v193 )
    _libc_free(v264);
  if ( v261 != v194 )
    _libc_free((unsigned __int64)v261);
  v63 = (__int64)v101;
LABEL_45:
  if ( v224 )
    sub_B91220((__int64)&v224, v224);
  return v63;
}
