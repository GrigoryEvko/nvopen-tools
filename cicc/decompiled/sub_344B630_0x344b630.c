// Function: sub_344B630
// Address: 0x344b630
//
unsigned __int8 *__fastcall sub_344B630(__int64 a1, __int64 a2, __int64 *a3, int a4, int a5, __int64 a6, __m128i a7)
{
  __int64 v11; // rsi
  unsigned __int16 *v12; // rax
  int v13; // ebx
  __int64 v14; // rdx
  __int64 *v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rsi
  int v18; // eax
  __int16 v19; // bx
  __int64 v20; // rdx
  __int64 v21; // rdx
  unsigned __int16 v22; // bx
  __int64 v23; // rdx
  __int64 v24; // rdx
  bool v25; // zf
  const __m128i *v26; // rax
  __m128i v27; // xmm1
  __int64 v28; // rcx
  __int32 v29; // eax
  __m128i *v30; // rax
  __int64 v31; // r8
  __int64 v32; // rsi
  __int64 v33; // rbx
  char v34; // bl
  __int64 v35; // r9
  int v36; // eax
  __int64 v37; // r11
  __int64 v38; // r10
  __int64 v39; // rdx
  __int64 v40; // rdx
  __m128i v41; // xmm2
  unsigned __int16 v42; // cx
  __int64 v43; // rax
  unsigned int *v44; // rax
  __int64 v45; // rdx
  __int64 v46; // r9
  unsigned __int8 *v47; // rax
  __int64 v48; // r8
  __int64 v49; // r9
  __int64 v50; // r10
  unsigned __int8 *v51; // rbx
  __int64 v52; // rax
  unsigned __int8 *v53; // r12
  __int64 v55; // rdx
  __int64 v56; // rcx
  __int64 v57; // r8
  __int64 v58; // rdx
  __int64 v59; // rcx
  __int64 v60; // r8
  bool v61; // al
  __int64 v62; // rcx
  __int64 v63; // r8
  unsigned __int16 *v64; // rcx
  __m128i v65; // xmm0
  unsigned __int32 v66; // r13d
  __int64 v67; // rdx
  __int64 v68; // rcx
  __int64 *v69; // rsi
  __int64 *v70; // rdi
  __int64 v71; // rax
  __int64 v72; // rsi
  int v73; // eax
  __int16 v74; // r12
  __int64 v75; // rdx
  __int64 v76; // rdx
  __m128i *v77; // rax
  char v78; // bl
  __int64 v79; // r9
  int v80; // eax
  __int64 v81; // rdx
  __int64 *v82; // r10
  __int64 v83; // rdx
  unsigned __int64 v84; // r13
  unsigned __int8 *v85; // rax
  __int64 v86; // rdx
  unsigned __int8 *v87; // rbx
  __int64 (__fastcall *v88)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v89; // rdx
  unsigned int v90; // edx
  __int64 v91; // rdx
  int v92; // r9d
  __int64 v93; // rdx
  __int64 v94; // r9
  __int64 v95; // rsi
  __int64 v96; // rdx
  __int128 v97; // rax
  __int64 v98; // r9
  int v99; // r9d
  __int64 v100; // rdx
  unsigned __int8 *v101; // rax
  __int64 v102; // rdx
  unsigned __int8 *v103; // r12
  unsigned __int64 v104; // r13
  __int64 v105; // rax
  __int64 v106; // rdx
  unsigned __int8 *v107; // rax
  __int64 v108; // r8
  __int64 v109; // r9
  __int64 v110; // r10
  unsigned __int8 *v111; // rbx
  __int64 v112; // rdx
  unsigned __int64 v113; // rcx
  __int64 v114; // rax
  __int64 v115; // rdx
  __int64 v116; // rcx
  unsigned __int8 *v117; // rax
  __int64 v118; // r8
  __int64 v119; // r9
  __int64 v120; // r10
  unsigned __int8 *v121; // rbx
  unsigned __int8 *v122; // r12
  __int64 v123; // rdx
  unsigned __int64 v124; // r13
  __int64 v125; // rax
  __int64 v126; // rdx
  __int64 v127; // rcx
  unsigned __int8 *v128; // rax
  __int64 v129; // r8
  __int64 v130; // r9
  __int64 v131; // r10
  unsigned __int8 *v132; // rbx
  unsigned __int8 *v133; // r12
  __int64 v134; // rdx
  unsigned __int64 v135; // r13
  __int64 v136; // rax
  __int64 v137; // rdx
  int v138; // eax
  __int128 v139; // rax
  __int64 v140; // r9
  __int128 v141; // rax
  __int64 v142; // r8
  __int64 v143; // r9
  unsigned __int64 v144; // rcx
  __int64 v145; // r10
  __int64 v146; // rbx
  __int64 v147; // rax
  __int64 v148; // rdx
  __int64 v149; // rcx
  unsigned __int8 *v150; // rax
  __int64 v151; // r8
  __int64 v152; // r9
  __int64 v153; // r10
  unsigned __int8 *v154; // rbx
  __int64 v155; // rdx
  unsigned __int64 v156; // rcx
  __int64 v157; // rax
  __int64 v158; // rcx
  __int64 v159; // rdx
  __int64 v160; // rcx
  __int64 v161; // r8
  bool v162; // al
  __int64 v163; // r8
  __int16 v164; // ax
  __int64 v165; // rdx
  unsigned __int8 *v166; // rax
  __int64 v167; // r8
  unsigned __int8 *v168; // rbx
  __int64 v169; // rdx
  __int64 v170; // rax
  unsigned int v171; // edx
  __int64 v172; // r9
  unsigned __int8 *v173; // rax
  unsigned int v174; // edx
  unsigned int v175; // edx
  unsigned int v176; // edx
  unsigned int v177; // edx
  unsigned __int8 *v178; // rax
  unsigned int v179; // edx
  unsigned __int16 v180; // ax
  __int64 v181; // rsi
  unsigned int v182; // esi
  int v183; // eax
  __int64 *v184; // rdx
  __int64 *v185; // rcx
  int v186; // edx
  __int64 *v187; // r9
  unsigned int v188; // ebx
  bool v189; // al
  __int64 v190; // rax
  char v191; // al
  __int64 v192; // rdx
  __int64 v193; // rdx
  __int64 v194; // rdx
  __int64 v195; // rsi
  __int128 v196; // rax
  __int64 v197; // r9
  __int64 v198; // r8
  __int64 v199; // rdx
  __int64 v200; // r8
  __int64 v201; // r9
  unsigned int v202; // edx
  __int64 v203; // rdx
  __int64 v204; // rdx
  __int64 v205; // rax
  __int64 v206; // rdx
  __int64 *v207; // rax
  __int64 v208; // rdx
  int v209; // r9d
  __int64 *v210; // rax
  __int64 v211; // rdx
  unsigned __int8 *v212; // rax
  __int64 v213; // rdx
  __int64 v214; // rax
  __int64 v215; // rdx
  __int64 v216; // rax
  unsigned __int16 v217; // cx
  unsigned __int64 v218; // rax
  const __m128i *v219; // rsi
  unsigned __int64 v220; // rax
  bool v221; // di
  __int64 *v222; // r13
  unsigned int v223; // eax
  __int64 v224; // r8
  __int64 *v225; // rdx
  bool v226; // al
  __int64 v227; // rdx
  __int64 v228; // rcx
  __int64 v229; // r8
  unsigned __int16 v230; // ax
  __int128 v231; // [rsp-30h] [rbp-750h]
  __int128 v232; // [rsp-20h] [rbp-740h]
  __int128 v233; // [rsp-20h] [rbp-740h]
  __int128 v234; // [rsp-20h] [rbp-740h]
  __int128 v235; // [rsp-20h] [rbp-740h]
  __int128 v236; // [rsp-20h] [rbp-740h]
  __int128 v237; // [rsp-20h] [rbp-740h]
  __int128 v238; // [rsp-20h] [rbp-740h]
  __int128 v239; // [rsp-10h] [rbp-730h]
  __int128 v240; // [rsp-10h] [rbp-730h]
  __int128 v241; // [rsp-10h] [rbp-730h]
  __int128 v242; // [rsp-10h] [rbp-730h]
  __int128 v243; // [rsp-10h] [rbp-730h]
  __int64 v244; // [rsp-8h] [rbp-728h]
  unsigned int v245; // [rsp+0h] [rbp-720h]
  int v246; // [rsp+8h] [rbp-718h]
  unsigned __int8 *v247; // [rsp+10h] [rbp-710h]
  __int16 v248; // [rsp+10h] [rbp-710h]
  __int64 v249; // [rsp+10h] [rbp-710h]
  unsigned __int8 *v250; // [rsp+18h] [rbp-708h]
  __int128 v251; // [rsp+20h] [rbp-700h]
  unsigned int v252; // [rsp+20h] [rbp-700h]
  __int128 v253; // [rsp+20h] [rbp-700h]
  unsigned int v254; // [rsp+20h] [rbp-700h]
  __int64 v255; // [rsp+30h] [rbp-6F0h]
  __int64 *v256; // [rsp+30h] [rbp-6F0h]
  unsigned int v257; // [rsp+30h] [rbp-6F0h]
  __m128i v258; // [rsp+40h] [rbp-6E0h] BYREF
  __int128 v259; // [rsp+50h] [rbp-6D0h]
  __int128 v260; // [rsp+60h] [rbp-6C0h]
  __int64 *v261; // [rsp+70h] [rbp-6B0h]
  __int64 v262; // [rsp+78h] [rbp-6A8h]
  __int64 *v263; // [rsp+80h] [rbp-6A0h]
  __int64 *v264; // [rsp+88h] [rbp-698h]
  __int128 v265; // [rsp+90h] [rbp-690h]
  unsigned __int8 *v266; // [rsp+A0h] [rbp-680h]
  __int64 *v267; // [rsp+A8h] [rbp-678h]
  __int128 v268; // [rsp+B0h] [rbp-670h]
  __int128 v269; // [rsp+C0h] [rbp-660h]
  unsigned __int8 *v270; // [rsp+D0h] [rbp-650h]
  __int64 v271; // [rsp+D8h] [rbp-648h]
  unsigned __int8 *v272; // [rsp+E0h] [rbp-640h]
  __int64 v273; // [rsp+E8h] [rbp-638h]
  unsigned __int8 *v274; // [rsp+F0h] [rbp-630h]
  __int64 v275; // [rsp+F8h] [rbp-628h]
  unsigned __int8 *v276; // [rsp+100h] [rbp-620h]
  __int64 v277; // [rsp+108h] [rbp-618h]
  unsigned __int8 *v278; // [rsp+110h] [rbp-610h]
  __int64 v279; // [rsp+118h] [rbp-608h]
  unsigned __int8 *v280; // [rsp+120h] [rbp-600h]
  __int64 v281; // [rsp+128h] [rbp-5F8h]
  __int64 v282; // [rsp+130h] [rbp-5F0h]
  __int64 v283; // [rsp+138h] [rbp-5E8h]
  unsigned __int8 *v284; // [rsp+140h] [rbp-5E0h]
  __int64 v285; // [rsp+148h] [rbp-5D8h]
  unsigned __int8 *v286; // [rsp+150h] [rbp-5D0h]
  __int64 v287; // [rsp+158h] [rbp-5C8h]
  unsigned __int8 *v288; // [rsp+160h] [rbp-5C0h]
  __int64 v289; // [rsp+168h] [rbp-5B8h]
  __int64 v290; // [rsp+170h] [rbp-5B0h]
  __int64 v291; // [rsp+178h] [rbp-5A8h]
  unsigned __int8 *v292; // [rsp+180h] [rbp-5A0h]
  __int64 v293; // [rsp+188h] [rbp-598h]
  __int64 v294; // [rsp+190h] [rbp-590h]
  __int64 v295; // [rsp+198h] [rbp-588h]
  __int64 v296; // [rsp+1A0h] [rbp-580h]
  __int64 v297; // [rsp+1A8h] [rbp-578h]
  __int64 v298; // [rsp+1B0h] [rbp-570h]
  __int64 v299; // [rsp+1B8h] [rbp-568h]
  unsigned __int8 *v300; // [rsp+1C0h] [rbp-560h]
  __int64 v301; // [rsp+1C8h] [rbp-558h]
  char v302; // [rsp+1DFh] [rbp-541h] BYREF
  __int64 v303; // [rsp+1E0h] [rbp-540h] BYREF
  int v304; // [rsp+1E8h] [rbp-538h]
  __int64 v305; // [rsp+1F0h] [rbp-530h] BYREF
  __int64 v306; // [rsp+1F8h] [rbp-528h]
  __int16 v307; // [rsp+200h] [rbp-520h] BYREF
  __int64 v308; // [rsp+208h] [rbp-518h]
  __int64 v309; // [rsp+210h] [rbp-510h] BYREF
  __int64 v310; // [rsp+218h] [rbp-508h]
  __int16 v311; // [rsp+220h] [rbp-500h] BYREF
  __int64 v312; // [rsp+228h] [rbp-4F8h]
  unsigned int v313; // [rsp+230h] [rbp-4F0h] BYREF
  __int64 v314; // [rsp+238h] [rbp-4E8h]
  __int64 v315; // [rsp+240h] [rbp-4E0h]
  __int64 v316; // [rsp+248h] [rbp-4D8h]
  __int64 v317; // [rsp+250h] [rbp-4D0h] BYREF
  __int64 v318; // [rsp+258h] [rbp-4C8h]
  __int64 v319[2]; // [rsp+260h] [rbp-4C0h] BYREF
  const __m128i *v320; // [rsp+270h] [rbp-4B0h] BYREF
  __int64 v321; // [rsp+278h] [rbp-4A8h]
  __int64 (__fastcall *v322)(unsigned __int64 *, const __m128i **, int); // [rsp+280h] [rbp-4A0h]
  __int64 (__fastcall *v323)(__int64 *, __int64, __m128i); // [rsp+288h] [rbp-498h]
  __int64 v324; // [rsp+290h] [rbp-490h] BYREF
  __int64 v325; // [rsp+298h] [rbp-488h]
  __int64 (__fastcall *v326)(unsigned __int64 *, const __m128i **, int); // [rsp+2A0h] [rbp-480h]
  __int64 (__fastcall *v327)(__int64 *, __int64, __m128i); // [rsp+2A8h] [rbp-478h]
  const __m128i *v328; // [rsp+2B0h] [rbp-470h] BYREF
  __int64 v329; // [rsp+2B8h] [rbp-468h]
  __int64 (__fastcall *v330)(unsigned __int64 *, const __m128i **, int); // [rsp+2C0h] [rbp-460h] BYREF
  __int64 (__fastcall *v331)(__int64 *, __int64, __m128i); // [rsp+2C8h] [rbp-458h]
  __int64 *v332; // [rsp+3C0h] [rbp-360h] BYREF
  __int64 v333; // [rsp+3C8h] [rbp-358h]
  _QWORD v334[32]; // [rsp+3D0h] [rbp-350h] BYREF
  __int64 *v335; // [rsp+4D0h] [rbp-250h] BYREF
  __int64 v336; // [rsp+4D8h] [rbp-248h]
  _BYTE v337[256]; // [rsp+4E0h] [rbp-240h] BYREF
  __int64 *v338; // [rsp+5E0h] [rbp-140h] BYREF
  __int64 v339; // [rsp+5E8h] [rbp-138h]
  _QWORD v340[38]; // [rsp+5F0h] [rbp-130h] BYREF

  v11 = *(_QWORD *)(a2 + 80);
  LODWORD(v267) = a4;
  LODWORD(v266) = a5;
  v303 = v11;
  if ( v11 )
    sub_B96E90((__int64)&v303, v11, 1);
  v304 = *(_DWORD *)(a2 + 72);
  v12 = *(unsigned __int16 **)(a2 + 48);
  v13 = *v12;
  v14 = *((_QWORD *)v12 + 1);
  LOWORD(v305) = v13;
  v306 = v14;
  if ( (_WORD)v13 )
  {
    if ( (unsigned __int16)(v13 - 17) <= 0xD3u )
    {
      v14 = 0;
      LOWORD(v13) = word_4456580[v13 - 1];
    }
  }
  else
  {
    *(_QWORD *)&v268 = v14;
    *(_QWORD *)&v269 = &v305;
    v61 = sub_30070B0((__int64)&v305);
    v14 = v268;
    if ( v61 )
      LOWORD(v13) = sub_3009970((__int64)&v305, v11, v268, v62, v63);
  }
  v15 = (__int64 *)a3[5];
  v307 = v13;
  v308 = v14;
  v16 = sub_2E79000(v15);
  v17 = (unsigned int)v305;
  v18 = sub_2FE6750(a1, (unsigned int)v305, v306, v16);
  LODWORD(v309) = v18;
  v19 = v18;
  v310 = v20;
  if ( (_WORD)v18 )
  {
    if ( (unsigned __int16)(v18 - 17) > 0xD3u )
    {
LABEL_8:
      v21 = v310;
      goto LABEL_9;
    }
    v21 = 0;
    v19 = word_4456580[(unsigned __int16)v18 - 1];
  }
  else
  {
    *(_QWORD *)&v269 = &v309;
    if ( !sub_30070B0((__int64)&v309) )
      goto LABEL_8;
    v19 = sub_3009970((__int64)&v309, v17, v58, v59, v60);
  }
LABEL_9:
  v311 = v19;
  v22 = v305;
  v312 = v21;
  if ( (_WORD)v305 )
  {
    if ( (unsigned __int16)(v305 - 17) > 0xD3u )
    {
LABEL_11:
      v23 = v306;
      goto LABEL_12;
    }
    v23 = 0;
    v22 = word_4456580[(unsigned __int16)v305 - 1];
  }
  else
  {
    *(_QWORD *)&v269 = &v305;
    if ( !sub_30070B0((__int64)&v305) )
      goto LABEL_11;
    v22 = sub_3009970((__int64)&v305, v17, v55, v56, v57);
  }
LABEL_12:
  LOWORD(v324) = v22;
  v325 = v23;
  if ( v22 )
  {
    if ( v22 == 1 || (unsigned __int16)(v22 - 504) <= 7u )
      goto LABEL_184;
    *(_QWORD *)&v269 = *(_QWORD *)&byte_444C4A0[16 * v22 - 16];
  }
  else
  {
    v315 = sub_3007260((__int64)&v324);
    v316 = v24;
    *(_QWORD *)&v269 = v315;
  }
  v314 = 0;
  LOWORD(v313) = 0;
  if ( !(_WORD)v305 )
    goto LABEL_34;
  v25 = *(_QWORD *)(a1 + 8LL * (unsigned __int16)v305 + 112) == 0;
  LODWORD(v262) = v269;
  if ( v25 )
  {
    if ( (unsigned __int16)(v305 - 17) <= 0xD3u || *(_BYTE *)(a1 + (unsigned __int16)v305 + 524896) != 1 )
      goto LABEL_34;
    v88 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)a1 + 592LL);
    if ( v88 == sub_2D56A50 )
    {
      sub_2FE6CC0((__int64)&v338, a1, a3[8], v305, v306);
      LOWORD(v313) = v339;
      v314 = v340[0];
    }
    else
    {
      v313 = v88(a1, a3[8], v305, v306);
      v314 = v165;
    }
    if ( (v338 = (__int64 *)sub_2D5B750((unsigned __int16 *)&v313),
          v339 = v89,
          sub_CA1930(&v338) < (unsigned __int64)(unsigned int)(2 * v262))
      || (v90 = 1, (_WORD)v313 != 1)
      && (!(_WORD)v313 || (v90 = (unsigned __int16)v313, !*(_QWORD *)(a1 + 8LL * (unsigned __int16)v313 + 112)))
      || *(_BYTE *)(a1 + 500LL * v90 + 6472) )
    {
LABEL_34:
      v53 = 0;
      goto LABEL_35;
    }
  }
  v26 = *(const __m128i **)(a2 + 40);
  if ( (*(_BYTE *)(a2 + 28) & 4) == 0 )
  {
    v328 = (const __m128i *)&v330;
    v332 = v334;
    v261 = v334;
    v335 = (__int64 *)v337;
    v263 = (__int64 *)v337;
    v329 = 0x1000000000LL;
    v333 = 0x1000000000LL;
    v336 = 0x1000000000LL;
    v338 = v340;
    v339 = 0x1000000000LL;
    v27 = _mm_loadu_si128(v26);
    v264 = v340;
    v28 = v26[2].m128i_i64[1];
    v29 = v26[3].m128i_i32[0];
    *(_QWORD *)&v259 = &v328;
    *(_QWORD *)&v260 = &v332;
    *(_QWORD *)&v268 = v28;
    LODWORD(v265) = v29;
    v322 = 0;
    v258 = v27;
    v30 = (__m128i *)sub_22077B0(0x40u);
    if ( v30 )
    {
      v31 = v259;
      v32 = v260;
      v30->m128i_i64[1] = (__int64)a3;
      v30[1].m128i_i64[0] = (__int64)&v303;
      v30[1].m128i_i64[1] = (__int64)&v307;
      v30->m128i_i64[0] = v31;
      v30[2].m128i_i64[0] = v32;
      v30[2].m128i_i64[1] = (__int64)&v335;
      v30[3].m128i_i64[0] = (__int64)&v311;
      v30[3].m128i_i64[1] = (__int64)&v338;
    }
    v320 = v30;
    v33 = (unsigned int)v265;
    v323 = sub_3443590;
    *(_QWORD *)&v265 = &v324;
    v322 = sub_343FB90;
    v326 = 0;
    sub_343FB90((unsigned __int64 *)&v324, &v320, 2);
    v327 = v323;
    v326 = v322;
    v34 = sub_33CA8D0((_QWORD *)v268, v33, (__int64)&v324, 0, 0);
    if ( v326 )
      v326((unsigned __int64 *)v265, (const __m128i **)v265, 3);
    if ( v322 )
      v322((unsigned __int64 *)&v320, &v320, 3);
    if ( !v34 )
      goto LABEL_97;
    v265 = 0u;
    v260 = 0u;
    v259 = 0u;
    v36 = *(_DWORD *)(v268 + 24);
    if ( v36 == 156 )
    {
      *(_QWORD *)&v268 = &v303;
      *((_QWORD *)&v242 + 1) = (unsigned int)v329;
      *(_QWORD *)&v242 = v328;
      v247 = sub_33FC220(a3, 156, (__int64)&v303, v305, v306, (__int64)&v303, v242);
      v252 = v175;
      *((_QWORD *)&v237 + 1) = (unsigned int)v333;
      *(_QWORD *)&v237 = v332;
      v250 = sub_33FC220(a3, 156, v268, v305, v306, v268, v237);
      *(_QWORD *)&v265 = v250;
      *((_QWORD *)&v265 + 1) = v176 | *((_QWORD *)&v265 + 1) & 0xFFFFFFFF00000000LL;
      *((_QWORD *)&v243 + 1) = (unsigned int)v336;
      *(_QWORD *)&v243 = v335;
      *(_QWORD *)&v260 = sub_33FC220(a3, 156, v268, v309, v310, v268, v243);
      *((_QWORD *)&v260 + 1) = v177 | *((_QWORD *)&v260 + 1) & 0xFFFFFFFF00000000LL;
      *((_QWORD *)&v238 + 1) = (unsigned int)v339;
      *(_QWORD *)&v238 = v338;
      v178 = sub_33FC220(a3, 156, v268, v305, v306, v268, v238);
      v38 = v252;
      v37 = (__int64)v247;
      *(_QWORD *)&v259 = v178;
      *((_QWORD *)&v259 + 1) = v179 | *((_QWORD *)&v259 + 1) & 0xFFFFFFFF00000000LL;
    }
    else if ( v36 == 168 )
    {
      v200 = v328->m128i_i64[0];
      v201 = v328->m128i_i64[1];
      *(_QWORD *)&v268 = &v303;
      v249 = sub_3288900((__int64)a3, v305, v306, (int)&v303, v200, v201);
      v254 = v202;
      v298 = sub_3288900((__int64)a3, v305, v306, (int)&v303, *v332, v332[1]);
      v250 = (unsigned __int8 *)v298;
      *(_QWORD *)&v265 = v298;
      v299 = v203;
      *((_QWORD *)&v265 + 1) = (unsigned int)v203 | *((_QWORD *)&v265 + 1) & 0xFFFFFFFF00000000LL;
      v296 = sub_3288900((__int64)a3, v309, v310, (int)&v303, *v335, v335[1]);
      *(_QWORD *)&v260 = v296;
      v297 = v204;
      *((_QWORD *)&v260 + 1) = (unsigned int)v204 | *((_QWORD *)&v260 + 1) & 0xFFFFFFFF00000000LL;
      v205 = sub_3288900((__int64)a3, v305, v306, (int)&v303, *v338, v338[1]);
      v38 = v254;
      v37 = v249;
      v294 = v205;
      *(_QWORD *)&v259 = v205;
      v295 = v206;
      *((_QWORD *)&v259 + 1) = (unsigned int)v206 | *((_QWORD *)&v259 + 1) & 0xFFFFFFFF00000000LL;
    }
    else
    {
      v37 = v328->m128i_i64[0];
      v38 = v328->m128i_u32[2];
      *(_QWORD *)&v265 = *v332;
      v250 = (unsigned __int8 *)*v332;
      *((_QWORD *)&v265 + 1) = *((unsigned int *)v332 + 2) | *((_QWORD *)&v265 + 1) & 0xFFFFFFFF00000000LL;
      v39 = *((unsigned int *)v335 + 2);
      *(_QWORD *)&v260 = *v335;
      *((_QWORD *)&v260 + 1) = v39 | *((_QWORD *)&v260 + 1) & 0xFFFFFFFF00000000LL;
      v40 = *((unsigned int *)v338 + 2);
      *(_QWORD *)&v259 = *v338;
      *((_QWORD *)&v259 + 1) = v40 | *((_QWORD *)&v259 + 1) & 0xFFFFFFFF00000000LL;
    }
    v41 = _mm_load_si128(&v258);
    v42 = v305;
    *(_QWORD *)&v268 = v37;
    *((_QWORD *)&v268 + 1) = v38;
    if ( !(_WORD)v305 || !*(_QWORD *)(a1 + 8LL * (unsigned __int16)v305 + 112) )
    {
      v267 = &v303;
      v292 = sub_33FAF80((__int64)a3, 213, (__int64)&v303, v313, v314, v35, a7);
      *(_QWORD *)&v251 = v292;
      v293 = v91;
      *((_QWORD *)&v251 + 1) = (unsigned int)v91 | v41.m128i_i64[1] & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&v268 = sub_33FAF80((__int64)a3, 213, (__int64)v267, v313, v314, v92, a7);
      v290 = v268;
      v291 = v93;
      *((_QWORD *)&v268 + 1) = (unsigned int)v93 | *((_QWORD *)&v268 + 1) & 0xFFFFFFFF00000000LL;
      v288 = sub_3406EB0(a3, 0x3Au, (__int64)v267, v313, v314, v94, v251, v268);
      v95 = (unsigned int)v269;
      *(_QWORD *)&v268 = v288;
      v289 = v96;
      *(_QWORD *)&v269 = v267;
      *((_QWORD *)&v268 + 1) = (unsigned int)v96 | *((_QWORD *)&v268 + 1) & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&v97 = sub_3400E40((__int64)a3, v95, v313, v314, (__int64)v267, a7);
      v286 = sub_3406EB0(a3, 0xC0u, v269, v313, v314, v98, v268, v97);
      *(_QWORD *)&v268 = v286;
      v287 = v100;
      *((_QWORD *)&v268 + 1) = (unsigned int)v100 | *((_QWORD *)&v268 + 1) & 0xFFFFFFFF00000000LL;
LABEL_82:
      v101 = sub_33FAF80((__int64)a3, 216, v269, (unsigned int)v305, v306, v99, a7);
      v50 = v269;
      v255 = v102;
      v51 = v101;
      v52 = (unsigned int)v102;
      goto LABEL_83;
    }
    v43 = a1 + 500LL * (unsigned __int16)v305;
    if ( (_BYTE)v267 )
    {
      if ( *(_BYTE *)(v43 + 6587) )
      {
        if ( !*(_BYTE *)(v43 + 6477) )
          goto LABEL_32;
LABEL_120:
        if ( (unsigned __int16)(v305 - 17) <= 0xD3u )
        {
          v321 = 0;
          v180 = word_4456580[(unsigned __int16)v305 - 1];
          LOWORD(v320) = v180;
          if ( !v180 )
          {
            LODWORD(v267) = (unsigned __int16)v305;
            v214 = sub_3007260((__int64)&v320);
            v42 = (unsigned __int16)v267;
            v324 = v214;
            LODWORD(v181) = v214;
            v325 = v215;
LABEL_125:
            v182 = 2 * v181;
            switch ( v182 )
            {
              case 2u:
                LOWORD(v182) = 3;
                break;
              case 4u:
                LOWORD(v182) = 4;
                break;
              case 8u:
                LOWORD(v182) = 5;
                break;
              case 0x10u:
                LOWORD(v182) = 6;
                break;
              case 0x20u:
                LOWORD(v182) = 7;
                break;
              case 0x40u:
                LOWORD(v182) = 8;
                break;
              case 0x80u:
                LOWORD(v182) = 9;
                break;
              default:
                v183 = sub_3007020((_QWORD *)a3[8], v182);
                v185 = v184;
                v186 = v183;
                LOWORD(v182) = v183;
                v187 = v185;
                v42 = v305;
                HIWORD(v188) = HIWORD(v183);
                v267 = v187;
                if ( !(_WORD)v305 )
                {
                  v246 = v183;
                  v248 = v183;
                  v256 = v187;
                  v189 = sub_30070B0((__int64)&v305);
                  v187 = v256;
                  LOWORD(v182) = v248;
                  if ( !v189 )
                    goto LABEL_134;
                  v218 = sub_3007240((__int64)&v305);
                  v186 = v246;
                  v219 = (const __m128i *)v218;
                  v220 = HIDWORD(v218);
                  v320 = v219;
                  v221 = v220;
                  goto LABEL_167;
                }
LABEL_151:
                if ( (unsigned __int16)(v42 - 17) > 0xD3u )
                {
                  if ( (_BYTE)v266 )
                    goto LABEL_134;
                  v216 = v42;
                  if ( !*(_QWORD *)(a1 + 8LL * v42 + 112) )
                    goto LABEL_159;
LABEL_154:
                  if ( *(_BYTE *)(a1 + 500LL * (unsigned int)v216 + 6473) != 2 )
                    goto LABEL_134;
LABEL_155:
                  if ( (unsigned __int16)(v42 - 17) <= 0xD3u )
                  {
                    v217 = word_4456580[(int)v216 - 1];
LABEL_157:
                    if ( !v217 )
                      goto LABEL_134;
                    v216 = v217;
                  }
LABEL_159:
                  if ( *(_BYTE *)(a1 + 500 * v216 + 6479) == 4 )
                  {
LABEL_139:
                    LOWORD(v188) = v182;
                    v266 = (unsigned __int8 *)v187;
                    v267 = &v303;
                    v284 = sub_33FAF80((__int64)a3, 213, (__int64)&v303, v188, (__int64)v187, (_DWORD)v187, a7);
                    *(_QWORD *)&v253 = v284;
                    v285 = v192;
                    *((_QWORD *)&v253 + 1) = (unsigned int)v192 | v41.m128i_i64[1] & 0xFFFFFFFF00000000LL;
                    *(_QWORD *)&v268 = sub_33FAF80(
                                         (__int64)a3,
                                         213,
                                         (__int64)v267,
                                         v188,
                                         (__int64)v266,
                                         (_DWORD)v266,
                                         a7);
                    v282 = v268;
                    v283 = v193;
                    *((_QWORD *)&v268 + 1) = (unsigned int)v193 | *((_QWORD *)&v268 + 1) & 0xFFFFFFFF00000000LL;
                    v280 = sub_3406EB0(a3, 0x3Au, (__int64)v267, v188, (__int64)v266, (__int64)v266, v253, v268);
                    *(_QWORD *)&v268 = v280;
                    v281 = v194;
                    v195 = (unsigned int)v269;
                    *(_QWORD *)&v269 = v266;
                    *((_QWORD *)&v268 + 1) = (unsigned int)v194 | *((_QWORD *)&v268 + 1) & 0xFFFFFFFF00000000LL;
                    *(_QWORD *)&v196 = sub_3400E40((__int64)a3, v195, v188, (__int64)v266, (__int64)v267, a7);
                    v197 = v269;
                    v198 = v269;
                    *(_QWORD *)&v269 = v267;
                    v278 = sub_3406EB0(a3, 0xC0u, (__int64)v267, v188, v198, v197, v268, v196);
                    *(_QWORD *)&v268 = v278;
                    v279 = v199;
                    *((_QWORD *)&v268 + 1) = (unsigned int)v199 | *((_QWORD *)&v268 + 1) & 0xFFFFFFFF00000000LL;
                    goto LABEL_82;
                  }
LABEL_134:
                  v190 = 1;
                  if ( (_WORD)v182 != 1 )
                  {
                    if ( !(_WORD)v182 )
                      goto LABEL_97;
                    v190 = (unsigned __int16)v182;
                    if ( !*(_QWORD *)(a1 + 8LL * (unsigned __int16)v182 + 112) )
                      goto LABEL_97;
                  }
                  v191 = *(_BYTE *)(a1 + 500 * v190 + 6472);
                  if ( v191 )
                  {
                    if ( v191 != 4 )
                      goto LABEL_97;
                  }
                  goto LABEL_139;
                }
                v221 = (unsigned __int16)(v42 - 176) <= 0x34u;
                LODWORD(v219) = word_4456340[v42 - 1];
                LOBYTE(v220) = v221;
LABEL_167:
                LODWORD(v319[0]) = (_DWORD)v219;
                v222 = (__int64 *)a3[8];
                BYTE4(v319[0]) = v220;
                v257 = v186;
                if ( v221 )
                  v223 = sub_2D43AD0(v186, (int)v219);
                else
                  v223 = sub_2D43050(v186, (int)v219);
                v182 = v223;
                v187 = 0;
                if ( !(_WORD)v223 )
                {
                  v245 = sub_3009450(v222, v257, (__int64)v267, v319[0], v224, 0);
                  v182 = v245;
                  v187 = v225;
                }
                HIWORD(v188) = HIWORD(v245);
                if ( (_BYTE)v266 )
                  goto LABEL_134;
                v42 = v305;
                if ( !(_WORD)v305 )
                {
                  LODWORD(v266) = v182;
                  v267 = v187;
                  v226 = sub_30070B0((__int64)&v305);
                  v187 = v267;
                  LOWORD(v182) = (_WORD)v266;
                  if ( !v226 )
                    goto LABEL_134;
                  v230 = sub_3009970((__int64)&v305, (unsigned int)v266, v227, v228, v229);
                  LOWORD(v182) = (_WORD)v266;
                  v187 = v267;
                  v217 = v230;
                  goto LABEL_157;
                }
                v216 = (unsigned __int16)v305;
                if ( !*(_QWORD *)(a1 + 8LL * (unsigned __int16)v305 + 112) )
                  goto LABEL_155;
                goto LABEL_154;
            }
            v267 = 0;
            v187 = 0;
            v186 = (unsigned __int16)v182;
            v188 = (unsigned __int16)v182;
            goto LABEL_151;
          }
        }
        else
        {
          LOWORD(v320) = v305;
          v321 = v306;
          v180 = v305;
        }
        if ( v180 != 1 && (unsigned __int16)(v180 - 504) > 7u )
        {
          v181 = *(_QWORD *)&byte_444C4A0[16 * v180 - 16];
          goto LABEL_125;
        }
LABEL_184:
        BUG();
      }
    }
    else if ( (*(_BYTE *)(v43 + 6587) & 0xFB) != 0 )
    {
      if ( (*(_BYTE *)(v43 + 6477) & 0xFB) == 0 )
      {
LABEL_32:
        v44 = (unsigned int *)sub_33E5110(a3, (unsigned int)v305, v306, (unsigned int)v305, v306);
        *(_QWORD *)&v269 = &v303;
        v47 = sub_3411F20(a3, 63, (__int64)&v303, v44, v45, v46, *(_OWORD *)&v258, v268);
        v50 = v269;
        v51 = v47;
        v52 = 1;
        goto LABEL_83;
      }
      goto LABEL_120;
    }
    *(_QWORD *)&v269 = &v303;
    v212 = sub_3406EB0(a3, 0xADu, (__int64)&v303, (unsigned int)v305, v306, v35, *(_OWORD *)&v258, v268);
    v50 = v269;
    v51 = v212;
    v255 = v213;
    v52 = (unsigned int)v213;
LABEL_83:
    v103 = v51;
    v104 = v52 | v255 & 0xFFFFFFFF00000000LL;
    if ( v51 )
    {
      v105 = *(unsigned int *)(a6 + 8);
      if ( v105 + 1 > (unsigned __int64)*(unsigned int *)(a6 + 12) )
      {
        *(_QWORD *)&v269 = v50;
        sub_C8D5F0(a6, (const void *)(a6 + 16), v105 + 1, 8u, v48, v49);
        v105 = *(unsigned int *)(a6 + 8);
        v50 = v269;
      }
      v106 = *(_QWORD *)a6;
      *(_QWORD *)&v269 = v50;
      *(_QWORD *)(v106 + 8 * v105) = v51;
      ++*(_DWORD *)(a6 + 8);
      *((_QWORD *)&v239 + 1) = *((_QWORD *)&v265 + 1);
      *(_QWORD *)&v265 = v250;
      *(_QWORD *)&v239 = v250;
      v107 = sub_3406EB0(a3, 0x3Au, v50, (unsigned int)v305, v306, v49, *(_OWORD *)&v258, v239);
      v110 = v269;
      v276 = v107;
      v111 = v107;
      *(_QWORD *)&v265 = v107;
      v277 = v112;
      v113 = *(unsigned int *)(a6 + 12);
      *((_QWORD *)&v265 + 1) = (unsigned int)v112 | *((_QWORD *)&v265 + 1) & 0xFFFFFFFF00000000LL;
      v114 = *(unsigned int *)(a6 + 8);
      if ( v114 + 1 > v113 )
      {
        sub_C8D5F0(a6, (const void *)(a6 + 16), v114 + 1, 8u, v108, v109);
        v114 = *(unsigned int *)(a6 + 8);
        v110 = v269;
      }
      v115 = *(_QWORD *)a6;
      *(_QWORD *)&v269 = v110;
      *(_QWORD *)(v115 + 8 * v114) = v111;
      v116 = (unsigned int)v305;
      ++*(_DWORD *)(a6 + 8);
      *((_QWORD *)&v233 + 1) = v104;
      *(_QWORD *)&v233 = v103;
      v117 = sub_3406EB0(a3, 0x38u, v110, v116, v306, v109, v233, v265);
      v120 = v269;
      v274 = v117;
      v121 = v117;
      v122 = v117;
      v275 = v123;
      v124 = (unsigned int)v123 | v104 & 0xFFFFFFFF00000000LL;
      v125 = *(unsigned int *)(a6 + 8);
      if ( v125 + 1 > (unsigned __int64)*(unsigned int *)(a6 + 12) )
      {
        sub_C8D5F0(a6, (const void *)(a6 + 16), v125 + 1, 8u, v118, v119);
        v125 = *(unsigned int *)(a6 + 8);
        v120 = v269;
      }
      v126 = *(_QWORD *)a6;
      *(_QWORD *)&v269 = v120;
      *(_QWORD *)(v126 + 8 * v125) = v121;
      v127 = (unsigned int)v305;
      ++*(_DWORD *)(a6 + 8);
      *((_QWORD *)&v234 + 1) = v124;
      *(_QWORD *)&v234 = v122;
      v128 = sub_3406EB0(a3, 0xBFu, v120, v127, v306, v119, v234, v260);
      v131 = v269;
      v272 = v128;
      v132 = v128;
      v133 = v128;
      v273 = v134;
      v135 = (unsigned int)v134 | v124 & 0xFFFFFFFF00000000LL;
      v136 = *(unsigned int *)(a6 + 8);
      if ( v136 + 1 > (unsigned __int64)*(unsigned int *)(a6 + 12) )
      {
        sub_C8D5F0(a6, (const void *)(a6 + 16), v136 + 1, 8u, v129, v130);
        v136 = *(unsigned int *)(a6 + 8);
        v131 = v269;
      }
      v137 = *(_QWORD *)a6;
      *(_QWORD *)&v269 = v131;
      *(_QWORD *)(v137 + 8 * v136) = v132;
      v138 = v262;
      ++*(_DWORD *)(a6 + 8);
      *(_QWORD *)&v139 = sub_3400BD0((__int64)a3, (unsigned int)(v138 - 1), v131, (unsigned int)v309, v310, 0, a7, 0);
      *((_QWORD *)&v231 + 1) = v135;
      *(_QWORD *)&v231 = v133;
      *(_QWORD *)&v268 = v269;
      *(_QWORD *)&v141 = sub_3406EB0(a3, 0xC0u, v269, (unsigned int)v305, v306, v140, v231, v139);
      v144 = *(unsigned int *)(a6 + 12);
      v145 = v268;
      v269 = v141;
      v146 = v141;
      v147 = *(unsigned int *)(a6 + 8);
      if ( v147 + 1 > v144 )
      {
        sub_C8D5F0(a6, (const void *)(a6 + 16), v147 + 1, 8u, v142, v143);
        v147 = *(unsigned int *)(a6 + 8);
        v145 = v268;
      }
      v148 = *(_QWORD *)a6;
      *(_QWORD *)&v268 = v145;
      *(_QWORD *)(v148 + 8 * v147) = v146;
      v149 = (unsigned int)v305;
      ++*(_DWORD *)(a6 + 8);
      v150 = sub_3406EB0(a3, 0xBAu, v145, v149, v306, v143, v269, v259);
      v153 = v268;
      v270 = v150;
      v154 = v150;
      *(_QWORD *)&v269 = v150;
      v271 = v155;
      v156 = *(unsigned int *)(a6 + 12);
      *((_QWORD *)&v269 + 1) = (unsigned int)v155 | *((_QWORD *)&v269 + 1) & 0xFFFFFFFF00000000LL;
      v157 = *(unsigned int *)(a6 + 8);
      if ( v157 + 1 > v156 )
      {
        sub_C8D5F0(a6, (const void *)(a6 + 16), v157 + 1, 8u, v151, v152);
        v157 = *(unsigned int *)(a6 + 8);
        v153 = v268;
      }
      *(_QWORD *)(*(_QWORD *)a6 + 8 * v157) = v154;
      v158 = (unsigned int)v305;
      ++*(_DWORD *)(a6 + 8);
      *((_QWORD *)&v235 + 1) = v135;
      *(_QWORD *)&v235 = v133;
      v53 = sub_3406EB0(a3, 0x38u, v153, v158, v306, v152, v235, v269);
      goto LABEL_98;
    }
LABEL_97:
    v53 = 0;
LABEL_98:
    if ( v338 != v264 )
      _libc_free((unsigned __int64)v338);
    if ( v335 != v263 )
      _libc_free((unsigned __int64)v335);
    if ( v332 != v261 )
      _libc_free((unsigned __int64)v332);
    if ( v328 != (const __m128i *)&v330 )
      _libc_free((unsigned __int64)v328);
    goto LABEL_35;
  }
  v64 = *(unsigned __int16 **)(a2 + 48);
  v65 = _mm_loadu_si128(v26);
  v66 = v26[3].m128i_u32[0];
  LODWORD(v67) = *v64;
  v68 = *((_QWORD *)v64 + 1);
  v266 = (unsigned __int8 *)v26->m128i_i64[0];
  v69 = (__int64 *)v26[2].m128i_i64[1];
  v265 = (__int128)v65;
  LOWORD(v317) = v67;
  v267 = v69;
  v318 = v68;
  if ( (_WORD)v67 )
  {
    if ( (unsigned __int16)(v67 - 17) <= 0xD3u )
    {
      v68 = 0;
      LOWORD(v67) = word_4456580[(unsigned __int16)v67 - 1];
    }
  }
  else
  {
    v264 = (__int64 *)v68;
    LODWORD(v268) = v67;
    *(_QWORD *)&v269 = &v317;
    v162 = sub_30070B0((__int64)&v317);
    LOWORD(v67) = v268;
    v68 = (__int64)v264;
    if ( v162 )
    {
      v164 = sub_3009970((__int64)&v317, (__int64)v69, (unsigned int)v268, (__int64)v264, v163);
      v68 = v67;
      LOWORD(v67) = v164;
    }
  }
  v70 = (__int64 *)a3[5];
  LOWORD(v319[0]) = v67;
  v319[1] = v68;
  v71 = sub_2E79000(v70);
  v72 = (unsigned int)v317;
  v73 = sub_2FE6750(a1, (unsigned int)v317, v318, v71);
  LODWORD(v320) = v73;
  v74 = v73;
  v321 = v75;
  if ( (_WORD)v73 )
  {
    if ( (unsigned __int16)(v73 - 17) <= 0xD3u )
    {
      v74 = word_4456580[(unsigned __int16)v73 - 1];
      v76 = 0;
      goto LABEL_55;
    }
  }
  else
  {
    *(_QWORD *)&v269 = &v320;
    if ( sub_30070B0((__int64)&v320) )
    {
      v74 = sub_3009970((__int64)&v320, v72, v159, v160, v161);
      goto LABEL_55;
    }
  }
  v76 = v321;
LABEL_55:
  LOWORD(v324) = v74;
  v263 = (__int64 *)v337;
  v335 = (__int64 *)v337;
  v325 = v76;
  v302 = 0;
  v336 = 0x1000000000LL;
  v264 = v340;
  v338 = v340;
  v339 = 0x1000000000LL;
  v330 = 0;
  v77 = (__m128i *)sub_22077B0(0x38u);
  if ( v77 )
  {
    v77->m128i_i64[1] = (__int64)&v335;
    v77->m128i_i64[0] = (__int64)&v302;
    v77[2].m128i_i64[0] = (__int64)&v324;
    v77[1].m128i_i64[0] = (__int64)a3;
    v77[1].m128i_i64[1] = (__int64)&v303;
    v77[2].m128i_i64[1] = (__int64)&v338;
    v77[3].m128i_i64[0] = (__int64)v319;
  }
  v328 = v77;
  v331 = sub_34419E0;
  v334[0] = 0;
  v330 = sub_343FCB0;
  sub_343FCB0((unsigned __int64 *)&v332, &v328, 2);
  v334[1] = v331;
  v334[0] = v330;
  v78 = sub_33CA8D0(v267, v66, (__int64)&v332, 0, 0);
  if ( v334[0] )
    ((void (__fastcall *)(__int64 **, __int64 **, __int64))v334[0])(&v332, &v332, 3);
  if ( v330 )
    v330((unsigned __int64 *)&v328, &v328, 3);
  if ( v78 )
  {
    v268 = 0u;
    v269 = 0u;
    v80 = *((_DWORD *)v267 + 6);
    if ( v80 == 156 )
    {
      v267 = &v303;
      *((_QWORD *)&v241 + 1) = (unsigned int)v336;
      *(_QWORD *)&v241 = v335;
      *(_QWORD *)&v268 = sub_33FC220(a3, 156, (__int64)&v303, (__int64)v320, v321, v79, v241);
      *((_QWORD *)&v268 + 1) = v171 | *((_QWORD *)&v268 + 1) & 0xFFFFFFFF00000000LL;
      *((_QWORD *)&v236 + 1) = (unsigned int)v339;
      *(_QWORD *)&v236 = v338;
      v173 = sub_33FC220(a3, 156, (__int64)&v303, v317, v318, v172, v236);
      v82 = &v303;
      *(_QWORD *)&v269 = v173;
      *((_QWORD *)&v269 + 1) = v174 | *((_QWORD *)&v269 + 1) & 0xFFFFFFFF00000000LL;
    }
    else if ( v80 == 168 )
    {
      if ( *(_DWORD *)(*v335 + 24) == 51 )
      {
        v332 = 0;
        LODWORD(v333) = 0;
        v207 = sub_33F17F0(a3, 51, (__int64)&v332, (unsigned int)v320, v321);
        if ( v332 )
        {
          *(_QWORD *)&v260 = &v303;
          v262 = v208;
          v267 = v207;
          sub_B91220((__int64)&v332, (__int64)v332);
          v207 = v267;
          LODWORD(v208) = v262;
        }
      }
      else
      {
        v244 = v335[1];
        v267 = &v303;
        v207 = (__int64 *)sub_33FAF80((__int64)a3, 168, (__int64)&v303, (__int64)v320, v321, v79, v65);
        v209 = v244;
      }
      *(_QWORD *)&v268 = v207;
      *((_QWORD *)&v268 + 1) = (unsigned int)v208 | *((_QWORD *)&v268 + 1) & 0xFFFFFFFF00000000LL;
      if ( *(_DWORD *)(*v338 + 24) == 51 )
      {
        v267 = &v303;
        v332 = 0;
        LODWORD(v333) = 0;
        v210 = sub_33F17F0(a3, 51, (__int64)&v332, v317, v318);
        v82 = &v303;
        if ( v332 )
        {
          *(_QWORD *)&v260 = v267;
          v262 = v211;
          v267 = v210;
          sub_B91220((__int64)&v332, (__int64)v332);
          v210 = v267;
          LODWORD(v211) = v262;
          v82 = (__int64 *)v260;
        }
      }
      else
      {
        v267 = &v303;
        v210 = (__int64 *)sub_33FAF80((__int64)a3, 168, (__int64)&v303, v317, v318, v209, v65);
        v82 = &v303;
      }
      *(_QWORD *)&v269 = v210;
      *((_QWORD *)&v269 + 1) = (unsigned int)v211 | *((_QWORD *)&v269 + 1) & 0xFFFFFFFF00000000LL;
    }
    else
    {
      v81 = *((unsigned int *)v335 + 2);
      v82 = &v303;
      *(_QWORD *)&v268 = *v335;
      *((_QWORD *)&v268 + 1) = v81 | *((_QWORD *)&v268 + 1) & 0xFFFFFFFF00000000LL;
      v83 = *((unsigned int *)v338 + 2);
      *(_QWORD *)&v269 = *v338;
      *((_QWORD *)&v269 + 1) = v83 | *((_QWORD *)&v269 + 1) & 0xFFFFFFFF00000000LL;
    }
    v84 = *((_QWORD *)&v265 + 1);
    if ( v302 )
    {
      v240 = v268;
      *(_QWORD *)&v268 = v82;
      v166 = sub_3405C90(a3, 0xBFu, (__int64)v82, (unsigned int)v317, v318, 4, v65, v265, v240);
      v82 = (__int64 *)v268;
      v300 = v166;
      v168 = v166;
      v266 = v166;
      v301 = v169;
      v84 = (unsigned int)v169 | *((_QWORD *)&v265 + 1) & 0xFFFFFFFF00000000LL;
      v170 = *(unsigned int *)(a6 + 8);
      if ( v170 + 1 > (unsigned __int64)*(unsigned int *)(a6 + 12) )
      {
        sub_C8D5F0(a6, (const void *)(a6 + 16), v170 + 1, 8u, v167, v79);
        v170 = *(unsigned int *)(a6 + 8);
        v82 = (__int64 *)v268;
      }
      *(_QWORD *)(*(_QWORD *)a6 + 8 * v170) = v168;
      ++*(_DWORD *)(a6 + 8);
    }
    *((_QWORD *)&v232 + 1) = v84;
    *(_QWORD *)&v232 = v266;
    v85 = sub_3406EB0(a3, 0x3Au, (__int64)v82, (unsigned int)v317, v318, v79, v232, v269);
    *(_QWORD *)&v260 = v86;
    v87 = v85;
  }
  else
  {
    v87 = 0;
  }
  if ( v338 != v264 )
    _libc_free((unsigned __int64)v338);
  if ( v335 != v263 )
    _libc_free((unsigned __int64)v335);
  v53 = v87;
LABEL_35:
  if ( v303 )
    sub_B91220((__int64)&v303, v303);
  return v53;
}
