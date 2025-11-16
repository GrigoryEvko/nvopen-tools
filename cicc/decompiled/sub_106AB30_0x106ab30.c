// Function: sub_106AB30
// Address: 0x106ab30
//
__int64 *__fastcall sub_106AB30(
        __int64 *a1,
        __int64 a2,
        __int64 **a3,
        char **a4,
        __int64 a5,
        const __m128i *a6,
        char a7,
        char a8)
{
  __int64 v10; // rax
  __int64 *v12; // rsi
  __int64 **v13; // rdx
  __int64 v14; // rcx
  _QWORD *v15; // rax
  _QWORD *v16; // rdx
  char v17; // cl
  _QWORD *v18; // rax
  _QWORD *v19; // rdx
  char v20; // cl
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  unsigned int v24; // edx
  _QWORD *v25; // r12
  _QWORD *v26; // rbx
  __int64 v27; // rsi
  __int64 v28; // rdx
  unsigned int v29; // eax
  __int64 v30; // rdx
  char **v31; // r14
  char **v32; // r12
  char *v33; // rsi
  __int64 v34; // rbx
  __m128i *v35; // rdi
  __int64 *v36; // rsi
  __int64 v37; // r13
  __int64 v38; // rax
  _QWORD *v39; // rbx
  _QWORD *v40; // r14
  __int64 v41; // rsi
  _QWORD *v42; // rsi
  __int64 v43; // rax
  unsigned int v44; // eax
  _QWORD *v45; // r12
  _QWORD *v46; // r13
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rax
  unsigned int v50; // eax
  _QWORD *v51; // r12
  _QWORD *v52; // r13
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rax
  __int64 v56; // rsi
  __int64 v57; // rsi
  __int64 v58; // rcx
  unsigned __int64 v59; // rdx
  __int64 v60; // rbx
  __m128i *v61; // rdi
  __int64 *v62; // r12
  __int64 *v64; // rax
  __int64 **v65; // r14
  __int64 v66; // rdx
  _BYTE *v67; // rsi
  __int64 v68; // r15
  __int64 v69; // rax
  int v70; // edx
  bool v71; // r15
  __int64 v72; // rax
  __int64 v73; // rdx
  __int64 **v74; // rcx
  __m128i *v75; // rdi
  size_t v76; // r8
  __int64 *v77; // rsi
  __int64 *v78; // rdx
  __int64 *v79; // rsi
  __int64 *v80; // r14
  __int64 *v81; // rbx
  __int64 v82; // r15
  const char *v83; // rax
  unsigned __int64 v84; // rdx
  __int64 v85; // rax
  __int64 v86; // rdx
  __int64 v87; // r8
  __int64 v88; // r9
  __int64 v89; // r15
  __int64 v90; // rcx
  __int64 *v91; // r14
  __int64 *v92; // rbx
  __int64 v93; // r15
  const char *v94; // rax
  unsigned __int64 v95; // rdx
  __int64 v96; // rax
  __int64 v97; // rdx
  __int64 v98; // rcx
  __int64 v99; // r8
  __int64 v100; // r9
  __int64 v101; // r15
  __int64 v102; // rsi
  __int64 v103; // rdx
  __int64 *v104; // r14
  __int64 *v105; // rbx
  __int64 v106; // r15
  const char *v107; // rax
  unsigned __int64 v108; // rdx
  __int64 v109; // rax
  __int64 v110; // rdx
  __int64 v111; // rcx
  __int64 v112; // r8
  __int64 v113; // r9
  __int64 v114; // r15
  __int64 v115; // rdx
  __int64 v116; // rcx
  __int64 v117; // r8
  __int64 v118; // r9
  size_t v119; // rbx
  __m128i *j; // r14
  __int64 **v121; // r12
  __int64 v122; // r15
  unsigned __int64 v123; // rdx
  unsigned __int64 v124; // rcx
  __int64 v125; // r13
  unsigned __int64 k; // rax
  unsigned __int64 v127; // rdx
  __int64 v128; // rax
  __int64 v129; // r15
  __int64 v130; // r8
  __int64 v131; // r9
  unsigned __int64 v132; // rax
  __int64 v133; // rdx
  __int64 v134; // rcx
  __int64 *v135; // rax
  __int64 *v136; // rdx
  __int64 v137; // rsi
  unsigned int v138; // esi
  __int64 v139; // r10
  unsigned int v140; // esi
  __int64 v141; // r10
  __int64 v142; // rbx
  __int64 v143; // r14
  _QWORD *v144; // r13
  __int64 v145; // rsi
  unsigned __int64 v146; // rdx
  __int64 v147; // r8
  __int64 v148; // r9
  __int64 v149; // rbx
  __int64 v150; // r13
  _QWORD *v151; // rax
  size_t v152; // rdx
  __int64 v153; // r13
  _QWORD *v154; // rax
  __int64 v155; // rdx
  int v156; // r14d
  unsigned int m; // r15d
  _QWORD *v158; // rax
  __int64 v159; // rax
  __int64 v160; // rdx
  __int64 v161; // rcx
  __int64 v162; // r8
  __int64 v163; // r9
  __int64 v164; // rsi
  unsigned __int8 v165; // al
  __int64 *v166; // r15
  __int64 v167; // rdi
  __int64 v168; // rdx
  unsigned int v169; // eax
  __int64 v170; // rdx
  unsigned int v171; // r11d
  int v172; // r14d
  unsigned int v173; // r15d
  unsigned int v174; // esi
  __int64 v175; // rax
  __int64 v176; // rdx
  __int64 v177; // rcx
  __int64 v178; // r8
  __int64 v179; // r9
  __int64 v180; // rax
  __int64 *v181; // rbx
  unsigned __int64 v182; // rcx
  __int64 *v183; // r13
  __int64 *v184; // rdi
  __int64 v185; // rax
  __int64 *v186; // r13
  __int64 *v187; // rbx
  unsigned __int8 *v188; // rdi
  unsigned __int8 *v189; // rax
  __int64 v190; // r14
  __int64 v191; // r15
  __int64 v192; // rcx
  __int64 *v193; // r14
  __int64 *v194; // rbx
  unsigned __int8 *v195; // rdi
  unsigned __int8 *v196; // rax
  __int64 v197; // r13
  __int64 v198; // r15
  __int64 v199; // rcx
  char v200; // al
  __int64 *jj; // r14
  __int64 v202; // rdi
  __int64 v203; // rax
  __int64 v204; // rdx
  __int64 v205; // rax
  __int64 v206; // rdx
  __int64 v207; // rax
  __int64 v208; // rbx
  int v209; // eax
  unsigned int v210; // r12d
  int v211; // r14d
  unsigned int v212; // esi
  __m128i *v213; // r15
  unsigned int v214; // eax
  _QWORD *v215; // r12
  _QWORD *v216; // rbx
  __int64 v217; // rsi
  unsigned int v218; // eax
  _QWORD *v219; // r12
  _QWORD *v220; // rbx
  __int64 v221; // rsi
  __int64 v222; // r13
  __int64 v223; // rax
  __int64 v224; // rdx
  __int64 v225; // r13
  __int64 v226; // rcx
  __m128i *v227; // rax
  __int64 v228; // rcx
  __m128i *v229; // rax
  __int64 v230; // rcx
  __m128i *v231; // rax
  __int64 v232; // r15
  __int64 *v233; // r15
  __int64 v234; // rdi
  _BYTE *v235; // rsi
  __m128i *v236; // rdi
  size_t v237; // rcx
  __int64 *v238; // rdx
  __int64 *v239; // rsi
  __int64 v240; // rcx
  __m128i *v241; // rax
  __int64 v242; // rcx
  __m128i *v243; // rax
  __int64 v244; // rcx
  __m128i *v245; // rax
  __int64 **v246; // r15
  __m128i *v247; // rax
  __int64 v248; // rsi
  __m128i *v249; // rax
  __int64 v250; // rcx
  __m128i *v251; // rax
  __int64 v252; // rcx
  __m128i *v253; // rax
  __int64 v254; // rcx
  __m128i *v255; // rax
  __int64 v256; // rcx
  __m128i *v257; // rax
  __int64 v258; // rcx
  __m128i *v259; // rax
  __int64 *ii; // r14
  __int64 v261; // rdi
  size_t v262; // rdx
  unsigned int v263; // r11d
  __int64 v264; // r14
  __int64 **v265; // rbx
  __int64 v266; // rcx
  __int64 *v267; // r13
  __int64 **v268; // rax
  unsigned __int64 v269; // r14
  __int64 *v270; // rax
  __int64 v271; // rax
  __int64 v272; // rdx
  __int64 v273; // rax
  __int64 v274; // rdx
  __int64 v275; // rax
  size_t v276; // rdx
  __int64 v277; // [rsp+8h] [rbp-718h]
  __int64 v278; // [rsp+10h] [rbp-710h]
  char v279; // [rsp+47h] [rbp-6D9h]
  __int64 *v280; // [rsp+48h] [rbp-6D8h]
  __int64 *v281; // [rsp+78h] [rbp-6A8h]
  __int64 *v283; // [rsp+90h] [rbp-690h]
  __int64 *v284; // [rsp+98h] [rbp-688h]
  __int64 **v285; // [rsp+A0h] [rbp-680h]
  char **v286; // [rsp+A0h] [rbp-680h]
  __int64 *v287; // [rsp+A0h] [rbp-680h]
  __int64 **v288; // [rsp+A0h] [rbp-680h]
  _QWORD *v289; // [rsp+A0h] [rbp-680h]
  _QWORD *v290; // [rsp+A0h] [rbp-680h]
  _QWORD *v291; // [rsp+A0h] [rbp-680h]
  __int64 v292; // [rsp+A0h] [rbp-680h]
  __int64 **v293; // [rsp+A0h] [rbp-680h]
  __int64 **v294; // [rsp+A0h] [rbp-680h]
  __int64 *v295; // [rsp+A0h] [rbp-680h]
  __int64 **v296; // [rsp+A0h] [rbp-680h]
  bool v297; // [rsp+A0h] [rbp-680h]
  _QWORD v299[2]; // [rsp+B0h] [rbp-670h] BYREF
  __int64 v300; // [rsp+C0h] [rbp-660h] BYREF
  _QWORD v301[2]; // [rsp+D0h] [rbp-650h] BYREF
  __m128i v302; // [rsp+E0h] [rbp-640h] BYREF
  __m128i *v303; // [rsp+F0h] [rbp-630h] BYREF
  __int64 v304; // [rsp+F8h] [rbp-628h]
  __m128i v305; // [rsp+100h] [rbp-620h] BYREF
  __m128i *v306; // [rsp+110h] [rbp-610h] BYREF
  __int64 v307; // [rsp+118h] [rbp-608h]
  __m128i v308; // [rsp+120h] [rbp-600h] BYREF
  __m128i v309; // [rsp+130h] [rbp-5F0h] BYREF
  __m128i v310; // [rsp+140h] [rbp-5E0h] BYREF
  __m128i *v311; // [rsp+150h] [rbp-5D0h] BYREF
  __int64 v312; // [rsp+158h] [rbp-5C8h]
  __m128i v313; // [rsp+160h] [rbp-5C0h] BYREF
  __m128i v314; // [rsp+170h] [rbp-5B0h] BYREF
  __m128i v315; // [rsp+180h] [rbp-5A0h] BYREF
  __m128i *v316; // [rsp+190h] [rbp-590h] BYREF
  __int64 v317; // [rsp+198h] [rbp-588h]
  __m128i v318; // [rsp+1A0h] [rbp-580h] BYREF
  __m128i v319; // [rsp+1B0h] [rbp-570h] BYREF
  __m128i v320; // [rsp+1C0h] [rbp-560h] BYREF
  __m128i *v321; // [rsp+1D0h] [rbp-550h] BYREF
  __int64 v322; // [rsp+1D8h] [rbp-548h]
  __m128i v323; // [rsp+1E0h] [rbp-540h] BYREF
  __m128i v324; // [rsp+1F0h] [rbp-530h] BYREF
  __m128i v325; // [rsp+200h] [rbp-520h] BYREF
  __int16 v326; // [rsp+210h] [rbp-510h]
  __int64 v327[2]; // [rsp+220h] [rbp-500h] BYREF
  _QWORD v328[2]; // [rsp+230h] [rbp-4F0h] BYREF
  __int64 v329; // [rsp+240h] [rbp-4E0h]
  __int64 v330; // [rsp+248h] [rbp-4D8h]
  __int64 v331; // [rsp+250h] [rbp-4D0h]
  __m128i v332; // [rsp+260h] [rbp-4C0h] BYREF
  __int64 v333; // [rsp+270h] [rbp-4B0h] BYREF
  __int64 v334; // [rsp+278h] [rbp-4A8h]
  __int64 v335; // [rsp+280h] [rbp-4A0h]
  __int64 v336; // [rsp+288h] [rbp-498h]
  __int64 v337; // [rsp+290h] [rbp-490h]
  __m128i *v338; // [rsp+2A0h] [rbp-480h] BYREF
  size_t n; // [rsp+2A8h] [rbp-478h] BYREF
  __int64 *v340; // [rsp+2B0h] [rbp-470h] BYREF
  __int64 v341; // [rsp+2B8h] [rbp-468h]
  __int64 i; // [rsp+2C0h] [rbp-460h]
  __int64 v343; // [rsp+2C8h] [rbp-458h]
  __int64 v344; // [rsp+2D0h] [rbp-450h]
  __int64 **v345; // [rsp+2E0h] [rbp-440h] BYREF
  __int64 *v346; // [rsp+2E8h] [rbp-438h]
  __m128i v347; // [rsp+2F0h] [rbp-430h] BYREF
  __int64 v348; // [rsp+300h] [rbp-420h]
  __int64 v349; // [rsp+308h] [rbp-418h]
  _QWORD v350[2]; // [rsp+310h] [rbp-410h] BYREF
  __int64 v351; // [rsp+320h] [rbp-400h]
  __int64 v352; // [rsp+328h] [rbp-3F8h]
  unsigned int v353; // [rsp+330h] [rbp-3F0h]
  _BYTE *v354; // [rsp+338h] [rbp-3E8h]
  __int64 v355; // [rsp+340h] [rbp-3E0h]
  _BYTE v356[128]; // [rsp+348h] [rbp-3D8h] BYREF
  _BYTE *v357; // [rsp+3C8h] [rbp-358h]
  __int64 v358; // [rsp+3D0h] [rbp-350h]
  _BYTE v359[128]; // [rsp+3D8h] [rbp-348h] BYREF
  _BYTE *v360; // [rsp+458h] [rbp-2C8h]
  __int64 v361; // [rsp+460h] [rbp-2C0h]
  _BYTE v362[128]; // [rsp+468h] [rbp-2B8h] BYREF
  __int64 v363; // [rsp+4E8h] [rbp-238h]
  char *v364; // [rsp+4F0h] [rbp-230h]
  __int64 v365; // [rsp+4F8h] [rbp-228h]
  int v366; // [rsp+500h] [rbp-220h]
  char v367; // [rsp+504h] [rbp-21Ch]
  char v368; // [rsp+508h] [rbp-218h] BYREF
  __int64 v369; // [rsp+588h] [rbp-198h]
  _QWORD v370[2]; // [rsp+590h] [rbp-190h] BYREF
  _QWORD v371[2]; // [rsp+5A0h] [rbp-180h] BYREF
  __int64 v372; // [rsp+5B0h] [rbp-170h]
  __int64 v373; // [rsp+5B8h] [rbp-168h] BYREF
  _QWORD *v374; // [rsp+5C0h] [rbp-160h]
  __int64 v375; // [rsp+5C8h] [rbp-158h]
  unsigned int v376; // [rsp+5D0h] [rbp-150h]
  __int64 v377; // [rsp+5D8h] [rbp-148h]
  _QWORD *v378; // [rsp+5E0h] [rbp-140h]
  __int64 v379; // [rsp+5E8h] [rbp-138h]
  unsigned int v380; // [rsp+5F0h] [rbp-130h]
  char v381; // [rsp+5F8h] [rbp-128h]
  __int64 v382; // [rsp+608h] [rbp-118h] BYREF
  _QWORD *v383; // [rsp+610h] [rbp-110h]
  __int64 v384; // [rsp+618h] [rbp-108h]
  unsigned int v385; // [rsp+620h] [rbp-100h]
  _QWORD *v386; // [rsp+630h] [rbp-F0h]
  unsigned int v387; // [rsp+640h] [rbp-E0h]
  char v388; // [rsp+648h] [rbp-D8h]
  __int64 v389; // [rsp+658h] [rbp-C8h]
  __int64 v390; // [rsp+660h] [rbp-C0h]
  __int64 v391; // [rsp+668h] [rbp-B8h]
  __int64 v392; // [rsp+670h] [rbp-B0h]
  unsigned __int64 v393; // [rsp+678h] [rbp-A8h]
  __int64 *v394; // [rsp+680h] [rbp-A0h]
  __int64 v395; // [rsp+688h] [rbp-98h]
  __int64 v396; // [rsp+690h] [rbp-90h]
  __int64 v397; // [rsp+698h] [rbp-88h]
  __int64 v398; // [rsp+6A0h] [rbp-80h]
  __int64 v399; // [rsp+6A8h] [rbp-78h]
  __int64 *v400; // [rsp+6B0h] [rbp-70h]
  __int64 v401; // [rsp+6B8h] [rbp-68h]
  __int64 v402; // [rsp+6C0h] [rbp-60h]
  char v403; // [rsp+6C8h] [rbp-58h]
  char v404; // [rsp+6C9h] [rbp-57h]
  char v405; // [rsp+6CAh] [rbp-56h]
  __int64 v406; // [rsp+6D0h] [rbp-50h] BYREF
  char v407; // [rsp+6D8h] [rbp-48h]
  _DWORD *v408; // [rsp+6E0h] [rbp-40h] BYREF
  int v409; // [rsp+6E8h] [rbp-38h]

  v10 = a6[1].m128i_i64[1];
  v334 = v10;
  if ( (v10 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    if ( (v10 & 2) != 0 && (v10 & 4) != 0 )
    {
      v285 = a3;
      (*(void (__fastcall **)(__m128i *, const __m128i *))((v10 & 0xFFFFFFFFFFFFFFF8LL) + 8))(&v332, a6);
      (*(void (__fastcall **)(const __m128i *))((v334 & 0xFFFFFFFFFFFFFFF8LL) + 16))(a6);
      v10 = v334;
      a3 = v285;
    }
    else
    {
      v332 = _mm_loadu_si128(a6);
      v333 = a6[1].m128i_i64[0];
    }
    a6[1].m128i_i64[1] = 0;
  }
  v12 = *a3;
  *a3 = 0;
  v13 = *(__int64 ***)a2;
  v346 = v12;
  v14 = a2 + 8;
  v345 = v13;
  v349 = v334;
  if ( (v10 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    if ( (v334 & 2) != 0 && (v334 & 4) != 0 )
    {
      (*(void (__fastcall **)(__m128i *, __m128i *))((v334 & 0xFFFFFFFFFFFFFFF8LL) + 8))(&v347, &v332);
      (*(void (__fastcall **)(__m128i *))((v349 & 0xFFFFFFFFFFFFFFF8LL) + 16))(&v332);
      v14 = a2 + 8;
    }
    else
    {
      v347 = _mm_load_si128(&v332);
      v348 = v333;
    }
    v334 = 0;
  }
  v369 = v14;
  v350[0] = off_49E5E78;
  v354 = v356;
  v355 = 0x1000000000LL;
  v358 = 0x1000000000LL;
  v361 = 0x1000000000LL;
  v364 = &v368;
  v357 = v359;
  v370[0] = off_49E5ED8;
  v370[1] = &v345;
  v371[1] = &v345;
  v360 = v362;
  v350[1] = 0;
  v351 = 0;
  v352 = 0;
  v353 = 0;
  v363 = 0;
  v365 = 16;
  v366 = 0;
  v367 = 1;
  v371[0] = off_49E5EF8;
  v372 = a2 + 72;
  v373 = 0;
  v376 = 128;
  v15 = (_QWORD *)sub_C7D670(0x2000, 8);
  v375 = 0;
  v374 = v15;
  n = 2;
  v16 = &v15[8 * (unsigned __int64)v376];
  v338 = (__m128i *)&unk_49DD7B0;
  v340 = 0;
  v341 = -4096;
  for ( i = 0; v16 != v15; v15 += 8 )
  {
    if ( v15 )
    {
      v17 = n;
      v15[2] = 0;
      v15[3] = -4096;
      *v15 = &unk_49DD7B0;
      v15[1] = v17 & 6;
      v15[4] = i;
    }
  }
  v381 = 0;
  v382 = 0;
  v385 = 128;
  v18 = (_QWORD *)sub_C7D670(0x2000, 8);
  v384 = 0;
  v383 = v18;
  n = 2;
  v19 = &v18[8 * (unsigned __int64)v385];
  v338 = (__m128i *)&unk_49DD7B0;
  v340 = 0;
  v341 = -4096;
  for ( i = 0; v19 != v18; v18 += 8 )
  {
    if ( v18 )
    {
      v20 = n;
      v18[2] = 0;
      v18[3] = -4096;
      *v18 = &unk_49DD7B0;
      v18[1] = v20 & 6;
      v18[4] = i;
    }
  }
  v404 = a8;
  v403 = a7;
  v405 = 0;
  v388 = 0;
  v389 = 0;
  v390 = 0;
  v391 = 0;
  v392 = 0;
  v393 = 0;
  v394 = 0;
  v395 = 0;
  v396 = 0;
  v397 = 0;
  v398 = 0;
  v399 = 0;
  v400 = 0;
  v401 = 0;
  v402 = 0;
  v407 = 0;
  sub_FC75A0((__int64 *)&v408, (__int64)&v373, 6, (__int64)v350, (__int64)v370, 0);
  v409 = sub_FC7760((__int64 *)&v408, (__int64)&v382, (__int64)v371, v21, v22, v23);
  if ( v381 )
  {
    v24 = v380;
    if ( v380 )
    {
      v286 = a4;
      v25 = v378;
      v26 = &v378[2 * v380];
      do
      {
        if ( *v25 != -8192 && *v25 != -4096 )
        {
          v27 = v25[1];
          if ( v27 )
            sub_B91220((__int64)(v25 + 1), v27);
        }
        v25 += 2;
      }
      while ( v26 != v25 );
      a4 = v286;
      v24 = v380;
    }
    sub_C7D6A0((__int64)v378, 16LL * v24, 8);
    ++v377;
    v28 = *(_QWORD *)(a2 + 80);
    v29 = *(_DWORD *)(a2 + 96);
    *(_QWORD *)(a2 + 80) = 0;
    ++*(_QWORD *)(a2 + 72);
    v378 = (_QWORD *)v28;
    v30 = *(_QWORD *)(a2 + 88);
    v380 = v29;
    v379 = v30;
    *(_QWORD *)(a2 + 88) = 0;
    *(_DWORD *)(a2 + 96) = 0;
  }
  else
  {
    v381 = 1;
    v377 = 1;
    v168 = *(_QWORD *)(a2 + 80);
    v169 = *(_DWORD *)(a2 + 96);
    *(_QWORD *)(a2 + 80) = 0;
    ++*(_QWORD *)(a2 + 72);
    v378 = (_QWORD *)v168;
    v170 = *(_QWORD *)(a2 + 88);
    v380 = v169;
    v379 = v170;
    *(_QWORD *)(a2 + 88) = 0;
    *(_DWORD *)(a2 + 96) = 0;
  }
  v31 = &a4[a5];
  if ( v31 != a4 )
  {
    v32 = a4;
    do
    {
      v33 = *v32++;
      sub_10634E0((__int64)&v345, v33);
    }
    while ( v31 != v32 );
  }
  if ( a7 )
  {
    v207 = sub_BA8DC0((__int64)v346, (__int64)"llvm.dbg.cu", 11);
    v208 = v207;
    if ( v207 )
    {
      v209 = sub_B91A00(v207);
      if ( v209 )
      {
        v210 = 0;
        v211 = v209;
        do
        {
          v212 = v210++;
          v213 = (__m128i *)sub_B91A10(v208, v212);
          sub_BA6610(v213, 4u, 0);
          sub_BA6610(v213, 8u, 0);
          sub_BA6610(v213, 5u, 0);
          sub_BA6610(v213, 6u, 0);
          sub_BA6610(v213, 7u, 0);
        }
        while ( v211 != v210 );
      }
    }
  }
  if ( (v334 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    v34 = (v334 >> 1) & 1;
    if ( (v334 & 4) != 0 )
    {
      v35 = &v332;
      if ( !(_BYTE)v34 )
        v35 = (__m128i *)v332.m128i_i64[0];
      (*(void (__fastcall **)(__m128i *))((v334 & 0xFFFFFFFFFFFFFFF8LL) + 16))(v35);
    }
    if ( !(_BYTE)v34 )
      sub_C7D6A0(v332.m128i_i64[0], v332.m128i_i64[1], v333);
  }
  v36 = (__int64 *)v346[20];
  v283 = v346;
  if ( v36 )
  {
    (*(void (__fastcall **)(__m128i **))(*v36 + 32))(&v338);
    if ( ((unsigned __int64)v338 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      *a1 = (unsigned __int64)v338 & 0xFFFFFFFFFFFFFFFELL | 1;
      goto LABEL_43;
    }
    v283 = v346;
  }
  v64 = v283;
  v65 = v345;
  v279 = *((_BYTE *)v283 + 872);
  if ( *((_BYTE *)v345 + 872) )
  {
    if ( !*((_BYTE *)v283 + 872) )
    {
      v233 = (__int64 *)v283[4];
      if ( v233 != v283 + 3 )
      {
        do
        {
          v234 = (__int64)(v233 - 7);
          if ( !v233 )
            v234 = 0;
          sub_B2B950(v234);
          v233 = (__int64 *)v233[1];
        }
        while ( v283 + 3 != v233 );
        v65 = v345;
        v64 = v346;
      }
      *((_BYTE *)v283 + 872) = 1;
    }
  }
  else
  {
    v64 = v283;
    if ( v279 )
    {
      v166 = (__int64 *)v283[4];
      if ( v166 != v283 + 3 )
      {
        do
        {
          v167 = (__int64)(v166 - 7);
          if ( !v166 )
            v167 = 0;
          sub_B2B9A0(v167);
          v166 = (__int64 *)v166[1];
        }
        while ( v283 + 3 != v166 );
        v65 = v345;
        v64 = v346;
      }
      *((_BYTE *)v283 + 872) = 0;
    }
  }
  if ( !v65[96] )
  {
    sub_BA9570((__int64)v65, (__int64)(v64 + 39));
    v65 = v345;
    v64 = v346;
  }
  v66 = v64[30];
  if ( !v65[30] && v66 )
  {
    v235 = (_BYTE *)v64[29];
    v295 = v64;
    v338 = (__m128i *)&v340;
    sub_10614E0((__int64 *)&v338, v235, (__int64)&v235[v66]);
    i = v295[33];
    v343 = v295[34];
    v344 = v295[35];
    v236 = (__m128i *)v65[29];
    if ( v338 == (__m128i *)&v340 )
    {
      v276 = n;
      if ( n )
      {
        if ( n == 1 )
          v236->m128i_i8[0] = (char)v340;
        else
          memcpy(v236, &v340, n);
        v276 = n;
        v236 = (__m128i *)v65[29];
      }
      v65[30] = (__int64 *)v276;
      v236->m128i_i8[v276] = 0;
      v236 = v338;
      goto LABEL_357;
    }
    v237 = n;
    v238 = v340;
    if ( v236 == (__m128i *)(v65 + 31) )
    {
      v65[29] = (__int64 *)v338;
      v65[30] = (__int64 *)v237;
      v65[31] = v238;
    }
    else
    {
      v239 = v65[31];
      v65[29] = (__int64 *)v338;
      v65[30] = (__int64 *)v237;
      v65[31] = v238;
      if ( v236 )
      {
        v338 = v236;
        v340 = v239;
LABEL_357:
        n = 0;
        v236->m128i_i8[0] = 0;
        v65[33] = (__int64 *)i;
        v65[34] = (__int64 *)v343;
        v65[35] = (__int64 *)v344;
        if ( v338 != (__m128i *)&v340 )
          j_j___libc_free_0(v338, (char *)v340 + 1);
        v64 = v346;
        v66 = v346[30];
        goto LABEL_123;
      }
    }
    v338 = (__m128i *)&v340;
    v236 = (__m128i *)&v340;
    goto LABEL_357;
  }
LABEL_123:
  v67 = (_BYTE *)v64[29];
  v287 = v64;
  v327[0] = (__int64)v328;
  sub_10614E0(v327, v67, (__int64)&v67[v66]);
  v68 = (__int64)v345;
  v329 = v287[33];
  v69 = v287[35];
  v330 = v287[34];
  v331 = v69;
  v332.m128i_i64[0] = (__int64)&v333;
  sub_10614E0(v332.m128i_i64, v345[29], (__int64)v345[30] + (_QWORD)v345[29]);
  v70 = *(_DWORD *)(v68 + 264);
  v335 = *(_QWORD *)(v68 + 264);
  v336 = *(_QWORD *)(v68 + 272);
  v337 = *(_QWORD *)(v68 + 280);
  if ( (unsigned int)(v329 - 42) > 1 || (unsigned int)(v70 - 42) > 1 )
    goto LABEL_496;
  v71 = 1;
  if ( v346[96] )
    v71 = (unsigned int)sub_2241AC0(v346 + 95, "e-i64:64-v16:16-v32:32-n16:32:64") == 0;
  if ( (_DWORD)v330 == 8
    && (v271 = sub_CC7380(v327), v272 == 7)
    && *(_DWORD *)v271 == 1819635815
    && *(_WORD *)(v271 + 4) == 25193
    && *(_BYTE *)(v271 + 6) == 115
    || (v72 = sub_CC72D0(v327), v73 == 7)
    && *(_DWORD *)v72 == 1852534389
    && *(_WORD *)(v72 + 4) == 30575
    && *(_BYTE *)(v72 + 6) == 110
    && (v273 = sub_CC7380(v327), v274 == 7)
    && *(_DWORD *)v273 == 1852534389
    && *(_WORD *)(v273 + 4) == 30575
    && *(_BYTE *)(v273 + 6) == 110 )
  {
    if ( v71 )
      goto LABEL_131;
    v297 = sub_AE4640((__int64)(v346 + 39), (__int64)(v345 + 39));
    if ( v297 )
      goto LABEL_131;
  }
  else
  {
LABEL_496:
    if ( sub_AE4640((__int64)(v346 + 39), (__int64)(v345 + 39)) )
    {
LABEL_130:
      if ( v346[30] && !(unsigned __int8)sub_CC7FA0(v327, &v332) )
      {
        v296 = v345;
        v280 = v346;
        sub_8FD6D0((__int64)&v306, "Linking two modules of different target triples: '", v346 + 21);
        sub_94F930(&v309, (__int64)&v306, "' is '");
        v241 = (__m128i *)sub_2241490(&v309, v280[29], v280[30], v240);
        v311 = &v313;
        if ( (__m128i *)v241->m128i_i64[0] == &v241[1] )
        {
          v313 = _mm_loadu_si128(v241 + 1);
        }
        else
        {
          v311 = (__m128i *)v241->m128i_i64[0];
          v313.m128i_i64[0] = v241[1].m128i_i64[0];
        }
        v312 = v241->m128i_i64[1];
        v241->m128i_i64[0] = (__int64)v241[1].m128i_i64;
        v241->m128i_i64[1] = 0;
        v241[1].m128i_i8[0] = 0;
        sub_94F930(&v314, (__int64)&v311, "' whereas '");
        v243 = (__m128i *)sub_2241490(&v314, v296[21], v296[22], v242);
        v316 = &v318;
        if ( (__m128i *)v243->m128i_i64[0] == &v243[1] )
        {
          v318 = _mm_loadu_si128(v243 + 1);
        }
        else
        {
          v316 = (__m128i *)v243->m128i_i64[0];
          v318.m128i_i64[0] = v243[1].m128i_i64[0];
        }
        v317 = v243->m128i_i64[1];
        v243->m128i_i64[0] = (__int64)v243[1].m128i_i64;
        v243->m128i_i64[1] = 0;
        v243[1].m128i_i8[0] = 0;
        sub_94F930(&v319, (__int64)&v316, "' is '");
        v245 = (__m128i *)sub_2241490(&v319, v296[29], v296[30], v244);
        v321 = &v323;
        if ( (__m128i *)v245->m128i_i64[0] == &v245[1] )
        {
          v323 = _mm_loadu_si128(v245 + 1);
        }
        else
        {
          v321 = (__m128i *)v245->m128i_i64[0];
          v323.m128i_i64[0] = v245[1].m128i_i64[0];
        }
        v322 = v245->m128i_i64[1];
        v245->m128i_i64[0] = (__int64)v245[1].m128i_i64;
        v245->m128i_i64[1] = 0;
        v245[1].m128i_i8[0] = 0;
        sub_94F930(&v324, (__int64)&v321, "'\n");
        LOWORD(i) = 260;
        v338 = &v324;
        v278 = *v346;
        sub_1061A30((__int64)&v303, 1, (__int64)&v338);
        sub_B6EB20(v278, (__int64)&v303);
        sub_2240A30(&v324);
        sub_2240A30(&v321);
        sub_2240A30(&v319);
        sub_2240A30(&v316);
        sub_2240A30(&v314);
        sub_2240A30(&v311);
        sub_2240A30(&v309);
        sub_2240A30(&v306);
      }
      goto LABEL_131;
    }
    v297 = 1;
  }
  v246 = v345;
  v281 = v346;
  sub_8FD6D0((__int64)v299, "Linking two modules of different data layouts: '", v346 + 21);
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v299[1]) <= 5 )
    goto LABEL_489;
  v247 = (__m128i *)sub_2241490(v299, "' is '", 6, v281);
  v301[0] = &v302;
  if ( (__m128i *)v247->m128i_i64[0] == &v247[1] )
  {
    v302 = _mm_loadu_si128(v247 + 1);
  }
  else
  {
    v301[0] = v247->m128i_i64[0];
    v302.m128i_i64[0] = v247[1].m128i_i64[0];
  }
  v248 = v247->m128i_i64[1];
  v247[1].m128i_i8[0] = 0;
  v301[1] = v248;
  v247->m128i_i64[0] = (__int64)v247[1].m128i_i64;
  v247->m128i_i64[1] = 0;
  v249 = (__m128i *)sub_2241490(v301, v281[95], v281[96], v281);
  v303 = &v305;
  if ( (__m128i *)v249->m128i_i64[0] == &v249[1] )
  {
    v305 = _mm_loadu_si128(v249 + 1);
  }
  else
  {
    v303 = (__m128i *)v249->m128i_i64[0];
    v305.m128i_i64[0] = v249[1].m128i_i64[0];
  }
  v250 = v249->m128i_i64[1];
  v304 = v250;
  v249->m128i_i64[0] = (__int64)v249[1].m128i_i64;
  v249->m128i_i64[1] = 0;
  v249[1].m128i_i8[0] = 0;
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v304) <= 0xA )
    goto LABEL_489;
  v251 = (__m128i *)sub_2241490(&v303, "' whereas '", 11, v250);
  v306 = &v308;
  if ( (__m128i *)v251->m128i_i64[0] == &v251[1] )
  {
    v308 = _mm_loadu_si128(v251 + 1);
  }
  else
  {
    v306 = (__m128i *)v251->m128i_i64[0];
    v308.m128i_i64[0] = v251[1].m128i_i64[0];
  }
  v307 = v251->m128i_i64[1];
  v252 = v307;
  v251->m128i_i64[0] = (__int64)v251[1].m128i_i64;
  v251->m128i_i64[1] = 0;
  v251[1].m128i_i8[0] = 0;
  v253 = (__m128i *)sub_2241490(&v306, v246[21], v246[22], v252);
  v309.m128i_i64[0] = (__int64)&v310;
  if ( (__m128i *)v253->m128i_i64[0] == &v253[1] )
  {
    v310 = _mm_loadu_si128(v253 + 1);
  }
  else
  {
    v309.m128i_i64[0] = v253->m128i_i64[0];
    v310.m128i_i64[0] = v253[1].m128i_i64[0];
  }
  v254 = v253->m128i_i64[1];
  v309.m128i_i64[1] = v254;
  v253->m128i_i64[0] = (__int64)v253[1].m128i_i64;
  v253->m128i_i64[1] = 0;
  v253[1].m128i_i8[0] = 0;
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v309.m128i_i64[1]) <= 5 )
    goto LABEL_489;
  v255 = (__m128i *)sub_2241490(&v309, "' is '", 6, v254);
  v311 = &v313;
  if ( (__m128i *)v255->m128i_i64[0] == &v255[1] )
  {
    v313 = _mm_loadu_si128(v255 + 1);
  }
  else
  {
    v311 = (__m128i *)v255->m128i_i64[0];
    v313.m128i_i64[0] = v255[1].m128i_i64[0];
  }
  v312 = v255->m128i_i64[1];
  v256 = v312;
  v255->m128i_i64[0] = (__int64)v255[1].m128i_i64;
  v255->m128i_i64[1] = 0;
  v255[1].m128i_i8[0] = 0;
  v257 = (__m128i *)sub_2241490(&v311, v246[95], v246[96], v256);
  v314.m128i_i64[0] = (__int64)&v315;
  if ( (__m128i *)v257->m128i_i64[0] == &v257[1] )
  {
    v315 = _mm_loadu_si128(v257 + 1);
  }
  else
  {
    v314.m128i_i64[0] = v257->m128i_i64[0];
    v315.m128i_i64[0] = v257[1].m128i_i64[0];
  }
  v258 = v257->m128i_i64[1];
  v314.m128i_i64[1] = v258;
  v257->m128i_i64[0] = (__int64)v257[1].m128i_i64;
  v257->m128i_i64[1] = 0;
  v257[1].m128i_i8[0] = 0;
  if ( v314.m128i_i64[1] == 0x3FFFFFFFFFFFFFFFLL || v314.m128i_i64[1] == 4611686018427387902LL )
    goto LABEL_489;
  v259 = (__m128i *)sub_2241490(&v314, "'\n", 2, v258);
  v324.m128i_i64[0] = (__int64)&v325;
  if ( (__m128i *)v259->m128i_i64[0] == &v259[1] )
  {
    v325 = _mm_loadu_si128(v259 + 1);
  }
  else
  {
    v324.m128i_i64[0] = v259->m128i_i64[0];
    v325.m128i_i64[0] = v259[1].m128i_i64[0];
  }
  v324.m128i_i64[1] = v259->m128i_i64[1];
  v259->m128i_i64[0] = (__int64)v259[1].m128i_i64;
  v259->m128i_i64[1] = 0;
  v259[1].m128i_i8[0] = 0;
  v338 = &v324;
  LOWORD(i) = 260;
  v277 = *v346;
  sub_1061A30((__int64)&v321, 1, (__int64)&v338);
  sub_B6EB20(v277, (__int64)&v321);
  if ( (__m128i *)v324.m128i_i64[0] != &v325 )
    j_j___libc_free_0(v324.m128i_i64[0], v325.m128i_i64[0] + 1);
  if ( (__m128i *)v314.m128i_i64[0] != &v315 )
    j_j___libc_free_0(v314.m128i_i64[0], v315.m128i_i64[0] + 1);
  if ( v311 != &v313 )
    j_j___libc_free_0(v311, v313.m128i_i64[0] + 1);
  if ( (__m128i *)v309.m128i_i64[0] != &v310 )
    j_j___libc_free_0(v309.m128i_i64[0], v310.m128i_i64[0] + 1);
  if ( v306 != &v308 )
    j_j___libc_free_0(v306, v308.m128i_i64[0] + 1);
  if ( v303 != &v305 )
    j_j___libc_free_0(v303, v305.m128i_i64[0] + 1);
  if ( (__m128i *)v301[0] != &v302 )
    j_j___libc_free_0(v301[0], v302.m128i_i64[0] + 1);
  if ( (__int64 *)v299[0] != &v300 )
    j_j___libc_free_0(v299[0], v300 + 1);
  if ( v297 )
    goto LABEL_130;
LABEL_131:
  v288 = v345;
  sub_CC80F0((__int64 *)&v321, (__int64)v327, (__int64)&v332);
  v324.m128i_i64[0] = (__int64)&v321;
  v326 = 260;
  sub_CC9F70((__int64)&v338, (void **)&v324);
  v74 = v288;
  v75 = (__m128i *)v288[29];
  if ( v338 == (__m128i *)&v340 )
  {
    v262 = n;
    if ( n )
    {
      if ( n == 1 )
      {
        v75->m128i_i8[0] = (char)v340;
      }
      else
      {
        memcpy(v75, &v340, n);
        v74 = v288;
      }
      v262 = n;
      v75 = (__m128i *)v288[29];
    }
    v74[30] = (__int64 *)v262;
    v75->m128i_i8[v262] = 0;
    v75 = v338;
  }
  else
  {
    v76 = n;
    v77 = v340;
    if ( v75 == (__m128i *)(v288 + 31) )
    {
      v288[29] = (__int64 *)v338;
      v288[30] = (__int64 *)v76;
      v288[31] = v77;
    }
    else
    {
      v78 = v288[31];
      v288[29] = (__int64 *)v338;
      v288[30] = (__int64 *)v76;
      v288[31] = v77;
      if ( v75 )
      {
        v338 = v75;
        v340 = v78;
        goto LABEL_135;
      }
    }
    v338 = (__m128i *)&v340;
    v75 = (__m128i *)&v340;
  }
LABEL_135:
  n = 0;
  v75->m128i_i8[0] = 0;
  v74[33] = (__int64 *)i;
  v74[34] = (__int64 *)v343;
  v74[35] = (__int64 *)v344;
  if ( v338 != (__m128i *)&v340 )
    j_j___libc_free_0(v338, (char *)v340 + 1);
  if ( v321 != &v323 )
    j_j___libc_free_0(v321, v323.m128i_i64[0] + 1);
  v79 = v346;
  v80 = (__int64 *)v346[2];
  v81 = v346 + 1;
  if ( v346 + 1 != v80 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        if ( !v80 )
LABEL_493:
          BUG();
        if ( (*((_BYTE *)v80 - 49) & 0x10) == 0 )
          goto LABEL_141;
        if ( (*(_BYTE *)(v80 - 3) & 0xFu) - 7 <= 1 )
          goto LABEL_141;
        v82 = (__int64)v345;
        v83 = sub_BD5D20((__int64)(v80 - 7));
        v85 = sub_BA8B30(v82, (__int64)v83, v84);
        v89 = v85;
        if ( !v85 )
          goto LABEL_141;
        v90 = *(_BYTE *)(v85 + 32) & 0xF;
        if ( (unsigned int)(v90 - 7) <= 1 )
          goto LABEL_141;
        if ( !*(_BYTE *)v85 && (*(_BYTE *)(v85 + 33) & 0x20) != 0 && !*((_BYTE *)v80 - 56) )
          break;
LABEL_152:
        if ( (_BYTE)v90 == 6 && (*(_BYTE *)(v80 - 3) & 0xF) == 6 )
        {
          sub_1062A30((__int64)v350, *(_QWORD *)(*(_QWORD *)(v89 + 24) + 24LL), *(_QWORD *)(*(v80 - 4) + 24));
          goto LABEL_141;
        }
        sub_1062A30((__int64)v350, *(_QWORD *)(v89 + 8), *(v80 - 6));
        v80 = (__int64 *)v80[1];
        if ( v81 == v80 )
        {
LABEL_155:
          v79 = v346;
          goto LABEL_156;
        }
      }
      v289 = *(_QWORD **)(v85 + 24);
      if ( v289 == sub_10651F0((__int64)v350, *(v80 - 4), v86, v90, v87, v88) )
      {
        LOBYTE(v90) = *(_BYTE *)(v89 + 32) & 0xF;
        goto LABEL_152;
      }
LABEL_141:
      v80 = (__int64 *)v80[1];
      if ( v81 == v80 )
        goto LABEL_155;
    }
  }
LABEL_156:
  v91 = (__int64 *)v79[4];
  v92 = v79 + 3;
  if ( v91 != v79 + 3 )
  {
    do
    {
      while ( 1 )
      {
        if ( !v91 )
          goto LABEL_493;
        if ( (*((_BYTE *)v91 - 49) & 0x10) != 0 && (*(_BYTE *)(v91 - 3) & 0xFu) - 7 > 1 )
        {
          v93 = (__int64)v345;
          v94 = sub_BD5D20((__int64)(v91 - 7));
          v96 = sub_BA8B30(v93, (__int64)v94, v95);
          v101 = v96;
          if ( v96 )
          {
            if ( (*(_BYTE *)(v96 + 32) & 0xFu) - 7 > 1 )
            {
              if ( *(_BYTE *)v96
                || (*(_BYTE *)(v96 + 33) & 0x20) == 0
                || *((_BYTE *)v91 - 56)
                || (v290 = *(_QWORD **)(v96 + 24), v290 == sub_10651F0((__int64)v350, *(v91 - 4), v97, v98, v99, v100)) )
              {
                v102 = *(_QWORD *)(v101 + 8);
                v103 = *(v91 - 6);
                if ( v102 != v103 )
                  break;
              }
            }
          }
        }
        v91 = (__int64 *)v91[1];
        if ( v92 == v91 )
          goto LABEL_170;
      }
      sub_1062A30((__int64)v350, v102, v103);
      v91 = (__int64 *)v91[1];
    }
    while ( v92 != v91 );
LABEL_170:
    v79 = v346;
  }
  v104 = (__int64 *)v79[6];
  v105 = v79 + 5;
  if ( v79 + 5 != v104 )
  {
    do
    {
      while ( 1 )
      {
        if ( !v104 )
          goto LABEL_493;
        if ( (*((_BYTE *)v104 - 41) & 0x10) != 0 && (*(_BYTE *)(v104 - 2) & 0xFu) - 7 > 1 )
        {
          v106 = (__int64)v345;
          v107 = sub_BD5D20((__int64)(v104 - 6));
          v109 = sub_BA8B30(v106, (__int64)v107, v108);
          v114 = v109;
          if ( v109 )
          {
            if ( (*(_BYTE *)(v109 + 32) & 0xFu) - 7 > 1 )
            {
              if ( *(_BYTE *)v109 )
                break;
              if ( (*(_BYTE *)(v109 + 33) & 0x20) == 0 )
                break;
              if ( *((_BYTE *)v104 - 48) )
                break;
              v291 = *(_QWORD **)(v109 + 24);
              if ( v291 == sub_10651F0((__int64)v350, *(v104 - 3), v110, v111, v112, v113) )
                break;
            }
          }
        }
        v104 = (__int64 *)v104[1];
        if ( v105 == v104 )
          goto LABEL_184;
      }
      sub_1062A30((__int64)v350, *(_QWORD *)(v114 + 8), *(v104 - 5));
      v104 = (__int64 *)v104[1];
    }
    while ( v105 != v104 );
LABEL_184:
    v79 = v346;
  }
  sub_BA9860(&v338, (__int64)v79);
  v119 = n;
  for ( j = v338; (__m128i *)v119 != j; j = (__m128i *)((char *)j + 8) )
  {
    v121 = (__int64 **)j->m128i_i64[0];
    if ( *(_QWORD *)(j->m128i_i64[0] + 24) && !sub_1061E80(v369, j->m128i_i64[0]) )
    {
      v122 = sub_BCB490((__int64)v121);
      v124 = v123;
      v125 = v123;
      for ( k = v123; k; --k )
      {
        v127 = k - 1;
        if ( *(_BYTE *)(k + v122 - 1) == 46 )
        {
          if ( k > 1 && *(_BYTE *)(v122 + v124 - 1) != 46 && (unsigned int)*(unsigned __int8 *)(v122 + k) - 48 <= 9 )
          {
            if ( v124 <= v127 )
              v127 = v124;
            v125 = v127;
          }
          break;
        }
      }
      sub_BCB490((__int64)v121);
      if ( v115 != v125 )
      {
        v128 = sub_BCBBB0(*v121, v122, v125);
        v129 = v128;
        if ( v128 )
        {
          if ( sub_1061E80(v369, v128) )
            sub_1062A30((__int64)v350, v129, (__int64)v121);
        }
      }
    }
  }
  v42 = v350;
  sub_1065380((unsigned __int64 *)&v324, (__int64)v350, v115, v116, v117, v118);
  v132 = v324.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (v324.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    v324.m128i_i64[0] = v132 | 1;
    if ( v407 )
    {
      if ( (v406 & 1) != 0 || (v406 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        goto LABEL_306;
      v406 |= v132 | 1;
    }
    else
    {
      v406 = v132 | 1;
      v407 = 1;
    }
  }
  if ( v338 )
    j_j___libc_free_0(v338, (char *)v340 - (char *)v338);
  v133 = (__int64)v394;
  v134 = v393;
  if ( v394 != (__int64 *)v393 )
  {
    v135 = v394 - 1;
    if ( v393 >= (unsigned __int64)(v394 - 1) )
      goto LABEL_211;
    v136 = (__int64 *)v393;
    do
    {
      v134 = *v136;
      v137 = *v135;
      ++v136;
      --v135;
      *(v136 - 1) = v137;
      v135[1] = v134;
    }
    while ( v135 > v136 );
LABEL_208:
    v133 = (__int64)v394;
LABEL_209:
    if ( v133 != v393 )
    {
      while ( 1 )
      {
        v135 = (__int64 *)(v133 - 8);
LABEL_211:
        v130 = *(_QWORD *)(v133 - 8);
        v394 = v135;
        v133 = (__int64)v135;
        if ( v376 )
        {
          v131 = v376 - 1;
          v138 = v131 & (((unsigned int)v130 >> 9) ^ ((unsigned int)v130 >> 4));
          v134 = (__int64)&v374[8 * (unsigned __int64)v138];
          v139 = *(_QWORD *)(v134 + 24);
          if ( v130 == v139 )
          {
LABEL_213:
            if ( (_QWORD *)v134 != &v374[8 * (unsigned __int64)v376] )
              goto LABEL_209;
          }
          else
          {
            v134 = 1;
            while ( v139 != -4096 )
            {
              v171 = v134 + 1;
              v138 = v131 & (v134 + v138);
              v134 = (__int64)&v374[8 * (unsigned __int64)v138];
              v139 = *(_QWORD *)(v134 + 24);
              if ( v130 == v139 )
                goto LABEL_213;
              v134 = v171;
            }
          }
        }
        if ( v385 )
        {
          v131 = v385 - 1;
          v140 = v131 & (((unsigned int)v130 >> 9) ^ ((unsigned int)v130 >> 4));
          v134 = (__int64)&v383[8 * (unsigned __int64)v140];
          v141 = *(_QWORD *)(v134 + 24);
          if ( v130 == v141 )
          {
LABEL_216:
            if ( (_QWORD *)v134 != &v383[8 * (unsigned __int64)v385] )
              goto LABEL_209;
          }
          else
          {
            v134 = 1;
            while ( v141 != -4096 )
            {
              v263 = v134 + 1;
              v140 = v131 & (v134 + v140);
              v134 = (__int64)&v383[8 * (unsigned __int64)v140];
              v141 = *(_QWORD *)(v134 + 24);
              if ( v130 == v141 )
                goto LABEL_216;
              v134 = v263;
            }
          }
        }
        v36 = (__int64 *)v130;
        sub_FCD360((__int64 *)&v408, v130, (__int64)v135, v134, v130, v131);
        if ( v407 )
          break;
        v142 = v396;
        v143 = v397;
        if ( v396 == v397 )
          goto LABEL_208;
        do
        {
          v144 = *(_QWORD **)v142;
          v145 = *(_QWORD *)(v142 + 8);
          v142 += 16;
          sub_BD84D0((__int64)v144, v145);
          sub_B30810(v144);
        }
        while ( v143 != v142 );
        v133 = (__int64)v394;
        if ( v396 == v397 )
          goto LABEL_209;
        v397 = v396;
        if ( v394 == (__int64 *)v393 )
          goto LABEL_222;
      }
      v275 = v406;
      v406 = 0;
      *a1 = v275 | 1;
      goto LABEL_281;
    }
  }
LABEL_222:
  v405 = 1;
  sub_FCD200(&v408, 8, v133, v134, v130, v131);
  v149 = v346[10];
  v292 = v346[108];
  v284 = v346 + 9;
  while ( v284 != (__int64 *)v149 )
  {
    if ( v292 == v149 )
      goto LABEL_261;
    if ( v403 )
    {
      v203 = sub_B91B20(v149);
      if ( v204 == 22
        && !(*(_QWORD *)v203 ^ 0x6573702E6D766C6CLL | *(_QWORD *)(v203 + 8) ^ 0x626F72705F6F6475LL)
        && *(_DWORD *)(v203 + 16) == 1701076837
        && *(_WORD *)(v203 + 20) == 25459 )
      {
        v222 = (__int64)v345;
        v223 = sub_B91B20(v149);
        if ( !sub_BA8DC0(v222, v223, v224) )
        {
          v225 = (__int64)v345;
          sub_8FD6D0((__int64)&v316, "Pseudo-probe ignored: source module '", v346 + 21);
          if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v317) <= 0x49 )
            goto LABEL_489;
          v227 = (__m128i *)sub_2241490(
                              &v316,
                              "' is compiled with -fpseudo-probe-for-profiling while destination module '",
                              74,
                              v226);
          v319.m128i_i64[0] = (__int64)&v320;
          if ( (__m128i *)v227->m128i_i64[0] == &v227[1] )
          {
            v320 = _mm_loadu_si128(v227 + 1);
          }
          else
          {
            v319.m128i_i64[0] = v227->m128i_i64[0];
            v320.m128i_i64[0] = v227[1].m128i_i64[0];
          }
          v319.m128i_i64[1] = v227->m128i_i64[1];
          v228 = v319.m128i_i64[1];
          v227->m128i_i64[0] = (__int64)v227[1].m128i_i64;
          v227->m128i_i64[1] = 0;
          v227[1].m128i_i8[0] = 0;
          v229 = (__m128i *)sub_2241490(&v319, *(_QWORD *)(v225 + 168), *(_QWORD *)(v225 + 176), v228);
          v321 = &v323;
          if ( (__m128i *)v229->m128i_i64[0] == &v229[1] )
          {
            v323 = _mm_loadu_si128(v229 + 1);
          }
          else
          {
            v321 = (__m128i *)v229->m128i_i64[0];
            v323.m128i_i64[0] = v229[1].m128i_i64[0];
          }
          v230 = v229->m128i_i64[1];
          v322 = v230;
          v229->m128i_i64[0] = (__int64)v229[1].m128i_i64;
          v229->m128i_i64[1] = 0;
          v229[1].m128i_i8[0] = 0;
          if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v322) <= 8 )
            goto LABEL_489;
          v231 = (__m128i *)sub_2241490(&v321, "' is not\n", 9, v230);
          v324.m128i_i64[0] = (__int64)&v325;
          if ( (__m128i *)v231->m128i_i64[0] == &v231[1] )
          {
            v325 = _mm_loadu_si128(v231 + 1);
          }
          else
          {
            v324.m128i_i64[0] = v231->m128i_i64[0];
            v325.m128i_i64[0] = v231[1].m128i_i64[0];
          }
          v324.m128i_i64[1] = v231->m128i_i64[1];
          v231->m128i_i64[0] = (__int64)v231[1].m128i_i64;
          v231->m128i_i64[1] = 0;
          v231[1].m128i_i8[0] = 0;
          LOWORD(i) = 260;
          v338 = &v324;
          v232 = *v346;
          sub_1061A30((__int64)&v314, 1, (__int64)&v338);
          sub_B6EB20(v232, (__int64)&v314);
          if ( (__m128i *)v324.m128i_i64[0] != &v325 )
            j_j___libc_free_0(v324.m128i_i64[0], v325.m128i_i64[0] + 1);
          if ( v321 != &v323 )
            j_j___libc_free_0(v321, v323.m128i_i64[0] + 1);
          if ( (__m128i *)v319.m128i_i64[0] != &v320 )
            j_j___libc_free_0(v319.m128i_i64[0], v320.m128i_i64[0] + 1);
          if ( v316 != &v318 )
            j_j___libc_free_0(v316, v318.m128i_i64[0] + 1);
        }
        goto LABEL_261;
      }
      if ( v403 )
      {
        v205 = sub_B91B20(v149);
        if ( v206 == 10 )
        {
          v146 = 0x6174732E6D766C6CLL;
          if ( *(_QWORD *)v205 == 0x6174732E6D766C6CLL && *(_WORD *)(v205 + 8) == 29556 )
            goto LABEL_261;
        }
      }
    }
    v150 = (__int64)v345;
    v151 = (_QWORD *)sub_B91B20(v149);
    v153 = sub_BA8E40(v150, v151, v152);
    v154 = (_QWORD *)sub_B91B20(v149);
    if ( v155 != 16 || *v154 ^ 0x6E6E612E6D76766ELL | v154[1] ^ 0x736E6F697461746FLL )
    {
      v172 = sub_B91A00(v149);
      if ( v172 )
      {
        v173 = 0;
        do
        {
          v174 = v173++;
          v175 = sub_B91A10(v149, v174);
          v180 = sub_FCD270((__int64 *)&v408, v175, v176, v177, v178, v179);
          sub_B979A0(v153, v180);
        }
        while ( v172 != v173 );
      }
    }
    else
    {
      v156 = sub_B91A00(v149);
      if ( v156 )
      {
        for ( m = 0; m != v156; ++m )
        {
          v159 = sub_B91A10(v149, m);
          v164 = sub_FCD270((__int64 *)&v408, v159, v160, v161, v162, v163);
          v165 = *(_BYTE *)(v164 - 16);
          if ( (v165 & 2) != 0 )
          {
            if ( !*(_DWORD *)(v164 - 24) )
              goto LABEL_232;
            v158 = *(_QWORD **)(v164 - 32);
          }
          else
          {
            if ( (*(_WORD *)(v164 - 16) & 0x3C0) == 0 )
            {
LABEL_232:
              sub_B979A0(v153, v164);
              continue;
            }
            v146 = -16 - 8LL * ((v165 >> 2) & 0xF);
            v158 = (_QWORD *)(v164 + v146);
          }
          if ( *v158 )
            goto LABEL_232;
        }
      }
    }
LABEL_261:
    v149 = *(_QWORD *)(v149 + 8);
  }
  v181 = v400;
  v182 = (unsigned int)v401;
  v183 = &v400[(unsigned int)v402];
  if ( (_DWORD)v401 && v400 != v183 )
  {
    while ( *v181 == -8192 || *v181 == -4096 )
    {
      if ( ++v181 == v183 )
        goto LABEL_263;
    }
LABEL_425:
    if ( v183 != v181 )
    {
      v264 = *v181;
      if ( sub_B2FC80(*v181) )
        sub_FCD2B0((__int64 *)&v408, v264);
      while ( ++v181 != v183 )
      {
        if ( *v181 != -8192 && *v181 != -4096 )
          goto LABEL_425;
      }
    }
  }
LABEL_263:
  v184 = v346;
  if ( !v403 )
  {
    v185 = v346[12];
    if ( !v185 )
      goto LABEL_265;
    v265 = v345;
    if ( (unsigned int)(v329 - 36) <= 1 )
    {
      sub_8FD6D0((__int64)&v338, ".text\n.balign 2\n.thumb\n", v346 + 11);
    }
    else if ( (unsigned int)(v329 - 1) > 1 )
    {
      v338 = (__m128i *)&v340;
      sub_10614E0((__int64 *)&v338, (_BYTE *)v346[11], v346[11] + v185);
    }
    else
    {
      sub_8FD6D0((__int64)&v338, ".text\n.balign 4\n.arm\n", v346 + 11);
    }
    if ( n <= 0x3FFFFFFFFFFFFFFFLL - (__int64)v265[12] )
    {
      sub_2241490(v265 + 11, v338, n, v266);
      v267 = v265[12];
      if ( v267 )
      {
        v268 = (__int64 **)v265[11];
        if ( *((_BYTE *)v267 + (_QWORD)v268 - 1) != 10 )
        {
          v269 = (unsigned __int64)v267 + 1;
          if ( v268 == v265 + 13 )
            v146 = 15;
          else
            v146 = (unsigned __int64)v265[13];
          if ( v269 > v146 )
          {
            sub_2240BB0(v265 + 11, v265[12], 0, 0, 1);
            v268 = (__int64 **)v265[11];
          }
          *((_BYTE *)v267 + (_QWORD)v268) = 10;
          v270 = v265[11];
          v265[12] = (__int64 *)v269;
          *((_BYTE *)v267 + (_QWORD)v270 + 1) = 0;
        }
      }
      if ( v338 != (__m128i *)&v340 )
        j_j___libc_free_0(v338, (char *)v340 + 1);
      goto LABEL_326;
    }
LABEL_489:
    sub_4262D8((__int64)"basic_string::append");
  }
  v338 = (__m128i *)&v345;
  sub_C137E0(v346, (__int64)sub_1061A20, (__int64)&v338);
LABEL_326:
  v184 = v346;
LABEL_265:
  v186 = (__int64 *)v184[2];
  v187 = v184 + 1;
  if ( v184 + 1 != v186 )
  {
    while ( v186 )
    {
      if ( (*(_BYTE *)(v186 - 3) & 0xF) != 6 )
      {
        v188 = (unsigned __int8 *)sub_FCD360((__int64 *)&v408, (__int64)(v186 - 7), v146, v182, v147, v148);
        if ( v188 )
        {
          v189 = sub_BD3990(v188, (__int64)(v186 - 7));
          v190 = (__int64)v189;
          if ( *v189 == 3 )
          {
            sub_B30110(v189);
            v191 = (__int64)v345;
            v293 = v345 + 1;
            sub_BA85C0((__int64)(v345 + 1), v190);
            v192 = *(_QWORD *)(v191 + 8);
            *(_QWORD *)(v190 + 64) = v293;
            v182 = v192 & 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v190 + 56) = v182 | *(_QWORD *)(v190 + 56) & 7LL;
            *(_QWORD *)(v182 + 8) = v190 + 56;
            v146 = *(_QWORD *)(v191 + 8) & 7LL | (v190 + 56);
            *(_QWORD *)(v191 + 8) = v146;
          }
        }
      }
      v186 = (__int64 *)v186[1];
      if ( v187 == v186 )
        goto LABEL_272;
    }
LABEL_492:
    BUG();
  }
LABEL_272:
  if ( v404 )
  {
    v193 = (__int64 *)v346[4];
    v194 = v346 + 3;
    if ( v193 != v346 + 3 )
    {
      while ( v193 )
      {
        if ( (*(_BYTE *)(v193 - 3) & 0xF) != 6 )
        {
          v195 = (unsigned __int8 *)sub_FCD360((__int64 *)&v408, (__int64)(v193 - 7), v146, v182, v147, v148);
          if ( v195 )
          {
            v196 = sub_BD3990(v195, (__int64)(v193 - 7));
            v197 = (__int64)v196;
            if ( !*v196 )
            {
              sub_B2C2B0(v196);
              v198 = (__int64)v345;
              v294 = v345 + 3;
              sub_BA8540((__int64)(v345 + 3), v197);
              v199 = *(_QWORD *)(v198 + 24);
              *(_QWORD *)(v197 + 64) = v294;
              v182 = v199 & 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v197 + 56) = v182 | *(_QWORD *)(v197 + 56) & 7LL;
              *(_QWORD *)(v182 + 8) = v197 + 56;
              v146 = *(_QWORD *)(v198 + 24) & 7LL | (v197 + 56);
              *(_QWORD *)(v198 + 24) = v146;
            }
          }
        }
        v193 = (__int64 *)v193[1];
        if ( v194 == v193 )
          goto LABEL_280;
      }
      goto LABEL_492;
    }
  }
LABEL_280:
  v36 = (__int64 *)&v345;
  sub_1068BA0(a1, (__int64 *)&v345);
LABEL_281:
  if ( (__int64 *)v332.m128i_i64[0] != &v333 )
  {
    v36 = (__int64 *)(v333 + 1);
    j_j___libc_free_0(v332.m128i_i64[0], v333 + 1);
  }
  if ( (_QWORD *)v327[0] != v328 )
  {
    v36 = (__int64 *)(v328[0] + 1LL);
    j_j___libc_free_0(v327[0], v328[0] + 1LL);
  }
  v200 = *((_BYTE *)v283 + 872);
  if ( v279 )
  {
    if ( !v200 )
    {
      for ( ii = (__int64 *)v283[4]; v283 + 3 != ii; ii = (__int64 *)ii[1] )
      {
        v261 = (__int64)(ii - 7);
        if ( !ii )
          v261 = 0;
        sub_B2B950(v261);
      }
      *((_BYTE *)v283 + 872) = 1;
    }
  }
  else if ( v200 )
  {
    for ( jj = (__int64 *)v283[4]; v283 + 3 != jj; jj = (__int64 *)jj[1] )
    {
      v202 = (__int64)(jj - 7);
      if ( !jj )
        v202 = 0;
      sub_B2B9A0(v202);
    }
    *((_BYTE *)v283 + 872) = 0;
  }
LABEL_43:
  sub_B7C530(*(__int64 ***)a2, (__int64)v36);
  v37 = v372;
  v38 = *(unsigned int *)(v372 + 24);
  if ( (_DWORD)v38 )
  {
    v39 = *(_QWORD **)(v372 + 8);
    v40 = &v39[2 * v38];
    do
    {
      if ( *v39 != -8192 && *v39 != -4096 )
      {
        v41 = v39[1];
        if ( v41 )
          sub_B91220((__int64)(v39 + 1), v41);
      }
      v39 += 2;
    }
    while ( v40 != v39 );
    LODWORD(v38) = *(_DWORD *)(v37 + 24);
  }
  v42 = (_QWORD *)(16LL * (unsigned int)v38);
  sub_C7D6A0(*(_QWORD *)(v37 + 8), (__int64)v42, 8);
  *(_QWORD *)(v37 + 16) = 0;
  *(_QWORD *)(v37 + 8) = 0;
  *(_DWORD *)(v37 + 24) = 0;
  ++*(_QWORD *)v37;
  ++v377;
  v43 = *(_QWORD *)(v37 + 8);
  *(_QWORD *)(v37 + 8) = v378;
  v378 = (_QWORD *)v43;
  LODWORD(v43) = *(_DWORD *)(v37 + 16);
  *(_DWORD *)(v37 + 16) = v379;
  LODWORD(v379) = v43;
  LODWORD(v43) = *(_DWORD *)(v37 + 20);
  *(_DWORD *)(v37 + 20) = HIDWORD(v379);
  HIDWORD(v379) = v43;
  LODWORD(v43) = *(_DWORD *)(v37 + 24);
  *(_DWORD *)(v37 + 24) = v380;
  v380 = v43;
  sub_FC7680((__int64 *)&v408, (__int64)v42);
  if ( v407 )
  {
    v407 = 0;
    if ( (v406 & 1) != 0 || (v406 & 0xFFFFFFFFFFFFFFFELL) != 0 )
LABEL_306:
      sub_C63C30(&v406, (__int64)v42);
  }
  sub_C7D6A0((__int64)v400, 8LL * (unsigned int)v402, 8);
  if ( v396 )
    j_j___libc_free_0(v396, v398 - v396);
  if ( v393 )
    j_j___libc_free_0(v393, v395 - v393);
  sub_C7D6A0(v390, 8LL * (unsigned int)v392, 8);
  if ( v388 )
  {
    v214 = v387;
    v388 = 0;
    if ( v387 )
    {
      v215 = v386;
      v216 = &v386[2 * v387];
      do
      {
        if ( *v215 != -8192 && *v215 != -4096 )
        {
          v217 = v215[1];
          if ( v217 )
            sub_B91220((__int64)(v215 + 1), v217);
        }
        v215 += 2;
      }
      while ( v216 != v215 );
      v214 = v387;
    }
    sub_C7D6A0((__int64)v386, 16LL * v214, 8);
  }
  v44 = v385;
  if ( v385 )
  {
    v45 = v383;
    v332.m128i_i64[1] = 2;
    v333 = 0;
    v46 = &v383[8 * (unsigned __int64)v385];
    v334 = -4096;
    v47 = -4096;
    v332.m128i_i64[0] = (__int64)&unk_49DD7B0;
    v335 = 0;
    n = 2;
    v340 = 0;
    v341 = -8192;
    v338 = (__m128i *)&unk_49DD7B0;
    i = 0;
    while ( 1 )
    {
      v48 = v45[3];
      if ( v47 != v48 )
      {
        v47 = v341;
        if ( v48 != v341 )
        {
          v49 = v45[7];
          if ( v49 != -4096 && v49 != 0 && v49 != -8192 )
          {
            sub_BD60C0(v45 + 5);
            v48 = v45[3];
          }
          v47 = v48;
        }
      }
      *v45 = &unk_49DB368;
      if ( v47 != -4096 && v47 != 0 && v47 != -8192 )
        sub_BD60C0(v45 + 1);
      v45 += 8;
      if ( v46 == v45 )
        break;
      v47 = v334;
    }
    v338 = (__m128i *)&unk_49DB368;
    if ( v341 != -4096 && v341 != 0 && v341 != -8192 )
      sub_BD60C0(&n);
    v332.m128i_i64[0] = (__int64)&unk_49DB368;
    if ( v334 != -4096 && v334 != 0 && v334 != -8192 )
      sub_BD60C0(&v332.m128i_i64[1]);
    v44 = v385;
  }
  sub_C7D6A0((__int64)v383, (unsigned __int64)v44 << 6, 8);
  if ( v381 )
  {
    v218 = v380;
    v381 = 0;
    if ( v380 )
    {
      v219 = v378;
      v220 = &v378[2 * v380];
      do
      {
        if ( *v219 != -8192 && *v219 != -4096 )
        {
          v221 = v219[1];
          if ( v221 )
            sub_B91220((__int64)(v219 + 1), v221);
        }
        v219 += 2;
      }
      while ( v220 != v219 );
      v218 = v380;
    }
    sub_C7D6A0((__int64)v378, 16LL * v218, 8);
  }
  v50 = v376;
  if ( v376 )
  {
    v51 = v374;
    v332.m128i_i64[1] = 2;
    v333 = 0;
    v52 = &v374[8 * (unsigned __int64)v376];
    v334 = -4096;
    v53 = -4096;
    v332.m128i_i64[0] = (__int64)&unk_49DD7B0;
    v335 = 0;
    n = 2;
    v340 = 0;
    v341 = -8192;
    v338 = (__m128i *)&unk_49DD7B0;
    i = 0;
    while ( 1 )
    {
      v54 = v51[3];
      if ( v53 != v54 )
      {
        v53 = v341;
        if ( v54 != v341 )
        {
          v55 = v51[7];
          if ( v55 != 0 && v55 != -4096 && v55 != -8192 )
          {
            sub_BD60C0(v51 + 5);
            v54 = v51[3];
          }
          v53 = v54;
        }
      }
      *v51 = &unk_49DB368;
      if ( v53 != 0 && v53 != -4096 && v53 != -8192 )
        sub_BD60C0(v51 + 1);
      v51 += 8;
      if ( v52 == v51 )
        break;
      v53 = v334;
    }
    v338 = (__m128i *)&unk_49DB368;
    if ( v341 != 0 && v341 != -4096 && v341 != -8192 )
      sub_BD60C0(&n);
    v332.m128i_i64[0] = (__int64)&unk_49DB368;
    if ( v334 != -4096 && v334 != 0 && v334 != -8192 )
      sub_BD60C0(&v332.m128i_i64[1]);
    v50 = v376;
  }
  v56 = (unsigned __int64)v50 << 6;
  sub_C7D6A0((__int64)v374, v56, 8);
  v350[0] = off_49E5E78;
  if ( !v367 )
    _libc_free(v364, v56);
  if ( v360 != v362 )
    _libc_free(v360, v56);
  if ( v357 != v359 )
    _libc_free(v357, v56);
  if ( v354 != v356 )
    _libc_free(v354, v56);
  v57 = 16LL * v353;
  sub_C7D6A0(v351, v57, 8);
  v59 = v349 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v349 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    v60 = (v349 >> 1) & 1;
    if ( (v349 & 4) != 0 )
    {
      v61 = &v347;
      if ( !(_BYTE)v60 )
        v61 = (__m128i *)v347.m128i_i64[0];
      (*(void (__fastcall **)(__m128i *))(v59 + 16))(v61);
    }
    if ( !(_BYTE)v60 )
    {
      v57 = v347.m128i_i64[1];
      sub_C7D6A0(v347.m128i_i64[0], v347.m128i_i64[1], v348);
    }
  }
  v62 = v346;
  if ( v346 )
  {
    sub_BA9C10((_QWORD **)v346, v57, v59, v58);
    j_j___libc_free_0(v62, 880);
  }
  return a1;
}
