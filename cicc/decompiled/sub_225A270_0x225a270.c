// Function: sub_225A270
// Address: 0x225a270
//
char *__fastcall sub_225A270(__int64 *a1, _DWORD *a2, unsigned int a3, __int64 *a4, __int64 a5, __m128i a6)
{
  __int64 v6; // rsi
  size_t v7; // r12
  unsigned __int8 *v8; // r13
  _QWORD *v9; // rsi
  char *(*v10)(); // rax
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rax
  __int64 v14; // rdx
  char *v15; // rsi
  __int64 v16; // rax
  int *v17; // r12
  int *v18; // r14
  void *v19; // r13
  size_t v20; // r12
  int *v21; // rbx
  int *v22; // r14
  size_t v23; // r15
  size_t v24; // rdx
  int v25; // eax
  __int64 v26; // r15
  size_t v27; // r15
  size_t v28; // rdx
  signed __int64 v29; // rax
  size_t v30; // r15
  __int64 *v31; // rbx
  __int64 *v32; // r12
  size_t v33; // r14
  size_t v34; // rdx
  int v35; // eax
  __int64 v36; // r14
  size_t v37; // rbx
  _QWORD *v38; // r15
  size_t v39; // r14
  size_t v40; // rdx
  signed __int64 v41; // rax
  __int64 *v42; // rax
  size_t v43; // r13
  size_t v44; // rdx
  int v45; // eax
  __int64 *v46; // r12
  char *v47; // rsi
  char v48; // r12
  __int64 v49; // rax
  char *v50; // rdx
  unsigned __int64 v51; // r12
  _BYTE *v52; // rsi
  int v53; // eax
  int v54; // edx
  char v55; // cl
  __int64 v56; // rax
  __int64 v57; // rcx
  __int64 v58; // rax
  _BYTE *v59; // rbx
  unsigned __int64 v60; // r12
  unsigned __int64 v61; // rdi
  unsigned __int64 v62; // rax
  __int64 v63; // rbx
  __int64 v64; // r13
  __int64 v65; // rbx
  __int64 v66; // r15
  __int64 v67; // r14
  unsigned __int64 v68; // rdx
  int v69; // ebx
  __int64 v70; // rbx
  void (__fastcall *v71)(_QWORD, _QWORD, _QWORD); // rdx
  __int64 v72; // rsi
  void (__fastcall *v73)(_QWORD, _QWORD, _QWORD); // r9
  __int64 v74; // r12
  __int64 v75; // r15
  _BYTE *p_src; // rdi
  __int64 v77; // rdx
  __int64 v78; // rsi
  char v79; // r12
  __int64 v80; // rax
  __int64 v81; // rdx
  __int64 v82; // rax
  __int64 *v83; // r15
  size_t v84; // r14
  __int64 *v85; // r12
  void *v86; // r13
  size_t v87; // rbx
  size_t v88; // rdx
  int v89; // eax
  __int64 v90; // rbx
  const void *v91; // r8
  size_t v92; // r13
  size_t v93; // rdx
  signed __int64 v94; // rax
  __int64 v95; // rsi
  __int64 *v96; // r15
  void *v97; // r14
  size_t v98; // rbx
  size_t v99; // r13
  size_t v100; // rdx
  int v101; // eax
  __int64 v102; // r13
  __m128 *v103; // rdi
  size_t v104; // r14
  unsigned __int64 v105; // rax
  unsigned __int8 **v106; // r15
  __int64 v107; // rax
  _DWORD *v108; // rdx
  __int64 v109; // rdi
  __int64 v110; // rax
  void *v111; // rdx
  __int64 v112; // rdi
  __int64 v113; // rax
  void *v114; // rdx
  __int64 *v115; // r14
  void *v116; // r15
  __int64 *v117; // r13
  size_t v118; // r12
  __int64 *v119; // rbx
  size_t v120; // r14
  size_t v121; // rdx
  int v122; // eax
  __int64 v123; // r14
  __int64 *v124; // rax
  __int64 *v125; // r14
  size_t v126; // rbx
  size_t v127; // rcx
  size_t v128; // rdx
  int v129; // eax
  __int64 v130; // rbx
  __int64 v131; // rax
  void *v132; // rdx
  __int64 v133; // rdi
  __int64 v134; // rax
  _WORD *v135; // rdx
  _WORD *v136; // rdi
  unsigned __int64 v137; // rax
  __m128i *v138; // r12
  char *v139; // rdi
  size_t v140; // r13
  _BYTE *v141; // rbx
  unsigned __int64 v142; // r12
  unsigned __int64 v143; // rdi
  char *v144; // r13
  char *v145; // r12
  __int64 v146; // rdx
  __int64 v147; // rcx
  __int64 v149; // rax
  size_t v150; // rdx
  __int64 v151; // rax
  __int64 v152; // rax
  __m128i *v153; // rdi
  unsigned __int64 v154; // rax
  __m128i *v155; // r14
  __m128i si128; // xmm0
  __m128i v157; // xmm0
  __m128i *v158; // rdi
  __m128i *v159; // rdi
  __int64 v160; // rax
  __int64 v161; // rax
  _BYTE *v162; // r15
  __int64 v163; // r14
  __int64 v164; // rax
  unsigned __int64 v165; // r13
  char v166; // al
  __int64 v167; // rdx
  __int64 v168; // rcx
  char *v169; // r12
  char v170; // bl
  __int64 v171; // rbx
  __int64 v172; // rdx
  _QWORD *v173; // rdx
  _BYTE *v174; // rax
  _QWORD **v175; // r12
  __int64 v176; // rax
  _QWORD *v177; // r14
  _QWORD *v178; // rbx
  __int64 v179; // rsi
  __int64 v180; // rax
  __int64 *v181; // r15
  __int64 v182; // rbx
  __int64 v183; // rsi
  __int64 v184; // r14
  _BYTE *v185; // rsi
  __int64 v186; // rdx
  char *v187; // rdi
  void (__fastcall *v188)(_QWORD, _QWORD, _QWORD); // rdx
  __int64 v189; // rsi
  void (__fastcall *v190)(_QWORD, _QWORD, _QWORD); // r9
  _BYTE *v191; // rsi
  __int64 v192; // rdx
  bool v193; // zf
  __int64 v194; // rsi
  int v195; // eax
  __int64 v196; // rdx
  __int64 v197; // rcx
  __int64 v198; // r8
  unsigned __int64 *v199; // r9
  void *v200; // r14
  __int64 *v201; // rax
  __int64 v202; // rax
  _QWORD *v203; // r12
  _QWORD *v204; // rbx
  __int64 v205; // rsi
  __int64 v206; // rdi
  __int64 v207; // rax
  __int64 v208; // rdx
  __int64 v209; // rcx
  __int64 v210; // rax
  __int64 v211; // rdi
  __m128i *v212; // rax
  __m128i v213; // xmm0
  __int64 v214; // r9
  char v215; // dl
  char *v216; // r14
  __int64 v217; // rdx
  __m128i *v218; // rax
  __m128i v219; // xmm0
  __m128i *v220; // rdi
  unsigned __int8 *v221; // rsi
  __int64 v222; // rdx
  __int64 v223; // rcx
  void *v224; // r12
  __int64 v225; // rax
  __int64 v226; // rdi
  __m128i *v227; // rax
  __m128i v228; // xmm0
  __int64 v229; // r12
  __int64 v230; // rax
  __int64 v231; // rbx
  __int64 *v232; // r15
  __int64 *v233; // r14
  __m128i *v234; // rax
  __m128i v235; // xmm0
  __m128i *v236; // r14
  unsigned __int8 *v237; // r13
  size_t v238; // rax
  void *v239; // rdi
  size_t v240; // r15
  __int64 v241; // r8
  char *v242; // rax
  __m128i *v243; // rax
  __m128i v244; // xmm0
  __m128i *v245; // r14
  unsigned __int8 *v246; // r12
  size_t v247; // rax
  void *v248; // rdi
  size_t v249; // r15
  __int64 *v250; // rax
  __m128i *v251; // rdi
  unsigned __int8 *v252; // r12
  size_t v253; // rax
  size_t v254; // rbx
  bool v257; // [rsp+18h] [rbp-778h]
  __int64 v258; // [rsp+20h] [rbp-770h]
  void *v259; // [rsp+28h] [rbp-768h]
  __int64 v263; // [rsp+98h] [rbp-6F8h]
  __int64 *v264; // [rsp+A0h] [rbp-6F0h]
  unsigned __int64 v265; // [rsp+A0h] [rbp-6F0h]
  size_t v266; // [rsp+A0h] [rbp-6F0h]
  char *v267; // [rsp+B0h] [rbp-6E0h]
  __int64 v268; // [rsp+B8h] [rbp-6D8h]
  bool v269; // [rsp+B8h] [rbp-6D8h]
  __int64 v270; // [rsp+D8h] [rbp-6B8h]
  __int64 v271; // [rsp+E0h] [rbp-6B0h]
  unsigned __int64 v272; // [rsp+E0h] [rbp-6B0h]
  __int64 v273; // [rsp+E0h] [rbp-6B0h]
  __int64 v274; // [rsp+E0h] [rbp-6B0h]
  __int64 v275; // [rsp+E8h] [rbp-6A8h]
  int v276; // [rsp+E8h] [rbp-6A8h]
  __int64 v277; // [rsp+E8h] [rbp-6A8h]
  _QWORD *v278; // [rsp+F8h] [rbp-698h] BYREF
  char *v279[2]; // [rsp+100h] [rbp-690h] BYREF
  unsigned __int64 v280; // [rsp+110h] [rbp-680h] BYREF
  _BYTE *v281; // [rsp+118h] [rbp-678h]
  _BYTE *v282; // [rsp+120h] [rbp-670h]
  void *s2; // [rsp+130h] [rbp-660h] BYREF
  size_t v284; // [rsp+138h] [rbp-658h]
  _BYTE v285[16]; // [rsp+140h] [rbp-650h] BYREF
  const __m128i *v286[4]; // [rsp+150h] [rbp-640h] BYREF
  unsigned __int64 v287[2]; // [rsp+170h] [rbp-620h] BYREF
  __int64 *v288; // [rsp+180h] [rbp-610h]
  __int64 v289; // [rsp+188h] [rbp-608h]
  __int64 *v290; // [rsp+190h] [rbp-600h]
  __int64 v291; // [rsp+1A0h] [rbp-5F0h] BYREF
  int v292; // [rsp+1A8h] [rbp-5E8h] BYREF
  int *v293; // [rsp+1B0h] [rbp-5E0h]
  int *v294; // [rsp+1B8h] [rbp-5D8h]
  int *v295; // [rsp+1C0h] [rbp-5D0h]
  __int64 v296; // [rsp+1C8h] [rbp-5C8h]
  unsigned __int8 *v297; // [rsp+1D0h] [rbp-5C0h] BYREF
  size_t v298; // [rsp+1D8h] [rbp-5B8h]
  __int64 v299; // [rsp+1E0h] [rbp-5B0h] BYREF
  __int64 v300; // [rsp+1E8h] [rbp-5A8h]
  void *dest; // [rsp+1F0h] [rbp-5A0h]
  __int64 v302; // [rsp+1F8h] [rbp-598h]
  void *v303; // [rsp+200h] [rbp-590h]
  char *v304; // [rsp+210h] [rbp-580h] BYREF
  size_t v305; // [rsp+218h] [rbp-578h]
  unsigned __int64 v306[2]; // [rsp+220h] [rbp-570h] BYREF
  _QWORD v307[2]; // [rsp+230h] [rbp-560h] BYREF
  __int64 v308; // [rsp+240h] [rbp-550h]
  int v309; // [rsp+248h] [rbp-548h]
  void *v310; // [rsp+250h] [rbp-540h]
  size_t v311; // [rsp+258h] [rbp-538h]
  _QWORD v312[2]; // [rsp+260h] [rbp-530h] BYREF
  _QWORD *v313; // [rsp+270h] [rbp-520h]
  __int64 v314; // [rsp+278h] [rbp-518h]
  _QWORD v315[2]; // [rsp+280h] [rbp-510h] BYREF
  unsigned __int64 v316; // [rsp+290h] [rbp-500h]
  __int64 v317; // [rsp+298h] [rbp-4F8h]
  __int64 v318; // [rsp+2A0h] [rbp-4F0h]
  _BYTE *v319; // [rsp+2A8h] [rbp-4E8h]
  __int64 v320; // [rsp+2B0h] [rbp-4E0h]
  _BYTE v321[200]; // [rsp+2B8h] [rbp-4D8h] BYREF
  char *s; // [rsp+380h] [rbp-410h] BYREF
  size_t v323; // [rsp+388h] [rbp-408h]
  __int64 v324[2]; // [rsp+390h] [rbp-400h] BYREF
  char *v325; // [rsp+3A0h] [rbp-3F0h]
  __int64 v326; // [rsp+3A8h] [rbp-3E8h]
  __int64 v327; // [rsp+3B0h] [rbp-3E0h]
  char v328; // [rsp+3B8h] [rbp-3D8h] BYREF
  __int64 *v329; // [rsp+3C0h] [rbp-3D0h]
  __int64 v330; // [rsp+3C8h] [rbp-3C8h]
  _QWORD *v331; // [rsp+3D0h] [rbp-3C0h] BYREF
  unsigned int v332; // [rsp+3E0h] [rbp-3B0h]
  char *v333; // [rsp+400h] [rbp-390h]
  __int64 v334; // [rsp+408h] [rbp-388h]
  char v335; // [rsp+410h] [rbp-380h] BYREF
  char *v336; // [rsp+430h] [rbp-360h]
  __int64 v337; // [rsp+438h] [rbp-358h]
  char v338; // [rsp+440h] [rbp-350h] BYREF
  char *v339; // [rsp+490h] [rbp-300h]
  __int64 v340; // [rsp+498h] [rbp-2F8h]
  char v341; // [rsp+4A0h] [rbp-2F0h] BYREF
  char *v342; // [rsp+540h] [rbp-250h]
  __int64 v343; // [rsp+548h] [rbp-248h]
  char v344; // [rsp+550h] [rbp-240h] BYREF
  __int16 v345; // [rsp+560h] [rbp-230h]
  __int64 v346; // [rsp+568h] [rbp-228h]
  __m128i v347; // [rsp+570h] [rbp-220h] BYREF
  void (__fastcall *src)(_QWORD, _QWORD, _QWORD); // [rsp+580h] [rbp-210h] BYREF
  _BYTE *v349; // [rsp+588h] [rbp-208h]
  void *v350[14]; // [rsp+590h] [rbp-200h] BYREF
  char v351; // [rsp+600h] [rbp-190h] BYREF
  char *v352; // [rsp+620h] [rbp-170h]
  __int64 v353; // [rsp+628h] [rbp-168h]
  char v354; // [rsp+630h] [rbp-160h] BYREF
  char *v355; // [rsp+680h] [rbp-110h]
  __int64 v356; // [rsp+688h] [rbp-108h]
  char v357; // [rsp+690h] [rbp-100h] BYREF
  char *v358; // [rsp+730h] [rbp-60h]
  __int64 v359; // [rsp+738h] [rbp-58h]
  char v360; // [rsp+740h] [rbp-50h] BYREF
  __int16 v361; // [rsp+750h] [rbp-40h]
  __int64 v362; // [rsp+758h] [rbp-38h]

  v6 = *(unsigned __int8 *)(a5 + 232);
  v294 = &v292;
  v295 = &v292;
  v280 = 0;
  v281 = 0;
  v282 = 0;
  v292 = 0;
  v293 = 0;
  v296 = 0;
  v287[0] = 0;
  v287[1] = 0;
  v288 = 0;
  v289 = 0;
  v290 = 0;
  sub_B6F950(a4, v6);
  v258 = a1[1];
  if ( *a1 != v258 )
  {
    v263 = *a1;
    v259 = a1 + 10;
    v257 = 0;
    while ( 1 )
    {
      v7 = 14;
      v8 = (unsigned __int8 *)"Unknown buffer";
      sub_C7DA90(
        &v278,
        *(_QWORD *)(v263 + 16),
        *(_QWORD *)(v263 + 24),
        *(const char **)v263,
        *(const char **)(v263 + 8),
        1);
      v9 = v278;
      v304 = 0;
      v305 = 0;
      v306[0] = (unsigned __int64)v307;
      v306[1] = 0;
      v310 = v312;
      LOBYTE(v307[0]) = 0;
      v313 = v315;
      v308 = 0;
      v319 = v321;
      v309 = 0;
      v311 = 0;
      LOBYTE(v312[0]) = 0;
      v314 = 0;
      LOBYTE(v315[0]) = 0;
      v316 = 0;
      v317 = 0;
      v318 = 0;
      v320 = 0x400000000LL;
      v10 = *(char *(**)())(*v278 + 16LL);
      if ( v10 != sub_C1E8B0 )
      {
        v149 = ((__int64 (__fastcall *)(_QWORD *))v10)(v278);
        v9 = v278;
        v8 = (unsigned __int8 *)v149;
        v7 = v150;
      }
      memset(v350, 0, 0x58u);
      sub_C7E010(v286, v9);
      sub_E46810(
        (unsigned __int64 *)&s,
        (__int64)&v304,
        (__int64)a4,
        (__int64)&v347,
        a6,
        v11,
        v12,
        v286[0],
        (unsigned __int64)v286[1],
        (__int64)v286[2]->m128i_i64,
        (__int64)v286[3]->m128i_i64);
      v267 = s;
      if ( LOBYTE(v350[10]) )
      {
        LOBYTE(v350[10]) = 0;
        if ( v350[8] )
          ((void (__fastcall *)(void **, void **, __int64))v350[8])(&v350[6], &v350[6], 3);
      }
      if ( LOBYTE(v350[5]) )
      {
        LOBYTE(v350[5]) = 0;
        if ( v350[3] )
          ((void (__fastcall *)(void **, void **, __int64))v350[3])(&v350[1], &v350[1], 3);
      }
      if ( LOBYTE(v350[0]) && (LOBYTE(v350[0]) = 0, src) )
      {
        src(&v347, &v347, 3);
        if ( !v267 )
        {
LABEL_204:
          v350[1] = (void *)0x100000000LL;
          v347.m128i_i64[1] = 0;
          src = 0;
          v347.m128i_i64[0] = (__int64)&unk_49DD210;
          v349 = 0;
          v350[2] = v259;
          v350[0] = 0;
          sub_CB5980((__int64)&v347, 0, 0, 0);
          v136 = v350[0];
          v137 = v349 - (char *)v350[0];
          if ( v349 - (char *)v350[0] < v7 )
          {
            sub_CB6200((__int64)&v347, v8, v7);
            v136 = v350[0];
            v137 = v349 - (char *)v350[0];
            if ( (_DWORD)v308 == -1 )
              goto LABEL_208;
          }
          else
          {
            if ( v7 )
            {
              memcpy(v350[0], v8, v7);
              v136 = (char *)v350[0] + v7;
              v350[0] = v136;
              v137 = v349 - (_BYTE *)v136;
            }
            if ( (_DWORD)v308 == -1 )
            {
LABEL_208:
              if ( v137 > 7 )
                goto LABEL_209;
LABEL_273:
              v160 = sub_CB6200((__int64)&v347, ": parse ", 8u);
              v139 = *(char **)(v160 + 32);
              v138 = (__m128i *)v160;
              goto LABEL_210;
            }
          }
          if ( v137 <= 1 )
          {
            v158 = (__m128i *)sub_CB6200((__int64)&v347, (unsigned __int8 *)" (", 2u);
          }
          else
          {
            *v136 = 10272;
            v158 = &v347;
            v350[0] = (char *)v350[0] + 2;
          }
          sub_CB59F0((__int64)v158, (int)v308);
          if ( HIDWORD(v308) != -1 )
          {
            if ( (unsigned __int64)(v349 - (char *)v350[0]) <= 1 )
            {
              v159 = (__m128i *)sub_CB6200((__int64)&v347, (unsigned __int8 *)", ", 2u);
            }
            else
            {
              v159 = &v347;
              *(_WORD *)v350[0] = 8236;
              v350[0] = (char *)v350[0] + 2;
            }
            sub_CB59F0((__int64)v159, SHIDWORD(v308));
          }
          if ( v349 == v350[0] )
          {
            sub_CB6200((__int64)&v347, (unsigned __int8 *)")", 1u);
            v136 = v350[0];
          }
          else
          {
            *(_BYTE *)v350[0] = 41;
            v136 = ++v350[0];
          }
          if ( (unsigned __int64)(v349 - (_BYTE *)v136) <= 7 )
            goto LABEL_273;
LABEL_209:
          v138 = &v347;
          *(_QWORD *)v136 = 0x206573726170203ALL;
          v139 = (char *)v350[0] + 8;
          v350[0] = (char *)v350[0] + 8;
LABEL_210:
          v140 = v311;
          v6 = (__int64)v310;
          if ( v311 > v138[1].m128i_i64[1] - (__int64)v139 )
          {
            sub_CB6200((__int64)v138, (unsigned __int8 *)v310, v311);
          }
          else if ( v311 )
          {
            memcpy(v139, v310, v311);
            v138[2].m128i_i64[0] += v140;
          }
          *a2 = 9;
LABEL_214:
          v347.m128i_i64[0] = (__int64)&unk_49DD210;
          sub_CB5840((__int64)&v347);
          v141 = v319;
          v142 = (unsigned __int64)&v319[48 * (unsigned int)v320];
          if ( v319 != (_BYTE *)v142 )
          {
            do
            {
              v142 -= 48LL;
              v143 = *(_QWORD *)(v142 + 16);
              if ( v143 != v142 + 32 )
              {
                v6 = *(_QWORD *)(v142 + 32) + 1LL;
                j_j___libc_free_0(v143);
              }
            }
            while ( v141 != (_BYTE *)v142 );
            v142 = (unsigned __int64)v319;
          }
          if ( (_BYTE *)v142 != v321 )
            _libc_free(v142);
          if ( v316 )
          {
            v6 = v318 - v316;
            j_j___libc_free_0(v316);
          }
          if ( v313 != v315 )
          {
            v6 = v315[0] + 1LL;
            j_j___libc_free_0((unsigned __int64)v313);
          }
          if ( v310 != v312 )
          {
            v6 = v312[0] + 1LL;
            j_j___libc_free_0((unsigned __int64)v310);
          }
          if ( (_QWORD *)v306[0] != v307 )
          {
            v6 = v307[0] + 1LL;
            j_j___libc_free_0(v306[0]);
          }
          if ( v278 )
            (*(void (__fastcall **)(_QWORD *, __int64))(*v278 + 8LL))(v278, v6);
          goto LABEL_231;
        }
      }
      else if ( !v267 )
      {
        goto LABEL_204;
      }
      if ( (unsigned int)sub_2259720((__int64)a1, (__int64)v267, a3) )
      {
        v6 = 0;
        v350[1] = (void *)0x100000000LL;
        v347.m128i_i64[1] = 0;
        src = 0;
        v347.m128i_i64[0] = (__int64)&unk_49DD210;
        v349 = 0;
        v350[2] = v259;
        v350[0] = 0;
        sub_CB5980((__int64)&v347, 0, 0, 0);
        v153 = (__m128i *)v350[0];
        v154 = v349 - (char *)v350[0];
        if ( v349 - (char *)v350[0] < v7 )
        {
          v6 = (__int64)v8;
          v161 = sub_CB6200((__int64)&v347, v8, v7);
          v153 = *(__m128i **)(v161 + 32);
          v155 = (__m128i *)v161;
          v154 = *(_QWORD *)(v161 + 24) - (_QWORD)v153;
        }
        else
        {
          v155 = &v347;
          if ( v7 )
          {
            v6 = (__int64)v8;
            memcpy(v350[0], v8, v7);
            v153 = (__m128i *)((char *)v350[0] + v7);
            v350[0] = v153;
            v154 = v349 - (_BYTE *)v153;
          }
        }
        if ( v154 <= 0x56 )
        {
          v6 = (__int64)": error: incompatible IR detected. Possible mix of compiler/IR from different releases.";
          sub_CB6200(
            (__int64)v155,
            ": error: incompatible IR detected. Possible mix of compiler/IR from different releases.",
            0x57u);
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_4281880);
          v153[5].m128i_i32[0] = 1935762796;
          v153[5].m128i_i16[2] = 29541;
          *v153 = si128;
          v157 = _mm_load_si128((const __m128i *)&xmmword_4281890);
          v153[5].m128i_i8[6] = 46;
          v153[1] = v157;
          v153[2] = _mm_load_si128((const __m128i *)&xmmword_42818A0);
          v153[3] = _mm_load_si128((const __m128i *)&xmmword_42818B0);
          v153[4] = _mm_load_si128((const __m128i *)&xmmword_42818C0);
          v155[2].m128i_i64[0] += 87;
        }
        *a2 = 3;
        goto LABEL_214;
      }
      v302 = 0x100000000LL;
      v298 = 0;
      v299 = 0;
      v297 = (unsigned __int8 *)&unk_49DD210;
      v300 = 0;
      v303 = v259;
      dest = 0;
      sub_CB5980((__int64)&v297, 0, 0, 0);
      v275 = *((_QWORD *)v267 + 2);
      if ( (char *)v275 == v267 + 8 )
        goto LABEL_74;
      do
      {
        v13 = 0;
        if ( v275 )
          v13 = v275 - 56;
        v271 = v13;
        v15 = (char *)sub_BD5D20(v13);
        if ( v15 )
        {
          s2 = v285;
          sub_22579E0((__int64 *)&s2, v15, (__int64)&v15[v14]);
        }
        else
        {
          LOBYTE(v285[0]) = 0;
          v284 = 0;
          s2 = v285;
        }
        if ( (*(_BYTE *)(v271 + 32) & 0xF) != 0 )
          goto LABEL_70;
        v16 = *(_QWORD *)(v271 + 24);
        v268 = v16;
        if ( *(_BYTE *)(v16 + 8) == 15 )
        {
          if ( (*(_BYTE *)(v16 + 9) & 1) == 0 )
            goto LABEL_70;
          v17 = v293;
          if ( !v293 )
          {
LABEL_164:
            v32 = (__int64 *)&v292;
            goto LABEL_56;
          }
        }
        else
        {
          v17 = v293;
          if ( !v293 )
            goto LABEL_164;
        }
        v18 = v17;
        v264 = (__int64 *)v17;
        v19 = s2;
        v20 = v284;
        v21 = v18;
        v22 = &v292;
        do
        {
          while ( 1 )
          {
            v23 = *((_QWORD *)v21 + 5);
            v24 = v20;
            if ( v23 <= v20 )
              v24 = *((_QWORD *)v21 + 5);
            if ( v24 )
            {
              v25 = memcmp(*((const void **)v21 + 4), v19, v24);
              if ( v25 )
                break;
            }
            v26 = v23 - v20;
            if ( v26 >= 0x80000000LL )
              goto LABEL_28;
            if ( v26 > (__int64)0xFFFFFFFF7FFFFFFFLL )
            {
              v25 = v26;
              break;
            }
LABEL_19:
            v21 = (int *)*((_QWORD *)v21 + 3);
            if ( !v21 )
              goto LABEL_29;
          }
          if ( v25 < 0 )
            goto LABEL_19;
LABEL_28:
          v22 = v21;
          v21 = (int *)*((_QWORD *)v21 + 2);
        }
        while ( v21 );
LABEL_29:
        if ( v22 == &v292 )
          goto LABEL_37;
        v27 = *((_QWORD *)v22 + 5);
        v28 = v20;
        if ( v27 <= v20 )
          v28 = *((_QWORD *)v22 + 5);
        if ( v28 )
        {
          LODWORD(v29) = memcmp(v19, *((const void **)v22 + 4), v28);
          if ( (_DWORD)v29 )
          {
LABEL_36:
            if ( (int)v29 < 0 )
              goto LABEL_37;
LABEL_128:
            v78 = *(_QWORD *)(v271 + 40);
            s = 0;
            v325 = &v328;
            v329 = (__int64 *)&v331;
            v330 = 0x600000000LL;
            v333 = &v335;
            v334 = 0x400000000LL;
            v336 = &v338;
            v337 = 0xA00000000LL;
            v339 = &v341;
            v340 = 0x800000000LL;
            v342 = &v344;
            v323 = 0;
            v324[0] = 0;
            v324[1] = 0;
            v326 = 0;
            v327 = 8;
            v343 = 0;
            v344 = 0;
            v345 = 768;
            v346 = 0;
            sub_AE1EA0((__int64)&s, v78 + 312);
            v79 = sub_AE5020((__int64)&s, v268);
            v80 = sub_9208B0((__int64)&s, v268);
            v347.m128i_i64[1] = v81;
            v347.m128i_i64[0] = ((1LL << v79) + ((unsigned __int64)(v80 + 7) >> 3) - 1) >> v79 << v79;
            v82 = sub_CA1930(&v347);
            v83 = (__int64 *)v293;
            v272 = v82;
            if ( !v293 )
            {
              v85 = (__int64 *)&v292;
              goto LABEL_148;
            }
            v84 = v284;
            v85 = (__int64 *)&v292;
            v86 = s2;
            while ( 2 )
            {
              while ( 2 )
              {
                v87 = v83[5];
                v88 = v84;
                if ( v87 <= v84 )
                  v88 = v83[5];
                if ( v88 && (v89 = memcmp((const void *)v83[4], v86, v88)) != 0 )
                {
LABEL_138:
                  if ( v89 < 0 )
                  {
LABEL_130:
                    v83 = (__int64 *)v83[3];
                    if ( !v83 )
                      goto LABEL_140;
                    continue;
                  }
                }
                else
                {
                  v90 = v87 - v84;
                  if ( v90 < 0x80000000LL )
                  {
                    if ( v90 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
                      goto LABEL_130;
                    v89 = v90;
                    goto LABEL_138;
                  }
                }
                break;
              }
              v85 = v83;
              v83 = (__int64 *)v83[2];
              if ( !v83 )
              {
LABEL_140:
                v91 = v86;
                if ( v85 == (__int64 *)&v292 )
                  goto LABEL_148;
                v92 = v85[5];
                v93 = v84;
                if ( v92 <= v84 )
                  v93 = v85[5];
                if ( v93 && (LODWORD(v94) = memcmp(v91, (const void *)v85[4], v93), (_DWORD)v94) )
                {
LABEL_147:
                  if ( (int)v94 < 0 )
                    goto LABEL_148;
                }
                else
                {
                  v94 = v84 - v92;
                  if ( (__int64)(v84 - v92) < 0x80000000LL )
                  {
                    if ( v94 > (__int64)0xFFFFFFFF7FFFFFFFLL )
                      goto LABEL_147;
LABEL_148:
                    v347.m128i_i64[0] = (__int64)&s2;
                    v85 = sub_22595D0(&v291, v85, v347.m128i_i64);
                  }
                }
                v95 = *(_QWORD *)(v85[8] + 40);
                v360 = 0;
                v350[0] = &v350[3];
                v350[4] = &v350[6];
                v350[5] = (void *)0x600000000LL;
                v350[12] = &v351;
                v350[13] = (void *)0x400000000LL;
                v352 = &v354;
                v353 = 0xA00000000LL;
                v355 = &v357;
                v356 = 0x800000000LL;
                v358 = &v360;
                v347 = 0u;
                src = 0;
                v349 = 0;
                v350[1] = 0;
                v350[2] = (void *)8;
                v359 = 0;
                v361 = 768;
                v362 = 0;
                sub_AE1EA0((__int64)&v347, v95 + 312);
                v96 = (__int64 *)v293;
                if ( !v293 )
                {
                  v46 = (__int64 *)&v292;
                  goto LABEL_67;
                }
                v97 = s2;
                v98 = v284;
                v46 = (__int64 *)&v292;
                while ( 2 )
                {
                  v99 = v96[5];
                  v100 = v98;
                  if ( v99 <= v98 )
                    v100 = v96[5];
                  if ( v100 && (v101 = memcmp((const void *)v96[4], v97, v100)) != 0 )
                  {
LABEL_160:
                    if ( v101 < 0 )
                      goto LABEL_151;
                  }
                  else
                  {
                    v102 = v99 - v98;
                    if ( v102 < 0x80000000LL )
                    {
                      if ( v102 > (__int64)0xFFFFFFFF7FFFFFFFLL )
                      {
                        v101 = v102;
                        goto LABEL_160;
                      }
LABEL_151:
                      v96 = (__int64 *)v96[3];
LABEL_152:
                      if ( !v96 )
                      {
                        if ( v46 == (__int64 *)&v292 )
                          goto LABEL_67;
                        v43 = v46[5];
                        v44 = v98;
                        if ( v43 <= v98 )
                          v44 = v46[5];
                        if ( v44 && (v45 = memcmp(v97, (const void *)v46[4], v44)) != 0 )
                        {
LABEL_66:
                          if ( v45 < 0 )
                            goto LABEL_67;
                        }
                        else if ( (__int64)(v98 - v43) < 0x80000000LL )
                        {
                          if ( (__int64)(v98 - v43) > (__int64)0xFFFFFFFF7FFFFFFFLL )
                          {
                            v45 = v98 - v43;
                            goto LABEL_66;
                          }
LABEL_67:
                          v279[0] = (char *)&s2;
                          v46 = sub_22595D0(&v291, v46, (__int64 *)v279);
                        }
                        v47 = *(char **)(v46[8] + 24);
                        v48 = sub_AE5020((__int64)&v347, (__int64)v47);
                        v49 = sub_9208B0((__int64)&v347, (__int64)v47);
                        v279[1] = v50;
                        v279[0] = (char *)((((unsigned __int64)(v49 + 7) >> 3) + (1LL << v48) - 1) >> v48 << v48);
                        v51 = sub_CA1930(v279);
                        v269 = v272 != 0 && v272 != v51 && v51 != 0;
                        if ( !v269 )
                          goto LABEL_69;
                        v103 = (__m128 *)dest;
                        v104 = v311;
                        v105 = v300 - (_QWORD)dest;
                        if ( v311 > v300 - (__int64)dest )
                        {
                          v151 = sub_CB6200((__int64)&v297, (unsigned __int8 *)v310, v311);
                          v103 = *(__m128 **)(v151 + 32);
                          v106 = (unsigned __int8 **)v151;
                          if ( *(_QWORD *)(v151 + 24) - (_QWORD)v103 > 0x17u )
                            goto LABEL_169;
                        }
                        else
                        {
                          v106 = &v297;
                          if ( v311 )
                          {
                            memcpy(dest, v310, v311);
                            v103 = (__m128 *)((char *)dest + v104);
                            dest = v103;
                            v105 = v300 - (_QWORD)v103;
                          }
                          if ( v105 > 0x17 )
                          {
LABEL_169:
                            a6 = _mm_load_si128((const __m128i *)&xmmword_42818D0);
                            v103[1].m128_u64[0] = 0x20726F6620686374LL;
                            *v103 = (__m128)a6;
                            v106[4] += 24;
                            goto LABEL_170;
                          }
                        }
                        v106 = (unsigned __int8 **)sub_CB6200((__int64)v106, "Size does not match for ", 0x18u);
LABEL_170:
                        v107 = sub_CB6200((__int64)v106, (unsigned __int8 *)s2, v284);
                        v108 = *(_DWORD **)(v107 + 32);
                        v109 = v107;
                        if ( *(_QWORD *)(v107 + 24) - (_QWORD)v108 <= 3u )
                        {
                          v109 = sub_CB6200(v107, (unsigned __int8 *)" in ", 4u);
                        }
                        else
                        {
                          *v108 = 544106784;
                          *(_QWORD *)(v107 + 32) += 4LL;
                        }
                        v110 = sub_CB6200(v109, *((unsigned __int8 **)v267 + 21), *((_QWORD *)v267 + 22));
                        v111 = *(void **)(v110 + 32);
                        v112 = v110;
                        if ( *(_QWORD *)(v110 + 24) - (_QWORD)v111 <= 0xAu )
                        {
                          v112 = sub_CB6200(v110, " with size ", 0xBu);
                        }
                        else
                        {
                          qmemcpy(v111, " with size ", 11);
                          *(_QWORD *)(v110 + 32) += 11LL;
                        }
                        v113 = sub_CB59D0(v112, v272);
                        v114 = *(void **)(v113 + 32);
                        v273 = v113;
                        if ( *(_QWORD *)(v113 + 24) - (_QWORD)v114 > 0xDu )
                        {
                          qmemcpy(v114, " specified in ", 14);
                          v115 = (__int64 *)v293;
                          *(_QWORD *)(v113 + 32) += 14LL;
                          if ( v115 )
                            goto LABEL_176;
LABEL_254:
                          v125 = (__int64 *)&v292;
                          goto LABEL_196;
                        }
                        v152 = sub_CB6200(v113, " specified in ", 0xEu);
                        v115 = (__int64 *)v293;
                        v273 = v152;
                        if ( !v293 )
                          goto LABEL_254;
LABEL_176:
                        v265 = v51;
                        v116 = s2;
                        v117 = v115;
                        v118 = v284;
                        v119 = (__int64 *)&v292;
                        while ( 2 )
                        {
                          while ( 2 )
                          {
                            v120 = v117[5];
                            v121 = v118;
                            if ( v120 <= v118 )
                              v121 = v117[5];
                            if ( v121 && (v122 = memcmp((const void *)v117[4], v116, v121)) != 0 )
                            {
LABEL_185:
                              if ( v122 < 0 )
                              {
LABEL_177:
                                v117 = (__int64 *)v117[3];
                                if ( !v117 )
                                  goto LABEL_187;
                                continue;
                              }
                            }
                            else
                            {
                              v123 = v120 - v118;
                              if ( v123 < 0x80000000LL )
                              {
                                if ( v123 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
                                  goto LABEL_177;
                                v122 = v123;
                                goto LABEL_185;
                              }
                            }
                            break;
                          }
                          v119 = v117;
                          v117 = (__int64 *)v117[2];
                          if ( !v117 )
                          {
LABEL_187:
                            v124 = v119;
                            v125 = v119;
                            v126 = v118;
                            v51 = v265;
                            if ( v124 == (__int64 *)&v292 )
                              goto LABEL_196;
                            v127 = v124[5];
                            v128 = v126;
                            if ( v127 <= v126 )
                              v128 = v124[5];
                            if ( v128
                              && (v266 = v124[5], v129 = memcmp(v116, (const void *)v124[4], v128), v127 = v266, v129) )
                            {
LABEL_195:
                              if ( v129 < 0 )
                                goto LABEL_196;
                            }
                            else
                            {
                              v130 = v126 - v127;
                              if ( v130 < 0x80000000LL )
                              {
                                if ( v130 > (__int64)0xFFFFFFFF7FFFFFFFLL )
                                {
                                  v129 = v130;
                                  goto LABEL_195;
                                }
LABEL_196:
                                v279[0] = (char *)&s2;
                                v125 = sub_22595D0(&v291, v125, (__int64 *)v279);
                              }
                            }
                            v131 = sub_CB6200(
                                     v273,
                                     *(unsigned __int8 **)(*(_QWORD *)(v125[8] + 40) + 168LL),
                                     *(_QWORD *)(*(_QWORD *)(v125[8] + 40) + 176LL));
                            v132 = *(void **)(v131 + 32);
                            v133 = v131;
                            if ( *(_QWORD *)(v131 + 24) - (_QWORD)v132 <= 0xAu )
                            {
                              v133 = sub_CB6200(v131, " with size ", 0xBu);
                            }
                            else
                            {
                              qmemcpy(v132, " with size ", 11);
                              *(_QWORD *)(v131 + 32) += 11LL;
                            }
                            v47 = (char *)v51;
                            v134 = sub_CB59D0(v133, v51);
                            v135 = *(_WORD **)(v134 + 32);
                            if ( *(_QWORD *)(v134 + 24) - (_QWORD)v135 <= 1u )
                            {
                              v47 = ".\n";
                              sub_CB6200(v134, (unsigned __int8 *)".\n", 2u);
                            }
                            else
                            {
                              *v135 = 2606;
                              *(_QWORD *)(v134 + 32) += 2LL;
                            }
                            v257 = v269;
LABEL_69:
                            sub_AE4030(&v347, (__int64)v47);
                            sub_AE4030(&s, (__int64)v47);
LABEL_70:
                            v38 = s2;
                            goto LABEL_71;
                          }
                          continue;
                        }
                      }
                      continue;
                    }
                  }
                  break;
                }
                v46 = v96;
                v96 = (__int64 *)v96[2];
                goto LABEL_152;
              }
              continue;
            }
          }
        }
        v29 = v20 - v27;
        if ( (__int64)(v20 - v27) >= 0x80000000LL )
          goto LABEL_128;
        if ( v29 > (__int64)0xFFFFFFFF7FFFFFFFLL )
          goto LABEL_36;
LABEL_37:
        v30 = v20;
        v31 = v264;
        v32 = (__int64 *)&v292;
        while ( 2 )
        {
          while ( 2 )
          {
            v33 = v31[5];
            v34 = v30;
            if ( v33 <= v30 )
              v34 = v31[5];
            if ( !v34 || (v35 = memcmp((const void *)v31[4], v19, v34)) == 0 )
            {
              v36 = v33 - v30;
              if ( v36 >= 0x80000000LL )
                goto LABEL_47;
              if ( v36 > (__int64)0xFFFFFFFF7FFFFFFFLL )
              {
                v35 = v36;
                break;
              }
LABEL_38:
              v31 = (__int64 *)v31[3];
              if ( !v31 )
                goto LABEL_48;
              continue;
            }
            break;
          }
          if ( v35 < 0 )
            goto LABEL_38;
LABEL_47:
          v32 = v31;
          v31 = (__int64 *)v31[2];
          if ( v31 )
            continue;
          break;
        }
LABEL_48:
        v37 = v30;
        v38 = v19;
        if ( v32 == (__int64 *)&v292 )
          goto LABEL_56;
        v39 = v32[5];
        v40 = v37;
        if ( v39 <= v37 )
          v40 = v32[5];
        if ( v40 && (LODWORD(v41) = memcmp(v19, (const void *)v32[4], v40), (_DWORD)v41) )
        {
LABEL_55:
          if ( (int)v41 < 0 )
            goto LABEL_56;
        }
        else
        {
          v41 = v37 - v39;
          if ( (__int64)(v37 - v39) < 0x80000000LL )
          {
            if ( v41 > (__int64)0xFFFFFFFF7FFFFFFFLL )
              goto LABEL_55;
LABEL_56:
            v347.m128i_i64[0] = (__int64)&s2;
            v42 = sub_22595D0(&v291, v32, v347.m128i_i64);
            v38 = s2;
            v32 = v42;
          }
        }
        v32[8] = v271;
LABEL_71:
        if ( v38 != v285 )
          j_j___libc_free_0((unsigned __int64)v38);
        v275 = *(_QWORD *)(v275 + 8);
      }
      while ( v267 + 8 != (char *)v275 );
LABEL_74:
      v52 = v281;
      v347.m128i_i64[0] = (__int64)v267;
      if ( v281 == v282 )
      {
        sub_2259040((__int64)&v280, v281, &v347);
        v267 = (char *)v347.m128i_i64[0];
      }
      else
      {
        if ( v281 )
        {
          *(_QWORD *)v281 = v267;
          v52 = v281;
        }
        v281 = v52 + 8;
      }
      v53 = sub_2241AC0((__int64)(v267 + 232), off_4C5D110);
      v6 = (__int64)v288;
      v54 = v53;
      if ( v288 == v290 )
      {
        sub_2257D90((__int64)v287, v288, v289, v53 == 0);
      }
      else
      {
        v55 = v289;
        if ( (_DWORD)v289 == 63 )
        {
          LODWORD(v289) = 0;
          ++v288;
        }
        else
        {
          LODWORD(v289) = v289 + 1;
        }
        v56 = 1LL << v55;
        v57 = (1LL << v55) | *(_QWORD *)v6;
        v58 = *(_QWORD *)v6 & ~v56;
        if ( !v54 )
          v58 = v57;
        *(_QWORD *)v6 = v58;
      }
      v297 = (unsigned __int8 *)&unk_49DD210;
      sub_CB5840((__int64)&v297);
      v59 = v319;
      v60 = (unsigned __int64)&v319[48 * (unsigned int)v320];
      if ( v319 != (_BYTE *)v60 )
      {
        do
        {
          v60 -= 48LL;
          v61 = *(_QWORD *)(v60 + 16);
          if ( v61 != v60 + 32 )
          {
            v6 = *(_QWORD *)(v60 + 32) + 1LL;
            j_j___libc_free_0(v61);
          }
        }
        while ( v59 != (_BYTE *)v60 );
        v60 = (unsigned __int64)v319;
      }
      if ( (_BYTE *)v60 != v321 )
        _libc_free(v60);
      if ( v316 )
      {
        v6 = v318 - v316;
        j_j___libc_free_0(v316);
      }
      if ( v313 != v315 )
      {
        v6 = v315[0] + 1LL;
        j_j___libc_free_0((unsigned __int64)v313);
      }
      if ( v310 != v312 )
      {
        v6 = v312[0] + 1LL;
        j_j___libc_free_0((unsigned __int64)v310);
      }
      if ( (_QWORD *)v306[0] != v307 )
      {
        v6 = v307[0] + 1LL;
        j_j___libc_free_0(v306[0]);
      }
      if ( v278 )
        (*(void (__fastcall **)(_QWORD *, __int64))(*v278 + 8LL))(v278, v6);
      v263 += 32;
      if ( v258 == v263 )
      {
        if ( !v257 )
          break;
        *a2 = 9;
LABEL_281:
        v267 = 0;
LABEL_231:
        v144 = 0;
        v145 = 0;
        goto LABEL_232;
      }
    }
  }
  sub_CEAEC0();
  v62 = v280;
  v63 = (__int64)&v281[-v280] >> 3;
  if ( (_DWORD)v63 )
  {
    v64 = (unsigned int)v63;
    v65 = 0;
    v66 = (__int64)&v281[-v280] >> 3;
    while ( 1 )
    {
      v6 = (__int64)off_4C5D110;
      v276 = v65;
      v67 = 8 * v65;
      if ( sub_2241AC0(*(_QWORD *)(v62 + 8 * v65) + 232LL, off_4C5D110) )
        break;
      v62 = v280;
      ++v65;
      v68 = v280;
      if ( v64 == v65 )
      {
        v276 = 0;
        v69 = v66;
        v67 = 0;
        goto LABEL_108;
      }
    }
    v68 = v280;
    v69 = v66;
LABEL_108:
    if ( !*(_QWORD *)(*(_QWORD *)(v68 + v67) + 240LL) )
    {
      v350[1] = (void *)0x100000000LL;
      v347.m128i_i64[1] = 0;
      v347.m128i_i64[0] = (__int64)&unk_49DD210;
      src = 0;
      v350[2] = a1 + 10;
      v349 = 0;
      v350[0] = 0;
      sub_CB5980((__int64)&v347, 0, 0, 0);
      v225 = *(_QWORD *)(v280 + v67);
      v6 = *(_QWORD *)(v225 + 168);
      v226 = sub_CB6200((__int64)&v347, (unsigned __int8 *)v6, *(_QWORD *)(v225 + 176));
      v227 = *(__m128i **)(v226 + 32);
      if ( *(_QWORD *)(v226 + 24) - (_QWORD)v227 <= 0x3Eu )
      {
        v6 = (__int64)": error: Module does not contain a triple, should be 'nvptx64-'";
        sub_CB6200(v226, ": error: Module does not contain a triple, should be 'nvptx64-'", 0x3Fu);
      }
      else
      {
        v228 = _mm_load_si128((const __m128i *)&xmmword_42818E0);
        qmemcpy(&v227[3], "d be 'nvptx64-'", 15);
        *v227 = v228;
        v227[1] = _mm_load_si128((const __m128i *)&xmmword_42818F0);
        v227[2] = _mm_load_si128((const __m128i *)&xmmword_4281900);
        *(_QWORD *)(v226 + 32) += 63LL;
      }
      v144 = 0;
      v145 = 0;
      *a2 = 9;
      v347.m128i_i64[0] = (__int64)&unk_49DD210;
      sub_CB5840((__int64)&v347);
      v267 = 0;
      goto LABEL_232;
    }
    if ( v69 == 1 )
    {
      v145 = *(char **)v68;
    }
    else
    {
      LOBYTE(v306[0]) = 0;
      v70 = 0;
      v304 = (char *)v306;
      v305 = 0;
      v270 = (unsigned int)((__int64)&v281[-v68] >> 3);
      if ( (unsigned int)((__int64)&v281[-v68] >> 3) )
      {
        while ( v276 == (_DWORD)v70 )
        {
LABEL_118:
          if ( v270 == ++v70 )
            goto LABEL_353;
        }
        v74 = *(_QWORD *)(v280 + 8 * v70);
        if ( !*(_QWORD *)(v74 + 240) )
        {
          v350[1] = (void *)0x100000000LL;
          v347.m128i_i64[1] = 0;
          v347.m128i_i64[0] = (__int64)&unk_49DD210;
          src = 0;
          v350[2] = a1 + 10;
          v349 = 0;
          v350[0] = 0;
          sub_CB5980((__int64)&v347, 0, 0, 0);
          v210 = *(_QWORD *)(v280 + 8 * v70);
          v6 = *(_QWORD *)(v210 + 168);
          v211 = sub_CB6200((__int64)&v347, (unsigned __int8 *)v6, *(_QWORD *)(v210 + 176));
          v212 = *(__m128i **)(v211 + 32);
          if ( *(_QWORD *)(v211 + 24) - (_QWORD)v212 <= 0x3Eu )
          {
            v6 = (__int64)": error: Module does not contain a triple, should be 'nvptx64-'";
            sub_CB6200(v211, ": error: Module does not contain a triple, should be 'nvptx64-'", 0x3Fu);
          }
          else
          {
            v213 = _mm_load_si128((const __m128i *)&xmmword_42818E0);
            qmemcpy(&v212[3], "d be 'nvptx64-'", 15);
            *v212 = v213;
            v212[1] = _mm_load_si128((const __m128i *)&xmmword_42818F0);
            v212[2] = _mm_load_si128((const __m128i *)&xmmword_4281900);
            *(_QWORD *)(v211 + 32) += 63LL;
          }
          *a2 = 9;
          v347.m128i_i64[0] = (__int64)&unk_49DD210;
          sub_CB5840((__int64)&v347);
          goto LABEL_372;
        }
        v75 = *(_QWORD *)(v280 + v67);
        v347.m128i_i64[0] = (__int64)&src;
        sub_2257AB0(v347.m128i_i64, *(_BYTE **)(v75 + 232), *(_QWORD *)(v75 + 232) + *(_QWORD *)(v75 + 240));
        v350[0] = *(void **)(v75 + 264);
        v350[1] = *(void **)(v75 + 272);
        v350[2] = *(void **)(v75 + 280);
        p_src = *(_BYTE **)(v74 + 232);
        if ( (void (__fastcall **)(_QWORD, _QWORD, _QWORD))v347.m128i_i64[0] == &src )
        {
          v77 = v347.m128i_i64[1];
          if ( v347.m128i_i64[1] )
          {
            if ( v347.m128i_i64[1] == 1 )
              *p_src = (_BYTE)src;
            else
              memcpy(p_src, &src, v347.m128i_u64[1]);
            v77 = v347.m128i_i64[1];
            p_src = *(_BYTE **)(v74 + 232);
          }
          *(_QWORD *)(v74 + 240) = v77;
          p_src[v77] = 0;
          p_src = (_BYTE *)v347.m128i_i64[0];
          goto LABEL_115;
        }
        v71 = src;
        v72 = v347.m128i_i64[1];
        if ( p_src == (_BYTE *)(v74 + 248) )
        {
          *(_QWORD *)(v74 + 232) = v347.m128i_i64[0];
          *(_QWORD *)(v74 + 240) = v72;
          *(_QWORD *)(v74 + 248) = v71;
        }
        else
        {
          v73 = *(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(v74 + 248);
          *(_QWORD *)(v74 + 232) = v347.m128i_i64[0];
          *(_QWORD *)(v74 + 240) = v72;
          *(_QWORD *)(v74 + 248) = v71;
          if ( p_src )
          {
            v347.m128i_i64[0] = (__int64)p_src;
            src = v73;
LABEL_115:
            v347.m128i_i64[1] = 0;
            *p_src = 0;
            *(void **)(v74 + 264) = v350[0];
            *(void **)(v74 + 272) = v350[1];
            *(void **)(v74 + 280) = v350[2];
            if ( (void (__fastcall **)(_QWORD, _QWORD, _QWORD))v347.m128i_i64[0] != &src )
              j_j___libc_free_0(v347.m128i_u64[0]);
            sub_BA9570(*(_QWORD *)(v280 + 8 * v70), *(_QWORD *)(v280 + v67) + 312LL);
            goto LABEL_118;
          }
        }
        v347.m128i_i64[0] = (__int64)&src;
        p_src = &src;
        goto LABEL_115;
      }
LABEL_353:
      v323 = 0;
      v6 = (__int64)v287;
      s = (char *)v324;
      LOBYTE(v324[0]) = 0;
      v145 = (char *)sub_3099E10(&v280, v287, &v304, &s, a5);
      if ( !v145 )
      {
        v350[1] = (void *)0x100000000LL;
        v347.m128i_i64[1] = 0;
        v347.m128i_i64[0] = (__int64)&unk_49DD210;
        src = 0;
        v349 = 0;
        v350[2] = a1 + 10;
        v350[0] = 0;
        sub_CB5980((__int64)&v347, 0, 0, 0);
        v241 = sub_CB6200((__int64)&v347, (unsigned __int8 *)s, v323);
        v242 = *(char **)(v241 + 32);
        if ( *(_QWORD *)(v241 + 24) - (_QWORD)v242 <= 0xDu )
        {
          v241 = sub_CB6200(v241, (unsigned __int8 *)": link error: ", 0xEu);
        }
        else
        {
          qmemcpy(v242, ": link error: ", 0xEu);
          *(_QWORD *)(v241 + 32) += 14LL;
        }
        v6 = (__int64)v304;
        sub_CB6200(v241, (unsigned __int8 *)v304, v305);
        *a2 = 9;
        v347.m128i_i64[0] = (__int64)&unk_49DD210;
        sub_CB5840((__int64)&v347);
        if ( s != (char *)v324 )
        {
          v6 = v324[0] + 1;
          j_j___libc_free_0((unsigned __int64)s);
        }
LABEL_372:
        if ( v304 != (char *)v306 )
        {
          v6 = v306[0] + 1;
          j_j___libc_free_0((unsigned __int64)v304);
        }
        goto LABEL_281;
      }
      if ( v305 )
      {
        v6 = (__int64)v304;
        sub_2241490((unsigned __int64 *)a1 + 10, v304, v305);
      }
      if ( s != (char *)v324 )
      {
        v6 = v324[0] + 1;
        j_j___libc_free_0((unsigned __int64)s);
      }
      if ( v304 != (char *)v306 )
      {
        v6 = v306[0] + 1;
        j_j___libc_free_0((unsigned __int64)v304);
      }
    }
    sub_A84F90(v145);
  }
  else
  {
    v145 = 0;
  }
  if ( (a3 & 1) == 0 )
  {
LABEL_276:
    v144 = 0;
    v267 = 0;
    *a2 = 0;
    goto LABEL_232;
  }
  if ( (unsigned __int8)sub_2C72F80() )
  {
    v6 = 0;
    v350[1] = (void *)0x100000000LL;
    v347.m128i_i64[1] = 0;
    v347.m128i_i64[0] = (__int64)&unk_49DD210;
    src = 0;
    v349 = 0;
    v350[2] = a1 + 10;
    v350[0] = 0;
    sub_CB5980((__int64)&v347, 0, 0, 0);
    s = 0;
    sub_CEAF80((__int64 *)&s);
    v234 = (__m128i *)v350[0];
    if ( (unsigned __int64)(v349 - (char *)v350[0]) <= 0x15 )
    {
      v6 = (__int64)"builtins: link error: ";
      v236 = (__m128i *)sub_CB6200((__int64)&v347, "builtins: link error: ", 0x16u);
    }
    else
    {
      v235 = _mm_load_si128((const __m128i *)&xmmword_4281910);
      *((_DWORD *)v350[0] + 4) = 1919906418;
      v234[1].m128i_i16[2] = 8250;
      v236 = &v347;
      *v234 = v235;
      v350[0] = (char *)v350[0] + 22;
    }
    v237 = (unsigned __int8 *)s;
    if ( !s )
      goto LABEL_434;
    v238 = strlen(s);
    v239 = (void *)v236[2].m128i_i64[0];
    v240 = v238;
    if ( v238 > v236[1].m128i_i64[1] - (__int64)v239 )
    {
      v6 = (__int64)v237;
      sub_CB6200((__int64)v236, v237, v238);
      v237 = (unsigned __int8 *)s;
    }
    else
    {
      if ( !v238 )
        goto LABEL_433;
      v6 = (__int64)v237;
      memcpy(v239, v237, v238);
      v236[2].m128i_i64[0] += v240;
      v237 = (unsigned __int8 *)s;
    }
    if ( !v237 )
    {
LABEL_434:
      *a2 = 9;
      v347.m128i_i64[0] = (__int64)&unk_49DD210;
      sub_CB5840((__int64)&v347);
      v267 = 0;
      goto LABEL_365;
    }
LABEL_433:
    j_j___libc_free_0_0((unsigned __int64)v237);
    goto LABEL_434;
  }
  v162 = (_BYTE *)*((_QWORD *)v145 + 21);
  v163 = *((_QWORD *)v145 + 22);
  v164 = sub_22077B0(0x370u);
  v165 = v164;
  if ( v164 )
    sub_BA8740(v164, v162, v163, (__int64)a4);
  sub_E48650(&s, v165);
  v304 = v145;
  src = 0;
  v166 = sub_E49E40((__int64 *)&s, (__int64)&v304, 4u, &v347);
  v169 = v304;
  v170 = v166;
  if ( v304 )
  {
    sub_BA9C10((_QWORD **)v304, (__int64)&v304, v167, v168);
    j_j___libc_free_0((unsigned __int64)v169);
  }
  if ( src )
    src(&v347, &v347, 3);
  if ( v170 )
  {
    v350[1] = (void *)0x100000000LL;
    v347.m128i_i64[1] = 0;
    v347.m128i_i64[0] = (__int64)&unk_49DD210;
    src = 0;
    v349 = 0;
    v350[2] = a1 + 10;
    v350[0] = 0;
    sub_CB5980((__int64)&v347, 0, 0, 0);
    v304 = 0;
    sub_CEAF80((__int64 *)&v304);
    v243 = (__m128i *)v350[0];
    if ( (unsigned __int64)(v349 - (char *)v350[0]) <= 0x15 )
    {
      v245 = (__m128i *)sub_CB6200((__int64)&v347, "builtins: link error: ", 0x16u);
    }
    else
    {
      v244 = _mm_load_si128((const __m128i *)&xmmword_4281910);
      *((_DWORD *)v350[0] + 4) = 1919906418;
      v243[1].m128i_i16[2] = 8250;
      v245 = &v347;
      *v243 = v244;
      v350[0] = (char *)v350[0] + 22;
    }
    v246 = (unsigned __int8 *)v304;
    if ( !v304 )
      goto LABEL_450;
    v247 = strlen(v304);
    v248 = (void *)v245[2].m128i_i64[0];
    v249 = v247;
    if ( v247 > v245[1].m128i_i64[1] - (__int64)v248 )
    {
      sub_CB6200((__int64)v245, v246, v247);
      v246 = (unsigned __int8 *)v304;
    }
    else
    {
      if ( !v247 )
        goto LABEL_449;
      memcpy(v248, v246, v247);
      v245[2].m128i_i64[0] += v249;
      v246 = (unsigned __int8 *)v304;
    }
    if ( !v246 )
    {
LABEL_450:
      v175 = (_QWORD **)v165;
      v165 = 0;
      *a2 = 9;
      v347.m128i_i64[0] = (__int64)&unk_49DD210;
      sub_CB5840((__int64)&v347);
      v267 = 0;
      goto LABEL_303;
    }
LABEL_449:
    j_j___libc_free_0_0((unsigned __int64)v246);
    goto LABEL_450;
  }
  v171 = a1[3];
  v274 = a1[4];
  if ( v274 == v171 )
  {
LABEL_338:
    sub_B848C0(&v304);
    v201 = (__int64 *)sub_2D028E0(a5 + 208);
    sub_B8B500((__int64)&v304, v201, 1u);
    if ( v165 && (unsigned __int8)sub_B89FE0((__int64)&v304, v165) && !LOBYTE(qword_502D788[8]) )
    {
      sub_B848C0(&v347);
      v250 = (__int64 *)sub_2D1B100();
      sub_B8B500((__int64)&v347, v250, 0);
      sub_B89FE0((__int64)&v347, v165);
      sub_B82680(&v347);
    }
    sub_B82680(&v304);
    v202 = v332;
    if ( v332 )
    {
      v203 = v331;
      v204 = &v331[2 * v332];
      do
      {
        if ( *v203 != -8192 && *v203 != -4096 )
        {
          v205 = v203[1];
          if ( v205 )
            sub_B91220((__int64)(v203 + 1), v205);
        }
        v203 += 2;
      }
      while ( v204 != v203 );
      v202 = v332;
    }
    sub_C7D6A0((__int64)v331, 16 * v202, 8);
    if ( (_DWORD)v329 )
    {
      v229 = sub_1061AC0();
      v230 = sub_1061AD0();
      v206 = v327;
      v231 = v230;
      v207 = (unsigned int)v329;
      v232 = (__int64 *)v327;
      v233 = (__int64 *)(v327 + 8LL * (unsigned int)v329);
      if ( (__int64 *)v327 != v233 )
      {
        do
        {
          if ( !sub_1061B40(*v232, v229) )
            sub_1061B40(*v232, v231);
          ++v232;
        }
        while ( v233 != v232 );
        v206 = v327;
        v207 = (unsigned int)v329;
      }
    }
    else
    {
      v206 = v327;
      v207 = 0;
    }
    v145 = (char *)v165;
    sub_C7D6A0(v206, 8 * v207, 8);
    v6 = 8LL * (unsigned int)v325;
    sub_C7D6A0(v324[0], v6, 8);
    goto LABEL_276;
  }
  while ( 1 )
  {
    v172 = *(_QWORD *)(v171 + 24);
    if ( !v172 )
      goto LABEL_337;
    sub_C7DA90(&v278, *(_QWORD *)(v171 + 16), v172, *(const char **)v171, *(const char **)(v171 + 8), 1);
    v173 = v278;
    v174 = (_BYTE *)v278[1];
    if ( v174 == (_BYTE *)v278[2] )
      goto LABEL_300;
    if ( *v174 == 0xDE )
    {
      if ( v174[1] != 0xC0 || v174[2] != 23 || v174[3] != 11 )
      {
LABEL_300:
        v267 = 0;
        *a2 = 9;
        goto LABEL_301;
      }
    }
    else if ( *v174 != 66 || v174[1] != 67 || v174[2] != 0xC0 || v174[3] != 0xDE )
    {
      goto LABEL_300;
    }
    v297 = (unsigned __int8 *)&v299;
    memset(v350, 0, 0x58u);
    v298 = 0;
    LOBYTE(v299) = 0;
    sub_C7EC60(&v304, v278);
    sub_A011E0((__int64)&s2, (__int64)a4, 0, 0, (__int64)&v347, v214, a6, (const __m128i *)v304, v305);
    if ( LOBYTE(v350[10]) )
    {
      LOBYTE(v350[10]) = 0;
      if ( v350[8] )
        ((void (__fastcall *)(void **, void **, __int64))v350[8])(&v350[6], &v350[6], 3);
    }
    if ( LOBYTE(v350[5]) )
    {
      LOBYTE(v350[5]) = 0;
      if ( v350[3] )
        ((void (__fastcall *)(void **, void **, __int64))v350[3])(&v350[1], &v350[1], 3);
    }
    if ( LOBYTE(v350[0]) )
    {
      LOBYTE(v350[0]) = 0;
      if ( src )
        src(&v347, &v347, 3);
    }
    v215 = v284 & 1;
    LOBYTE(v284) = (2 * (v284 & 1)) | v284 & 0xFD;
    if ( v215 || (v216 = (char *)s2, s2 = 0, !v216) )
    {
      v267 = 0;
      goto LABEL_393;
    }
    if ( !v165 || v298 )
      break;
    if ( (unsigned __int8)sub_CF2A80(v165, (__int64)v216, (__int64)&v297) )
    {
      sub_BA97D0(&v347, (__int64)v216);
      if ( (v347.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        v347.m128i_i64[0] = v347.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL | 1;
        sub_C63C30(&v347, (__int64)v216);
      }
    }
    v185 = *(_BYTE **)(v165 + 232);
    v186 = *(_QWORD *)(v165 + 240);
    v347.m128i_i64[0] = (__int64)&src;
    sub_2257AB0(v347.m128i_i64, v185, (__int64)&v185[v186]);
    v350[0] = *(void **)(v165 + 264);
    v350[1] = *(void **)(v165 + 272);
    v350[2] = *(void **)(v165 + 280);
    v187 = (char *)*((_QWORD *)v216 + 29);
    if ( (void (__fastcall **)(_QWORD, _QWORD, _QWORD))v347.m128i_i64[0] == &src )
    {
      v217 = v347.m128i_i64[1];
      if ( v347.m128i_i64[1] )
      {
        if ( v347.m128i_i64[1] == 1 )
          *v187 = (char)src;
        else
          memcpy(v187, &src, v347.m128i_u64[1]);
        v217 = v347.m128i_i64[1];
        v187 = (char *)*((_QWORD *)v216 + 29);
      }
      *((_QWORD *)v216 + 30) = v217;
      v187[v217] = 0;
      v187 = (char *)v347.m128i_i64[0];
      goto LABEL_324;
    }
    v188 = src;
    v189 = v347.m128i_i64[1];
    if ( v187 == v216 + 248 )
    {
      *((_QWORD *)v216 + 29) = v347.m128i_i64[0];
      *((_QWORD *)v216 + 30) = v189;
      *((_QWORD *)v216 + 31) = v188;
LABEL_414:
      v347.m128i_i64[0] = (__int64)&src;
      v187 = (char *)&src;
      goto LABEL_324;
    }
    v190 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))*((_QWORD *)v216 + 31);
    *((_QWORD *)v216 + 29) = v347.m128i_i64[0];
    *((_QWORD *)v216 + 30) = v189;
    *((_QWORD *)v216 + 31) = v188;
    if ( !v187 )
      goto LABEL_414;
    v347.m128i_i64[0] = (__int64)v187;
    src = v190;
LABEL_324:
    v347.m128i_i64[1] = 0;
    *v187 = 0;
    *((void **)v216 + 33) = v350[0];
    *((void **)v216 + 34) = v350[1];
    *((void **)v216 + 35) = v350[2];
    if ( (void (__fastcall **)(_QWORD, _QWORD, _QWORD))v347.m128i_i64[0] != &src )
      j_j___libc_free_0(v347.m128i_u64[0]);
    sub_BA9570((__int64)v216, v165 + 312);
    v191 = *(_BYTE **)(v165 + 232);
    v192 = *(_QWORD *)(v165 + 240);
    v304 = (char *)v306;
    sub_2257AB0((__int64 *)&v304, v191, (__int64)&v191[v192]);
    v193 = *(_DWORD *)(v165 + 276) == 21;
    v194 = (__int64)v216;
    v307[0] = *(_QWORD *)(v165 + 264);
    v307[1] = *(_QWORD *)(v165 + 272);
    v308 = *(_QWORD *)(v165 + 280);
    v195 = sub_3099D80(&s, v216, &v297, v193);
    v199 = v306;
    if ( v195 )
    {
      v350[1] = (void *)0x100000000LL;
      v347.m128i_i64[1] = 0;
      v347.m128i_i64[0] = (__int64)&unk_49DD210;
      src = 0;
      v349 = 0;
      v350[2] = a1 + 10;
      v350[0] = 0;
      sub_CB5980((__int64)&v347, 0, 0, 0);
      v279[0] = 0;
      sub_CEAF80((__int64 *)v279);
      if ( (unsigned __int64)(v349 - (char *)v350[0]) <= 0x15 )
      {
        v251 = (__m128i *)sub_CB6200((__int64)&v347, "builtins: link error: ", 0x16u);
      }
      else
      {
        qmemcpy(v350[0], "builtins: link error: ", 0x16u);
        v251 = &v347;
        v350[0] = (char *)v350[0] + 22;
      }
      v221 = v297;
      sub_CB6200((__int64)v251, v297, v298);
      v252 = (unsigned __int8 *)v279[0];
      if ( v279[0] )
      {
        v253 = strlen(v279[0]);
        v254 = v253;
        if ( v253 > v349 - (char *)v350[0] )
        {
          v221 = v252;
          sub_CB6200((__int64)&v347, v252, v253);
          v252 = (unsigned __int8 *)v279[0];
        }
        else if ( v253 )
        {
          v221 = v252;
          memcpy(v350[0], v252, v253);
          v350[0] = (char *)v350[0] + v254;
          v252 = (unsigned __int8 *)v279[0];
        }
        if ( v252 )
          j_j___libc_free_0_0((unsigned __int64)v252);
      }
      *a2 = 9;
      v347.m128i_i64[0] = (__int64)&unk_49DD210;
      sub_CB5840((__int64)&v347);
      if ( v304 != (char *)v306 )
      {
        v221 = (unsigned __int8 *)(v306[0] + 1);
        j_j___libc_free_0((unsigned __int64)v304);
      }
      v267 = 0;
      goto LABEL_396;
    }
    if ( v304 != (char *)v306 )
    {
      v194 = v306[0] + 1;
      j_j___libc_free_0((unsigned __int64)v304);
    }
    if ( (v284 & 2) != 0 )
      goto LABEL_410;
    v200 = s2;
    if ( (v284 & 1) != 0 )
    {
      if ( s2 )
        (*(void (__fastcall **)(void *, __int64, __int64, __int64, __int64, unsigned __int64 *))(*(_QWORD *)s2 + 8LL))(
          s2,
          v194,
          v196,
          v197,
          v198,
          v199);
    }
    else if ( s2 )
    {
      sub_BA9C10((_QWORD **)s2, v194, v196, v197);
      v194 = 880;
      j_j___libc_free_0((unsigned __int64)v200);
    }
    if ( v297 != (unsigned __int8 *)&v299 )
    {
      v194 = v299 + 1;
      j_j___libc_free_0((unsigned __int64)v297);
    }
    if ( v278 )
      (*(void (__fastcall **)(_QWORD *, __int64, __int64, __int64, __int64, unsigned __int64 *))(*v278 + 8LL))(
        v278,
        v194,
        v196,
        v197,
        v198,
        v199);
LABEL_337:
    v171 += 32;
    if ( v274 == v171 )
      goto LABEL_338;
  }
  v267 = v216;
LABEL_393:
  v350[1] = (void *)0x100000000LL;
  v347.m128i_i64[1] = 0;
  v347.m128i_i64[0] = (__int64)&unk_49DD210;
  src = 0;
  v349 = 0;
  v350[2] = a1 + 10;
  v350[0] = 0;
  sub_CB5980((__int64)&v347, 0, 0, 0);
  v218 = (__m128i *)v350[0];
  if ( (unsigned __int64)(v349 - (char *)v350[0]) <= 0x15 )
  {
    v220 = (__m128i *)sub_CB6200((__int64)&v347, "builtins: link error: ", 0x16u);
  }
  else
  {
    v219 = _mm_load_si128((const __m128i *)&xmmword_4281910);
    *((_DWORD *)v350[0] + 4) = 1919906418;
    v218[1].m128i_i16[2] = 8250;
    v220 = &v347;
    *v218 = v219;
    v350[0] = (char *)v350[0] + 22;
  }
  v221 = v297;
  sub_CB6200((__int64)v220, v297, v298);
  *a2 = 9;
  v347.m128i_i64[0] = (__int64)&unk_49DD210;
  sub_CB5840((__int64)&v347);
LABEL_396:
  if ( (v284 & 2) != 0 )
LABEL_410:
    sub_904700(&s2);
  v224 = s2;
  if ( (v284 & 1) != 0 )
  {
    if ( s2 )
      (*(void (__fastcall **)(void *))(*(_QWORD *)s2 + 8LL))(s2);
  }
  else if ( s2 )
  {
    sub_BA9C10((_QWORD **)s2, (__int64)v221, v222, v223);
    j_j___libc_free_0((unsigned __int64)v224);
  }
  if ( v297 != (unsigned __int8 *)&v299 )
    j_j___libc_free_0((unsigned __int64)v297);
  v173 = v278;
  if ( v278 )
LABEL_301:
    (*(void (__fastcall **)(_QWORD *))(*v173 + 8LL))(v173);
  v175 = 0;
LABEL_303:
  v176 = v332;
  if ( v332 )
  {
    v177 = v331;
    v178 = &v331[2 * v332];
    do
    {
      if ( *v177 != -4096 && *v177 != -8192 )
      {
        v179 = v177[1];
        if ( v179 )
          sub_B91220((__int64)(v177 + 1), v179);
      }
      v177 += 2;
    }
    while ( v178 != v177 );
    v176 = v332;
  }
  sub_C7D6A0((__int64)v331, 16 * v176, 8);
  if ( (_DWORD)v329 )
  {
    v277 = sub_1061AC0();
    v180 = sub_1061AD0();
    v181 = (__int64 *)v327;
    v182 = v180;
    v183 = 8LL * (unsigned int)v329;
    v184 = v327 + v183;
    if ( v327 != v327 + v183 )
    {
      do
      {
        if ( !sub_1061B40(*v181, v277) )
          sub_1061B40(*v181, v182);
        ++v181;
      }
      while ( (__int64 *)v184 != v181 );
      v184 = v327;
      v183 = 8LL * (unsigned int)v329;
    }
  }
  else
  {
    v184 = v327;
    v183 = 0;
  }
  sub_C7D6A0(v184, v183, 8);
  v6 = 8LL * (unsigned int)v325;
  sub_C7D6A0(v324[0], v6, 8);
  if ( v175 )
  {
    sub_BA9C10(v175, v6, v208, v209);
    v6 = 880;
    j_j___libc_free_0((unsigned __int64)v175);
  }
  v145 = (char *)v165;
LABEL_365:
  v144 = v145;
  v145 = 0;
LABEL_232:
  if ( v287[0] )
  {
    v6 = (__int64)v290 - v287[0];
    j_j___libc_free_0(v287[0]);
  }
  sub_22581A0(v293);
  if ( v280 )
  {
    v6 = (__int64)&v282[-v280];
    j_j___libc_free_0(v280);
  }
  if ( v267 )
  {
    sub_BA9C10((_QWORD **)v267, v6, v146, v147);
    v6 = 880;
    j_j___libc_free_0((unsigned __int64)v267);
  }
  if ( v144 )
  {
    sub_BA9C10((_QWORD **)v144, v6, v146, v147);
    j_j___libc_free_0((unsigned __int64)v144);
  }
  return v145;
}
