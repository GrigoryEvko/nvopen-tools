// Function: sub_12C06E0
// Address: 0x12c06e0
//
__int64 __fastcall sub_12C06E0(_QWORD *a1, _DWORD *a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rsi
  __int64 v6; // r12
  char *v7; // r13
  char **v8; // rsi
  char *(*v9)(); // rax
  void **v10; // rax
  __int64 v11; // rdx
  _BYTE *v12; // rsi
  __int64 v13; // rax
  int *v14; // r12
  int *v15; // r14
  void *v16; // r13
  size_t v17; // r12
  int *v18; // rbx
  int *v19; // r14
  size_t v20; // r15
  size_t v21; // rdx
  int v22; // eax
  __int64 v23; // r15
  size_t v24; // r15
  size_t v25; // rdx
  signed __int64 v26; // rax
  size_t v27; // r15
  __int64 *v28; // rbx
  __int64 *v29; // r12
  size_t v30; // r14
  size_t v31; // rdx
  int v32; // eax
  __int64 v33; // r14
  size_t v34; // rbx
  _QWORD *v35; // r15
  size_t v36; // r14
  size_t v37; // rdx
  signed __int64 v38; // rax
  __int64 *v39; // rax
  __int64 v40; // rax
  unsigned __int64 v41; // r12
  _BYTE *v42; // rsi
  int v43; // eax
  __int64 *v44; // rsi
  int v45; // edx
  char v46; // cl
  __int64 v47; // rax
  __int64 v48; // rcx
  __int64 v49; // rax
  _BYTE *v50; // rbx
  _BYTE *v51; // r12
  _BYTE *v52; // rdi
  _QWORD *v53; // rax
  __int64 v54; // rcx
  __int64 v55; // r13
  __int64 v56; // rbx
  int v57; // r15d
  __int64 v58; // r14
  __int64 *v59; // rcx
  __int64 v60; // r13
  __int64 v61; // rdx
  void **v62; // rdi
  void **v63; // rax
  size_t v64; // rdi
  __m128i *v65; // rsi
  __m128i *v66; // r9
  __int64 v67; // r12
  __int64 v68; // rax
  _QWORD *v69; // rdx
  _QWORD *v70; // rbx
  __int64 v71; // rax
  _BYTE *v72; // rsi
  size_t v73; // rdx
  __int64 v74; // r15
  __int64 v75; // rax
  unsigned int v76; // eax
  __int64 v77; // rdx
  unsigned __int64 v78; // r12
  __int64 v79; // rax
  __int64 *v80; // r14
  void *v81; // r13
  size_t v82; // rbx
  __int64 *v83; // r12
  size_t v84; // r15
  size_t v85; // rdx
  int v86; // eax
  __int64 v87; // r15
  unsigned __int64 v88; // r15
  size_t v89; // rdx
  int v90; // eax
  __int64 v91; // rbx
  __int64 v92; // rax
  __int64 *v93; // r13
  void *v94; // r14
  size_t v95; // rbx
  __int64 *v96; // r12
  size_t v97; // r15
  size_t v98; // rdx
  int v99; // eax
  __int64 v100; // r15
  unsigned __int64 v101; // r15
  size_t v102; // rdx
  int v103; // eax
  __int64 v104; // rbx
  __int64 v105; // rax
  __int64 v106; // r12
  __int64 v107; // r15
  unsigned __int64 v108; // rbx
  __int64 v109; // rbx
  __int64 v110; // r13
  __int64 v111; // rsi
  unsigned __int64 v112; // r14
  __int64 v113; // rax
  __int64 v114; // rax
  __int64 v115; // rsi
  __int64 v116; // r13
  __int64 v117; // r15
  unsigned __int64 v118; // r9
  __m128i *v119; // rdi
  size_t v120; // rbx
  unsigned __int64 v121; // rax
  __int64 **v122; // r15
  __m128i v123; // xmm0
  __int64 v124; // rax
  _DWORD *v125; // rdx
  __int64 v126; // rdi
  __int64 v127; // rax
  void *v128; // rdx
  __int64 v129; // rdi
  __int64 v130; // rax
  void *v131; // rdx
  __int64 *v132; // r14
  void *v133; // r15
  size_t v134; // r12
  __int64 *v135; // rbx
  __int64 *v136; // r14
  size_t v137; // r13
  size_t v138; // rdx
  int v139; // eax
  __int64 v140; // r13
  size_t v141; // rbx
  unsigned __int64 v142; // rcx
  size_t v143; // rdx
  int v144; // eax
  __int64 v145; // rbx
  __int64 v146; // rax
  void *v147; // rdx
  __int64 v148; // rdi
  __int64 v149; // rax
  _WORD *v150; // rdx
  __int64 v151; // rax
  __int64 v152; // rdx
  __int64 v153; // rax
  __int64 v154; // rax
  __int64 v155; // rax
  __int64 v156; // rax
  __int64 v157; // rax
  unsigned __int64 *v158; // rdi
  void ***v159; // r12
  size_t v160; // r13
  const char *v161; // rsi
  void ***v162; // rdi
  _BYTE *v163; // rbx
  _BYTE *v164; // r12
  _BYTE *v165; // rdi
  __int64 v166; // r12
  __int64 v167; // r13
  __int64 result; // rax
  __int64 v169; // rax
  __int64 v170; // rsi
  unsigned int v171; // eax
  __int64 v172; // rsi
  int v173; // eax
  __int64 v174; // rax
  _QWORD *v175; // rax
  int v176; // eax
  __int64 *v177; // rax
  unsigned __int64 v178; // rdx
  void ***v179; // rdi
  void ***v180; // rdi
  __m128i *v181; // rax
  __m128i v182; // xmm0
  __m128i v183; // xmm0
  _QWORD *v184; // r14
  __int64 v185; // rdx
  int v186; // r8d
  int v187; // r9d
  char **v188; // rdi
  char *v189; // rax
  _QWORD *v190; // rbx
  _QWORD *v191; // r12
  __int64 v192; // r12
  __int64 v193; // rax
  _QWORD *v194; // rdi
  __int64 v195; // rbx
  _QWORD *v196; // r15
  _QWORD *v197; // r14
  __int64 v198; // rcx
  __int64 v199; // rax
  int v200; // eax
  __int64 v201; // rax
  __int64 v202; // r14
  unsigned __int64 v203; // rax
  _QWORD *v204; // rax
  int v205; // eax
  __int64 v206; // rax
  __int64 v207; // rax
  _BYTE *v208; // rsi
  __int64 v209; // rdx
  char *v210; // rdi
  char *v211; // rax
  char *v212; // rcx
  size_t v213; // rsi
  char *v214; // rdi
  __int64 v215; // rax
  __int64 v216; // rsi
  __int64 v217; // rdx
  void *v218; // r15
  __int64 v219; // rsi
  _QWORD *v220; // rbx
  _QWORD *v221; // r12
  __int64 v222; // r12
  __int64 v223; // rax
  _QWORD *v224; // rdi
  __int64 v225; // rbx
  _QWORD *v226; // r15
  _QWORD *v227; // r14
  __int64 v228; // rdi
  __m128i *v229; // rax
  __m128i v230; // xmm0
  __int64 v231; // rdi
  __m128i *v232; // rax
  __m128i si128; // xmm0
  char *v234; // rax
  __int64 (*v235)(void); // rax
  char v236; // dl
  void **v237; // rbx
  char v238; // al
  size_t v239; // r10
  __int64 v240; // rax
  void *v241; // r12
  __int64 v242; // rdx
  char ***v243; // rdi
  char *v244; // r12
  size_t v245; // rax
  size_t v246; // r14
  __int64 v247; // r8
  char *v248; // rax
  __int64 v249; // rax
  __m128i *v250; // rax
  __m128i v251; // xmm0
  void ***v252; // r14
  char *v253; // r12
  size_t v254; // rax
  void **v255; // rdi
  size_t v256; // r15
  unsigned __int64 v257; // [rsp+8h] [rbp-728h]
  __int64 v258; // [rsp+10h] [rbp-720h]
  unsigned __int64 v259; // [rsp+18h] [rbp-718h]
  __int64 v260; // [rsp+18h] [rbp-718h]
  unsigned __int64 v261; // [rsp+20h] [rbp-710h]
  __int64 v262; // [rsp+20h] [rbp-710h]
  _QWORD *v265; // [rsp+38h] [rbp-6F8h]
  bool v266; // [rsp+40h] [rbp-6F0h]
  unsigned __int64 v267; // [rsp+40h] [rbp-6F0h]
  __int64 v268; // [rsp+40h] [rbp-6F0h]
  _QWORD *v269; // [rsp+48h] [rbp-6E8h]
  int v271; // [rsp+68h] [rbp-6C8h]
  _QWORD *v273; // [rsp+A0h] [rbp-690h]
  __int64 *v274; // [rsp+A8h] [rbp-688h]
  bool v275; // [rsp+A8h] [rbp-688h]
  __int64 v276; // [rsp+A8h] [rbp-688h]
  __int64 v277; // [rsp+A8h] [rbp-688h]
  unsigned __int64 v278; // [rsp+A8h] [rbp-688h]
  unsigned __int64 v279; // [rsp+A8h] [rbp-688h]
  unsigned __int64 v280; // [rsp+A8h] [rbp-688h]
  void **v281; // [rsp+B0h] [rbp-680h]
  __int64 v282; // [rsp+B8h] [rbp-678h]
  unsigned __int64 v283; // [rsp+B8h] [rbp-678h]
  __int64 v284; // [rsp+B8h] [rbp-678h]
  __int64 v285; // [rsp+B8h] [rbp-678h]
  __int64 v286; // [rsp+E0h] [rbp-650h]
  _QWORD *v287; // [rsp+E0h] [rbp-650h]
  void **v288; // [rsp+E8h] [rbp-648h]
  int v289; // [rsp+E8h] [rbp-648h]
  __int64 v290; // [rsp+E8h] [rbp-648h]
  char **v291; // [rsp+F0h] [rbp-640h] BYREF
  char *s; // [rsp+F8h] [rbp-638h] BYREF
  _QWORD *v293; // [rsp+100h] [rbp-630h] BYREF
  _BYTE *v294; // [rsp+108h] [rbp-628h]
  _BYTE *v295; // [rsp+110h] [rbp-620h]
  void *s2; // [rsp+120h] [rbp-610h] BYREF
  size_t v297; // [rsp+128h] [rbp-608h]
  _QWORD v298[2]; // [rsp+130h] [rbp-600h] BYREF
  __int64 v299[4]; // [rsp+140h] [rbp-5F0h] BYREF
  _QWORD v300[2]; // [rsp+160h] [rbp-5D0h] BYREF
  __int64 *v301; // [rsp+170h] [rbp-5C0h]
  __int64 v302; // [rsp+178h] [rbp-5B8h]
  __int64 *v303; // [rsp+180h] [rbp-5B0h]
  __int64 v304; // [rsp+190h] [rbp-5A0h] BYREF
  int v305; // [rsp+198h] [rbp-598h] BYREF
  int *v306; // [rsp+1A0h] [rbp-590h]
  int *v307; // [rsp+1A8h] [rbp-588h]
  int *v308; // [rsp+1B0h] [rbp-580h]
  __int64 v309; // [rsp+1B8h] [rbp-578h]
  __int64 *v310; // [rsp+1C0h] [rbp-570h] BYREF
  __int64 v311; // [rsp+1C8h] [rbp-568h]
  __int64 v312; // [rsp+1D0h] [rbp-560h] BYREF
  void *dest; // [rsp+1D8h] [rbp-558h]
  int v314; // [rsp+1E0h] [rbp-550h]
  _QWORD *v315; // [rsp+1E8h] [rbp-548h]
  char **v316; // [rsp+1F0h] [rbp-540h] BYREF
  __int64 v317; // [rsp+1F8h] [rbp-538h]
  char *v318; // [rsp+200h] [rbp-530h] BYREF
  void *v319; // [rsp+208h] [rbp-528h]
  _QWORD v320[2]; // [rsp+210h] [rbp-520h] BYREF
  __int64 v321; // [rsp+220h] [rbp-510h]
  int v322; // [rsp+228h] [rbp-508h]
  void *src; // [rsp+230h] [rbp-500h]
  size_t n; // [rsp+238h] [rbp-4F8h]
  _QWORD v325[2]; // [rsp+240h] [rbp-4F0h] BYREF
  _QWORD *v326; // [rsp+250h] [rbp-4E0h]
  __int64 v327; // [rsp+258h] [rbp-4D8h]
  _QWORD v328[2]; // [rsp+260h] [rbp-4D0h] BYREF
  __int64 v329; // [rsp+270h] [rbp-4C0h]
  __int64 v330; // [rsp+278h] [rbp-4B8h]
  __int64 v331; // [rsp+280h] [rbp-4B0h]
  _BYTE *v332; // [rsp+288h] [rbp-4A8h]
  __int64 v333; // [rsp+290h] [rbp-4A0h]
  _BYTE v334[200]; // [rsp+298h] [rbp-498h] BYREF
  char *v335; // [rsp+360h] [rbp-3D0h] BYREF
  size_t v336; // [rsp+368h] [rbp-3C8h]
  char *v337; // [rsp+370h] [rbp-3C0h] BYREF
  __int64 v338; // [rsp+378h] [rbp-3B8h]
  int v339; // [rsp+380h] [rbp-3B0h]
  _QWORD *v340; // [rsp+388h] [rbp-3A8h]
  void **p_s2; // [rsp+530h] [rbp-200h] BYREF
  size_t v342; // [rsp+538h] [rbp-1F8h]
  __m128i *v343; // [rsp+540h] [rbp-1F0h] BYREF
  __m128i *v344; // [rsp+548h] [rbp-1E8h]
  int v345; // [rsp+550h] [rbp-1E0h]
  _QWORD *v346; // [rsp+558h] [rbp-1D8h]
  _QWORD *v347; // [rsp+560h] [rbp-1D0h]
  int v348; // [rsp+570h] [rbp-1C0h]
  _QWORD *v349; // [rsp+580h] [rbp-1B0h]
  unsigned int v350; // [rsp+590h] [rbp-1A0h]

  v5 = *(unsigned __int8 *)(a5 + 240);
  v271 = a4;
  v307 = &v305;
  v308 = &v305;
  v293 = 0;
  v294 = 0;
  v295 = 0;
  v305 = 0;
  v306 = 0;
  v309 = 0;
  v300[0] = 0;
  v300[1] = 0;
  v301 = 0;
  v302 = 0;
  v303 = 0;
  sub_16033C0(a4, v5);
  v265 = (_QWORD *)a1[1];
  if ( (_QWORD *)*a1 == v265 )
  {
LABEL_91:
    sub_1C3E900();
    v53 = v293;
    v54 = (v294 - (_BYTE *)v293) >> 3;
    v289 = v54;
    if ( !(_DWORD)v54 )
    {
      v167 = 0;
      goto LABEL_314;
    }
    v55 = (unsigned int)v54;
    v56 = 0;
    while ( 1 )
    {
      v57 = v56;
      v58 = v56;
      if ( (unsigned int)sub_2241AC0(v53[v56] + 240LL, off_4CD49B0) )
        break;
      v53 = v293;
      ++v56;
      v59 = v293;
      if ( v55 == v56 )
      {
        v58 = 0;
        v57 = 0;
        goto LABEL_96;
      }
    }
    v59 = v293;
LABEL_96:
    if ( !*(_QWORD *)(v59[v58] + 248) )
    {
      v345 = 1;
      v344 = 0;
      v343 = 0;
      p_s2 = (void **)&unk_49EFBE0;
      v342 = 0;
      v346 = a1 + 10;
      v231 = sub_16E7EE0(&p_s2, *(_QWORD *)(v59[v58] + 176), *(_QWORD *)(v59[v58] + 184));
      v232 = *(__m128i **)(v231 + 24);
      if ( *(_QWORD *)(v231 + 16) - (_QWORD)v232 <= 0x3Eu )
      {
        sub_16E7EE0(v231, ": error: Module does not contain a triple, should be 'nvptx64-'", 63);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_42818E0);
        qmemcpy(&v232[3], "d be 'nvptx64-'", 15);
        *v232 = si128;
        v232[1] = _mm_load_si128((const __m128i *)&xmmword_42818F0);
        v232[2] = _mm_load_si128((const __m128i *)&xmmword_4281900);
        *(_QWORD *)(v231 + 24) += 63LL;
      }
      v166 = 0;
      v167 = 0;
      *a2 = 9;
      sub_16E7BC0(&p_s2);
      v281 = 0;
      goto LABEL_270;
    }
    if ( v289 == 1 )
    {
      v167 = *v59;
LABEL_354:
      sub_1C3DFC0(v167);
LABEL_314:
      if ( (a3 & 1) == 0 )
      {
LABEL_315:
        v166 = 0;
        v281 = 0;
        *a2 = 0;
        goto LABEL_270;
      }
      if ( !(unsigned __int8)sub_1BF83F0() )
      {
        sub_167C560(&p_s2, v167);
        v184 = (_QWORD *)a1[3];
        v287 = (_QWORD *)a1[4];
        if ( v287 != v184 )
        {
          while ( 1 )
          {
            v185 = v184[3];
            if ( !v185 )
              goto LABEL_399;
            sub_16C2450(&v291, v184[2], v185, *v184, v184[1], 1);
            v188 = v291;
            v189 = v291[1];
            if ( v189 == v291[2] )
              goto LABEL_331;
            if ( *v189 == -34 )
            {
              if ( v189[1] != -64 || v189[2] != 23 || v189[3] != 11 )
              {
LABEL_331:
                v281 = 0;
                *a2 = 9;
                goto LABEL_332;
              }
            }
            else if ( *v189 != 66 || v189[1] != 67 || v189[2] != -64 || v189[3] != -34 )
            {
              goto LABEL_331;
            }
            LOBYTE(v312) = 0;
            v310 = &v312;
            v311 = 0;
            v234 = v291[2];
            v335 = v291[1];
            v336 = v234 - v335;
            v235 = (__int64 (*)(void))*((_QWORD *)*v291 + 2);
            if ( (char *)v235 == (char *)sub_12BCB10 )
            {
              v338 = 14;
              v337 = "Unknown buffer";
            }
            else
            {
              v337 = (char *)v235();
              v338 = v242;
            }
            sub_15099C0((unsigned int)&s2, v271, 0, 0, v186, v187, (__int64)v335, v336, (__int64)v337, v338);
            v236 = v297 & 1;
            LOBYTE(v297) = (2 * (v297 & 1)) | v297 & 0xFD;
            if ( v236 || (v237 = (void **)s2, s2 = 0, !v237) )
            {
              v281 = 0;
LABEL_445:
              v339 = 1;
              v338 = 0;
              v335 = (char *)&unk_49EFBE0;
              v337 = 0;
              v336 = 0;
              v340 = a1 + 10;
              v240 = sub_16E7EE0(&v335, "builtins: link error: ", 22);
              v216 = (__int64)v310;
              sub_16E7EE0(v240, v310, v311);
              *a2 = 9;
              sub_16E7BC0(&v335);
              goto LABEL_446;
            }
            if ( !v167 || v311 )
            {
              v281 = v237;
              goto LABEL_445;
            }
            v238 = sub_1CCEBE0(v167, v237, &v310);
            v239 = 0;
            if ( v238 )
            {
              sub_16330A0(&v335, v237);
              v239 = 0;
              if ( ((unsigned __int64)v335 & 0xFFFFFFFFFFFFFFFELL) != 0 )
              {
                v335 = (char *)((unsigned __int64)v335 & 0xFFFFFFFFFFFFFFFELL | 1);
                sub_16BCAE0(&v335);
              }
            }
            v208 = *(_BYTE **)(v167 + 240);
            if ( !v208 )
              break;
            v209 = *(_QWORD *)(v167 + 248);
            v335 = (char *)&v337;
            sub_12BCC40((__int64 *)&v335, v208, (__int64)&v208[v209]);
            v210 = (char *)v237[30];
            v211 = v210;
            if ( v335 == (char *)&v337 )
            {
              v239 = v336;
              if ( v336 )
              {
                if ( v336 == 1 )
                  *v210 = (char)v337;
                else
                  memcpy(v210, &v337, v336);
                v239 = v336;
                v210 = (char *)v237[30];
              }
              goto LABEL_443;
            }
            v212 = v337;
            v213 = v336;
            if ( v210 == (char *)(v237 + 32) )
            {
              v237[30] = v335;
              v237[31] = (void *)v213;
              v237[32] = v212;
LABEL_459:
              v335 = (char *)&v337;
              v211 = (char *)&v337;
              goto LABEL_386;
            }
            v214 = (char *)v237[32];
            v237[30] = v335;
            v237[31] = (void *)v213;
            v237[32] = v212;
            if ( !v211 )
              goto LABEL_459;
            v335 = v211;
            v337 = v214;
LABEL_386:
            v336 = 0;
            *v211 = 0;
            if ( v335 != (char *)&v337 )
              j_j___libc_free_0(v335, v337 + 1);
            v215 = sub_1632FA0(v167);
            sub_1632B40(v237, v215);
            v316 = (char **)(v167 + 240);
            LOWORD(v318) = 260;
            sub_16E1010(&v335);
            v216 = (__int64)v237;
            if ( (unsigned int)sub_12F5580(&p_s2, v237, &v310, HIDWORD(v340) == 23) )
            {
              LODWORD(v320[0]) = 1;
              v319 = 0;
              v318 = 0;
              v316 = (char **)&unk_49EFBE0;
              v317 = 0;
              s = 0;
              v320[1] = a1 + 10;
              sub_1C3E9C0(&s);
              if ( (unsigned __int64)(v318 - (_BYTE *)v319) <= 0x15 )
              {
                v243 = (char ***)sub_16E7EE0(&v316, "builtins: link error: ", 22);
              }
              else
              {
                qmemcpy(v319, "builtins: link error: ", 0x16u);
                v243 = &v316;
                v319 = (char *)v319 + 22;
              }
              v216 = (__int64)v310;
              sub_16E7EE0(v243, v310, v311);
              v244 = s;
              if ( s )
              {
                v245 = strlen(s);
                v246 = v245;
                if ( v245 > v318 - (_BYTE *)v319 )
                {
                  v216 = (__int64)v244;
                  sub_16E7EE0(&v316, v244, v245);
                  v244 = s;
                }
                else if ( v245 )
                {
                  v216 = (__int64)v244;
                  memcpy(v319, v244, v245);
                  v319 = (char *)v319 + v246;
                  v244 = s;
                }
                if ( v244 )
                  j_j___libc_free_0_0(v244);
              }
              *a2 = 9;
              sub_16E7BC0(&v316);
              if ( v335 != (char *)&v337 )
              {
                v216 = (__int64)(v337 + 1);
                j_j___libc_free_0(v335, v337 + 1);
              }
              v281 = 0;
LABEL_446:
              if ( (v297 & 2) != 0 )
LABEL_457:
                sub_1264230(&s2, v216, v217);
              v241 = s2;
              if ( (v297 & 1) != 0 )
              {
                if ( s2 )
                  (*(void (__fastcall **)(void *))(*(_QWORD *)s2 + 8LL))(s2);
              }
              else if ( s2 )
              {
                sub_1633490(s2);
                j_j___libc_free_0(v241, 736);
              }
              if ( v310 != &v312 )
                j_j___libc_free_0(v310, v312 + 1);
              v188 = v291;
              if ( v291 )
LABEL_332:
                (*((void (__fastcall **)(char **))*v188 + 1))(v188);
              if ( v350 )
              {
                v190 = v349;
                v191 = &v349[2 * v350];
                do
                {
                  if ( *v190 != -4 && *v190 != -8 && v190[1] )
                    sub_161E7C0(v190 + 1);
                  v190 += 2;
                }
                while ( v191 != v190 );
              }
              j___libc_free_0(v349);
              if ( v348 )
              {
                v192 = sub_16704E0();
                v193 = sub_16704F0();
                v194 = v347;
                v195 = v193;
                v196 = v347;
                v197 = &v347[v348];
                if ( v347 == v197 )
                {
LABEL_427:
                  j___libc_free_0(v194);
                  j___libc_free_0(v343);
                  goto LABEL_428;
                }
                do
                {
                  if ( !(unsigned __int8)sub_1670560(*v196, v192) )
                    sub_1670560(*v196, v195);
                  ++v196;
                }
                while ( v197 != v196 );
              }
              v194 = v347;
              goto LABEL_427;
            }
            if ( v335 != (char *)&v337 )
            {
              v216 = (__int64)(v337 + 1);
              j_j___libc_free_0(v335, v337 + 1);
            }
            if ( (v297 & 2) != 0 )
              goto LABEL_457;
            v218 = s2;
            if ( (v297 & 1) != 0 )
            {
              if ( s2 )
                (*(void (__fastcall **)(void *))(*(_QWORD *)s2 + 8LL))(s2);
            }
            else if ( s2 )
            {
              sub_1633490(s2);
              j_j___libc_free_0(v218, 736);
            }
            if ( v310 != &v312 )
              j_j___libc_free_0(v310, v312 + 1);
            if ( v291 )
              (*((void (__fastcall **)(char **))*v291 + 1))(v291);
LABEL_399:
            v184 += 4;
            if ( v287 == v184 )
              goto LABEL_400;
          }
          LOBYTE(v337) = 0;
          v335 = (char *)&v337;
          v210 = (char *)v237[30];
LABEL_443:
          v237[31] = (void *)v239;
          v210[v239] = 0;
          v211 = v335;
          goto LABEL_386;
        }
LABEL_400:
        sub_1611EE0(&v316);
        v219 = sub_1CB9110(a5 + 208);
        sub_1619140(&v316, v219, 1);
        if ( v167 && (unsigned __int8)sub_1619BD0(&v316, v167) && !LOBYTE(qword_4F96820[20]) )
        {
          sub_1611EE0(&v335);
          v249 = sub_1CC63C0();
          sub_1619140(&v335, v249, 0);
          sub_1619BD0(&v335, v167);
          sub_160FE50(&v335);
        }
        sub_160FE50(&v316);
        if ( v350 )
        {
          v220 = v349;
          v221 = &v349[2 * v350];
          do
          {
            if ( *v220 != -8 && *v220 != -4 && v220[1] )
              sub_161E7C0(v220 + 1);
            v220 += 2;
          }
          while ( v221 != v220 );
        }
        j___libc_free_0(v349);
        if ( v348 )
        {
          v222 = sub_16704E0();
          v223 = sub_16704F0();
          v224 = v347;
          v225 = v223;
          v226 = v347;
          v227 = &v347[v348];
          if ( v347 == v227 )
            goto LABEL_507;
          do
          {
            if ( !(unsigned __int8)sub_1670560(*v226, v222) )
              sub_1670560(*v226, v225);
            ++v226;
          }
          while ( v227 != v226 );
        }
        v224 = v347;
LABEL_507:
        j___libc_free_0(v224);
        j___libc_free_0(v343);
        goto LABEL_315;
      }
      v345 = 1;
      v344 = 0;
      v343 = 0;
      p_s2 = (void **)&unk_49EFBE0;
      v342 = 0;
      v335 = 0;
      v346 = a1 + 10;
      sub_1C3E9C0(&v335);
      v250 = v344;
      if ( (unsigned __int64)((char *)v343 - (char *)v344) <= 0x15 )
      {
        v252 = (void ***)sub_16E7EE0(&p_s2, "builtins: link error: ", 22);
      }
      else
      {
        v251 = _mm_load_si128((const __m128i *)&xmmword_4281910);
        v344[1].m128i_i32[0] = 1919906418;
        v250[1].m128i_i16[2] = 8250;
        *v250 = v251;
        v344 = (__m128i *)((char *)v344 + 22);
        v252 = &p_s2;
      }
      v253 = v335;
      if ( !v335 )
      {
LABEL_502:
        *a2 = 9;
        sub_16E7BC0(&p_s2);
        v281 = 0;
LABEL_428:
        v166 = v167;
        v167 = 0;
        goto LABEL_270;
      }
      v254 = strlen(v335);
      v255 = v252[3];
      v256 = v254;
      if ( v254 > (char *)v252[2] - (char *)v255 )
      {
        sub_16E7EE0(v252, v253, v254);
        v253 = v335;
      }
      else
      {
        if ( !v254 )
        {
LABEL_501:
          j_j___libc_free_0_0(v253);
          goto LABEL_502;
        }
        memcpy(v255, v253, v254);
        v252[3] = (void **)((char *)v252[3] + v256);
        v253 = v335;
      }
      if ( !v253 )
        goto LABEL_502;
      goto LABEL_501;
    }
    LOBYTE(v318) = 0;
    v60 = 0;
    v316 = &v318;
    v317 = 0;
    v290 = (unsigned int)((v294 - (_BYTE *)v59) >> 3);
    if ( !(unsigned int)((v294 - (_BYTE *)v59) >> 3) )
    {
LABEL_347:
      v336 = 0;
      v335 = (char *)&v337;
      LOBYTE(v337) = 0;
      v167 = sub_12F5610(&v293, v300, &v316, &v335, a5);
      if ( !v167 )
      {
        v345 = 1;
        v344 = 0;
        v343 = 0;
        p_s2 = (void **)&unk_49EFBE0;
        v342 = 0;
        v346 = a1 + 10;
        v247 = sub_16E7EE0(&p_s2, v335, v336);
        v248 = *(char **)(v247 + 24);
        if ( *(_QWORD *)(v247 + 16) - (_QWORD)v248 <= 0xDu )
        {
          v247 = sub_16E7EE0(v247, ": link error: ", 14);
        }
        else
        {
          qmemcpy(v248, ": link error: ", 0xEu);
          *(_QWORD *)(v247 + 24) += 14LL;
        }
        sub_16E7EE0(v247, v316, v317);
        *a2 = 9;
        sub_16E7BC0(&p_s2);
        if ( v335 != (char *)&v337 )
          j_j___libc_free_0(v335, v337 + 1);
LABEL_420:
        if ( v316 != &v318 )
          j_j___libc_free_0(v316, v318 + 1);
        goto LABEL_319;
      }
      if ( v317 )
        sub_2241490(a1 + 10, v316, v317, v198);
      if ( v335 != (char *)&v337 )
        j_j___libc_free_0(v335, v337 + 1);
      if ( v316 != &v318 )
        j_j___libc_free_0(v316, v318 + 1);
      goto LABEL_354;
    }
    while ( v57 == (_DWORD)v60 )
    {
LABEL_107:
      if ( v290 == ++v60 )
        goto LABEL_347;
    }
    v69 = &v293[v60];
    v70 = (_QWORD *)*v69;
    if ( !*(_QWORD *)(*v69 + 248LL) )
    {
      v345 = 1;
      v344 = 0;
      v343 = 0;
      p_s2 = (void **)&unk_49EFBE0;
      v342 = 0;
      v346 = a1 + 10;
      v228 = sub_16E7EE0(&p_s2, *(_QWORD *)(*v69 + 176LL), *(_QWORD *)(*v69 + 184LL));
      v229 = *(__m128i **)(v228 + 24);
      if ( *(_QWORD *)(v228 + 16) - (_QWORD)v229 <= 0x3Eu )
      {
        sub_16E7EE0(v228, ": error: Module does not contain a triple, should be 'nvptx64-'", 63);
      }
      else
      {
        v230 = _mm_load_si128((const __m128i *)&xmmword_42818E0);
        qmemcpy(&v229[3], "d be 'nvptx64-'", 15);
        *v229 = v230;
        v229[1] = _mm_load_si128((const __m128i *)&xmmword_42818F0);
        v229[2] = _mm_load_si128((const __m128i *)&xmmword_4281900);
        *(_QWORD *)(v228 + 24) += 63LL;
      }
      *a2 = 9;
      sub_16E7BC0(&p_s2);
      goto LABEL_420;
    }
    v71 = v293[v58];
    v72 = *(_BYTE **)(v71 + 240);
    if ( v72 )
    {
      v61 = *(_QWORD *)(v71 + 248);
      p_s2 = (void **)&v343;
      sub_12BCC40((__int64 *)&p_s2, v72, (__int64)&v72[v61]);
      v62 = (void **)v70[30];
      v63 = v62;
      if ( p_s2 != (void **)&v343 )
      {
        v64 = v342;
        v65 = v343;
        if ( v63 == v70 + 32 )
        {
          v70[30] = p_s2;
          v70[31] = v64;
          v70[32] = v65;
        }
        else
        {
          v66 = (__m128i *)v70[32];
          v70[30] = p_s2;
          v70[31] = v64;
          v70[32] = v65;
          if ( v63 )
          {
            p_s2 = v63;
            v343 = v66;
LABEL_104:
            v342 = 0;
            *(_BYTE *)v63 = 0;
            if ( p_s2 != (void **)&v343 )
              j_j___libc_free_0(p_s2, &v343->m128i_i8[1]);
            v67 = v293[v60];
            v68 = sub_1632FA0(v293[v58]);
            sub_1632B40(v67, v68);
            goto LABEL_107;
          }
        }
        p_s2 = (void **)&v343;
        v63 = (void **)&v343;
        goto LABEL_104;
      }
      v73 = v342;
      if ( v342 )
      {
        if ( v342 == 1 )
          *(_BYTE *)v62 = (_BYTE)v343;
        else
          memcpy(v62, &v343, v342);
        v73 = v342;
        v62 = (void **)v70[30];
      }
    }
    else
    {
      LOBYTE(v343) = 0;
      v73 = 0;
      p_s2 = (void **)&v343;
      v62 = (void **)v70[30];
    }
    v70[31] = v73;
    *((_BYTE *)v62 + v73) = 0;
    v63 = p_s2;
    goto LABEL_104;
  }
  v273 = (_QWORD *)*a1;
  v269 = a1 + 10;
  v266 = 0;
  do
  {
    v6 = 14;
    v7 = "Unknown buffer";
    sub_16C2450(&v291, v273[2], v273[3], *v273, v273[1], 1);
    v8 = v291;
    v316 = 0;
    v317 = 0;
    v318 = (char *)v320;
    v319 = 0;
    src = v325;
    LOBYTE(v320[0]) = 0;
    v326 = v328;
    v321 = 0;
    v332 = v334;
    v322 = 0;
    n = 0;
    LOBYTE(v325[0]) = 0;
    v327 = 0;
    LOBYTE(v328[0]) = 0;
    v329 = 0;
    v330 = 0;
    v331 = 0;
    v333 = 0x400000000LL;
    v9 = (char *(*)())*((_QWORD *)*v291 + 2);
    if ( v9 != sub_12BCB10 )
    {
      v151 = ((__int64 (__fastcall *)(char **))v9)(v291);
      v8 = v291;
      v7 = (char *)v151;
      v6 = v152;
    }
    sub_16C2FC0(v299, v8);
    sub_166F050(
      (unsigned int)&p_s2,
      (unsigned int)&v316,
      v271,
      1,
      (unsigned int)byte_3F871B3,
      0,
      v299[0],
      v299[1],
      v299[2],
      v299[3]);
    v281 = p_s2;
    if ( !p_s2 )
    {
      v345 = 1;
      v344 = 0;
      v343 = 0;
      p_s2 = (void **)&unk_49EFBE0;
      v342 = 0;
      v346 = v269;
      if ( v6 )
      {
        sub_16E7EE0(&p_s2, v7, v6);
        v177 = (__int64 *)v344;
        v178 = (char *)v343 - (char *)v344;
        if ( (_DWORD)v321 == -1 )
          goto LABEL_311;
        if ( v178 > 1 )
        {
          v179 = &p_s2;
          v344->m128i_i16[0] = 10272;
          v344 = (__m128i *)((char *)v344 + 2);
          goto LABEL_304;
        }
      }
      else if ( (_DWORD)v321 == -1 )
      {
        goto LABEL_244;
      }
      v179 = (void ***)sub_16E7EE0(&p_s2, " (", 2);
LABEL_304:
      sub_16E7AB0(v179, (int)v321);
      if ( HIDWORD(v321) != -1 )
      {
        if ( (unsigned __int64)((char *)v343 - (char *)v344) <= 1 )
        {
          v180 = (void ***)sub_16E7EE0(&p_s2, ", ", 2);
        }
        else
        {
          v180 = &p_s2;
          v344->m128i_i16[0] = 8236;
          v344 = (__m128i *)((char *)v344 + 2);
        }
        sub_16E7AB0(v180, SHIDWORD(v321));
      }
      if ( v343 == v344 )
      {
        sub_16E7EE0(&p_s2, ")", 1);
        v177 = (__int64 *)v344;
      }
      else
      {
        v344->m128i_i8[0] = 41;
        v177 = (__int64 *)((char *)v344->m128i_i64 + 1);
        v344 = (__m128i *)((char *)v344 + 1);
      }
      v178 = (char *)v343 - (char *)v177;
LABEL_311:
      if ( v178 > 7 )
      {
        v159 = &p_s2;
        *v177 = 0x206573726170203ALL;
        v158 = &v344->m128i_u64[1];
        v344 = (__m128i *)((char *)v344 + 8);
        goto LABEL_245;
      }
LABEL_244:
      v157 = sub_16E7EE0(&p_s2, ": parse ", 8);
      v158 = *(unsigned __int64 **)(v157 + 24);
      v159 = (void ***)v157;
LABEL_245:
      v160 = n;
      v161 = (const char *)src;
      if ( n > (char *)v159[2] - (char *)v158 )
      {
        sub_16E7EE0(v159, src, n);
      }
      else if ( n )
      {
        memcpy(v158, src, n);
        v159[3] = (void **)((char *)v159[3] + v160);
      }
      *a2 = 9;
      sub_16E7BC0(&p_s2);
      goto LABEL_252;
    }
    if ( (unsigned int)sub_12BFF60((__int64)a1, (__int64)p_s2, a3) )
    {
      v345 = 1;
      v344 = 0;
      v162 = &p_s2;
      v343 = 0;
      p_s2 = (void **)&unk_49EFBE0;
      v342 = 0;
      v346 = v269;
      if ( v6
        && (v161 = v7,
            v162 = (void ***)sub_16E7EE0(&p_s2, v7, v6),
            v181 = (__m128i *)v162[3],
            (unsigned __int64)((char *)v162[2] - (char *)v181) > 0x56) )
      {
        v182 = _mm_load_si128((const __m128i *)&xmmword_4281880);
        v181[5].m128i_i32[0] = 1935762796;
        v181[5].m128i_i16[2] = 29541;
        *v181 = v182;
        v183 = _mm_load_si128((const __m128i *)&xmmword_4281890);
        v181[5].m128i_i8[6] = 46;
        v181[1] = v183;
        v181[2] = _mm_load_si128((const __m128i *)&xmmword_42818A0);
        v181[3] = _mm_load_si128((const __m128i *)&xmmword_42818B0);
        v181[4] = _mm_load_si128((const __m128i *)&xmmword_42818C0);
        v162[3] = (void **)((char *)v162[3] + 87);
      }
      else
      {
        v161 = ": error: incompatible IR detected. Possible mix of compiler/IR from different releases.";
        sub_16E7EE0(v162, ": error: incompatible IR detected. Possible mix of compiler/IR from different releases.", 87);
      }
      *a2 = 3;
      sub_16E7BC0(&p_s2);
LABEL_252:
      v163 = v332;
      v164 = &v332[48 * (unsigned int)v333];
      if ( v332 != v164 )
      {
        do
        {
          v164 -= 48;
          v165 = (_BYTE *)*((_QWORD *)v164 + 2);
          if ( v165 != v164 + 32 )
          {
            v161 = (const char *)(*((_QWORD *)v164 + 4) + 1LL);
            j_j___libc_free_0(v165, v161);
          }
        }
        while ( v163 != v164 );
        v164 = v332;
      }
      if ( v164 != v334 )
        _libc_free(v164, v161);
      if ( v329 )
      {
        v161 = (const char *)(v331 - v329);
        j_j___libc_free_0(v329, v331 - v329);
      }
      if ( v326 != v328 )
      {
        v161 = (const char *)(v328[0] + 1LL);
        j_j___libc_free_0(v326, v328[0] + 1LL);
      }
      if ( src != v325 )
      {
        v161 = (const char *)(v325[0] + 1LL);
        j_j___libc_free_0(src, v325[0] + 1LL);
      }
      if ( v318 != (char *)v320 )
      {
        v161 = (const char *)(v320[0] + 1LL);
        j_j___libc_free_0(v318, v320[0] + 1LL);
      }
      if ( v291 )
        (*((void (__fastcall **)(char **, const char *))*v291 + 1))(v291, v161);
      goto LABEL_269;
    }
    v314 = 1;
    dest = 0;
    v312 = 0;
    v310 = (__int64 *)&unk_49EFBE0;
    v311 = 0;
    v315 = v269;
    v288 = (void **)v281[2];
    if ( v288 == v281 + 1 )
      goto LABEL_62;
    do
    {
      v10 = 0;
      if ( v288 )
        v10 = v288 - 7;
      v286 = (__int64)v10;
      v12 = (_BYTE *)sub_1649960(v10);
      if ( v12 )
      {
        s2 = v298;
        sub_12BCC40((__int64 *)&s2, v12, (__int64)&v12[v11]);
      }
      else
      {
        LOBYTE(v298[0]) = 0;
        v297 = 0;
        s2 = v298;
      }
      if ( (*(_BYTE *)(v286 + 32) & 0xF) != 0 )
        goto LABEL_58;
      v13 = *(_QWORD *)(v286 + 24);
      v282 = v13;
      if ( *(_BYTE *)(v13 + 8) == 13 )
      {
        if ( (*(_BYTE *)(v13 + 9) & 1) == 0 )
          goto LABEL_58;
        v14 = v306;
        if ( !v306 )
        {
LABEL_187:
          v29 = (__int64 *)&v305;
          goto LABEL_53;
        }
      }
      else
      {
        v14 = v306;
        if ( !v306 )
          goto LABEL_187;
      }
      v15 = v14;
      v274 = (__int64 *)v14;
      v16 = s2;
      v17 = v297;
      v18 = v15;
      v19 = &v305;
      do
      {
        while ( 1 )
        {
          v20 = *((_QWORD *)v18 + 5);
          v21 = v17;
          if ( v20 <= v17 )
            v21 = *((_QWORD *)v18 + 5);
          if ( v21 )
          {
            v22 = memcmp(*((const void **)v18 + 4), v16, v21);
            if ( v22 )
              break;
          }
          v23 = v20 - v17;
          if ( v23 >= 0x80000000LL )
            goto LABEL_25;
          if ( v23 > (__int64)0xFFFFFFFF7FFFFFFFLL )
          {
            v22 = v23;
            break;
          }
LABEL_16:
          v18 = (int *)*((_QWORD *)v18 + 3);
          if ( !v18 )
            goto LABEL_26;
        }
        if ( v22 < 0 )
          goto LABEL_16;
LABEL_25:
        v19 = v18;
        v18 = (int *)*((_QWORD *)v18 + 2);
      }
      while ( v18 );
LABEL_26:
      if ( v19 == &v305 )
        goto LABEL_34;
      v24 = *((_QWORD *)v19 + 5);
      v25 = v17;
      if ( v24 <= v17 )
        v25 = *((_QWORD *)v19 + 5);
      if ( v25 )
      {
        LODWORD(v26) = memcmp(v16, *((const void **)v19 + 4), v25);
        if ( (_DWORD)v26 )
        {
LABEL_33:
          if ( (int)v26 < 0 )
            goto LABEL_34;
LABEL_114:
          v74 = 1;
          v75 = sub_1632FA0(*(_QWORD *)(v286 + 40));
          sub_12BDBC0((__int64)&v335, v75);
          v76 = sub_15A9FE0(&v335, v282);
          v77 = v282;
          v78 = v76;
          while ( 2 )
          {
            switch ( *(_BYTE *)(v77 + 8) )
            {
              case 0:
              case 8:
              case 0xA:
              case 0xC:
              case 0x10:
                v113 = *(_QWORD *)(v77 + 32);
                v77 = *(_QWORD *)(v77 + 24);
                v74 *= v113;
                continue;
              case 1:
                v79 = 16;
                break;
              case 2:
                v79 = 32;
                break;
              case 3:
              case 9:
                v79 = 64;
                break;
              case 4:
                v79 = 80;
                break;
              case 5:
              case 6:
                v79 = 128;
                break;
              case 7:
                v79 = 8 * (unsigned int)sub_15A9520(&v335, 0);
                break;
              case 0xB:
                v79 = *(_DWORD *)(v77 + 8) >> 8;
                break;
              case 0xD:
                v79 = 8LL * *(_QWORD *)sub_15A9930(&v335, v77);
                break;
              case 0xE:
                v109 = *(_QWORD *)(v77 + 32);
                v110 = 1;
                v111 = *(_QWORD *)(v77 + 24);
                v112 = (unsigned int)sub_15A9FE0(&v335, v111);
                while ( 2 )
                {
                  switch ( *(_BYTE *)(v111 + 8) )
                  {
                    case 0:
                    case 8:
                    case 0xA:
                    case 0xC:
                    case 0x10:
                      v169 = *(_QWORD *)(v111 + 32);
                      v111 = *(_QWORD *)(v111 + 24);
                      v110 *= v169;
                      continue;
                    case 1:
                      v155 = 16;
                      goto LABEL_239;
                    case 2:
                      v155 = 32;
                      goto LABEL_239;
                    case 3:
                    case 9:
                      v155 = 64;
                      goto LABEL_239;
                    case 4:
                      v155 = 80;
                      goto LABEL_239;
                    case 5:
                    case 6:
                      v155 = 128;
                      goto LABEL_239;
                    case 7:
                      v155 = 8 * (unsigned int)sub_15A9520(&v335, 0);
                      goto LABEL_239;
                    case 0xB:
                      v155 = *(_DWORD *)(v111 + 8) >> 8;
                      goto LABEL_239;
                    case 0xD:
                      v155 = 8LL * *(_QWORD *)sub_15A9930(&v335, v111);
                      goto LABEL_239;
                    case 0xE:
                      v276 = *(_QWORD *)(v111 + 32);
                      v170 = *(_QWORD *)(v111 + 24);
                      v285 = 1;
                      v261 = (unsigned int)sub_15A9FE0(&v335, v170);
                      while ( 2 )
                      {
                        switch ( *(_BYTE *)(v170 + 8) )
                        {
                          case 0:
                          case 8:
                          case 0xA:
                          case 0xC:
                          case 0x10:
                            v207 = v285 * *(_QWORD *)(v170 + 32);
                            v170 = *(_QWORD *)(v170 + 24);
                            v285 = v207;
                            continue;
                          case 1:
                            v206 = 16;
                            goto LABEL_370;
                          case 2:
                            v206 = 32;
                            goto LABEL_370;
                          case 3:
                          case 9:
                            v206 = 64;
                            goto LABEL_370;
                          case 4:
                            v206 = 80;
                            goto LABEL_370;
                          case 5:
                          case 6:
                            v206 = 128;
                            goto LABEL_370;
                          case 7:
                            v206 = 8 * (unsigned int)sub_15A9520(&v335, 0);
                            goto LABEL_370;
                          case 0xB:
                            v206 = *(_DWORD *)(v170 + 8) >> 8;
                            goto LABEL_370;
                          case 0xD:
                            v206 = 8LL * *(_QWORD *)sub_15A9930(&v335, v170);
                            goto LABEL_370;
                          case 0xE:
                            v260 = *(_QWORD *)(v170 + 32);
                            v206 = 8 * v260 * sub_12BE0A0((__int64)&v335, *(_QWORD *)(v170 + 24));
                            goto LABEL_370;
                          case 0xF:
                            v206 = 8 * (unsigned int)sub_15A9520(&v335, *(_DWORD *)(v170 + 8) >> 8);
LABEL_370:
                            v155 = 8 * v261 * v276 * ((v261 + ((unsigned __int64)(v285 * v206 + 7) >> 3) - 1) / v261);
                            goto LABEL_239;
                          default:
                            goto LABEL_508;
                        }
                      }
                    case 0xF:
                      v155 = 8 * (unsigned int)sub_15A9520(&v335, *(_DWORD *)(v111 + 8) >> 8);
LABEL_239:
                      v79 = 8 * v112 * v109 * ((v112 + ((unsigned __int64)(v110 * v155 + 7) >> 3) - 1) / v112);
                      break;
                    default:
LABEL_508:
                      MEMORY[0x20] &= 0xFFFFFFF0;
                      BUG();
                  }
                  return result;
                }
              case 0xF:
                v79 = 8 * (unsigned int)sub_15A9520(&v335, *(_DWORD *)(v77 + 8) >> 8);
                break;
            }
            break;
          }
          v80 = (__int64 *)v306;
          v283 = v78 * ((v78 + ((unsigned __int64)(v74 * v79 + 7) >> 3) - 1) / v78);
          if ( !v306 )
          {
            v83 = (__int64 *)&v305;
            goto LABEL_138;
          }
          v81 = s2;
          v82 = v297;
          v83 = (__int64 *)&v305;
          while ( 2 )
          {
            while ( 2 )
            {
              v84 = v80[5];
              v85 = v82;
              if ( v84 <= v82 )
                v85 = v80[5];
              if ( !v85 || (v86 = memcmp((const void *)v80[4], v81, v85)) == 0 )
              {
                v87 = v84 - v82;
                if ( v87 >= 0x80000000LL )
                  goto LABEL_128;
                if ( v87 > (__int64)0xFFFFFFFF7FFFFFFFLL )
                {
                  v86 = v87;
                  break;
                }
LABEL_119:
                v80 = (__int64 *)v80[3];
                if ( !v80 )
                  goto LABEL_129;
                continue;
              }
              break;
            }
            if ( v86 < 0 )
              goto LABEL_119;
LABEL_128:
            v83 = v80;
            v80 = (__int64 *)v80[2];
            if ( v80 )
              continue;
            break;
          }
LABEL_129:
          if ( v83 == (__int64 *)&v305 )
            goto LABEL_138;
          v88 = v83[5];
          v89 = v82;
          if ( v88 <= v82 )
            v89 = v83[5];
          if ( v89 && (v90 = memcmp(v81, (const void *)v83[4], v89)) != 0 )
          {
LABEL_137:
            if ( v90 < 0 )
              goto LABEL_138;
          }
          else
          {
            v91 = v82 - v88;
            if ( v91 < 0x80000000LL )
            {
              if ( v91 > (__int64)0xFFFFFFFF7FFFFFFFLL )
              {
                v90 = v91;
                goto LABEL_137;
              }
LABEL_138:
              p_s2 = &s2;
              v83 = sub_12BFA60(&v304, v83, (__int64 *)&p_s2);
            }
          }
          v92 = sub_1632FA0(*(_QWORD *)(v83[8] + 40));
          sub_12BDBC0((__int64)&p_s2, v92);
          v93 = (__int64 *)v306;
          if ( !v306 )
          {
            v96 = (__int64 *)&v305;
            goto LABEL_160;
          }
          v94 = s2;
          v95 = v297;
          v96 = (__int64 *)&v305;
          while ( 2 )
          {
            while ( 2 )
            {
              v97 = v93[5];
              v98 = v95;
              if ( v97 <= v95 )
                v98 = v93[5];
              if ( !v98 || (v99 = memcmp((const void *)v93[4], v94, v98)) == 0 )
              {
                v100 = v97 - v95;
                if ( v100 >= 0x80000000LL )
                  goto LABEL_150;
                if ( v100 > (__int64)0xFFFFFFFF7FFFFFFFLL )
                {
                  v99 = v100;
                  break;
                }
LABEL_141:
                v93 = (__int64 *)v93[3];
                if ( !v93 )
                  goto LABEL_151;
                continue;
              }
              break;
            }
            if ( v99 < 0 )
              goto LABEL_141;
LABEL_150:
            v96 = v93;
            v93 = (__int64 *)v93[2];
            if ( v93 )
              continue;
            break;
          }
LABEL_151:
          if ( v96 == (__int64 *)&v305 )
            goto LABEL_160;
          v101 = v96[5];
          v102 = v95;
          if ( v101 <= v95 )
            v102 = v96[5];
          if ( v102 && (v103 = memcmp(v94, (const void *)v96[4], v102)) != 0 )
          {
LABEL_159:
            if ( v103 < 0 )
              goto LABEL_160;
          }
          else
          {
            v104 = v95 - v101;
            if ( v104 < 0x80000000LL )
            {
              if ( v104 > (__int64)0xFFFFFFFF7FFFFFFFLL )
              {
                v103 = v104;
                goto LABEL_159;
              }
LABEL_160:
              s = (char *)&s2;
              v96 = sub_12BFA60(&v304, v96, (__int64 *)&s);
            }
          }
          v105 = v96[8];
          v106 = 1;
          v107 = *(_QWORD *)(v105 + 24);
          v108 = (unsigned int)sub_15A9FE0(&p_s2, v107);
          while ( 2 )
          {
            switch ( *(_BYTE *)(v107 + 8) )
            {
              case 1:
                v40 = 16;
                goto LABEL_56;
              case 2:
                v40 = 32;
                goto LABEL_56;
              case 3:
              case 9:
                v40 = 64;
                goto LABEL_56;
              case 4:
                v40 = 80;
                goto LABEL_56;
              case 5:
              case 6:
                v40 = 128;
                goto LABEL_56;
              case 7:
                v40 = 8 * (unsigned int)sub_15A9520(&p_s2, 0);
                goto LABEL_56;
              case 0xB:
                v40 = *(_DWORD *)(v107 + 8) >> 8;
                goto LABEL_56;
              case 0xD:
                v40 = 8LL * *(_QWORD *)sub_15A9930(&p_s2, v107);
                goto LABEL_56;
              case 0xE:
                v115 = *(_QWORD *)(v107 + 24);
                v116 = *(_QWORD *)(v107 + 32);
                v117 = 1;
                v118 = (unsigned int)sub_15A9FE0(&p_s2, v115);
                while ( 2 )
                {
                  switch ( *(_BYTE *)(v115 + 8) )
                  {
                    case 1:
                      v156 = 16;
                      goto LABEL_241;
                    case 2:
                      v156 = 32;
                      goto LABEL_241;
                    case 3:
                    case 9:
                      v156 = 64;
                      goto LABEL_241;
                    case 4:
                      v156 = 80;
                      goto LABEL_241;
                    case 5:
                    case 6:
                      v156 = 128;
                      goto LABEL_241;
                    case 7:
                      v280 = v118;
                      v176 = sub_15A9520(&p_s2, 0);
                      v118 = v280;
                      v156 = (unsigned int)(8 * v176);
                      goto LABEL_241;
                    case 0xB:
                      v156 = *(_DWORD *)(v115 + 8) >> 8;
                      goto LABEL_241;
                    case 0xD:
                      v279 = v118;
                      v175 = (_QWORD *)sub_15A9930(&p_s2, v115);
                      v118 = v279;
                      v156 = 8LL * *v175;
                      goto LABEL_241;
                    case 0xE:
                      v257 = v118;
                      v262 = *(_QWORD *)(v115 + 32);
                      v258 = *(_QWORD *)(v115 + 24);
                      v171 = sub_15A9FE0(&p_s2, v258);
                      v118 = v257;
                      v277 = 1;
                      v172 = v258;
                      v259 = v171;
                      while ( 2 )
                      {
                        switch ( *(_BYTE *)(v172 + 8) )
                        {
                          case 1:
                            v201 = 16;
                            goto LABEL_359;
                          case 2:
                            v201 = 32;
                            goto LABEL_359;
                          case 3:
                          case 9:
                            v201 = 64;
                            goto LABEL_359;
                          case 4:
                            v201 = 80;
                            goto LABEL_359;
                          case 5:
                          case 6:
                            v201 = 128;
                            goto LABEL_359;
                          case 7:
                            v205 = sub_15A9520(&p_s2, 0);
                            v118 = v257;
                            v201 = (unsigned int)(8 * v205);
                            goto LABEL_359;
                          case 0xB:
                            v201 = *(_DWORD *)(v172 + 8) >> 8;
                            goto LABEL_359;
                          case 0xD:
                            v204 = (_QWORD *)sub_15A9930(&p_s2, v172);
                            v118 = v257;
                            v201 = 8LL * *v204;
                            goto LABEL_359;
                          case 0xE:
                            v202 = *(_QWORD *)(v172 + 32);
                            v203 = sub_12BE0A0((__int64)&p_s2, *(_QWORD *)(v172 + 24));
                            v118 = v257;
                            v201 = 8 * v202 * v203;
                            goto LABEL_359;
                          case 0xF:
                            v200 = sub_15A9520(&p_s2, *(_DWORD *)(v172 + 8) >> 8);
                            v118 = v257;
                            v201 = (unsigned int)(8 * v200);
LABEL_359:
                            v156 = 8 * v259 * v262 * ((v259 + ((unsigned __int64)(v277 * v201 + 7) >> 3) - 1) / v259);
                            goto LABEL_241;
                          case 0x10:
                            v199 = v277 * *(_QWORD *)(v172 + 32);
                            v172 = *(_QWORD *)(v172 + 24);
                            v277 = v199;
                            continue;
                          default:
                            goto LABEL_508;
                        }
                      }
                    case 0xF:
                      v278 = v118;
                      v173 = sub_15A9520(&p_s2, *(_DWORD *)(v115 + 8) >> 8);
                      v118 = v278;
                      v156 = (unsigned int)(8 * v173);
LABEL_241:
                      v40 = 8 * v118 * v116 * ((v118 + ((unsigned __int64)(v156 * v117 + 7) >> 3) - 1) / v118);
                      goto LABEL_56;
                    case 0x10:
                      v174 = *(_QWORD *)(v115 + 32);
                      v115 = *(_QWORD *)(v115 + 24);
                      v117 *= v174;
                      continue;
                    default:
                      goto LABEL_508;
                  }
                }
              case 0xF:
                v40 = 8 * (unsigned int)sub_15A9520(&p_s2, *(_DWORD *)(v107 + 8) >> 8);
LABEL_56:
                v41 = v108 * ((v108 + ((unsigned __int64)(v40 * v106 + 7) >> 3) - 1) / v108);
                v275 = v283 != 0 && v41 != v283 && v41 != 0;
                if ( !v275 )
                  goto LABEL_57;
                v119 = (__m128i *)dest;
                v120 = n;
                v121 = v312 - (_QWORD)dest;
                if ( n > v312 - (__int64)dest )
                {
                  v154 = sub_16E7EE0(&v310, src, n);
                  v119 = *(__m128i **)(v154 + 24);
                  v122 = (__int64 **)v154;
                  v121 = *(_QWORD *)(v154 + 16) - (_QWORD)v119;
                }
                else
                {
                  v122 = &v310;
                  if ( n )
                  {
                    memcpy(dest, src, n);
                    v119 = (__m128i *)((char *)dest + v120);
                    dest = v119;
                    v121 = v312 - (_QWORD)v119;
                  }
                }
                if ( v121 <= 0x17 )
                {
                  v122 = (__int64 **)sub_16E7EE0(v122, "Size does not match for ", 24);
                }
                else
                {
                  v123 = _mm_load_si128((const __m128i *)&xmmword_42818D0);
                  v119[1].m128i_i64[0] = 0x20726F6620686374LL;
                  *v119 = v123;
                  v122[3] += 3;
                }
                v124 = sub_16E7EE0(v122, s2, v297);
                v125 = *(_DWORD **)(v124 + 24);
                v126 = v124;
                if ( *(_QWORD *)(v124 + 16) - (_QWORD)v125 <= 3u )
                {
                  v126 = sub_16E7EE0(v124, " in ", 4);
                }
                else
                {
                  *v125 = 544106784;
                  *(_QWORD *)(v124 + 24) += 4LL;
                }
                v127 = sub_16E7EE0(v126, v281[22], v281[23]);
                v128 = *(void **)(v127 + 24);
                v129 = v127;
                if ( *(_QWORD *)(v127 + 16) - (_QWORD)v128 <= 0xAu )
                {
                  v129 = sub_16E7EE0(v127, " with size ", 11);
                }
                else
                {
                  qmemcpy(v128, " with size ", 11);
                  *(_QWORD *)(v127 + 24) += 11LL;
                }
                v130 = sub_16E7A90(v129, v283);
                v131 = *(void **)(v130 + 24);
                v284 = v130;
                if ( *(_QWORD *)(v130 + 16) - (_QWORD)v131 <= 0xDu )
                {
                  v153 = sub_16E7EE0(v130, " specified in ", 14);
                  v132 = (__int64 *)v306;
                  v284 = v153;
                  if ( v306 )
                    goto LABEL_199;
                }
                else
                {
                  qmemcpy(v131, " specified in ", 14);
                  v132 = (__int64 *)v306;
                  *(_QWORD *)(v130 + 24) += 14LL;
                  if ( v132 )
                  {
LABEL_199:
                    v267 = v41;
                    v133 = s2;
                    v134 = v297;
                    v135 = v132;
                    v136 = (__int64 *)&v305;
                    while ( 1 )
                    {
                      v137 = v135[5];
                      v138 = v134;
                      if ( v137 <= v134 )
                        v138 = v135[5];
                      if ( v138 )
                      {
                        v139 = memcmp((const void *)v135[4], v133, v138);
                        if ( v139 )
                          goto LABEL_208;
                      }
                      v140 = v137 - v134;
                      if ( v140 >= 0x80000000LL )
                      {
LABEL_209:
                        v136 = v135;
                        v135 = (__int64 *)v135[2];
                        if ( !v135 )
                        {
LABEL_210:
                          v141 = v134;
                          v41 = v267;
                          if ( v136 == (__int64 *)&v305 )
                            goto LABEL_219;
                          v142 = v136[5];
                          v143 = v141;
                          if ( v142 <= v141 )
                            v143 = v136[5];
                          if ( v143
                            && (v268 = v136[5], v144 = memcmp(v133, (const void *)v136[4], v143), v142 = v268, v144) )
                          {
LABEL_218:
                            if ( v144 >= 0 )
                              goto LABEL_220;
                          }
                          else
                          {
                            v145 = v141 - v142;
                            if ( v145 >= 0x80000000LL )
                              goto LABEL_220;
                            if ( v145 > (__int64)0xFFFFFFFF7FFFFFFFLL )
                            {
                              v144 = v145;
                              goto LABEL_218;
                            }
                          }
LABEL_219:
                          s = (char *)&s2;
                          v136 = sub_12BFA60(&v304, v136, (__int64 *)&s);
LABEL_220:
                          v146 = sub_16E7EE0(
                                   v284,
                                   *(_QWORD *)(*(_QWORD *)(v136[8] + 40) + 176LL),
                                   *(_QWORD *)(*(_QWORD *)(v136[8] + 40) + 184LL));
                          v147 = *(void **)(v146 + 24);
                          v148 = v146;
                          if ( *(_QWORD *)(v146 + 16) - (_QWORD)v147 <= 0xAu )
                          {
                            v148 = sub_16E7EE0(v146, " with size ", 11);
                          }
                          else
                          {
                            qmemcpy(v147, " with size ", 11);
                            *(_QWORD *)(v146 + 24) += 11LL;
                          }
                          v149 = sub_16E7A90(v148, v41);
                          v150 = *(_WORD **)(v149 + 24);
                          if ( *(_QWORD *)(v149 + 16) - (_QWORD)v150 <= 1u )
                          {
                            sub_16E7EE0(v149, ".\n", 2);
                          }
                          else
                          {
                            *v150 = 2606;
                            *(_QWORD *)(v149 + 24) += 2LL;
                          }
                          v266 = v275;
LABEL_57:
                          sub_15A93E0(&p_s2);
                          sub_15A93E0(&v335);
LABEL_58:
                          v35 = s2;
                          goto LABEL_59;
                        }
                      }
                      else
                      {
                        if ( v140 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
                          goto LABEL_200;
                        v139 = v140;
LABEL_208:
                        if ( v139 >= 0 )
                          goto LABEL_209;
LABEL_200:
                        v135 = (__int64 *)v135[3];
                        if ( !v135 )
                          goto LABEL_210;
                      }
                    }
                  }
                }
                v136 = (__int64 *)&v305;
                goto LABEL_219;
              case 0x10:
                v114 = *(_QWORD *)(v107 + 32);
                v107 = *(_QWORD *)(v107 + 24);
                v106 *= v114;
                continue;
              default:
                goto LABEL_508;
            }
          }
        }
      }
      v26 = v17 - v24;
      if ( (__int64)(v17 - v24) >= 0x80000000LL )
        goto LABEL_114;
      if ( v26 > (__int64)0xFFFFFFFF7FFFFFFFLL )
        goto LABEL_33;
LABEL_34:
      v27 = v17;
      v28 = v274;
      v29 = (__int64 *)&v305;
      while ( 2 )
      {
        while ( 2 )
        {
          v30 = v28[5];
          v31 = v27;
          if ( v30 <= v27 )
            v31 = v28[5];
          if ( !v31 || (v32 = memcmp((const void *)v28[4], v16, v31)) == 0 )
          {
            v33 = v30 - v27;
            if ( v33 >= 0x80000000LL )
              goto LABEL_44;
            if ( v33 > (__int64)0xFFFFFFFF7FFFFFFFLL )
            {
              v32 = v33;
              break;
            }
LABEL_35:
            v28 = (__int64 *)v28[3];
            if ( !v28 )
              goto LABEL_45;
            continue;
          }
          break;
        }
        if ( v32 < 0 )
          goto LABEL_35;
LABEL_44:
        v29 = v28;
        v28 = (__int64 *)v28[2];
        if ( v28 )
          continue;
        break;
      }
LABEL_45:
      v34 = v27;
      v35 = v16;
      if ( v29 == (__int64 *)&v305 )
        goto LABEL_53;
      v36 = v29[5];
      v37 = v34;
      if ( v36 <= v34 )
        v37 = v29[5];
      if ( v37 && (LODWORD(v38) = memcmp(v16, (const void *)v29[4], v37), (_DWORD)v38) )
      {
LABEL_52:
        if ( (int)v38 < 0 )
          goto LABEL_53;
      }
      else
      {
        v38 = v34 - v36;
        if ( (__int64)(v34 - v36) < 0x80000000LL )
        {
          if ( v38 > (__int64)0xFFFFFFFF7FFFFFFFLL )
            goto LABEL_52;
LABEL_53:
          p_s2 = &s2;
          v39 = sub_12BFA60(&v304, v29, (__int64 *)&p_s2);
          v35 = s2;
          v29 = v39;
        }
      }
      v29[8] = v286;
LABEL_59:
      if ( v35 != v298 )
        j_j___libc_free_0(v35, v298[0] + 1LL);
      v288 = (void **)v288[1];
    }
    while ( v281 + 1 != v288 );
LABEL_62:
    v42 = v294;
    p_s2 = v281;
    if ( v294 == v295 )
    {
      sub_12BE8C0((__int64)&v293, v294, &p_s2);
      v281 = p_s2;
    }
    else
    {
      if ( v294 )
      {
        *(_QWORD *)v294 = v281;
        v42 = v294;
      }
      v294 = v42 + 8;
    }
    v43 = sub_2241AC0(v281 + 30, off_4CD49B0);
    v44 = v301;
    v45 = v43;
    if ( v301 == v303 )
    {
      sub_12BCF40((__int64)v300, v301, v302, v43 == 0);
    }
    else
    {
      v46 = v302;
      if ( (_DWORD)v302 == 63 )
      {
        LODWORD(v302) = 0;
        ++v301;
      }
      else
      {
        LODWORD(v302) = v302 + 1;
      }
      v47 = 1LL << v46;
      v48 = (1LL << v46) | *v44;
      v49 = *v44 & ~v47;
      if ( !v45 )
        v49 = v48;
      *v44 = v49;
    }
    sub_16E7BC0(&v310);
    v50 = v332;
    v51 = &v332[48 * (unsigned int)v333];
    if ( v332 != v51 )
    {
      do
      {
        v51 -= 48;
        v52 = (_BYTE *)*((_QWORD *)v51 + 2);
        if ( v52 != v51 + 32 )
        {
          v44 = (__int64 *)(*((_QWORD *)v51 + 4) + 1LL);
          j_j___libc_free_0(v52, v44);
        }
      }
      while ( v50 != v51 );
      v51 = v332;
    }
    if ( v51 != v334 )
      _libc_free(v51, v44);
    if ( v329 )
      j_j___libc_free_0(v329, v331 - v329);
    if ( v326 != v328 )
      j_j___libc_free_0(v326, v328[0] + 1LL);
    if ( src != v325 )
      j_j___libc_free_0(src, v325[0] + 1LL);
    if ( v318 != (char *)v320 )
      j_j___libc_free_0(v318, v320[0] + 1LL);
    if ( v291 )
      (*((void (__fastcall **)(char **))*v291 + 1))(v291);
    v273 += 4;
  }
  while ( v265 != v273 );
  if ( !v266 )
    goto LABEL_91;
  *a2 = 9;
LABEL_319:
  v281 = 0;
LABEL_269:
  v166 = 0;
  v167 = 0;
LABEL_270:
  if ( v300[0] )
    j_j___libc_free_0(v300[0], (char *)v303 - v300[0]);
  sub_12BD2E0(v306);
  if ( v293 )
    j_j___libc_free_0(v293, v295 - (_BYTE *)v293);
  if ( v281 )
  {
    sub_1633490(v281);
    j_j___libc_free_0(v281, 736);
  }
  if ( v166 )
  {
    sub_1633490(v166);
    j_j___libc_free_0(v166, 736);
  }
  return v167;
}
