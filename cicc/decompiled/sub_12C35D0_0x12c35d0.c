// Function: sub_12C35D0
// Address: 0x12c35d0
//
__int64 __fastcall sub_12C35D0(__int64 a1, int a2, unsigned int a3, __int64 a4)
{
  __int64 v5; // rsi
  int v7; // edx
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // rax
  _QWORD *v11; // r14
  unsigned __int64 v12; // rsi
  char v13; // al
  char *v14; // r12
  size_t v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  _QWORD *v21; // rdi
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  _QWORD *v28; // r12
  unsigned int v29; // r12d
  unsigned int (__fastcall *v30)(_QWORD, _QWORD); // rax
  _QWORD *v31; // rdx
  __int64 *v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // r14
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // r13
  __int64 v41; // rdi
  __int64 v42; // r8
  __int64 v43; // r13
  __int64 v44; // rbx
  __int64 v45; // rdi
  __int64 v46; // r8
  __int64 v47; // r13
  __int64 v48; // rbx
  __int64 v49; // rdi
  __int64 v50; // r8
  __int64 v51; // r13
  __int64 v52; // rbx
  __int64 v53; // rdi
  __int64 v54; // r8
  __int64 v55; // r13
  __int64 v56; // rbx
  __int64 v57; // rdi
  unsigned int v59; // eax
  __int64 v60; // rax
  size_t v61; // rax
  char *v62; // r14
  __int64 v63; // r15
  size_t v64; // rax
  char *v65; // r8
  const char *v66; // rsi
  __int64 v67; // rdx
  void (__fastcall *v68)(char *, const char *, __int64); // r14
  __int64 v69; // r15
  char **v70; // rax
  __int64 v71; // rax
  __int64 v72; // rdx
  char v73; // al
  unsigned int (__fastcall *v74)(_QWORD, _QWORD); // rax
  char *v75; // r12
  size_t v76; // rax
  __int64 v77; // rax
  __m128i *v78; // rdx
  __m128i si128; // xmm0
  _QWORD *v80; // rdx
  char **v81; // r12
  size_t v82; // rax
  char *v83; // rdi
  size_t v84; // rbx
  char *v85; // rax
  char *v86; // rbx
  char v87; // al
  int v88; // ebx
  size_t v89; // rax
  char *v90; // r8
  __int64 v91; // r9
  size_t v92; // rax
  char *v93; // r14
  void *v94; // rdi
  char *v95; // rsi
  __int64 v96; // rdx
  __int64 v97; // rcx
  __int64 v98; // r8
  __int64 v99; // r9
  void (__fastcall *v100)(char *, char *, __int64); // rbx
  __int64 v101; // rdx
  char **v102; // rax
  __int64 v103; // rax
  __int64 v104; // rcx
  __int64 v105; // r9
  char *v106; // r14
  char *v107; // rdi
  __int64 v108; // rbx
  __int64 v109; // r8
  _QWORD *v110; // r15
  size_t v111; // rax
  _QWORD *v112; // rdx
  unsigned int (__fastcall *v113)(char *, _QWORD); // rax
  char *v114; // r14
  void (__fastcall *v115)(char *, __int64, __int64); // rbx
  __int64 v116; // rdx
  char **v117; // rax
  __int64 v118; // rdx
  int v119; // ebx
  size_t v120; // rax
  char *v121; // r8
  __int64 v122; // r9
  size_t v123; // rax
  char *v124; // r14
  char v125; // al
  char *v126; // r14
  char v127; // bl
  _QWORD *v128; // r15
  size_t v129; // rax
  __int64 v130; // rdi
  int v131; // ebx
  size_t v132; // rax
  char *v133; // r8
  __int64 v134; // r9
  size_t v135; // rax
  char *v136; // r14
  void *v137; // rdi
  const char *v138; // rsi
  __int64 v139; // rdx
  __int64 v140; // rcx
  __int64 v141; // r8
  __int64 v142; // r9
  void (__fastcall *v143)(char *, const char *, __int64); // rbx
  __int64 v144; // rdx
  char **v145; // rax
  char *v146; // rdi
  char v147; // al
  __int64 v148; // rdx
  __int64 v149; // rcx
  __int64 v150; // r8
  __int64 v151; // r9
  char *v152; // r14
  char v153; // bl
  _QWORD *v154; // r15
  size_t v155; // rax
  __int64 v156; // rdx
  unsigned int (__fastcall *v157)(_QWORD, _QWORD); // rax
  char *v158; // [rsp-10h] [rbp-880h]
  __int64 v159; // [rsp-8h] [rbp-878h]
  char *v160; // [rsp+0h] [rbp-870h]
  char *v161; // [rsp+0h] [rbp-870h]
  __int64 v162; // [rsp+0h] [rbp-870h]
  _QWORD *v163; // [rsp+0h] [rbp-870h]
  __int64 v164; // [rsp+0h] [rbp-870h]
  __int64 v165; // [rsp+8h] [rbp-868h]
  char *v166; // [rsp+10h] [rbp-860h]
  char *v167; // [rsp+10h] [rbp-860h]
  const char *v170; // [rsp+28h] [rbp-848h]
  int v171; // [rsp+30h] [rbp-840h]
  int v172; // [rsp+30h] [rbp-840h]
  int v173; // [rsp+30h] [rbp-840h]
  int v174; // [rsp+30h] [rbp-840h]
  char *v175; // [rsp+30h] [rbp-840h]
  char *v176; // [rsp+30h] [rbp-840h]
  _QWORD *v177; // [rsp+30h] [rbp-840h]
  char *v178; // [rsp+30h] [rbp-840h]
  char *v179; // [rsp+30h] [rbp-840h]
  __int64 v180; // [rsp+30h] [rbp-840h]
  _QWORD *v181; // [rsp+30h] [rbp-840h]
  __int64 v182; // [rsp+38h] [rbp-838h]
  __int64 v183; // [rsp+38h] [rbp-838h]
  char *srcb; // [rsp+40h] [rbp-830h]
  const char *srcc; // [rsp+40h] [rbp-830h]
  char *srca; // [rsp+40h] [rbp-830h]
  __int64 v188; // [rsp+50h] [rbp-820h]
  __int64 v189; // [rsp+50h] [rbp-820h]
  __int64 v190; // [rsp+50h] [rbp-820h]
  __int64 v191; // [rsp+50h] [rbp-820h]
  _QWORD *v192; // [rsp+88h] [rbp-7E8h] BYREF
  unsigned __int8 v193; // [rsp+92h] [rbp-7DEh] BYREF
  char v194; // [rsp+93h] [rbp-7DDh] BYREF
  int v195; // [rsp+94h] [rbp-7DCh] BYREF
  int v196; // [rsp+98h] [rbp-7D8h] BYREF
  int v197; // [rsp+9Ch] [rbp-7D4h] BYREF
  int v198; // [rsp+A0h] [rbp-7D0h] BYREF
  unsigned int v199; // [rsp+A4h] [rbp-7CCh] BYREF
  __int64 v200; // [rsp+A8h] [rbp-7C8h] BYREF
  __int64 v201; // [rsp+B0h] [rbp-7C0h] BYREF
  __int64 v202; // [rsp+B8h] [rbp-7B8h] BYREF
  __int64 v203; // [rsp+C0h] [rbp-7B0h] BYREF
  char *v204; // [rsp+C8h] [rbp-7A8h] BYREF
  char *s; // [rsp+D0h] [rbp-7A0h] BYREF
  char *v206; // [rsp+D8h] [rbp-798h] BYREF
  char *v207; // [rsp+E0h] [rbp-790h] BYREF
  char v208[8]; // [rsp+E8h] [rbp-788h] BYREF
  char v209[8]; // [rsp+F0h] [rbp-780h] BYREF
  __int64 v210; // [rsp+F8h] [rbp-778h] BYREF
  int v211[2]; // [rsp+100h] [rbp-770h] BYREF
  char *v212; // [rsp+108h] [rbp-768h] BYREF
  int v213[2]; // [rsp+110h] [rbp-760h] BYREF
  int v214[2]; // [rsp+118h] [rbp-758h] BYREF
  __int64 v215; // [rsp+120h] [rbp-750h] BYREF
  __int64 v216; // [rsp+128h] [rbp-748h] BYREF
  __int64 v217; // [rsp+130h] [rbp-740h] BYREF
  __int64 v218; // [rsp+138h] [rbp-738h] BYREF
  __int64 v219; // [rsp+140h] [rbp-730h] BYREF
  __int64 v220; // [rsp+148h] [rbp-728h] BYREF
  __int64 v221; // [rsp+150h] [rbp-720h] BYREF
  __int64 v222; // [rsp+158h] [rbp-718h] BYREF
  _QWORD *v223; // [rsp+160h] [rbp-710h] BYREF
  __int64 v224; // [rsp+168h] [rbp-708h]
  _QWORD v225[2]; // [rsp+170h] [rbp-700h] BYREF
  char *v226; // [rsp+180h] [rbp-6F0h] BYREF
  __int64 v227; // [rsp+188h] [rbp-6E8h]
  _QWORD v228[2]; // [rsp+190h] [rbp-6E0h] BYREF
  char *v229; // [rsp+1A0h] [rbp-6D0h] BYREF
  __int64 v230; // [rsp+1A8h] [rbp-6C8h]
  __int64 v231; // [rsp+1B0h] [rbp-6C0h] BYREF
  __int64 v232; // [rsp+1B8h] [rbp-6B8h]
  int v233; // [rsp+1C0h] [rbp-6B0h]
  char **v234; // [rsp+1C8h] [rbp-6A8h]
  _DWORD v235[4]; // [rsp+1D0h] [rbp-6A0h] BYREF
  _QWORD *v236; // [rsp+1E0h] [rbp-690h]
  __int64 v237; // [rsp+1E8h] [rbp-688h]
  _QWORD v238[2]; // [rsp+1F0h] [rbp-680h] BYREF
  _QWORD *v239; // [rsp+200h] [rbp-670h]
  __int64 v240; // [rsp+208h] [rbp-668h]
  _QWORD v241[2]; // [rsp+210h] [rbp-660h] BYREF
  _QWORD *v242; // [rsp+220h] [rbp-650h]
  __int64 v243; // [rsp+228h] [rbp-648h]
  _QWORD v244[2]; // [rsp+230h] [rbp-640h] BYREF
  _QWORD *v245; // [rsp+240h] [rbp-630h]
  __int64 v246; // [rsp+248h] [rbp-628h]
  _QWORD v247[2]; // [rsp+250h] [rbp-620h] BYREF
  _QWORD *v248; // [rsp+260h] [rbp-610h]
  __int64 v249; // [rsp+268h] [rbp-608h]
  _QWORD v250[2]; // [rsp+270h] [rbp-600h] BYREF
  _QWORD *v251; // [rsp+280h] [rbp-5F0h]
  __int64 v252; // [rsp+288h] [rbp-5E8h]
  _QWORD v253[2]; // [rsp+290h] [rbp-5E0h] BYREF
  __int64 v254; // [rsp+2A0h] [rbp-5D0h]
  __int64 v255; // [rsp+2A8h] [rbp-5C8h]
  __int64 v256; // [rsp+2B0h] [rbp-5C0h]
  void *v257; // [rsp+2D0h] [rbp-5A0h] BYREF
  char v258[16]; // [rsp+2D8h] [rbp-598h] BYREF
  __int64 *v259; // [rsp+2E8h] [rbp-588h]
  __int64 v260; // [rsp+2F8h] [rbp-578h] BYREF
  __int64 *v261; // [rsp+308h] [rbp-568h]
  __int64 v262; // [rsp+318h] [rbp-558h] BYREF
  __int64 *v263; // [rsp+328h] [rbp-548h]
  __int64 v264; // [rsp+338h] [rbp-538h] BYREF
  __int64 *v265; // [rsp+348h] [rbp-528h]
  __int64 v266; // [rsp+358h] [rbp-518h] BYREF
  __int64 *v267; // [rsp+368h] [rbp-508h]
  __int64 v268; // [rsp+378h] [rbp-4F8h] BYREF
  __int64 *v269; // [rsp+388h] [rbp-4E8h]
  __int64 v270; // [rsp+398h] [rbp-4D8h] BYREF
  __int64 v271; // [rsp+3A8h] [rbp-4C8h]
  unsigned int v272; // [rsp+3B0h] [rbp-4C0h]
  int v273; // [rsp+3B4h] [rbp-4BCh]
  int *v274; // [rsp+3D0h] [rbp-4A0h]
  __int64 v275; // [rsp+3D8h] [rbp-498h] BYREF
  bool v276[8]; // [rsp+3E0h] [rbp-490h] BYREF
  __int64 v277; // [rsp+3E8h] [rbp-488h]
  __int64 v278; // [rsp+3F0h] [rbp-480h]
  void *v279; // [rsp+400h] [rbp-470h] BYREF
  char v280[16]; // [rsp+408h] [rbp-468h] BYREF
  __int64 *v281; // [rsp+418h] [rbp-458h]
  __int64 v282; // [rsp+428h] [rbp-448h] BYREF
  __int64 *v283; // [rsp+438h] [rbp-438h]
  __int64 v284; // [rsp+448h] [rbp-428h] BYREF
  __int64 *v285; // [rsp+458h] [rbp-418h]
  __int64 v286; // [rsp+468h] [rbp-408h] BYREF
  __int64 *v287; // [rsp+478h] [rbp-3F8h]
  __int64 v288; // [rsp+488h] [rbp-3E8h] BYREF
  __int64 *v289; // [rsp+498h] [rbp-3D8h]
  __int64 v290; // [rsp+4A8h] [rbp-3C8h] BYREF
  __int64 *v291; // [rsp+4B8h] [rbp-3B8h]
  __int64 v292; // [rsp+4C8h] [rbp-3A8h] BYREF
  __int64 v293; // [rsp+4D8h] [rbp-398h]
  unsigned int v294; // [rsp+4E0h] [rbp-390h]
  int v295; // [rsp+4E4h] [rbp-38Ch]
  unsigned int *v296; // [rsp+500h] [rbp-370h]
  __int64 v297; // [rsp+508h] [rbp-368h] BYREF
  bool v298[8]; // [rsp+510h] [rbp-360h] BYREF
  __int64 v299; // [rsp+518h] [rbp-358h]
  __int64 v300; // [rsp+520h] [rbp-350h]
  __int64 v301; // [rsp+528h] [rbp-348h]
  void *v302; // [rsp+530h] [rbp-340h] BYREF
  char v303[16]; // [rsp+538h] [rbp-338h] BYREF
  __int64 *v304; // [rsp+548h] [rbp-328h]
  __int64 v305; // [rsp+558h] [rbp-318h] BYREF
  __int64 *v306; // [rsp+568h] [rbp-308h]
  __int64 v307; // [rsp+578h] [rbp-2F8h] BYREF
  __int64 *v308; // [rsp+588h] [rbp-2E8h]
  __int64 v309; // [rsp+598h] [rbp-2D8h] BYREF
  __int64 *v310; // [rsp+5A8h] [rbp-2C8h]
  __int64 v311; // [rsp+5B8h] [rbp-2B8h] BYREF
  __int64 *v312; // [rsp+5C8h] [rbp-2A8h]
  __int64 v313; // [rsp+5D8h] [rbp-298h] BYREF
  __int64 *v314; // [rsp+5E8h] [rbp-288h]
  __int64 v315; // [rsp+5F8h] [rbp-278h] BYREF
  __int64 v316; // [rsp+608h] [rbp-268h]
  unsigned int v317; // [rsp+610h] [rbp-260h]
  unsigned int v318; // [rsp+614h] [rbp-25Ch]
  unsigned int *v319; // [rsp+630h] [rbp-240h]
  __int64 v320; // [rsp+638h] [rbp-238h] BYREF
  bool v321[8]; // [rsp+640h] [rbp-230h] BYREF
  __int64 v322; // [rsp+648h] [rbp-228h]
  _QWORD *v323; // [rsp+650h] [rbp-220h]
  __int64 v324; // [rsp+658h] [rbp-218h]
  unsigned int *v325; // [rsp+660h] [rbp-210h]
  char **v326; // [rsp+670h] [rbp-200h] BYREF
  char **v327; // [rsp+678h] [rbp-1F8h]
  unsigned __int8 *v328; // [rsp+680h] [rbp-1F0h]
  char **v329; // [rsp+688h] [rbp-1E8h]
  __int64 *v330; // [rsp+690h] [rbp-1E0h]
  char **v331; // [rsp+698h] [rbp-1D8h]
  _QWORD **v332; // [rsp+6A0h] [rbp-1D0h]

  v5 = a4;
  v192 = (_QWORD *)a1;
  v236 = v238;
  v239 = v241;
  v242 = v244;
  v199 = 0;
  v215 = 0;
  v216 = 0;
  v217 = 0;
  v218 = 0;
  v219 = 0;
  v220 = 0;
  v221 = 0;
  v222 = 0;
  v204 = 0;
  v235[2] = 0;
  v237 = 0;
  LOBYTE(v238[0]) = 0;
  v240 = 0;
  LOBYTE(v241[0]) = 0;
  v243 = 0;
  v251 = v253;
  v256 = 0x1000000000LL;
  v252 = 0;
  LOBYTE(v253[0]) = 0;
  v254 = 0;
  v255 = 0;
  v7 = *(_DWORD *)(a1 + 176);
  LOBYTE(v244[0]) = 0;
  v245 = v247;
  v246 = 0;
  LOBYTE(v247[0]) = 0;
  v248 = v250;
  v249 = 0;
  LOBYTE(v250[0]) = 0;
  if ( !(unsigned int)sub_12D2AA0(
                        a3,
                        a4,
                        v7,
                        (unsigned int)&v195,
                        (unsigned int)&v200,
                        (unsigned int)&v196,
                        (__int64)&v201,
                        (__int64)&v197,
                        (__int64)&v202,
                        (__int64)&v198,
                        (__int64)&v203,
                        (__int64)&v199,
                        (__int64)&v204,
                        (__int64)v235) )
  {
    if ( v195 != (_DWORD)v215 || v200 != v216 )
    {
      v171 = v195;
      v188 = v200;
      sub_12C7AC0(&v215, &v216);
      LODWORD(v215) = v171;
      v216 = v188;
    }
    if ( v196 != (_DWORD)v217 || v201 != v218 )
    {
      v172 = v196;
      v189 = v201;
      sub_12C7AC0(&v217, &v218);
      LODWORD(v217) = v172;
      v218 = v189;
    }
    if ( v197 != (_DWORD)v219 || v202 != v220 )
    {
      v173 = v197;
      v190 = v202;
      sub_12C7AC0(&v219, &v220);
      LODWORD(v219) = v173;
      v220 = v190;
    }
    if ( v198 != (_DWORD)v221 || v203 != v222 )
    {
      v174 = v198;
      v191 = v203;
      sub_12C7AC0(&v221, &v222);
      LODWORD(v221) = v174;
      v222 = v191;
    }
    v235[0] = a2;
    v257 = &unk_49E6A40;
    sub_12BE2E0((__int64)v258, (__int64)v235);
    v274 = &v195;
    v277 = 0;
    v275 = v200;
    v278 = 0;
    v276[0] = v235[0] == 0;
    v257 = &unk_49E7FF0;
    v8 = sub_22077B0(480);
    v9 = v8;
    if ( v8 )
      sub_12EC960(v8, "nvllc", 5);
    v279 = &unk_49E6A40;
    sub_12BE2E0((__int64)v280, (__int64)v235);
    v296 = (unsigned int *)&v198;
    v297 = v203;
    v298[0] = v235[0] == 0;
    v299 = 0;
    v300 = v9;
    v279 = &unk_49E7FD8;
    v301 = 0;
    v10 = sub_22077B0(512);
    v11 = (_QWORD *)v10;
    if ( v10 )
    {
      sub_12EC960(v10, "nvopt", 5);
      v11[61] = 0;
      v11[62] = 0;
      v11[63] = 0;
      *v11 = &unk_49E6A58;
      v11[60] = &unk_49E6B20;
    }
    v12 = (unsigned __int64)v235;
    v302 = &unk_49E6A40;
    sub_12BE2E0((__int64)v303, (__int64)v235);
    v319 = (unsigned int *)&v196;
    v322 = 0;
    v320 = v201;
    v321[0] = v235[0] == 0;
    v323 = v11;
    v302 = &unk_49E6B38;
    v324 = 0;
    v325 = &v199;
    if ( v195 > 0 )
    {
      v326 = &v229;
      *(_QWORD *)(__readfsqword(0) - 24) = &v326;
      *(_QWORD *)(__readfsqword(0) - 32) = sub_12BCC20;
      if ( !&_pthread_key_create )
        goto LABEL_294;
      v59 = pthread_once(&dword_4F92D9C, init_routine);
      if ( v59 )
        goto LABEL_295;
      nullsub_501(&v257);
      if ( v278 )
        (*(void (__fastcall **)(__int64, int *, __int64 *, _QWORD))(*(_QWORD *)v278 + 16LL))(v278, v274, &v275, 0);
      v12 = (unsigned int)*v274;
      sub_1C427B0(v276, v12, v275, byte_3F871B3);
      v13 = v199;
      if ( (v199 & 0x82) == 0 )
        goto LABEL_20;
    }
    else
    {
      v13 = v199;
      if ( (v199 & 0x82) == 0 )
      {
LABEL_20:
        if ( (v13 & 4) == 0 )
          goto LABEL_21;
LABEL_139:
        v326 = &v229;
        *(_QWORD *)(__readfsqword(0) - 24) = &v326;
        *(_QWORD *)(__readfsqword(0) - 32) = sub_12BCC20;
        if ( &_pthread_key_create )
        {
          v59 = pthread_once(&dword_4F92D9C, init_routine);
          if ( !v59 )
          {
            sub_12EE670(&v279);
            if ( v300 )
              (*(void (__fastcall **)(__int64, unsigned int *, __int64 *, _QWORD))(*(_QWORD *)v300 + 16LL))(
                v300,
                v296,
                &v297,
                0);
            v12 = *v296;
            sub_1C427B0(v298, v12, v297, byte_3F871B3);
            goto LABEL_21;
          }
LABEL_295:
          sub_4264C5(v59);
        }
LABEL_294:
        v59 = -1;
        goto LABEL_295;
      }
    }
    v326 = &v229;
    *(_QWORD *)(__readfsqword(0) - 24) = &v326;
    *(_QWORD *)(__readfsqword(0) - 32) = sub_12BCC20;
    if ( !&_pthread_key_create )
      goto LABEL_294;
    v59 = pthread_once(&dword_4F92D9C, init_routine);
    if ( v59 )
      goto LABEL_295;
    sub_12D4300(&v302);
    if ( v323 )
      (*(void (__fastcall **)(_QWORD *, unsigned int *, __int64 *, _QWORD))(*v323 + 16LL))(v323, v319, &v320, 0);
    v12 = *v319;
    sub_1C427B0(v321, v12, v320, byte_3F871B3);
    if ( (v199 & 4) == 0 )
    {
LABEL_21:
      s = 0;
      sub_1C3E9C0(&s);
      v14 = s;
      if ( s )
      {
        v15 = strlen(s);
        if ( v15 > 0x3FFFFFFFFFFFFFFFLL - v192[11] )
          goto LABEL_301;
        v12 = (unsigned __int64)v14;
        sub_2241490(v192 + 10, v14, v15, v16);
        if ( s )
          j_j___libc_free_0_0(s);
      }
      v193 = 0;
      v206 = "nvvmCompileProgram";
      v207 = "LibNVVM program compilation.";
      sub_1602D10(v208);
      sub_1602D10(v209);
      v210 = 0;
      if ( sub_16DA870(v209, v12, v17, v18, v19, v20) )
        sub_16DB3F0("NVVM Module Linker", 18, byte_3F871B3, 0);
      v21 = v192;
      v22 = (__int64)&v326;
      v23 = sub_12C06E0(v192, &v326, v199, (__int64)v208, (__int64)v235);
      v28 = (_QWORD *)v210;
      v210 = v23;
      if ( v28 )
      {
        sub_1633490(v28);
        v22 = 736;
        v21 = v28;
        j_j___libc_free_0(v28, 736);
      }
      v29 = (unsigned int)v326;
      if ( (_DWORD)v326 )
      {
        v60 = sub_16DA870(v21, v22, v24, v25, v26, v27);
        v32 = &v219;
        if ( v60 )
          sub_16DB5E0();
        goto LABEL_35;
      }
      if ( sub_16DA870(v21, v22, v24, v25, v26, v27) )
        sub_16DB5E0();
      v30 = (unsigned int (__fastcall *)(_QWORD, _QWORD))v192[26];
      if ( v30 )
      {
        v22 = 0;
        if ( v30(v192[27], 0) )
        {
          v29 = 10;
LABEL_35:
          v35 = v210;
          if ( v210 )
          {
            sub_1633490(v210);
            v22 = 736;
            j_j___libc_free_0(v35, 736);
          }
          sub_16025D0(v209, v22, v31, v32, v33, v34);
          sub_16025D0(v208, v22, v36, v37, v38, v39);
          v40 = v324;
          v302 = &unk_49E6B38;
          if ( v324 )
          {
            v41 = *(_QWORD *)(v324 + 4488);
            if ( v41 )
              j_j___libc_free_0(v41, *(_QWORD *)(v324 + 4504) - v41);
            j_j___libc_free_0(v40, 4512);
          }
          v302 = &unk_49E6A40;
          if ( v323 )
            (*(void (__fastcall **)(_QWORD *))(*v323 + 8LL))(v323);
          sub_1C428D0(v321);
          v5 = v318;
          if ( v318 )
          {
            v42 = v316;
            if ( v317 )
            {
              v43 = 8LL * v317;
              v44 = 0;
              do
              {
                v45 = *(_QWORD *)(v42 + v44);
                if ( v45 != -8 && v45 )
                {
                  _libc_free(v45, v5);
                  v42 = v316;
                }
                v44 += 8;
              }
              while ( v43 != v44 );
            }
          }
          else
          {
            v42 = v316;
          }
          _libc_free(v42, v5);
          if ( v314 != &v315 )
          {
            v5 = v315 + 1;
            j_j___libc_free_0(v314, v315 + 1);
          }
          if ( v312 != &v313 )
          {
            v5 = v313 + 1;
            j_j___libc_free_0(v312, v313 + 1);
          }
          if ( v310 != &v311 )
          {
            v5 = v311 + 1;
            j_j___libc_free_0(v310, v311 + 1);
          }
          if ( v308 != &v309 )
          {
            v5 = v309 + 1;
            j_j___libc_free_0(v308, v309 + 1);
          }
          if ( v306 != &v307 )
          {
            v5 = v307 + 1;
            j_j___libc_free_0(v306, v307 + 1);
          }
          if ( v304 != &v305 )
          {
            v5 = v305 + 1;
            j_j___libc_free_0(v304, v305 + 1);
          }
          v279 = &unk_49E7FD8;
          if ( v301 )
          {
            v5 = 4480;
            j_j___libc_free_0(v301, 4480);
          }
          v279 = &unk_49E6A40;
          if ( v300 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v300 + 8LL))(v300);
          sub_1C428D0(v298);
          if ( v295 )
          {
            v46 = v293;
            if ( v294 )
            {
              v47 = 8LL * v294;
              v48 = 0;
              do
              {
                v49 = *(_QWORD *)(v46 + v48);
                if ( v49 != -8 && v49 )
                {
                  _libc_free(v49, v5);
                  v46 = v293;
                }
                v48 += 8;
              }
              while ( v47 != v48 );
            }
          }
          else
          {
            v46 = v293;
          }
          _libc_free(v46, v5);
          if ( v291 != &v292 )
          {
            v5 = v292 + 1;
            j_j___libc_free_0(v291, v292 + 1);
          }
          if ( v289 != &v290 )
          {
            v5 = v290 + 1;
            j_j___libc_free_0(v289, v290 + 1);
          }
          if ( v287 != &v288 )
          {
            v5 = v288 + 1;
            j_j___libc_free_0(v287, v288 + 1);
          }
          if ( v285 != &v286 )
          {
            v5 = v286 + 1;
            j_j___libc_free_0(v285, v286 + 1);
          }
          if ( v283 != &v284 )
          {
            v5 = v284 + 1;
            j_j___libc_free_0(v283, v284 + 1);
          }
          if ( v281 != &v282 )
          {
            v5 = v282 + 1;
            j_j___libc_free_0(v281, v282 + 1);
          }
          v257 = &unk_49E6A40;
          if ( v278 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v278 + 8LL))(v278);
          sub_1C428D0(v276);
          if ( v273 )
          {
            v50 = v271;
            if ( v272 )
            {
              v51 = 8LL * v272;
              v52 = 0;
              do
              {
                v53 = *(_QWORD *)(v50 + v52);
                if ( v53 != -8 && v53 )
                {
                  _libc_free(v53, v5);
                  v50 = v271;
                }
                v52 += 8;
              }
              while ( v51 != v52 );
            }
          }
          else
          {
            v50 = v271;
          }
          _libc_free(v50, v5);
          if ( v269 != &v270 )
          {
            v5 = v270 + 1;
            j_j___libc_free_0(v269, v270 + 1);
          }
          if ( v267 != &v268 )
          {
            v5 = v268 + 1;
            j_j___libc_free_0(v267, v268 + 1);
          }
          if ( v265 != &v266 )
          {
            v5 = v266 + 1;
            j_j___libc_free_0(v265, v266 + 1);
          }
          if ( v263 != &v264 )
          {
            v5 = v264 + 1;
            j_j___libc_free_0(v263, v264 + 1);
          }
          if ( v261 != &v262 )
          {
            v5 = v262 + 1;
            j_j___libc_free_0(v261, v262 + 1);
          }
          if ( v259 != &v260 )
          {
            v5 = v260 + 1;
            j_j___libc_free_0(v259, v260 + 1);
          }
          goto LABEL_106;
        }
      }
      if ( (v199 & 1) != 0 )
      {
        v61 = 0;
        v62 = v207;
        if ( v207 )
          v61 = strlen(v207);
        v63 = v61;
        v64 = 0;
        v65 = v206;
        if ( v206 )
        {
          srcb = v206;
          v64 = strlen(v206);
          v65 = srcb;
        }
        v66 = "LNK";
        sub_16D8B50(
          (int)&v226,
          (int)"LNK",
          3,
          (int)"LibNVVM module linking step.",
          28,
          v193,
          v65,
          v64,
          (__int64)v62,
          v63);
        if ( v192[14] )
        {
          v331 = &v229;
          v326 = (char **)&unk_49EFBE0;
          v229 = (char *)&v231;
          v230 = 0;
          LOBYTE(v231) = 0;
          LODWORD(v330) = 1;
          v329 = 0;
          v328 = 0;
          v327 = 0;
          sub_153BF40(v210, &v326, 1, 0, 0, 0);
          v68 = (void (__fastcall *)(char *, const char *, __int64))v192[14];
          v69 = v192[15];
          if ( v329 == v327 )
          {
            v70 = v331;
            v66 = v331[1];
          }
          else
          {
            sub_16E7BA0(&v326);
            v70 = v331;
            v66 = v331[1];
            if ( v329 != v327 )
            {
              srcc = v331[1];
              sub_16E7BA0(&v326);
              v70 = v331;
              v66 = srcc;
            }
          }
          v68(*v70, v66, v69);
          sub_16E7BC0(&v326);
          if ( v229 != (char *)&v231 )
          {
            v66 = (const char *)(v231 + 1);
            j_j___libc_free_0(v229, v231 + 1);
          }
        }
        if ( v226 )
          sub_16D7950(v226, v66, v67);
      }
      v71 = sub_1632FA0(v210);
      v22 = *(_QWORD *)(v71 + 192);
      v72 = *(_QWORD *)(v71 + 200);
      v223 = v225;
      sub_12BCB70((__int64 *)&v223, (_BYTE *)v22, v22 + v72);
      v73 = v199;
      if ( (v199 & 8) != 0 )
      {
        if ( !v224 )
        {
LABEL_159:
          v32 = &v219;
          v194 = 0;
          v326 = &v206;
          v327 = &v207;
          v328 = &v193;
          v330 = &v210;
          v331 = (char **)&v194;
          v329 = (char **)&v219;
          v332 = &v192;
          if ( (v73 & 0xA0) == 0x20 && !(unsigned __int8)sub_12BD6B0((__int64)&v326) )
            goto LABEL_277;
          v74 = (unsigned int (__fastcall *)(_QWORD, _QWORD))v192[26];
          if ( v74 )
          {
            v22 = 0;
            if ( v74(v192[27], 0) )
            {
LABEL_162:
              v29 = 10;
              goto LABEL_163;
            }
          }
          v87 = v199;
          if ( (v199 & 0x82) != 0 )
          {
            v88 = v193;
            v89 = 0;
            v90 = v207;
            if ( v207 )
            {
              v160 = v207;
              v89 = strlen(v207);
              v90 = v160;
            }
            v91 = v89;
            v92 = 0;
            v93 = v206;
            if ( v206 )
            {
              v161 = v90;
              v165 = v91;
              v92 = strlen(v206);
              v90 = v161;
              v91 = v165;
            }
            v94 = v211;
            v95 = "OPT";
            sub_16D8B50(
              (int)v211,
              (int)"OPT",
              3,
              (int)"LibNVVM optimization step.",
              26,
              v88,
              v93,
              v92,
              (__int64)v90,
              v91);
            if ( v192[16] )
            {
              v234 = &v226;
              v229 = (char *)&unk_49EFBE0;
              v226 = (char *)v228;
              v227 = 0;
              LOBYTE(v228[0]) = 0;
              v233 = 1;
              v232 = 0;
              v231 = 0;
              v230 = 0;
              sub_153BF40(v210, &v229, 1, 0, 0, 0);
              v100 = (void (__fastcall *)(char *, char *, __int64))v192[16];
              v101 = v192[17];
              if ( v232 == v230 )
              {
                v102 = v234;
                v95 = v234[1];
              }
              else
              {
                v162 = v192[17];
                sub_16E7BA0(&v229);
                v102 = v234;
                v101 = v162;
                v95 = v234[1];
                if ( v232 != v230 )
                {
                  v166 = v234[1];
                  sub_16E7BA0(&v229);
                  v102 = v234;
                  v95 = v166;
                  v101 = v162;
                }
              }
              v100(*v102, v95, v101);
              sub_16E7BC0(&v229);
              v94 = v226;
              if ( v226 != (char *)v228 )
              {
                v95 = (char *)(v228[0] + 1LL);
                j_j___libc_free_0(v226, v228[0] + 1LL);
              }
            }
            if ( sub_16DA870(v94, v95, v96, v97, v98, v99) )
              sub_16DB3F0("NVVM Optimizer", 14, byte_3F871B3, 0);
            v22 = v210;
            v212 = 0;
            v103 = sub_12E7E70(
                     (unsigned int)&v302,
                     v210,
                     (unsigned int)&v212,
                     (unsigned int)v209,
                     (unsigned int)&v279,
                     (int)v192 + 48,
                     (__int64)(v192 + 23),
                     (__int64)(v192 + 26));
            v106 = v212;
            v107 = v158;
            v108 = v103;
            v109 = v159;
            if ( v212 )
            {
              v163 = v192;
              v110 = v192 + 10;
              v111 = strlen(v212);
              if ( v111 > 0x3FFFFFFFFFFFFFFFLL - v163[11] )
                goto LABEL_301;
              v22 = (__int64)v106;
              sub_2241490(v110, v106, v111, v163);
              v107 = v212;
              if ( v212 )
                j_j___libc_free_0_0(v212);
              v212 = 0;
            }
            v112 = v192;
            v113 = (unsigned int (__fastcall *)(char *, _QWORD))v192[26];
            if ( v113 )
            {
              v22 = 0;
              v107 = (char *)v192[27];
              if ( v113(v107, 0) )
              {
                v29 = 10;
LABEL_214:
                if ( sub_16DA870(v107, v22, v112, v104, v109, v105) )
                  sub_16DB5E0();
                if ( *(_QWORD *)v211 )
                  sub_16D7950(*(_QWORD *)v211, v22, v31);
                goto LABEL_163;
              }
            }
            if ( !v108 )
            {
              v29 = 9;
              goto LABEL_214;
            }
            v114 = (char *)v210;
            if ( v108 != v210 )
            {
              v210 = v108;
              if ( v114 )
              {
                sub_1633490(v114);
                v22 = 736;
                v107 = v114;
                j_j___libc_free_0(v114, 736);
              }
            }
            if ( v192[18] )
            {
              v234 = &v226;
              v229 = (char *)&unk_49EFBE0;
              v226 = (char *)v228;
              v227 = 0;
              LOBYTE(v228[0]) = 0;
              v233 = 1;
              v232 = 0;
              v231 = 0;
              v230 = 0;
              sub_153BF40(v210, &v229, 1, 0, 0, 0);
              v115 = (void (__fastcall *)(char *, __int64, __int64))v192[18];
              v116 = v192[19];
              if ( v232 == v230 )
              {
                v117 = v234;
                v22 = (__int64)v234[1];
              }
              else
              {
                v164 = v192[19];
                sub_16E7BA0(&v229);
                v117 = v234;
                v116 = v164;
                v22 = (__int64)v234[1];
                if ( v232 != v230 )
                {
                  v167 = v234[1];
                  sub_16E7BA0(&v229);
                  v117 = v234;
                  v22 = (__int64)v167;
                  v116 = v164;
                }
              }
              v115(*v117, v22, v116);
              sub_16E7BC0(&v229);
              v107 = v226;
              if ( v226 != (char *)v228 )
              {
                v22 = v228[0] + 1LL;
                j_j___libc_free_0(v226, v228[0] + 1LL);
              }
            }
            if ( sub_16DA870(v107, v22, v112, v104, v109, v105) )
              sub_16DB5E0();
            if ( *(_QWORD *)v211 )
              sub_16D7950(*(_QWORD *)v211, v22, v118);
            v87 = v199;
          }
          if ( v87 < 0 )
          {
            if ( !(unsigned __int8)sub_12BD6B0((__int64)&v326) )
            {
LABEL_277:
              v29 = 9;
              goto LABEL_163;
            }
            v87 = v199;
          }
          if ( (v87 & 0x40) != 0 )
          {
            v119 = v193;
            v120 = 0;
            v121 = v207;
            if ( v207 )
            {
              v175 = v207;
              v120 = strlen(v207);
              v121 = v175;
            }
            v122 = v120;
            v123 = 0;
            v124 = v206;
            if ( v206 )
            {
              v176 = v121;
              v182 = v122;
              v123 = strlen(v206);
              v121 = v176;
              v122 = v182;
            }
            sub_16D8B50(
              (int)v213,
              (int)"OPTIXIR",
              7,
              (int)"LibNVVM Optix IR step.",
              22,
              v119,
              v124,
              v123,
              (__int64)v121,
              v122);
            v22 = a4;
            v226 = 0;
            v125 = sub_12F9270(a3, a4, v210, v192 + 6, &v226);
            v126 = v226;
            v127 = v125;
            if ( v226 )
            {
              v177 = v192;
              v128 = v192 + 10;
              v129 = strlen(v226);
              if ( v129 > 0x3FFFFFFFFFFFFFFFLL - v177[11] )
                goto LABEL_301;
              v22 = (__int64)v126;
              sub_2241490(v128, v126, v129, v177);
              if ( v226 )
                j_j___libc_free_0_0(v226);
              v226 = 0;
            }
            v130 = *(_QWORD *)v213;
            if ( !v127 )
              goto LABEL_285;
            if ( *(_QWORD *)v213 )
              sub_16D7950(*(_QWORD *)v213, v22, v31);
            v87 = v199;
          }
          if ( (v87 & 4) == 0 )
            goto LABEL_271;
          v131 = v193;
          v132 = 0;
          v133 = v207;
          if ( v207 )
          {
            v178 = v207;
            v132 = strlen(v207);
            v133 = v178;
          }
          v134 = v132;
          v135 = 0;
          v136 = v206;
          if ( v206 )
          {
            v179 = v133;
            v183 = v134;
            v135 = strlen(v206);
            v133 = v179;
            v134 = v183;
          }
          v137 = v214;
          v138 = "LLC";
          sub_16D8B50(
            (int)v214,
            (int)"LLC",
            3,
            (int)"LibNVVM code-generation step.",
            29,
            v131,
            v136,
            v135,
            (__int64)v133,
            v134);
          if ( v192[20] )
          {
            v234 = &v226;
            v229 = (char *)&unk_49EFBE0;
            v226 = (char *)v228;
            v227 = 0;
            LOBYTE(v228[0]) = 0;
            v233 = 1;
            v232 = 0;
            v231 = 0;
            v230 = 0;
            sub_153BF40(v210, &v229, 1, 0, 0, 0);
            v143 = (void (__fastcall *)(char *, const char *, __int64))v192[20];
            v144 = v192[21];
            if ( v232 == v230 )
            {
              v145 = v234;
              v138 = v234[1];
            }
            else
            {
              v180 = v192[21];
              sub_16E7BA0(&v229);
              v145 = v234;
              v144 = v180;
              v138 = v234[1];
              if ( v232 != v230 )
              {
                v170 = v234[1];
                sub_16E7BA0(&v229);
                v145 = v234;
                v138 = v170;
                v144 = v180;
              }
            }
            v143(*v145, v138, v144);
            sub_16E7BC0(&v229);
            v137 = v226;
            if ( v226 != (char *)v228 )
            {
              v138 = (const char *)(v228[0] + 1LL);
              j_j___libc_free_0(v226, v228[0] + 1LL);
            }
          }
          if ( sub_16DA870(v137, v138, v139, v140, v141, v142) )
            sub_16DB3F0("NVVM CodeGen", 12, byte_3F871B3, 0);
          v22 = v210;
          v146 = (char *)&v279;
          v229 = 0;
          v147 = sub_12F5100(&v279, v210, v192 + 6, &v229, v192 + 26);
          v152 = v229;
          v153 = v147;
          if ( !v229 )
          {
LABEL_266:
            if ( v153 )
            {
              if ( sub_16DA870(v146, v22, v148, v149, v150, v151) )
                sub_16DB5E0();
              if ( *(_QWORD *)v214 )
                sub_16D7950(*(_QWORD *)v214, v22, v156);
LABEL_271:
              v31 = v192;
              v157 = (unsigned int (__fastcall *)(_QWORD, _QWORD))v192[26];
              if ( !v157 || (v22 = 0, !v157(v192[27], 0)) )
              {
                if ( v194 )
                  v29 = 100;
                goto LABEL_163;
              }
              goto LABEL_162;
            }
            if ( sub_16DA870(v146, v22, v148, v149, v150, v151) )
              sub_16DB5E0();
            v130 = *(_QWORD *)v214;
LABEL_285:
            if ( v130 )
              sub_16D7950(v130, v22, v31);
            goto LABEL_277;
          }
          v181 = v192;
          v154 = v192 + 10;
          v155 = strlen(v229);
          if ( v155 <= 0x3FFFFFFFFFFFFFFFLL - v181[11] )
          {
            v22 = (__int64)v152;
            sub_2241490(v154, v152, v155, v181);
            v146 = v229;
            if ( v229 )
              j_j___libc_free_0_0(v229);
            v229 = 0;
            goto LABEL_266;
          }
LABEL_301:
          sub_4262D8((__int64)"basic_string::append");
        }
      }
      else if ( !v224 )
      {
        v22 = (__int64)"DataLayoutError: Data Layout string is empty";
        LODWORD(v330) = 1;
        v329 = 0;
        v29 = 9;
        v328 = 0;
        v326 = (char **)&unk_49EFBE0;
        v327 = 0;
        v331 = (char **)(v192 + 10);
        sub_16E7EE0(&v326, "DataLayoutError: Data Layout string is empty", 44);
        sub_16E7BC0(&v326);
LABEL_163:
        if ( v223 != v225 )
        {
          v22 = v225[0] + 1LL;
          j_j___libc_free_0(v223, v225[0] + 1LL);
        }
        goto LABEL_35;
      }
      v77 = sub_1632FA0(v210);
      sub_12BDBC0((__int64)&v326, v77);
      if ( (unsigned __int8)sub_1C27B80(&v223, 1) )
        sub_1632B30(v210, v223, v224);
      v22 = sub_1632FA0(v210);
      sub_1C27E00(v213, v22, v208, 0, 1, 0);
      if ( (*(_QWORD *)v213 & 0xFFFFFFFFFFFFFFFELL) == 0 )
      {
        sub_15A93E0(&v326);
        v73 = v199;
        goto LABEL_159;
      }
      *(_QWORD *)v214 = *(_QWORD *)v213 & 0xFFFFFFFFFFFFFFFELL | 1;
      v233 = 1;
      v234 = (char **)(v192 + 10);
      v232 = 0;
      v229 = (char *)&unk_49EFBE0;
      v231 = 0;
      v230 = 0;
      *(_QWORD *)v213 = 0;
      sub_12BF440((__int64)&v226, (__int64 *)v214);
      sub_16E7EE0(&v229, v226, v227);
      if ( v226 != (char *)v228 )
        j_j___libc_free_0(v226, v228[0] + 1LL);
      if ( (v214[0] & 1) != 0 || (*(_QWORD *)v214 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_16BCAE0(v214);
      v78 = (__m128i *)v232;
      if ( (unsigned __int64)(v231 - v232) <= 0x1B )
      {
        sub_16E7EE0(&v229, "\nExample valid data layout:\n", 28);
        v80 = (_QWORD *)v232;
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_4281920);
        *(_DWORD *)(v232 + 24) = 171603061;
        v78[1].m128i_i64[0] = 0x6F79616C20617461LL;
        *v78 = si128;
        v80 = (_QWORD *)(v232 + 28);
        v232 += 28;
      }
      if ( (unsigned __int64)(v231 - (_QWORD)v80) <= 7 )
      {
        v81 = (char **)sub_16E7EE0(&v229, "64-bit: ", 8);
      }
      else
      {
        v81 = &v229;
        *v80 = 0x203A7469622D3436LL;
        v232 += 8;
      }
      v22 = (__int64)off_4CD4948[0];
      if ( off_4CD4948[0] )
      {
        srca = off_4CD4948[0];
        v82 = strlen(off_4CD4948[0]);
        v83 = v81[3];
        v22 = (__int64)srca;
        v84 = v82;
        v85 = v81[2];
        if ( v84 <= v85 - v83 )
        {
          if ( v84 )
          {
            memcpy(v83, srca, v84);
            v86 = &v81[3][v84];
            v85 = v81[2];
            v81[3] = v86;
            v83 = v86;
          }
          goto LABEL_188;
        }
        v81 = (char **)sub_16E7EE0(v81, srca, v84);
      }
      v85 = v81[2];
      v83 = v81[3];
LABEL_188:
      if ( v83 == v85 )
      {
        v22 = (__int64)"\n";
        sub_16E7EE0(v81, "\n", 1);
      }
      else
      {
        *v83 = 10;
        ++v81[3];
      }
      sub_16E7BC0(&v229);
      if ( (v213[0] & 1) != 0 || (*(_QWORD *)v213 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_16BCAE0(v213);
      v29 = 9;
      sub_15A93E0(&v326);
      goto LABEL_163;
    }
    goto LABEL_139;
  }
  v75 = v204;
  if ( v204 )
  {
    v76 = strlen(v204);
    v5 = 0;
    sub_2241130(v192 + 10, 0, v192[11], v75, v76);
    if ( v204 )
      j_j___libc_free_0_0(v204);
  }
  v29 = 7;
LABEL_106:
  v54 = v254;
  if ( HIDWORD(v255) && (_DWORD)v255 )
  {
    v55 = 8LL * (unsigned int)v255;
    v56 = 0;
    do
    {
      v57 = *(_QWORD *)(v54 + v56);
      if ( v57 != -8 && v57 )
      {
        _libc_free(v57, v5);
        v54 = v254;
      }
      v56 += 8;
    }
    while ( v56 != v55 );
  }
  _libc_free(v54, v5);
  if ( v251 != v253 )
    j_j___libc_free_0(v251, v253[0] + 1LL);
  if ( v248 != v250 )
    j_j___libc_free_0(v248, v250[0] + 1LL);
  if ( v245 != v247 )
    j_j___libc_free_0(v245, v247[0] + 1LL);
  if ( v242 != v244 )
    j_j___libc_free_0(v242, v244[0] + 1LL);
  if ( v239 != v241 )
    j_j___libc_free_0(v239, v241[0] + 1LL);
  if ( v236 != v238 )
    j_j___libc_free_0(v236, v238[0] + 1LL);
  sub_12C7AC0(&v221, &v222);
  sub_12C7AC0(&v219, &v220);
  sub_12C7AC0(&v217, &v218);
  sub_12C7AC0(&v215, &v216);
  return v29;
}
