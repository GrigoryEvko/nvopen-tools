// Function: sub_D555F0
// Address: 0xd555f0
//
__int64 __fastcall sub_D555F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rdi
  __int64 *v18; // r14
  unsigned __int64 v19; // rbx
  __int64 v20; // rdi
  __int64 v21; // rdi
  __int64 *v22; // r14
  unsigned __int64 v23; // rbx
  __int64 v24; // rdi
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  unsigned __int64 v29; // rbx
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 *v38; // rsi
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 v41; // rcx
  __int64 v42; // rdx
  __int64 v43; // rdi
  __int64 *v44; // r14
  unsigned __int64 v45; // rbx
  __int64 v46; // rdi
  __int64 v47; // rdi
  __int64 *v48; // r14
  unsigned __int64 v49; // rbx
  __int64 v50; // rdi
  __int64 v51; // rdi
  __int64 *v52; // r14
  unsigned __int64 v53; // rbx
  __int64 v54; // rdi
  __int64 v55; // rdi
  __int64 *v56; // r14
  unsigned __int64 v57; // rbx
  __int64 v58; // rdi
  __int64 v59; // rax
  __int64 v60; // r8
  __int64 v61; // rsi
  __int64 v62; // rdx
  char *v63; // r15
  __int64 v64; // r14
  __int64 v65; // rcx
  __int64 v66; // rbx
  unsigned int v67; // eax
  __int64 v68; // rdi
  __int64 v69; // rdx
  __int64 v70; // rcx
  __int64 v71; // r8
  __int64 v72; // r9
  __int64 v73; // rdx
  __int64 v74; // rcx
  __int64 v75; // r8
  __int64 v76; // r9
  __int64 v77; // rdx
  __int64 v78; // rcx
  __int64 v79; // r8
  __int64 v80; // r9
  __int64 v81; // rdx
  __int64 v82; // rcx
  __int64 v83; // r8
  __int64 v84; // r9
  __int64 v85; // rdx
  __int64 v86; // rcx
  __int64 v87; // r8
  __int64 v88; // r9
  __int64 v89; // rdx
  __int64 v90; // rcx
  __int64 v91; // r8
  __int64 v92; // r9
  __int64 v93; // rdx
  __int64 v94; // rcx
  __int64 v95; // r8
  __int64 v96; // r9
  char *v97; // rsi
  __int64 v98; // rcx
  __int64 v99; // r8
  __int64 v100; // r9
  __int64 result; // rax
  __int64 v102; // rdi
  __int64 *v103; // rbx
  unsigned __int64 v104; // r12
  __int64 v105; // rdi
  __int64 v106; // rdi
  __int64 *v107; // rbx
  unsigned __int64 v108; // r12
  __int64 v109; // rdi
  __int64 v110; // rdi
  __int64 *v111; // rbx
  unsigned __int64 v112; // r12
  __int64 v113; // rdi
  __int64 v114; // rdi
  __int64 *v115; // rbx
  unsigned __int64 v116; // r12
  __int64 v117; // rdi
  __int64 v118; // rdi
  __int64 *v119; // rbx
  unsigned __int64 v120; // r12
  __int64 v121; // rdi
  __int64 v122; // rdi
  __int64 *v123; // rbx
  unsigned __int64 v124; // r12
  __int64 v125; // rdi
  __int64 v126; // rdi
  __int64 *v127; // rbx
  unsigned __int64 v128; // r12
  __int64 v129; // rdi
  __int64 v130; // rdi
  __int64 *v131; // rbx
  unsigned __int64 v132; // r12
  __int64 v133; // rdi
  __int64 v134; // rdx
  __int64 v135; // rcx
  __int64 v136; // r8
  __int64 v137; // r9
  __int64 v138; // rdx
  __int64 v139; // rcx
  __int64 v140; // r8
  __int64 v141; // r9
  char *v142; // rsi
  __int64 v143; // rdx
  __int64 v144; // rcx
  __int64 v145; // r8
  __int64 v146; // r9
  __int64 v147; // rdx
  __int64 v148; // rcx
  __int64 v149; // r8
  __int64 v150; // r9
  __int64 v151; // rdi
  __int64 *v152; // r12
  unsigned __int64 v153; // rbx
  __int64 v154; // rdi
  __int64 v155; // rdi
  __int64 *v156; // r12
  unsigned __int64 v157; // rbx
  __int64 v158; // rdi
  __int64 v159; // rdx
  __int64 v160; // rcx
  __int64 v161; // r8
  __int64 v162; // r9
  char *v163; // rsi
  __int64 v164; // rdx
  __int64 v165; // rcx
  __int64 v166; // r8
  __int64 v167; // r9
  __int64 v168; // rax
  __int64 v169; // r8
  __int64 v170; // r9
  __int64 v171; // rdi
  __int64 *v172; // r12
  unsigned __int64 v173; // rbx
  __int64 v174; // rdi
  __int64 v175; // rdi
  __int64 *v176; // r12
  unsigned __int64 v177; // rbx
  __int64 v178; // rdi
  __int64 v179; // rax
  unsigned __int64 v180; // rcx
  __int64 v181; // rdx
  __int64 v182; // rcx
  __int64 v183; // r8
  __int64 v184; // r9
  __int64 v185; // rdx
  __int64 v186; // rcx
  __int64 v187; // r8
  __int64 v188; // r9
  __int64 v189; // rdx
  __int64 v190; // rcx
  __int64 v191; // r8
  __int64 v192; // r9
  __int64 v193; // rdx
  __int64 v194; // rcx
  __int64 v195; // r8
  __int64 v196; // r9
  __int64 v197; // rdx
  __int64 v198; // rcx
  __int64 v199; // r8
  __int64 v200; // r9
  __int64 v201; // rdx
  __int64 v202; // rcx
  __int64 v203; // r8
  __int64 v204; // r9
  __int64 v205; // rdx
  __int64 v206; // rcx
  __int64 v207; // r8
  __int64 v208; // r9
  char *v209; // rsi
  __int64 v210; // rcx
  __int64 v211; // r8
  __int64 v212; // r9
  __int64 v213; // rdi
  __int64 *v214; // rbx
  unsigned __int64 v215; // r13
  __int64 v216; // rdi
  __int64 v217; // rdi
  __int64 *v218; // rbx
  unsigned __int64 v219; // r13
  __int64 v220; // rdi
  __int64 v221; // rdi
  __int64 *v222; // rbx
  unsigned __int64 v223; // r13
  __int64 v224; // rdi
  __int64 v225; // rdi
  __int64 *v226; // rbx
  unsigned __int64 v227; // r13
  __int64 v228; // rdi
  __int64 v229; // rdi
  __int64 *v230; // rbx
  unsigned __int64 v231; // r13
  __int64 v232; // rdi
  __int64 v233; // rdi
  __int64 *v234; // rbx
  unsigned __int64 v235; // r13
  __int64 v236; // rdi
  __int64 v237; // rdi
  __int64 *v238; // rbx
  unsigned __int64 v239; // r13
  __int64 v240; // rdi
  __int64 v241; // rdi
  __int64 *v242; // rbx
  unsigned __int64 v243; // r13
  __int64 v244; // rdi
  __int64 v245; // rdi
  __int64 *v246; // rbx
  unsigned __int64 v247; // r12
  __int64 v248; // rdi
  __int64 v249; // rdi
  __int64 *v250; // rbx
  unsigned __int64 v251; // r12
  __int64 v252; // rdi
  void *v253; // rdi
  __int64 v254; // r9
  signed __int64 v255; // r11
  unsigned __int64 v256; // rcx
  char *v257; // rbx
  __int64 v258; // r10
  unsigned __int64 v259; // rdx
  __int64 v260; // rdx
  __int64 v261; // rdx
  __int64 v262; // rcx
  __int64 v263; // r8
  __int64 v264; // r9
  char *v265; // rsi
  __int64 v266; // rcx
  __int64 v267; // r8
  __int64 v268; // r9
  __int64 v269; // rdi
  __int64 *v270; // rbx
  unsigned __int64 v271; // r12
  __int64 v272; // rdi
  __int64 v273; // rdi
  __int64 *v274; // rbx
  unsigned __int64 v275; // r12
  __int64 v276; // rdi
  __int64 v278; // [rsp+20h] [rbp-7F0h]
  __int64 v279; // [rsp+28h] [rbp-7E8h]
  int v280; // [rsp+30h] [rbp-7E0h]
  signed __int64 v281; // [rsp+30h] [rbp-7E0h]
  __int64 v282; // [rsp+38h] [rbp-7D8h]
  __int64 v283; // [rsp+38h] [rbp-7D8h]
  unsigned __int64 v284; // [rsp+40h] [rbp-7D0h]
  __int64 v285; // [rsp+40h] [rbp-7D0h]
  __int64 v286; // [rsp+40h] [rbp-7D0h]
  __int64 v288; // [rsp+50h] [rbp-7C0h]
  __int64 v290; // [rsp+58h] [rbp-7B8h]
  char v291[8]; // [rsp+60h] [rbp-7B0h] BYREF
  __int64 v292; // [rsp+68h] [rbp-7A8h]
  char v293; // [rsp+7Ch] [rbp-794h]
  __int64 v294; // [rsp+C0h] [rbp-750h]
  __int64 v295; // [rsp+C8h] [rbp-748h]
  unsigned __int64 v296; // [rsp+E8h] [rbp-728h]
  __int64 v297; // [rsp+108h] [rbp-708h]
  char v298[8]; // [rsp+120h] [rbp-6F0h] BYREF
  __int64 v299; // [rsp+128h] [rbp-6E8h]
  char v300; // [rsp+13Ch] [rbp-6D4h]
  __int64 v301; // [rsp+180h] [rbp-690h]
  __int64 v302; // [rsp+188h] [rbp-688h]
  unsigned __int64 v303; // [rsp+1A8h] [rbp-668h]
  __int64 v304; // [rsp+1C8h] [rbp-648h]
  char v305[8]; // [rsp+1E0h] [rbp-630h] BYREF
  __int64 v306; // [rsp+1E8h] [rbp-628h]
  char v307; // [rsp+1FCh] [rbp-614h]
  __int64 v308; // [rsp+240h] [rbp-5D0h]
  __int64 v309; // [rsp+248h] [rbp-5C8h]
  unsigned __int64 v310; // [rsp+268h] [rbp-5A8h]
  __int64 v311; // [rsp+288h] [rbp-588h]
  __int64 v312; // [rsp+2A0h] [rbp-570h] BYREF
  __int64 v313; // [rsp+2A8h] [rbp-568h]
  __int64 v314; // [rsp+2B0h] [rbp-560h]
  unsigned __int64 v315; // [rsp+2B8h] [rbp-558h]
  __int64 v316; // [rsp+300h] [rbp-510h]
  __int64 v317; // [rsp+308h] [rbp-508h]
  unsigned __int64 v318; // [rsp+328h] [rbp-4E8h]
  __int64 v319; // [rsp+348h] [rbp-4C8h]
  __int64 v320; // [rsp+360h] [rbp-4B0h] BYREF
  __int64 v321; // [rsp+368h] [rbp-4A8h]
  __int64 v322; // [rsp+370h] [rbp-4A0h]
  __int64 v323; // [rsp+378h] [rbp-498h]
  __int64 v324; // [rsp+3C0h] [rbp-450h]
  __int64 v325; // [rsp+3C8h] [rbp-448h]
  unsigned __int64 v326; // [rsp+3E8h] [rbp-428h]
  __int64 v327; // [rsp+408h] [rbp-408h]
  __int64 v328; // [rsp+420h] [rbp-3F0h] BYREF
  __int64 v329; // [rsp+428h] [rbp-3E8h]
  __int64 v330; // [rsp+430h] [rbp-3E0h]
  unsigned __int64 v331; // [rsp+438h] [rbp-3D8h]
  __int64 v332; // [rsp+480h] [rbp-390h]
  __int64 v333; // [rsp+488h] [rbp-388h]
  unsigned __int64 v334; // [rsp+4A8h] [rbp-368h]
  __int64 v335; // [rsp+4C8h] [rbp-348h]
  char v336[8]; // [rsp+4E0h] [rbp-330h] BYREF
  __int64 v337; // [rsp+4E8h] [rbp-328h]
  char v338; // [rsp+4FCh] [rbp-314h]
  __int64 v339; // [rsp+540h] [rbp-2D0h]
  __int64 v340; // [rsp+548h] [rbp-2C8h]
  unsigned __int64 v341; // [rsp+568h] [rbp-2A8h]
  __int64 v342; // [rsp+588h] [rbp-288h]
  char v343[8]; // [rsp+5A0h] [rbp-270h] BYREF
  __int64 v344; // [rsp+5A8h] [rbp-268h]
  char v345; // [rsp+5BCh] [rbp-254h]
  __int64 v346; // [rsp+600h] [rbp-210h]
  __int64 v347; // [rsp+608h] [rbp-208h]
  unsigned __int64 v348; // [rsp+628h] [rbp-1E8h]
  __int64 v349; // [rsp+648h] [rbp-1C8h]
  char v350[8]; // [rsp+660h] [rbp-1B0h] BYREF
  __int64 v351; // [rsp+668h] [rbp-1A8h]
  char v352; // [rsp+67Ch] [rbp-194h]
  __int64 v353; // [rsp+6C0h] [rbp-150h]
  __int64 v354; // [rsp+6C8h] [rbp-148h]
  __int64 v355; // [rsp+6D0h] [rbp-140h]
  __int64 v356; // [rsp+6D8h] [rbp-138h]
  __int64 v357; // [rsp+6E0h] [rbp-130h]
  unsigned __int64 v358; // [rsp+6E8h] [rbp-128h]
  __int64 v359; // [rsp+6F0h] [rbp-120h]
  __int64 v360; // [rsp+6F8h] [rbp-118h]
  __int64 v361; // [rsp+708h] [rbp-108h]
  char v362[8]; // [rsp+720h] [rbp-F0h] BYREF
  __int64 v363; // [rsp+728h] [rbp-E8h]
  char v364; // [rsp+73Ch] [rbp-D4h]
  __int64 v365; // [rsp+780h] [rbp-90h]
  __int64 v366; // [rsp+788h] [rbp-88h]
  __int64 v367; // [rsp+790h] [rbp-80h]
  __int64 v368; // [rsp+798h] [rbp-78h]
  __int64 v369; // [rsp+7A0h] [rbp-70h]
  unsigned __int64 v370; // [rsp+7A8h] [rbp-68h]
  __int64 v371; // [rsp+7B0h] [rbp-60h]
  __int64 v372; // [rsp+7B8h] [rbp-58h]
  __int64 v373; // [rsp+7C0h] [rbp-50h]
  __int64 v374; // [rsp+7C8h] [rbp-48h]

  v7 = *(_QWORD *)a1;
  v278 = *(_QWORD *)a1;
  if ( a2 == *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) )
  {
    sub_D53210((__int64)v298, a4, v7, a4, a5, a6);
    sub_D53210((__int64)v291, a3, v134, v135, v136, v137);
    sub_D53210((__int64)v362, (__int64)v298, v138, v139, v140, v141);
    v142 = v291;
    sub_D53210((__int64)v350, (__int64)v291, v143, v144, v145, v146);
    v151 = v353;
    if ( v353 )
    {
      v152 = (__int64 *)v358;
      v153 = v361 + 8;
      if ( v361 + 8 > v358 )
      {
        do
        {
          v154 = *v152++;
          j_j___libc_free_0(v154, 512);
        }
        while ( v153 > (unsigned __int64)v152 );
        v151 = v353;
      }
      v142 = (char *)(8 * v354);
      j_j___libc_free_0(v151, 8 * v354);
    }
    if ( !v352 )
      _libc_free(v351, v142);
    v155 = v365;
    if ( v365 )
    {
      v156 = (__int64 *)v370;
      v157 = v374 + 8;
      if ( v374 + 8 > v370 )
      {
        do
        {
          v158 = *v156++;
          j_j___libc_free_0(v158, 512);
        }
        while ( v157 > (unsigned __int64)v156 );
        v155 = v365;
      }
      v142 = (char *)(8 * v366);
      j_j___libc_free_0(v155, 8 * v366);
    }
    if ( !v364 )
      _libc_free(v363, v142);
    sub_D53210((__int64)v362, (__int64)v298, v147, v148, v149, v150);
    sub_D53210((__int64)v350, (__int64)v291, v159, v160, v161, v162);
    v163 = v362;
    v168 = sub_D544F0((__int64)v350, (__int64)v362, v164, v165, v166, v167);
    v171 = v353;
    v288 = v168;
    if ( v353 )
    {
      v172 = (__int64 *)v358;
      v173 = v361 + 8;
      if ( v361 + 8 > v358 )
      {
        do
        {
          v174 = *v172++;
          j_j___libc_free_0(v174, 512);
        }
        while ( v173 > (unsigned __int64)v172 );
        v171 = v353;
      }
      v163 = (char *)(8 * v354);
      j_j___libc_free_0(v171, 8 * v354);
    }
    if ( !v352 )
      _libc_free(v351, v163);
    v175 = v365;
    if ( v365 )
    {
      v176 = (__int64 *)v370;
      v177 = v374 + 8;
      if ( v374 + 8 > v370 )
      {
        do
        {
          v178 = *v176++;
          j_j___libc_free_0(v178, 512);
        }
        while ( v177 > (unsigned __int64)v176 );
        v175 = v365;
      }
      v163 = (char *)(8 * v366);
      j_j___libc_free_0(v175, 8 * v366);
    }
    if ( !v364 )
      _libc_free(v363, v163);
    v179 = *(unsigned int *)(a1 + 8);
    v180 = *(unsigned int *)(a1 + 12);
    if ( v179 + v288 > v180 )
    {
      sub_C8D5F0(a1, (const void *)(a1 + 16), v179 + v288, 8u, v169, v170);
      v179 = *(unsigned int *)(a1 + 8);
    }
    v285 = *(_QWORD *)a1 + 8 * v179;
    sub_D53210((__int64)&v312, (__int64)v298, v285, v180, v169, v170);
    sub_D53210((__int64)v305, (__int64)v291, v181, v182, v183, v184);
    sub_D53210((__int64)&v328, (__int64)&v312, v185, v186, v187, v188);
    sub_D53210((__int64)&v320, (__int64)v305, v189, v190, v191, v192);
    sub_D53210((__int64)v336, (__int64)&v328, v193, v194, v195, v196);
    sub_D53210((__int64)v343, (__int64)&v320, v197, v198, v199, v200);
    sub_D53210((__int64)v350, (__int64)v336, v201, v202, v203, v204);
    sub_D53210((__int64)v362, (__int64)v343, v205, v206, v207, v208);
    v209 = v350;
    sub_D54790((__int64)v362, (__int64)v350, v285, v210, v211, v212);
    v213 = v365;
    if ( v365 )
    {
      v214 = (__int64 *)v370;
      v215 = v374 + 8;
      if ( v374 + 8 > v370 )
      {
        do
        {
          v216 = *v214++;
          j_j___libc_free_0(v216, 512);
        }
        while ( v215 > (unsigned __int64)v214 );
        v213 = v365;
      }
      v209 = (char *)(8 * v366);
      j_j___libc_free_0(v213, 8 * v366);
    }
    if ( !v364 )
      _libc_free(v363, v209);
    v217 = v353;
    if ( v353 )
    {
      v218 = (__int64 *)v358;
      v219 = v361 + 8;
      if ( v361 + 8 > v358 )
      {
        do
        {
          v220 = *v218++;
          j_j___libc_free_0(v220, 512);
        }
        while ( v219 > (unsigned __int64)v218 );
        v217 = v353;
      }
      v209 = (char *)(8 * v354);
      j_j___libc_free_0(v217, 8 * v354);
    }
    if ( !v352 )
      _libc_free(v351, v209);
    v221 = v346;
    if ( v346 )
    {
      v222 = (__int64 *)v348;
      v223 = v349 + 8;
      if ( v349 + 8 > v348 )
      {
        do
        {
          v224 = *v222++;
          j_j___libc_free_0(v224, 512);
        }
        while ( v223 > (unsigned __int64)v222 );
        v221 = v346;
      }
      v209 = (char *)(8 * v347);
      j_j___libc_free_0(v221, 8 * v347);
    }
    if ( !v345 )
      _libc_free(v344, v209);
    v225 = v339;
    if ( v339 )
    {
      v226 = (__int64 *)v341;
      v227 = v342 + 8;
      if ( v342 + 8 > v341 )
      {
        do
        {
          v228 = *v226++;
          j_j___libc_free_0(v228, 512);
        }
        while ( v227 > (unsigned __int64)v226 );
        v225 = v339;
      }
      v209 = (char *)(8 * v340);
      j_j___libc_free_0(v225, 8 * v340);
    }
    if ( !v338 )
      _libc_free(v337, v209);
    v229 = v324;
    if ( v324 )
    {
      v230 = (__int64 *)v326;
      v231 = v327 + 8;
      if ( v327 + 8 > v326 )
      {
        do
        {
          v232 = *v230++;
          j_j___libc_free_0(v232, 512);
        }
        while ( v231 > (unsigned __int64)v230 );
        v229 = v324;
      }
      v209 = (char *)(8 * v325);
      j_j___libc_free_0(v229, 8 * v325);
    }
    if ( !BYTE4(v323) )
      _libc_free(v321, v209);
    v233 = v332;
    if ( v332 )
    {
      v234 = (__int64 *)v334;
      v235 = v335 + 8;
      if ( v335 + 8 > v334 )
      {
        do
        {
          v236 = *v234++;
          j_j___libc_free_0(v236, 512);
        }
        while ( v235 > (unsigned __int64)v234 );
        v233 = v332;
      }
      v209 = (char *)(8 * v333);
      j_j___libc_free_0(v233, 8 * v333);
    }
    if ( !BYTE4(v331) )
      _libc_free(v329, v209);
    v237 = v308;
    if ( v308 )
    {
      v238 = (__int64 *)v310;
      v239 = v311 + 8;
      if ( v311 + 8 > v310 )
      {
        do
        {
          v240 = *v238++;
          j_j___libc_free_0(v240, 512);
        }
        while ( v239 > (unsigned __int64)v238 );
        v237 = v308;
      }
      v209 = (char *)(8 * v309);
      j_j___libc_free_0(v237, 8 * v309);
    }
    if ( !v307 )
      _libc_free(v306, v209);
    v241 = v316;
    if ( v316 )
    {
      v242 = (__int64 *)v318;
      v243 = v319 + 8;
      if ( v319 + 8 > v318 )
      {
        do
        {
          v244 = *v242++;
          j_j___libc_free_0(v244, 512);
        }
        while ( v243 > (unsigned __int64)v242 );
        v241 = v316;
      }
      v209 = (char *)(8 * v317);
      j_j___libc_free_0(v241, 8 * v317);
    }
    if ( !BYTE4(v315) )
      _libc_free(v313, v209);
    result = a1;
    v245 = v294;
    *(_DWORD *)(a1 + 8) += v288;
    if ( v245 )
    {
      v246 = (__int64 *)v296;
      v247 = v297 + 8;
      if ( v297 + 8 > v296 )
      {
        do
        {
          v248 = *v246++;
          j_j___libc_free_0(v248, 512);
        }
        while ( v247 > (unsigned __int64)v246 );
        v245 = v294;
      }
      v209 = (char *)(8 * v295);
      result = j_j___libc_free_0(v245, 8 * v295);
    }
    if ( !v293 )
      result = _libc_free(v292, v209);
    v249 = v301;
    if ( v301 )
    {
      v250 = (__int64 *)v303;
      v251 = v304 + 8;
      if ( v304 + 8 > v303 )
      {
        do
        {
          v252 = *v250++;
          j_j___libc_free_0(v252, 512);
        }
        while ( v251 > (unsigned __int64)v250 );
        v249 = v301;
      }
      v209 = (char *)(8 * v302);
      result = j_j___libc_free_0(v249, 8 * v302);
    }
    if ( !v300 )
      return _libc_free(v299, v209);
    return result;
  }
  sub_D53210((__int64)v362, a4, v7, a4, a5, a6);
  v8 = a3;
  sub_D53210((__int64)v350, a3, v9, v10, v11, v12);
  v17 = v353;
  if ( v353 )
  {
    v18 = (__int64 *)v358;
    v19 = v361 + 8;
    if ( v361 + 8 > v358 )
    {
      do
      {
        v20 = *v18++;
        j_j___libc_free_0(v20, 512);
      }
      while ( v19 > (unsigned __int64)v18 );
      v17 = v353;
    }
    v8 = 8 * v354;
    j_j___libc_free_0(v17, 8 * v354);
  }
  if ( v352 )
  {
    v21 = v365;
    if ( !v365 )
      goto LABEL_13;
  }
  else
  {
    _libc_free(v351, v8);
    v21 = v365;
    if ( !v365 )
      goto LABEL_13;
  }
  v22 = (__int64 *)v370;
  v23 = v374 + 8;
  if ( v374 + 8 > v370 )
  {
    do
    {
      v24 = *v22++;
      j_j___libc_free_0(v24, 512);
    }
    while ( v23 > (unsigned __int64)v22 );
    v21 = v365;
  }
  v8 = 8 * v366;
  j_j___libc_free_0(v21, 8 * v366);
LABEL_13:
  if ( !v364 )
    _libc_free(v363, v8);
  sub_D53210((__int64)v343, a4, v13, v14, v15, v16);
  sub_D53210((__int64)v336, a3, v25, v26, v27, v28);
  v29 = 0;
  sub_D53210((__int64)v350, (__int64)v343, v30, v31, v32, v33);
  sub_D53210((__int64)v362, (__int64)v336, v34, v35, v36, v37);
  while ( 1 )
  {
    v38 = (__int64 *)v358;
    v39 = v355;
    v40 = v370;
    v41 = ((v359 - v360) >> 5) + 16 * (((__int64)(v361 - v358) >> 3) - 1) + ((v357 - v355) >> 5);
    v42 = (v369 - v367) >> 5;
    if ( v41 == v42 + 16 * (((__int64)(v374 - v370) >> 3) - 1) + ((v371 - v372) >> 5) )
    {
      v330 = v357;
      v331 = v358;
      v322 = v373;
      v38 = &v320;
      v320 = v371;
      v313 = v368;
      v321 = v372;
      v323 = v374;
      v312 = v367;
      v314 = v369;
      v315 = v370;
      v328 = v355;
      v329 = v356;
      if ( (unsigned __int8)sub_D542B0(&v312, &v320, &v328) )
        break;
    }
    ++v29;
    sub_D53E10((__int64)v362, (__int64)v38, (__int64 *)v42, v41, v39, v40);
  }
  v43 = v365;
  v284 = v29;
  if ( v365 )
  {
    v44 = (__int64 *)v370;
    v45 = v374 + 8;
    if ( v374 + 8 > v370 )
    {
      do
      {
        v46 = *v44++;
        j_j___libc_free_0(v46, 512);
      }
      while ( v45 > (unsigned __int64)v44 );
      v43 = v365;
    }
    v38 = (__int64 *)(8 * v366);
    j_j___libc_free_0(v43, 8 * v366);
  }
  if ( v364 )
  {
    v47 = v353;
    if ( !v353 )
      goto LABEL_30;
  }
  else
  {
    _libc_free(v363, v38);
    v47 = v353;
    if ( !v353 )
      goto LABEL_30;
  }
  v48 = (__int64 *)v358;
  v49 = v361 + 8;
  if ( v361 + 8 > v358 )
  {
    do
    {
      v50 = *v48++;
      j_j___libc_free_0(v50, 512);
    }
    while ( v49 > (unsigned __int64)v48 );
    v47 = v353;
  }
  v38 = (__int64 *)(8 * v354);
  j_j___libc_free_0(v47, 8 * v354);
LABEL_30:
  if ( v352 )
  {
    v51 = v339;
    if ( !v339 )
      goto LABEL_36;
  }
  else
  {
    _libc_free(v351, v38);
    v51 = v339;
    if ( !v339 )
      goto LABEL_36;
  }
  v52 = (__int64 *)v341;
  v53 = v342 + 8;
  if ( v342 + 8 > v341 )
  {
    do
    {
      v54 = *v52++;
      j_j___libc_free_0(v54, 512);
    }
    while ( v53 > (unsigned __int64)v52 );
    v51 = v339;
  }
  v38 = (__int64 *)(8 * v340);
  j_j___libc_free_0(v51, 8 * v340);
LABEL_36:
  if ( v338 )
  {
    v55 = v346;
    if ( v346 )
    {
LABEL_38:
      v56 = (__int64 *)v348;
      v57 = v349 + 8;
      if ( v349 + 8 > v348 )
      {
        do
        {
          v58 = *v56++;
          j_j___libc_free_0(v58, 512);
        }
        while ( v57 > (unsigned __int64)v56 );
        v55 = v346;
      }
      v38 = (__int64 *)(8 * v347);
      j_j___libc_free_0(v55, 8 * v347);
    }
  }
  else
  {
    _libc_free(v337, v38);
    v55 = v346;
    if ( v346 )
      goto LABEL_38;
  }
  if ( !v345 )
    _libc_free(v344, v38);
  v59 = *(unsigned int *)(a1 + 8);
  if ( v59 + v284 > *(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v59 + v284, 8u, v39, v40);
    v59 = *(unsigned int *)(a1 + 8);
  }
  v60 = a2 - v278;
  v61 = *(_QWORD *)a1;
  v62 = 8 * v59 - (a2 - v278);
  v63 = (char *)(*(_QWORD *)a1 + a2 - v278);
  v64 = *(_QWORD *)a1 + 8 * v59;
  v65 = v62 >> 3;
  v66 = v62 >> 3;
  if ( v284 > v62 >> 3 )
  {
    v67 = v284 + v59;
    *(_DWORD *)(a1 + 8) = v67;
    if ( v63 != (char *)v64 )
    {
      v290 = v62 >> 3;
      v68 = v61 + 8LL * v67;
      v61 = (__int64)v63;
      memcpy((void *)(v68 - v62), v63, v62);
      v65 = v290;
    }
    if ( v65 )
    {
      do
      {
        v63 += 8;
        *((_QWORD *)v63 - 1) = **(_QWORD **)(a3 + 112);
        sub_D53E10(a3, v61, (__int64 *)v62, v65, v60, v40);
        --v66;
      }
      while ( v66 );
    }
    sub_D53210((__int64)&v312, a4, v62, v65, v60, v40);
    sub_D53210((__int64)v305, a3, v69, v70, v71, v72);
    sub_D53210((__int64)&v328, (__int64)&v312, v73, v74, v75, v76);
    sub_D53210((__int64)&v320, (__int64)v305, v77, v78, v79, v80);
    sub_D53210((__int64)v336, (__int64)&v328, v81, v82, v83, v84);
    sub_D53210((__int64)v343, (__int64)&v320, v85, v86, v87, v88);
    sub_D53210((__int64)v350, (__int64)v336, v89, v90, v91, v92);
    sub_D53210((__int64)v362, (__int64)v343, v93, v94, v95, v96);
    v97 = v350;
    result = sub_D54790((__int64)v362, (__int64)v350, v64, v98, v99, v100);
    v102 = v365;
    if ( v365 )
    {
      v103 = (__int64 *)v370;
      v104 = v374 + 8;
      if ( v374 + 8 > v370 )
      {
        do
        {
          v105 = *v103++;
          j_j___libc_free_0(v105, 512);
        }
        while ( v104 > (unsigned __int64)v103 );
        v102 = v365;
      }
      v97 = (char *)(8 * v366);
      result = j_j___libc_free_0(v102, 8 * v366);
    }
    if ( v364 )
    {
      v106 = v353;
      if ( !v353 )
        goto LABEL_62;
    }
    else
    {
      result = _libc_free(v363, v97);
      v106 = v353;
      if ( !v353 )
        goto LABEL_62;
    }
    v107 = (__int64 *)v358;
    v108 = v361 + 8;
    if ( v361 + 8 > v358 )
    {
      do
      {
        v109 = *v107++;
        j_j___libc_free_0(v109, 512);
      }
      while ( v108 > (unsigned __int64)v107 );
      v106 = v353;
    }
    v97 = (char *)(8 * v354);
    result = j_j___libc_free_0(v106, 8 * v354);
LABEL_62:
    if ( v352 )
    {
      v110 = v346;
      if ( !v346 )
        goto LABEL_68;
    }
    else
    {
      result = _libc_free(v351, v97);
      v110 = v346;
      if ( !v346 )
        goto LABEL_68;
    }
    v111 = (__int64 *)v348;
    v112 = v349 + 8;
    if ( v349 + 8 > v348 )
    {
      do
      {
        v113 = *v111++;
        j_j___libc_free_0(v113, 512);
      }
      while ( v112 > (unsigned __int64)v111 );
      v110 = v346;
    }
    v97 = (char *)(8 * v347);
    result = j_j___libc_free_0(v110, 8 * v347);
LABEL_68:
    if ( v345 )
    {
      v114 = v339;
      if ( !v339 )
        goto LABEL_74;
    }
    else
    {
      result = _libc_free(v344, v97);
      v114 = v339;
      if ( !v339 )
        goto LABEL_74;
    }
    v115 = (__int64 *)v341;
    v116 = v342 + 8;
    if ( v342 + 8 > v341 )
    {
      do
      {
        v117 = *v115++;
        j_j___libc_free_0(v117, 512);
      }
      while ( v116 > (unsigned __int64)v115 );
      v114 = v339;
    }
    v97 = (char *)(8 * v340);
    result = j_j___libc_free_0(v114, 8 * v340);
LABEL_74:
    if ( v338 )
    {
      v118 = v324;
      if ( !v324 )
        goto LABEL_80;
    }
    else
    {
      result = _libc_free(v337, v97);
      v118 = v324;
      if ( !v324 )
        goto LABEL_80;
    }
    v119 = (__int64 *)v326;
    v120 = v327 + 8;
    if ( v327 + 8 > v326 )
    {
      do
      {
        v121 = *v119++;
        j_j___libc_free_0(v121, 512);
      }
      while ( v120 > (unsigned __int64)v119 );
      v118 = v324;
    }
    v97 = (char *)(8 * v325);
    result = j_j___libc_free_0(v118, 8 * v325);
LABEL_80:
    if ( BYTE4(v323) )
    {
      v122 = v332;
      if ( !v332 )
        goto LABEL_86;
    }
    else
    {
      result = _libc_free(v321, v97);
      v122 = v332;
      if ( !v332 )
        goto LABEL_86;
    }
    v123 = (__int64 *)v334;
    v124 = v335 + 8;
    if ( v335 + 8 > v334 )
    {
      do
      {
        v125 = *v123++;
        j_j___libc_free_0(v125, 512);
      }
      while ( v124 > (unsigned __int64)v123 );
      v122 = v332;
    }
    v97 = (char *)(8 * v333);
    result = j_j___libc_free_0(v122, 8 * v333);
LABEL_86:
    if ( BYTE4(v331) )
    {
      v126 = v308;
      if ( !v308 )
        goto LABEL_92;
    }
    else
    {
      result = _libc_free(v329, v97);
      v126 = v308;
      if ( !v308 )
        goto LABEL_92;
    }
    v127 = (__int64 *)v310;
    v128 = v311 + 8;
    if ( v311 + 8 > v310 )
    {
      do
      {
        v129 = *v127++;
        j_j___libc_free_0(v129, 512);
      }
      while ( v128 > (unsigned __int64)v127 );
      v126 = v308;
    }
    v97 = (char *)(8 * v309);
    result = j_j___libc_free_0(v126, 8 * v309);
LABEL_92:
    if ( v307 )
    {
      v130 = v316;
      if ( v316 )
      {
LABEL_94:
        v131 = (__int64 *)v318;
        v132 = v319 + 8;
        if ( v319 + 8 > v318 )
        {
          do
          {
            v133 = *v131++;
            j_j___libc_free_0(v133, 512);
          }
          while ( v132 > (unsigned __int64)v131 );
          v130 = v316;
        }
        v97 = (char *)(8 * v317);
        result = j_j___libc_free_0(v130, 8 * v317);
      }
    }
    else
    {
      result = _libc_free(v306, v97);
      v130 = v316;
      if ( v316 )
        goto LABEL_94;
    }
    if ( !BYTE4(v315) )
      return _libc_free(v313, v97);
    return result;
  }
  v253 = (void *)(*(_QWORD *)a1 + 8 * v59);
  v254 = 8 * (v59 - v284);
  v255 = 8 * v59 - v254;
  v256 = *(unsigned int *)(a1 + 12);
  v257 = (char *)(v61 + v254);
  v258 = v255 >> 3;
  v259 = (v255 >> 3) + v59;
  if ( v259 > v256 )
  {
    v279 = v255 >> 3;
    v281 = 8 * v59 - v254;
    v283 = 8 * (v59 - v284);
    sub_C8D5F0(a1, (const void *)(a1 + 16), v259, 8u, v60, v254);
    LODWORD(v258) = v279;
    v255 = v281;
    v254 = v283;
    v59 = *(unsigned int *)(a1 + 8);
    v60 = a2 - v278;
    v253 = (void *)(*(_QWORD *)a1 + 8 * v59);
  }
  if ( (char *)v64 != v257 )
  {
    v280 = v258;
    v282 = v254;
    v286 = v60;
    memmove(v253, v257, v255);
    LODWORD(v258) = v280;
    v254 = v282;
    v60 = v286;
    LODWORD(v59) = *(_DWORD *)(a1 + 8);
  }
  v260 = a1;
  *(_DWORD *)(a1 + 8) = v258 + v59;
  if ( v63 != v257 )
    memmove((void *)(v64 - (v254 - v60)), v63, v254 - v60);
  sub_D53210((__int64)v362, a4, v260, v256, v60, v254);
  sub_D53210((__int64)v350, a3, v261, v262, v263, v264);
  v265 = v362;
  result = sub_D54790((__int64)v350, (__int64)v362, (__int64)v63, v266, v267, v268);
  v269 = v353;
  if ( v353 )
  {
    v270 = (__int64 *)v358;
    v271 = v361 + 8;
    if ( v361 + 8 > v358 )
    {
      do
      {
        v272 = *v270++;
        j_j___libc_free_0(v272, 512);
      }
      while ( v271 > (unsigned __int64)v270 );
      v269 = v353;
    }
    v265 = (char *)(8 * v354);
    result = j_j___libc_free_0(v269, 8 * v354);
  }
  if ( !v352 )
    result = _libc_free(v351, v265);
  v273 = v365;
  if ( v365 )
  {
    v274 = (__int64 *)v370;
    v275 = v374 + 8;
    if ( v374 + 8 > v370 )
    {
      do
      {
        v276 = *v274++;
        j_j___libc_free_0(v276, 512);
      }
      while ( v275 > (unsigned __int64)v274 );
      v273 = v365;
    }
    v265 = (char *)(8 * v366);
    result = j_j___libc_free_0(v273, 8 * v366);
  }
  if ( !v364 )
    return _libc_free(v363, v265);
  return result;
}
