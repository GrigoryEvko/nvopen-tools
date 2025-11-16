// Function: sub_32E81A0
// Address: 0x32e81a0
//
__int64 __fastcall sub_32E81A0(_QWORD *a1, __int64 a2)
{
  __int64 v4; // rdx
  __int16 *v5; // rax
  __int64 v6; // r8
  __int16 v7; // cx
  __int64 v8; // rsi
  __int64 v9; // rdi
  __int64 v10; // r14
  __int64 v11; // r10
  __int64 v12; // r11
  __int64 v13; // r9
  int v14; // ebx
  __int64 v15; // r15
  _QWORD *v16; // rax
  int v17; // eax
  int v18; // ecx
  int v19; // edx
  int v20; // eax
  __int64 v21; // rax
  int v22; // edx
  __int64 v23; // rax
  const __m128i *v24; // r8
  __int64 v25; // rdx
  unsigned __int16 v26; // bx
  __int64 v27; // rax
  __int16 v28; // cx
  __int64 v29; // rdx
  __int64 v30; // rax
  bool v31; // al
  bool v32; // cl
  __int64 v33; // rax
  unsigned __int64 v34; // rdx
  unsigned __int64 v35; // rcx
  unsigned __int64 v36; // rdx
  __int64 v37; // r9
  __m128i *v38; // rcx
  __int32 v39; // edi
  _BYTE *v40; // rsi
  _BYTE *v41; // rsi
  __int64 v42; // rsi
  __int64 v43; // r13
  __int64 v44; // r14
  __int64 v45; // r15
  __int64 result; // rax
  char v47; // al
  __int64 v48; // rdi
  _QWORD *v49; // rax
  char v50; // al
  bool v51; // zf
  unsigned int *v52; // rdx
  const __m128i *v53; // roff
  __int64 v54; // rsi
  unsigned __int16 *v55; // rcx
  __int64 v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rdi
  __int64 v59; // rax
  unsigned __int16 v60; // cx
  __int64 v61; // rax
  bool v62; // al
  __int64 v63; // rsi
  __int64 v64; // rcx
  __int16 v65; // ax
  __int64 v66; // rdx
  __int64 v67; // rax
  __int64 v68; // rcx
  __int64 v69; // rax
  __int64 v70; // rdx
  __int64 v71; // rax
  int v72; // edx
  __int64 v73; // rax
  __int64 v74; // rax
  unsigned int *v75; // rax
  __int64 v76; // rax
  __int16 v77; // dx
  __int64 v78; // rax
  __int64 v79; // rbx
  __int64 v80; // r15
  __int64 v81; // rax
  __int64 v82; // rdx
  _QWORD *v83; // rax
  __int64 v84; // rsi
  __int64 v85; // r14
  __int64 v86; // rax
  __int64 v87; // r8
  __int64 v88; // r9
  __int64 v89; // r14
  __int64 v90; // rdx
  __int64 v91; // r15
  __int64 v92; // r9
  __int64 v93; // r12
  __int64 v94; // rsi
  __int64 v95; // rbx
  __int64 v96; // rsi
  __int64 v97; // rax
  __int16 v98; // dx
  __int64 v99; // rax
  __int64 v100; // rdx
  __int64 v101; // rdx
  __int64 v102; // rcx
  __int64 v103; // rsi
  unsigned __int16 *v104; // rax
  int v105; // r8d
  bool v106; // al
  __int64 v107; // rdx
  __int64 v108; // r10
  unsigned __int16 *v109; // rax
  unsigned int v110; // ecx
  __int64 v111; // r11
  __int64 v112; // rsi
  __int16 v113; // ax
  __int64 v114; // rcx
  __int64 v115; // rcx
  __int16 v116; // ax
  __int64 v117; // rcx
  __int64 v118; // rdx
  __m128i v119; // rax
  __m128i v120; // rax
  unsigned __int64 v121; // rax
  unsigned int v122; // esi
  unsigned __int64 v123; // rax
  int v124; // esi
  unsigned int v125; // eax
  __int64 v126; // r9
  __int64 v127; // r8
  __int64 v128; // rax
  unsigned int v129; // r10d
  unsigned __int16 v130; // r8
  __int64 v131; // r11
  unsigned int v132; // edx
  char v133; // al
  _QWORD *v134; // rax
  char v135; // al
  __int64 v136; // rsi
  __int128 *v137; // rbx
  __int64 v138; // r13
  bool v139; // al
  __int64 v140; // r13
  bool v141; // al
  __int64 v142; // rdx
  __int16 v143; // ax
  __int64 v144; // rdx
  bool v145; // al
  bool v146; // al
  __int64 v147; // rdx
  __int64 v148; // rcx
  unsigned int v149; // eax
  __int64 v150; // rdx
  __int64 v151; // rax
  int v152; // edx
  bool v153; // al
  int v154; // eax
  __int64 v155; // rdx
  __m128i v156; // rax
  unsigned __int64 v157; // rax
  __m128i v158; // rax
  unsigned __int64 v159; // rax
  unsigned __int64 v160; // rcx
  unsigned int v161; // eax
  __int64 v162; // rdx
  __int64 v163; // rax
  unsigned int v164; // edx
  __int64 v165; // rax
  char v166; // al
  __int128 v167; // rax
  __int64 v168; // r15
  int v169; // r9d
  __int64 v170; // rax
  unsigned int v171; // edx
  __int64 v172; // rbx
  __int64 v173; // r13
  __int64 v174; // r14
  __int64 v175; // rbx
  int v176; // r15d
  __int64 v177; // rax
  __int64 v178; // rdx
  __int64 v179; // rbx
  __int64 v180; // r12
  unsigned int v181; // eax
  __int64 v182; // rdx
  __int64 v183; // rax
  __int64 v184; // rdx
  char v185; // si
  __int16 v186; // cx
  unsigned __int64 v187; // rax
  __int64 v188; // rdx
  bool v189; // al
  bool v190; // al
  __int128 *v191; // rbx
  __int64 v192; // r13
  bool v193; // al
  __int128 *v194; // rbx
  __m128i v195; // rax
  __m128i v196; // rax
  char v197; // al
  bool v198; // al
  __int128 v199; // [rsp-30h] [rbp-270h]
  __int128 v200; // [rsp-30h] [rbp-270h]
  __int128 v201; // [rsp-20h] [rbp-260h]
  __int128 v202; // [rsp-10h] [rbp-250h]
  __int128 v203; // [rsp-10h] [rbp-250h]
  __int128 v204; // [rsp-10h] [rbp-250h]
  __int16 v205; // [rsp+Ah] [rbp-236h]
  __int64 v206; // [rsp+10h] [rbp-230h]
  __int64 v207; // [rsp+18h] [rbp-228h]
  unsigned int v208; // [rsp+18h] [rbp-228h]
  unsigned __int64 v209; // [rsp+18h] [rbp-228h]
  __int64 v210; // [rsp+20h] [rbp-220h]
  unsigned __int64 v211; // [rsp+20h] [rbp-220h]
  __int64 v212; // [rsp+20h] [rbp-220h]
  __int64 v213; // [rsp+20h] [rbp-220h]
  __int64 v214; // [rsp+20h] [rbp-220h]
  __int64 v215; // [rsp+28h] [rbp-218h]
  __int64 v216; // [rsp+28h] [rbp-218h]
  __int64 v217; // [rsp+30h] [rbp-210h]
  unsigned int v218; // [rsp+30h] [rbp-210h]
  __int8 v219; // [rsp+30h] [rbp-210h]
  __int64 v220; // [rsp+30h] [rbp-210h]
  __int64 v221; // [rsp+30h] [rbp-210h]
  __int64 v222; // [rsp+30h] [rbp-210h]
  __int64 v223; // [rsp+30h] [rbp-210h]
  __int64 v224; // [rsp+30h] [rbp-210h]
  __int64 v225; // [rsp+38h] [rbp-208h]
  unsigned int v226; // [rsp+40h] [rbp-200h]
  __int64 v227; // [rsp+40h] [rbp-200h]
  unsigned int v228; // [rsp+40h] [rbp-200h]
  __int64 v229; // [rsp+40h] [rbp-200h]
  unsigned __int16 v230; // [rsp+40h] [rbp-200h]
  __int64 v231; // [rsp+40h] [rbp-200h]
  unsigned int v232; // [rsp+40h] [rbp-200h]
  __int64 v233; // [rsp+40h] [rbp-200h]
  __int16 v234; // [rsp+40h] [rbp-200h]
  __int16 v235; // [rsp+42h] [rbp-1FEh]
  __int64 v236; // [rsp+48h] [rbp-1F8h]
  unsigned __int16 v237; // [rsp+50h] [rbp-1F0h]
  __int64 v238; // [rsp+50h] [rbp-1F0h]
  unsigned int v239; // [rsp+50h] [rbp-1F0h]
  __int64 *v240; // [rsp+50h] [rbp-1F0h]
  __int64 v241; // [rsp+50h] [rbp-1F0h]
  __int64 v242; // [rsp+50h] [rbp-1F0h]
  int v243; // [rsp+50h] [rbp-1F0h]
  __int64 v244; // [rsp+50h] [rbp-1F0h]
  __int64 v245; // [rsp+58h] [rbp-1E8h]
  __int64 v246; // [rsp+58h] [rbp-1E8h]
  __int64 v247; // [rsp+60h] [rbp-1E0h]
  __int64 v248; // [rsp+60h] [rbp-1E0h]
  __int128 v249; // [rsp+60h] [rbp-1E0h]
  __int64 v250; // [rsp+60h] [rbp-1E0h]
  __int64 v251; // [rsp+60h] [rbp-1E0h]
  __int64 v252; // [rsp+60h] [rbp-1E0h]
  __int64 v253; // [rsp+60h] [rbp-1E0h]
  __int64 v254; // [rsp+70h] [rbp-1D0h]
  __int128 v255; // [rsp+70h] [rbp-1D0h]
  __int64 v256; // [rsp+70h] [rbp-1D0h]
  __int64 v257; // [rsp+70h] [rbp-1D0h]
  unsigned __int64 v258; // [rsp+70h] [rbp-1D0h]
  unsigned int v259; // [rsp+70h] [rbp-1D0h]
  unsigned int v260; // [rsp+70h] [rbp-1D0h]
  __int64 v261; // [rsp+70h] [rbp-1D0h]
  __int64 v262; // [rsp+70h] [rbp-1D0h]
  int v263; // [rsp+70h] [rbp-1D0h]
  unsigned __int16 v264; // [rsp+70h] [rbp-1D0h]
  __int64 v265; // [rsp+70h] [rbp-1D0h]
  __int64 v266; // [rsp+70h] [rbp-1D0h]
  __int64 v267; // [rsp+78h] [rbp-1C8h]
  __int64 v268; // [rsp+78h] [rbp-1C8h]
  __int64 v269; // [rsp+78h] [rbp-1C8h]
  __int64 v270; // [rsp+78h] [rbp-1C8h]
  __int64 v271; // [rsp+80h] [rbp-1C0h]
  __int128 v272; // [rsp+80h] [rbp-1C0h]
  __int64 v273; // [rsp+80h] [rbp-1C0h]
  __int64 v274; // [rsp+80h] [rbp-1C0h]
  __int64 v275; // [rsp+80h] [rbp-1C0h]
  __int64 v276; // [rsp+80h] [rbp-1C0h]
  __int64 v277; // [rsp+80h] [rbp-1C0h]
  __int64 v278; // [rsp+80h] [rbp-1C0h]
  __int64 v279; // [rsp+80h] [rbp-1C0h]
  __int64 v280; // [rsp+80h] [rbp-1C0h]
  __int64 v281; // [rsp+88h] [rbp-1B8h]
  __int64 v282; // [rsp+88h] [rbp-1B8h]
  __int64 v283; // [rsp+88h] [rbp-1B8h]
  __int64 v284; // [rsp+90h] [rbp-1B0h]
  unsigned __int16 v285; // [rsp+90h] [rbp-1B0h]
  __int64 v286; // [rsp+90h] [rbp-1B0h]
  __int64 v287; // [rsp+90h] [rbp-1B0h]
  __int64 v288; // [rsp+90h] [rbp-1B0h]
  __int64 v289; // [rsp+90h] [rbp-1B0h]
  __int64 v290; // [rsp+90h] [rbp-1B0h]
  const __m128i *v291; // [rsp+90h] [rbp-1B0h]
  __int64 v292; // [rsp+90h] [rbp-1B0h]
  __int64 v293; // [rsp+90h] [rbp-1B0h]
  __int64 v294; // [rsp+90h] [rbp-1B0h]
  __int64 v295; // [rsp+90h] [rbp-1B0h]
  __int64 v296; // [rsp+90h] [rbp-1B0h]
  __int64 v297; // [rsp+90h] [rbp-1B0h]
  __int64 v298; // [rsp+98h] [rbp-1A8h]
  const __m128i *v299; // [rsp+A0h] [rbp-1A0h]
  __int64 v300; // [rsp+A0h] [rbp-1A0h]
  __int64 v301; // [rsp+A0h] [rbp-1A0h]
  __int64 v302; // [rsp+A0h] [rbp-1A0h]
  __int64 v303; // [rsp+A0h] [rbp-1A0h]
  __int64 v304; // [rsp+A0h] [rbp-1A0h]
  __int8 *v305; // [rsp+A0h] [rbp-1A0h]
  unsigned __int64 v306; // [rsp+A0h] [rbp-1A0h]
  __int64 v307; // [rsp+A0h] [rbp-1A0h]
  __int64 v308; // [rsp+A0h] [rbp-1A0h]
  __int64 v309; // [rsp+A0h] [rbp-1A0h]
  __int64 v310; // [rsp+A0h] [rbp-1A0h]
  __int64 v311; // [rsp+A8h] [rbp-198h]
  __int128 v312; // [rsp+B0h] [rbp-190h]
  const __m128i *v313; // [rsp+B0h] [rbp-190h]
  int v314; // [rsp+B0h] [rbp-190h]
  int v315; // [rsp+C0h] [rbp-180h]
  __int64 v316; // [rsp+C0h] [rbp-180h]
  __int64 v317; // [rsp+C0h] [rbp-180h]
  __int64 v318; // [rsp+C0h] [rbp-180h]
  __int64 v319; // [rsp+C0h] [rbp-180h]
  __int64 v320; // [rsp+C0h] [rbp-180h]
  __int64 v321; // [rsp+C0h] [rbp-180h]
  unsigned __int64 v322; // [rsp+C8h] [rbp-178h]
  __int64 v323; // [rsp+C8h] [rbp-178h]
  __int64 v324; // [rsp+C8h] [rbp-178h]
  __int64 v325; // [rsp+C8h] [rbp-178h]
  __int64 v326; // [rsp+C8h] [rbp-178h]
  __int64 v327; // [rsp+C8h] [rbp-178h]
  __int64 v328; // [rsp+C8h] [rbp-178h]
  __int128 v329; // [rsp+D0h] [rbp-170h]
  __int64 v330; // [rsp+D0h] [rbp-170h]
  __int64 v331; // [rsp+D0h] [rbp-170h]
  __int64 v332; // [rsp+D0h] [rbp-170h]
  __int64 v333; // [rsp+D0h] [rbp-170h]
  __int64 v334; // [rsp+D0h] [rbp-170h]
  __int64 v335; // [rsp+D0h] [rbp-170h]
  int v336; // [rsp+D0h] [rbp-170h]
  __int64 v337; // [rsp+D0h] [rbp-170h]
  __int64 v338; // [rsp+D8h] [rbp-168h]
  __int64 v339; // [rsp+110h] [rbp-130h]
  __int64 v340; // [rsp+118h] [rbp-128h]
  unsigned int v341; // [rsp+120h] [rbp-120h] BYREF
  __int64 v342; // [rsp+128h] [rbp-118h]
  _QWORD v343[2]; // [rsp+130h] [rbp-110h] BYREF
  __int64 v344; // [rsp+140h] [rbp-100h] BYREF
  __int64 v345; // [rsp+148h] [rbp-F8h]
  __m128i v346; // [rsp+150h] [rbp-F0h] BYREF
  unsigned __int64 v347; // [rsp+160h] [rbp-E0h]
  __int64 v348; // [rsp+168h] [rbp-D8h]
  __int64 v349; // [rsp+170h] [rbp-D0h]
  __int64 v350; // [rsp+178h] [rbp-C8h]
  __m128i v351; // [rsp+180h] [rbp-C0h] BYREF
  _BYTE v352[176]; // [rsp+190h] [rbp-B0h] BYREF

  v4 = *(_QWORD *)(a2 + 40);
  v5 = *(__int16 **)(a2 + 48);
  v6 = *(_QWORD *)(v4 + 80);
  v7 = *v5;
  v8 = *((_QWORD *)v5 + 1);
  v9 = *(_QWORD *)(v6 + 96);
  LOWORD(v341) = *v5;
  v10 = *(_QWORD *)(v4 + 40);
  v342 = v8;
  v11 = *(_QWORD *)v4;
  v329 = (__int128)_mm_loadu_si128((const __m128i *)(v4 + 40));
  v12 = *(_QWORD *)(v4 + 8);
  v13 = *(_QWORD *)v4;
  v312 = (__int128)_mm_loadu_si128((const __m128i *)(v4 + 80));
  v14 = *(_DWORD *)(v4 + 8);
  v15 = *(unsigned int *)(v4 + 48);
  v16 = *(_QWORD **)(v9 + 24);
  if ( *(_DWORD *)(v9 + 32) > 0x40u )
    v16 = (_QWORD *)*v16;
  v322 = (unsigned __int64)v16;
  v17 = *(_DWORD *)(v10 + 24);
  if ( v17 == 51 )
    return v11;
  v315 = *(_DWORD *)(v4 + 88);
  if ( *(_DWORD *)(v13 + 24) != 51 )
    goto LABEL_46;
  if ( v17 == 161 )
  {
    v48 = *(_QWORD *)(v10 + 40);
    if ( v6 != *(_QWORD *)(v48 + 40) || *(_DWORD *)(v48 + 48) != v315 )
    {
LABEL_48:
      if ( *(_QWORD *)v48 != v13
        || *(_DWORD *)(v48 + 8) != v14
        || v6 != *(_QWORD *)(v48 + 40)
        || *(_DWORD *)(v48 + 48) != v315 )
      {
        goto LABEL_49;
      }
      return v11;
    }
    v97 = *(_QWORD *)(*(_QWORD *)v48 + 48LL) + 16LL * *(unsigned int *)(v48 + 8);
    v98 = *(_WORD *)v97;
    v99 = *(_QWORD *)(v97 + 8);
    v346.m128i_i16[0] = v98;
    v346.m128i_i64[1] = v99;
    if ( v98 == v7 )
    {
      if ( v7 || v99 == v8 )
        return *(_QWORD *)v48;
      v279 = v11;
      v282 = v12;
      v296 = v13;
      v309 = v6;
      v197 = sub_33CF170(v312, *((_QWORD *)&v312 + 1));
      v6 = v309;
      v13 = v296;
      v12 = v282;
      v11 = v279;
      if ( v197 )
      {
        v351 = _mm_loadu_si128(&v346);
        goto LABEL_207;
      }
    }
    else
    {
      v285 = v98;
      v254 = v11;
      v267 = v12;
      v271 = v13;
      v300 = v6;
      v47 = sub_33CF170(v312, *((_QWORD *)&v312 + 1));
      v6 = v300;
      v13 = v271;
      v12 = v267;
      v11 = v254;
      if ( v47 )
      {
        v351 = _mm_loadu_si128(&v346);
        if ( v285 )
        {
          if ( v285 == 1 || (unsigned __int16)(v285 - 504) <= 7u )
            goto LABEL_240;
          v306 = *(_QWORD *)&byte_444C4A0[16 * v285 - 16];
          v185 = byte_444C4A0[16 * v285 - 8];
LABEL_208:
          v186 = v341;
          if ( !(_WORD)v341 )
          {
            v244 = v11;
            v246 = v12;
            v253 = v13;
            v293 = v6;
            v187 = sub_3007260((__int64)&v341);
            v11 = v244;
            v12 = v246;
            v347 = v187;
            v13 = v253;
            v348 = v188;
            v186 = 0;
            v6 = v293;
LABEL_210:
            if ( ((_BYTE)v188 || !v185) && v187 >= v306 )
            {
              if ( v186 )
              {
                if ( (unsigned __int16)(v186 - 17) > 0x9Eu )
                {
LABEL_222:
                  v194 = *(__int128 **)(v10 + 40);
                  v173 = *a1;
                  v351.m128i_i64[0] = *(_QWORD *)(a2 + 80);
                  if ( v351.m128i_i64[0] )
                  {
                    v337 = v11;
                    v338 = v12;
                    sub_325F5D0(v351.m128i_i64);
                    v12 = v338;
                    v11 = v337;
                  }
                  v204 = v312;
                  v351.m128i_i32[2] = *(_DWORD *)(a2 + 72);
                  v201 = *v194;
                  goto LABEL_198;
                }
              }
              else
              {
                v280 = v11;
                v283 = v12;
                v297 = v13;
                v310 = v6;
                v198 = sub_30070D0((__int64)&v341);
                v6 = v310;
                v13 = v297;
                v12 = v283;
                v11 = v280;
                if ( !v198 )
                  goto LABEL_222;
              }
              v278 = v11;
              v281 = v12;
              v295 = v13;
              v307 = v6;
              v193 = sub_3280200((__int64)&v346);
              v6 = v307;
              v13 = v295;
              v12 = v281;
              v11 = v278;
              if ( !v193 )
                goto LABEL_222;
            }
            v266 = v11;
            v270 = v12;
            v277 = v13;
            v294 = v6;
            v189 = sub_3280910(&v341, v346.m128i_u32[0], v346.m128i_i64[1]);
            v6 = v294;
            v13 = v277;
            v12 = v270;
            v11 = v266;
            if ( v189 )
            {
              if ( !sub_3280200((__int64)&v341)
                || (v190 = sub_3280220((__int64)&v346), v6 = v294, v13 = v277, v12 = v270, v11 = v266, !v190) )
              {
                v191 = *(__int128 **)(v10 + 40);
                v192 = *a1;
                v351.m128i_i64[0] = *(_QWORD *)(a2 + 80);
                if ( v351.m128i_i64[0] )
                  sub_325F5D0(v351.m128i_i64);
                v351.m128i_i32[2] = *(_DWORD *)(a2 + 72);
                v328 = sub_3406EB0(v192, 161, (unsigned int)&v351, v341, v342, v13, *v191, v312);
                sub_9C6650(&v351);
                return v328;
              }
            }
            goto LABEL_45;
          }
          if ( (_WORD)v341 != 1 && (unsigned __int16)(v341 - 504) > 7u )
          {
            v187 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v341 - 16];
            LOBYTE(v188) = byte_444C4A0[16 * (unsigned __int16)v341 - 8];
            goto LABEL_210;
          }
LABEL_240:
          BUG();
        }
LABEL_207:
        v265 = v11;
        v269 = v12;
        v276 = v13;
        v292 = v6;
        v183 = sub_3007260((__int64)&v351);
        v12 = v269;
        v11 = v265;
        v349 = v183;
        v13 = v276;
        v306 = v183;
        v6 = v292;
        v350 = v184;
        v185 = v184;
        goto LABEL_208;
      }
    }
LABEL_45:
    v17 = *(_DWORD *)(v10 + 24);
LABEL_46:
    if ( v17 != 161 )
      goto LABEL_6;
    v48 = *(_QWORD *)(v10 + 40);
    goto LABEL_48;
  }
LABEL_6:
  v18 = *(_DWORD *)(v13 + 24);
  v19 = v18;
  if ( v17 != 168 )
    goto LABEL_50;
  if ( v18 != 51 )
    goto LABEL_8;
  v134 = *(_QWORD **)(v10 + 40);
  v241 = v11;
  v245 = v12;
  v250 = v13;
  v261 = v6;
  v275 = *v134;
  v289 = v134[1];
  v304 = *a1;
  v135 = sub_33E2390(*a1, *v134, v289, 1);
  LODWORD(v13) = v250;
  if ( !v135 && !(unsigned __int8)sub_33E2470(v304, v275, v289) )
  {
    v151 = *(_QWORD *)(v10 + 56);
    v6 = v261;
    v152 = 1;
    v13 = v250;
    v12 = v245;
    v11 = v241;
    if ( !v151 )
      goto LABEL_49;
    do
    {
      if ( *(_DWORD *)(v151 + 8) == (_DWORD)v15 )
      {
        if ( !v152 )
          goto LABEL_49;
        v151 = *(_QWORD *)(v151 + 32);
        if ( !v151 )
          goto LABEL_148;
        if ( *(_DWORD *)(v151 + 8) == (_DWORD)v15 )
          goto LABEL_49;
        v152 = 0;
      }
      v151 = *(_QWORD *)(v151 + 32);
    }
    while ( v151 );
    if ( v152 == 1 )
    {
LABEL_49:
      v18 = *(_DWORD *)(v13 + 24);
      v19 = v18;
LABEL_50:
      if ( v18 == 51 )
      {
        if ( *(_DWORD *)(v10 + 24) != 234
          || (v73 = **(_QWORD **)(v10 + 40), *(_DWORD *)(v73 + 24) != 161)
          || (v74 = *(_QWORD *)(v73 + 40), v6 != *(_QWORD *)(v74 + 40))
          || *(_DWORD *)(v74 + 48) != v315 )
        {
LABEL_52:
          v20 = *(_DWORD *)(v10 + 24);
          if ( v20 != 160 )
            goto LABEL_10;
          v49 = *(_QWORD **)(v10 + 40);
          if ( *(_DWORD *)(*v49 + 24LL) != 51 )
            goto LABEL_12;
          v301 = v11;
          v311 = v12;
          v317 = v13;
          v50 = sub_33CF170(v49[10], v49[11]);
          v13 = v317;
          v12 = v311;
          v11 = v301;
          if ( !v50 )
            goto LABEL_55;
          v166 = sub_33CF170(v312, *((_QWORD *)&v312 + 1));
          v12 = v311;
          v11 = v301;
          if ( !v166 )
          {
            v13 = v317;
LABEL_55:
            v19 = *(_DWORD *)(v13 + 24);
            if ( v19 != 234 && v19 != 51 )
              goto LABEL_13;
            goto LABEL_57;
          }
          v172 = *(_QWORD *)(v10 + 40);
          v173 = *a1;
          v351.m128i_i64[0] = *(_QWORD *)(a2 + 80);
          if ( v351.m128i_i64[0] )
          {
            sub_325F5D0(v351.m128i_i64);
            v12 = v311;
            v11 = v301;
          }
          v204 = v312;
          v351.m128i_i32[2] = *(_DWORD *)(a2 + 72);
          v201 = *(_OWORD *)(v172 + 40);
LABEL_198:
          *((_QWORD *)&v200 + 1) = v12;
          *(_QWORD *)&v200 = v11;
          v327 = sub_340F900(v173, 160, (unsigned int)&v351, v341, v342, v13, v200, v201, v204);
          sub_9C6650(&v351);
          return v327;
        }
        v256 = v11;
        v268 = v12;
        v273 = v13;
        v287 = v6;
        v340 = sub_3281590((__int64)&v341);
        v75 = *(unsigned int **)(**(_QWORD **)(v10 + 40) + 40LL);
        v76 = *(_QWORD *)(*(_QWORD *)v75 + 48LL) + 16LL * v75[2];
        v77 = *(_WORD *)v76;
        v78 = *(_QWORD *)(v76 + 8);
        LOWORD(v343[0]) = v77;
        v343[1] = v78;
        v339 = sub_3281590((__int64)v343);
        v6 = v287;
        v13 = v273;
        v12 = v268;
        v11 = v256;
        if ( (_DWORD)v340 == (_DWORD)v339 && BYTE4(v339) == BYTE4(v340) )
        {
          v195.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v341);
          v351 = v195;
          v308 = *(_QWORD *)(v10 + 40);
          v195.m128i_i64[0] = *(_QWORD *)(**(_QWORD **)(*(_QWORD *)v308 + 40LL) + 48LL)
                            + 16LL * *(unsigned int *)(*(_QWORD *)(*(_QWORD *)v308 + 40LL) + 8LL);
          v195.m128i_i64[1] = *(_QWORD *)(v195.m128i_i64[0] + 8);
          v195.m128i_i16[0] = *(_WORD *)v195.m128i_i64[0];
          v345 = v195.m128i_i64[1];
          LOWORD(v344) = v195.m128i_i16[0];
          v196.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v344);
          v6 = v287;
          v13 = v273;
          v12 = v268;
          v11 = v256;
          v346 = v196;
          if ( v196.m128i_i64[0] == v351.m128i_i64[0] && v346.m128i_i8[8] == v351.m128i_i8[8] )
            return sub_33FB890(
                     *a1,
                     v341,
                     v342,
                     **(_QWORD **)(*(_QWORD *)v308 + 40LL),
                     *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v308 + 40LL) + 8LL));
        }
        v18 = *(_DWORD *)(v13 + 24);
        v19 = v18;
      }
LABEL_8:
      if ( v18 != 234 )
        goto LABEL_77;
      v20 = *(_DWORD *)(v10 + 24);
      if ( v20 != 234 )
        goto LABEL_10;
      v52 = *(unsigned int **)(v13 + 40);
      v53 = *(const __m128i **)(v10 + 40);
      v54 = v53->m128i_i64[0];
      v302 = *(_QWORD *)v52;
      v286 = 16LL * v52[2];
      v55 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v52 + 48LL) + v286);
      v56 = v53->m128i_u32[2];
      v57 = *v55;
      v272 = (__int128)_mm_loadu_si128((const __m128i *)*(_QWORD *)(v13 + 40));
      v58 = *((_QWORD *)v55 + 1);
      v255 = (__int128)_mm_loadu_si128(v53);
      LOWORD(v344) = v57;
      v345 = v58;
      v59 = *(_QWORD *)(v54 + 48) + 16 * v56;
      v60 = *(_WORD *)v59;
      v61 = *(_QWORD *)(v59 + 8);
      v346.m128i_i16[0] = v60;
      v346.m128i_i64[1] = v61;
      if ( (_WORD)v57 )
      {
        if ( (unsigned __int16)(v57 - 17) > 0xD3u )
          goto LABEL_57;
      }
      else
      {
        v210 = v11;
        v215 = v12;
        v217 = v13;
        v226 = v57;
        v237 = v60;
        v247 = v6;
        v62 = sub_30070B0((__int64)&v344);
        v6 = v247;
        v60 = v237;
        v57 = v226;
        v13 = v217;
        v11 = v210;
        v12 = v215;
        if ( !v62 )
          goto LABEL_57;
      }
      if ( v60 )
      {
        if ( (unsigned __int16)(v60 - 17) <= 0xD3u )
        {
          v63 = 0;
          v64 = (unsigned __int16)word_4456580[v60 - 1];
LABEL_71:
          if ( (_WORD)v57 )
          {
            v65 = word_4456580[(unsigned __int16)v57 - 1];
            v66 = 0;
          }
          else
          {
            v214 = v11;
            v216 = v12;
            v224 = v13;
            v234 = v64;
            v252 = v6;
            v65 = sub_3009970((__int64)&v344, v63, v57, v64, v6);
            v11 = v214;
            v12 = v216;
            v13 = v224;
            LOWORD(v64) = v234;
            v6 = v252;
          }
          if ( v65 == (_WORD)v64 && (v65 || v63 == v66) )
          {
            v227 = v11;
            v236 = v12;
            v238 = v13;
            v248 = v6;
            v351.m128i_i64[0] = sub_3281590((__int64)&v341);
            v67 = sub_3281590((__int64)&v344);
            v6 = v248;
            v343[0] = v67;
            v13 = v238;
            v11 = v227;
            v12 = v236;
            if ( v351.m128i_i32[0] == (_DWORD)v67 && BYTE4(v343[0]) == v351.m128i_i8[4] )
            {
              v174 = *a1;
              v175 = *(_QWORD *)(*(_QWORD *)(v302 + 48) + v286 + 8);
              v176 = *(unsigned __int16 *)(*(_QWORD *)(v302 + 48) + v286);
              v351.m128i_i64[0] = *(_QWORD *)(a2 + 80);
              if ( v351.m128i_i64[0] )
                sub_325F5D0(v351.m128i_i64);
              v351.m128i_i32[2] = *(_DWORD *)(a2 + 72);
              v177 = sub_340F900(v174, 160, (unsigned int)&v351, v176, v175, v13, v272, v255, v312);
              v179 = v178;
              v180 = v177;
              sub_9C6650(&v351);
              return sub_33FB890(*a1, v341, v342, v180, v179);
            }
          }
          v18 = *(_DWORD *)(v13 + 24);
          v19 = v18;
LABEL_77:
          if ( v18 == 160 )
          {
            v68 = *(_QWORD *)(v13 + 40);
            v69 = *(_QWORD *)(v10 + 48) + 16LL * (unsigned int)v15;
            v70 = *(_QWORD *)(*(_QWORD *)(v68 + 40) + 48LL) + 16LL * *(unsigned int *)(v68 + 48);
            if ( *(_WORD *)v70 == *(_WORD *)v69
              && (*(_QWORD *)(v70 + 8) == *(_QWORD *)(v69 + 8) || *(_WORD *)v70)
              && v6 == *(_QWORD *)(v68 + 80)
              && *(_DWORD *)(v68 + 88) == v315 )
            {
              v140 = *a1;
              v351.m128i_i64[0] = *(_QWORD *)(a2 + 80);
              if ( v351.m128i_i64[0] )
              {
                v325 = v68;
                sub_325F5D0(v351.m128i_i64);
                v68 = v325;
              }
              v351.m128i_i32[2] = *(_DWORD *)(a2 + 72);
              result = sub_340F900(v140, 160, (unsigned int)&v351, v341, v342, v13, *(_OWORD *)v68, v329, v312);
LABEL_111:
              v96 = v351.m128i_i64[0];
              if ( !v351.m128i_i64[0] )
                return result;
LABEL_151:
              v335 = result;
              sub_B91220((__int64)&v351, v96);
              return v335;
            }
LABEL_85:
            v71 = *(_QWORD *)(v13 + 56);
            v72 = 1;
            if ( !v71 )
              goto LABEL_59;
            if ( *(_DWORD *)(v71 + 8) == v14 )
              goto LABEL_89;
            while ( 1 )
            {
              v71 = *(_QWORD *)(v71 + 32);
              if ( !v71 )
                break;
              if ( *(_DWORD *)(v71 + 8) == v14 )
              {
LABEL_89:
                if ( !v72 )
                  goto LABEL_59;
                v71 = *(_QWORD *)(v71 + 32);
                if ( !v71 )
                  goto LABEL_98;
                if ( *(_DWORD *)(v71 + 8) == v14 )
                  goto LABEL_59;
                v72 = 0;
              }
            }
            if ( v72 == 1 )
              goto LABEL_59;
LABEL_98:
            v79 = *(_QWORD *)(v13 + 40);
            v80 = *(_QWORD *)(v10 + 48) + 16 * v15;
            v81 = *(_QWORD *)(*(_QWORD *)(v79 + 40) + 48LL) + 16LL * *(unsigned int *)(v79 + 48);
            if ( *(_WORD *)v80 != *(_WORD *)v81 || *(_QWORD *)(v80 + 8) != *(_QWORD *)(v81 + 8) && !*(_WORD *)v80 )
              goto LABEL_59;
            v82 = *(_QWORD *)(*(_QWORD *)(v79 + 80) + 96LL);
            v83 = *(_QWORD **)(v82 + 24);
            if ( *(_DWORD *)(v82 + 32) > 0x40u )
              v83 = (_QWORD *)*v83;
            if ( (unsigned int)v83 <= v322 )
              goto LABEL_59;
            v84 = *(_QWORD *)(a2 + 80);
            v85 = *a1;
            v351.m128i_i64[0] = v84;
            if ( v84 )
            {
              v323 = v13;
              sub_B96E90((__int64)&v351, v84, 1);
              v13 = v323;
            }
            v351.m128i_i32[2] = *(_DWORD *)(a2 + 72);
            v324 = v13;
            v86 = sub_340F900(v85, 160, (unsigned int)&v351, v341, v342, v13, *(_OWORD *)v79, v329, v312);
            v88 = v324;
            v89 = v86;
            v91 = v90;
            if ( v351.m128i_i64[0] )
            {
              sub_B91220((__int64)&v351, v351.m128i_i64[0]);
              v88 = v324;
            }
            v334 = v88;
            sub_32B3E80((__int64)a1, v89, 1, 0, v87, v88);
            v92 = v334;
            v93 = *a1;
            v94 = *(_QWORD *)(v334 + 80);
            v95 = *(_QWORD *)(v334 + 40);
            v351.m128i_i64[0] = v94;
            if ( v94 )
            {
              sub_B96E90((__int64)&v351, v94, 1);
              v92 = v334;
            }
            v351.m128i_i32[2] = *(_DWORD *)(v92 + 72);
            *((_QWORD *)&v199 + 1) = v91;
            *(_QWORD *)&v199 = v89;
            result = sub_340F900(
                       v93,
                       160,
                       (unsigned int)&v351,
                       v341,
                       v342,
                       v92,
                       v199,
                       *(_OWORD *)(v95 + 40),
                       *(_OWORD *)(v95 + 80));
            goto LABEL_111;
          }
          if ( v18 != 51 )
          {
            if ( v18 != 234 )
            {
LABEL_14:
              if ( v19 == 159 )
              {
                v21 = *(_QWORD *)(v13 + 56);
                if ( v21 )
                {
                  v22 = 1;
                  do
                  {
                    while ( *(_DWORD *)(v21 + 8) != v14 )
                    {
                      v21 = *(_QWORD *)(v21 + 32);
                      if ( !v21 )
                        goto LABEL_23;
                    }
                    if ( !v22 )
                      goto LABEL_59;
                    v23 = *(_QWORD *)(v21 + 32);
                    if ( !v23 )
                      goto LABEL_24;
                    if ( *(_DWORD *)(v23 + 8) == v14 )
                      goto LABEL_59;
                    v21 = *(_QWORD *)(v23 + 32);
                    v22 = 0;
                  }
                  while ( v21 );
LABEL_23:
                  if ( v22 == 1 )
                    goto LABEL_59;
LABEL_24:
                  v24 = *(const __m128i **)(v13 + 40);
                  v25 = *(_QWORD *)(v10 + 48) + 16LL * (unsigned int)v15;
                  v26 = *(_WORD *)v25;
                  v27 = *(_QWORD *)(v24->m128i_i64[0] + 48) + 16LL * v24->m128i_u32[2];
                  v28 = *(_WORD *)v27;
                  if ( *(_WORD *)v27 == *(_WORD *)v25 )
                  {
                    v29 = *(_QWORD *)(v25 + 8);
                    v30 = *(_QWORD *)(v27 + 8);
                    if ( v28 || v30 == v29 )
                    {
                      v346.m128i_i16[0] = v28;
                      v346.m128i_i64[1] = v30;
                      if ( v28 )
                      {
                        v351.m128i_i16[0] = v26;
                        v351.m128i_i64[1] = v29;
                        v139 = (unsigned __int16)(v28 - 176) <= 0x34u;
                        v32 = v139;
                        if ( v26 )
                        {
                          if ( (unsigned __int16)(v26 - 176) <= 0x34u == v139 )
                          {
                            LODWORD(v33) = word_4456340[v26 - 1];
                            goto LABEL_31;
                          }
                          goto LABEL_59;
                        }
                      }
                      else
                      {
                        v316 = v13;
                        v313 = v24;
                        v330 = v29;
                        v31 = sub_3007100((__int64)&v346);
                        v29 = v330;
                        v351.m128i_i16[0] = v26;
                        v24 = v313;
                        v13 = v316;
                        v32 = v31;
                        v351.m128i_i64[1] = v330;
                      }
                      v284 = v13;
                      v299 = v24;
                      v331 = v29;
                      if ( sub_3007100((__int64)&v351) == v32 )
                      {
                        v351.m128i_i16[0] = v26;
                        v351.m128i_i64[1] = v331;
                        v33 = sub_3007240((__int64)&v351);
                        v24 = v299;
                        v13 = v284;
                        v344 = v33;
LABEL_31:
                        v34 = *(unsigned int *)(v13 + 64);
                        v351.m128i_i64[0] = (__int64)v352;
                        v34 *= 5LL;
                        v35 = v34;
                        v36 = 0xCCCCCCCCCCCCCCCDLL * v34;
                        v37 = (__int64)&v24->m128i_i64[v35];
                        v351.m128i_i64[1] = 0x800000000LL;
                        if ( v35 > 40 )
                        {
                          v291 = v24;
                          v305 = &v24->m128i_i8[v35 * 8];
                          v314 = v33;
                          v336 = v36;
                          sub_C8D5F0((__int64)&v351, v352, v36, 0x10u, (__int64)v24, v37);
                          v39 = v351.m128i_i32[2];
                          v40 = (_BYTE *)v351.m128i_i64[0];
                          LODWORD(v36) = v336;
                          LODWORD(v33) = v314;
                          v37 = (__int64)v305;
                          v24 = v291;
                          v38 = (__m128i *)(v351.m128i_i64[0] + 16LL * v351.m128i_u32[2]);
                        }
                        else
                        {
                          v38 = (__m128i *)v352;
                          v39 = 0;
                          v40 = v352;
                        }
                        if ( v24 != (const __m128i *)v37 )
                        {
                          do
                          {
                            if ( v38 )
                              *v38 = _mm_loadu_si128(v24);
                            v24 = (const __m128i *)((char *)v24 + 40);
                            ++v38;
                          }
                          while ( (const __m128i *)v37 != v24 );
                          v40 = (_BYTE *)v351.m128i_i64[0];
                          v39 = v351.m128i_i32[2];
                        }
                        v351.m128i_i32[2] = v39 + v36;
                        v41 = &v40[16 * (v322 / (unsigned int)v33)];
                        *(_QWORD *)v41 = v10;
                        *((_DWORD *)v41 + 2) = v15;
                        v42 = *(_QWORD *)(a2 + 80);
                        v43 = *a1;
                        v346.m128i_i64[0] = v42;
                        v44 = v351.m128i_i64[0];
                        v45 = v351.m128i_u32[2];
                        if ( v42 )
                          sub_B96E90((__int64)&v346, v42, 1);
                        *((_QWORD *)&v202 + 1) = v45;
                        *(_QWORD *)&v202 = v44;
                        v346.m128i_i32[2] = *(_DWORD *)(a2 + 72);
                        result = sub_33FC220(v43, 159, (unsigned int)&v346, v341, v342, v37, v202);
                        if ( v346.m128i_i64[0] )
                        {
                          v332 = result;
                          sub_B91220((__int64)&v346, v346.m128i_i64[0]);
                          result = v332;
                        }
                        if ( (_BYTE *)v351.m128i_i64[0] != v352 )
                        {
                          v333 = result;
                          _libc_free(v351.m128i_u64[0]);
                          return v333;
                        }
                        return result;
                      }
                    }
                  }
                }
              }
LABEL_59:
              v51 = (unsigned __int8)sub_32E2EF0((__int64)a1, a2, 0) == 0;
              result = 0;
              if ( !v51 )
                return a2;
              return result;
            }
            goto LABEL_57;
          }
          goto LABEL_52;
        }
      }
      else
      {
        v223 = v11;
        v225 = v12;
        v233 = v13;
        v251 = v6;
        v146 = sub_30070B0((__int64)&v346);
        v13 = v233;
        v11 = v223;
        v12 = v225;
        if ( v146 )
        {
          v149 = sub_3009970((__int64)&v346, v54, v147, v148, v251);
          v11 = v223;
          v12 = v225;
          v63 = v150;
          v13 = v233;
          v57 = (unsigned __int16)v344;
          v64 = v149;
          v6 = v251;
          goto LABEL_71;
        }
      }
LABEL_57:
      v20 = *(_DWORD *)(v10 + 24);
LABEL_10:
      if ( v20 != 234 )
      {
        v18 = *(_DWORD *)(v13 + 24);
LABEL_12:
        v19 = v18;
        goto LABEL_13;
      }
      v318 = v13;
      v303 = sub_33CF5B0(v11, v12);
      v274 = v100;
      *(_QWORD *)&v249 = sub_33CF5B0(v329, *((_QWORD *)&v329 + 1));
      v13 = v318;
      v102 = (unsigned int)v101;
      v103 = 16LL * (unsigned int)v274;
      *((_QWORD *)&v249 + 1) = v101;
      v104 = (unsigned __int16 *)(v103 + *(_QWORD *)(v303 + 48));
      v105 = *v104;
      v257 = *((_QWORD *)v104 + 1);
      v351.m128i_i16[0] = v105;
      v351.m128i_i64[1] = v257;
      if ( (_WORD)v105 )
      {
        if ( (unsigned __int16)(v105 - 17) <= 0xD3u )
        {
          v257 = 0;
          v105 = (unsigned __int16)word_4456580[(unsigned __int16)v105 - 1];
        }
      }
      else
      {
        v218 = v105;
        v239 = v101;
        v106 = sub_30070B0((__int64)&v351);
        v102 = v239;
        v103 = 16LL * (unsigned int)v274;
        v105 = v218;
        v13 = v318;
        if ( v106 )
        {
          v154 = sub_3009970((__int64)&v351, v103, v107, v239, v218);
          v13 = v318;
          v103 = 16LL * (unsigned int)v274;
          v257 = v155;
          v102 = v239;
          v105 = v154;
        }
      }
      v108 = 16 * v102;
      v109 = (unsigned __int16 *)(16 * v102 + *(_QWORD *)(v249 + 48));
      v110 = *v109;
      v111 = *((_QWORD *)v109 + 1);
      v351.m128i_i16[0] = v110;
      v351.m128i_i64[1] = v111;
      if ( (_WORD)v110 )
      {
        if ( (unsigned __int16)(v110 - 17) <= 0xD3u )
        {
          v111 = 0;
          LOWORD(v110) = word_4456580[(unsigned __int16)v110 - 1];
        }
      }
      else
      {
        v206 = v13;
        v208 = v105;
        v222 = v111;
        v232 = v110;
        v242 = v108;
        v141 = sub_30070B0((__int64)&v351);
        v108 = v242;
        LOWORD(v110) = v232;
        v111 = v222;
        LOWORD(v105) = v208;
        v13 = v206;
        if ( v141 )
        {
          v143 = sub_3009970((__int64)&v351, v103, v142, v232, v208);
          v13 = v206;
          LOWORD(v105) = v208;
          v108 = v242;
          LOWORD(v110) = v143;
          v111 = v144;
        }
      }
      LOWORD(v344) = v110;
      v345 = v111;
      v19 = *(_DWORD *)(v13 + 24);
      if ( v19 != 51 && ((_WORD)v110 != (_WORD)v105 || !(_WORD)v110 && v257 != v111) )
      {
LABEL_13:
        if ( v19 != 160 )
          goto LABEL_14;
        goto LABEL_85;
      }
      v112 = *(_QWORD *)(v303 + 48) + v103;
      v113 = *(_WORD *)v112;
      v114 = *(_QWORD *)(v112 + 8);
      v346.m128i_i16[0] = v113;
      v346.m128i_i64[1] = v114;
      if ( v113 )
      {
        if ( (unsigned __int16)(v113 - 17) > 0xD3u )
          goto LABEL_13;
      }
      else
      {
        v243 = v19;
        v262 = v13;
        v321 = v108;
        v145 = sub_30070B0((__int64)&v346);
        v108 = v321;
        v13 = v262;
        v19 = v243;
        if ( !v145 )
          goto LABEL_13;
      }
      v115 = v108 + *(_QWORD *)(v249 + 48);
      v116 = *(_WORD *)v115;
      v117 = *(_QWORD *)(v115 + 8);
      v351.m128i_i16[0] = v116;
      v351.m128i_i64[1] = v117;
      if ( v116 )
      {
        if ( (unsigned __int16)(v116 - 17) > 0xD3u )
          goto LABEL_13;
      }
      else
      {
        v263 = v19;
        v290 = v13;
        v153 = sub_30070B0((__int64)&v351);
        v13 = v290;
        v19 = v263;
        if ( !v153 )
          goto LABEL_13;
      }
      v346.m128i_i64[0] = *(_QWORD *)(a2 + 80);
      if ( v346.m128i_i64[0] )
      {
        v288 = v13;
        sub_325F5D0(v346.m128i_i64);
        v13 = v288;
      }
      v346.m128i_i32[2] = *(_DWORD *)(a2 + 72);
      v240 = *(__int64 **)(*a1 + 64LL);
      if ( (_WORD)v341 )
      {
        v118 = (unsigned __int16)v341 - 1;
        v219 = (unsigned __int16)(v341 - 176) <= 0x34u;
        v112 = word_4456340[v118];
        v228 = word_4456340[v118];
      }
      else
      {
        v213 = v13;
        v165 = sub_3007240((__int64)&v341);
        v13 = v213;
        v343[0] = v165;
        v228 = v165;
        v219 = BYTE4(v165);
      }
      v207 = v13;
      v258 = (unsigned int)sub_32844A0((unsigned __int16 *)&v341, v112);
      v119.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v344);
      v351 = v119;
      if ( v258 % sub_CA1930(&v351) )
      {
        v156.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v344);
        v351 = v156;
        v157 = sub_CA1930(&v351);
        v13 = v207;
        if ( v157 % v258
          || (v212 = v207,
              v158.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v344),
              v351 = v158,
              v159 = sub_CA1930(&v351),
              v13 = v207,
              v160 = v159 / v258,
              v228 % (unsigned int)(v159 / v258))
          || v322 % (unsigned int)v160 )
        {
LABEL_144:
          if ( v346.m128i_i64[0] )
          {
            v320 = v13;
            sub_B91220((__int64)&v346, v346.m128i_i64[0]);
            v13 = v320;
          }
          v19 = *(_DWORD *)(v13 + 24);
          goto LABEL_13;
        }
        v209 = v322 / (unsigned int)v160;
        LODWORD(v340) = v228 / (unsigned int)(v159 / v258);
        BYTE4(v340) = v219;
        v161 = sub_327FD70(v240, v344, v345, v340);
        v319 = v162;
        v235 = HIWORD(v161);
        v264 = v161;
        v163 = sub_3400EE0(*a1, v209, &v346, 0, v161);
        v13 = v212;
        HIWORD(v129) = v235;
        v131 = v163;
        v130 = v264;
        v298 = v164;
      }
      else
      {
        v120.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v344);
        v351 = v120;
        v121 = sub_CA1930(&v351);
        v122 = v228;
        v229 = v345;
        v123 = v258 / v121;
        v351.m128i_i8[4] = v219;
        v124 = v123 * v122;
        v211 = v123;
        v259 = v344;
        v351.m128i_i32[0] = v124;
        if ( v219 )
          v125 = sub_2D43AD0(v344, v124);
        else
          v125 = sub_2D43050(v344, v124);
        v126 = v207;
        v127 = v125;
        v319 = 0;
        if ( !(_WORD)v125 )
        {
          v181 = sub_3009450(v240, v259, v229, v351.m128i_i64[0], v125, v207);
          v126 = v207;
          v205 = HIWORD(v181);
          v127 = v181;
          v319 = v182;
        }
        v220 = v126;
        v230 = v127;
        v128 = sub_3400EE0(*a1, v322 * v211, &v346, 0, v127);
        HIWORD(v129) = v205;
        v130 = v230;
        v131 = v128;
        v13 = v220;
        v298 = v132;
      }
      v221 = v131;
      if ( v131 )
      {
        LOWORD(v129) = v130;
        v231 = v13;
        v260 = v129;
        v133 = sub_328A020(a1[1], 0xA0u, v130, v319, *((unsigned __int8 *)a1 + 33));
        v13 = v231;
        if ( v133 )
        {
          *(_QWORD *)&v167 = sub_33FB890(*a1, v260, v319, v303, v274);
          *((_QWORD *)&v203 + 1) = v298;
          v168 = *((_QWORD *)&v167 + 1);
          *(_QWORD *)&v203 = v221;
          v170 = sub_340F900(*a1, 160, (unsigned int)&v346, v260, v319, v169, v167, v249, v203);
          v326 = sub_33FB890(*a1, v341, v342, v170, v171 | v168 & 0xFFFFFFFF00000000LL);
          sub_9C6650(&v346);
          return v326;
        }
      }
      goto LABEL_144;
    }
  }
LABEL_148:
  v136 = *(_QWORD *)(a2 + 80);
  v137 = *(__int128 **)(v10 + 40);
  v138 = *a1;
  v351.m128i_i64[0] = v136;
  if ( v136 )
    sub_B96E90((__int64)&v351, v136, 1);
  v351.m128i_i32[2] = *(_DWORD *)(a2 + 72);
  result = sub_33FAF80(v138, 168, (unsigned int)&v351, v341, v342, v13, *v137);
  v96 = v351.m128i_i64[0];
  if ( v351.m128i_i64[0] )
    goto LABEL_151;
  return result;
}
