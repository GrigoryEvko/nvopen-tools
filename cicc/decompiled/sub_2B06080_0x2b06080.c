// Function: sub_2B06080
// Address: 0x2b06080
//
__int64 __fastcall sub_2B06080(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // r14
  __int64 v8; // r13
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 **v11; // r14
  __int64 **v12; // rax
  __int64 v13; // r13
  __int64 **v14; // r15
  bool v15; // zf
  __int64 v16; // r12
  __int64 v17; // rsi
  unsigned __int64 v18; // rdx
  unsigned __int16 v19; // cx
  unsigned __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // r9
  __int64 v24; // rdx
  unsigned __int64 v25; // r8
  const char *v26; // rax
  unsigned __int64 v27; // rdx
  __int64 v28; // r9
  __int64 v29; // r9
  __int64 v30; // r8
  unsigned __int64 v31; // rax
  unsigned __int64 v32; // rax
  unsigned __int64 v33; // r12
  __int64 v34; // r13
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  __int64 v37; // rax
  _QWORD **v38; // rdx
  __int64 **v39; // rax
  unsigned int v40; // eax
  char v41; // cl
  unsigned __int64 v42; // r15
  unsigned int v43; // r13d
  int v44; // r12d
  __int64 v45; // rdi
  unsigned __int64 v46; // rdi
  _QWORD *v47; // r12
  _QWORD *v48; // rbx
  unsigned __int64 v49; // rsi
  _QWORD *v50; // rax
  _QWORD *v51; // rdi
  __int64 v52; // rcx
  __int64 v53; // rdx
  __int64 v54; // rax
  _QWORD *v55; // rdi
  __int64 v56; // rcx
  __int64 v57; // rdx
  __int64 *v58; // r12
  unsigned __int64 v59; // rax
  __int64 v60; // rbx
  int v61; // eax
  __int64 v62; // rdx
  __int64 v63; // rcx
  __int64 v64; // r8
  __int64 v65; // r9
  __int64 v66; // rdx
  __int64 v67; // rcx
  __int64 v68; // r9
  __int64 v69; // r8
  __int64 v70; // rdx
  __int64 v71; // r8
  __int64 v72; // r9
  __int64 v73; // rcx
  __int64 v74; // rax
  __int64 v75; // rdx
  __int64 *v76; // r8
  unsigned __int64 v77; // rax
  __int64 *v78; // r12
  __int64 *v79; // r12
  __int64 *v80; // rbx
  __int64 v81; // r15
  __int64 v82; // rdi
  _QWORD **v83; // rdx
  unsigned int v84; // r10d
  unsigned int v85; // eax
  unsigned __int8 *v86; // r11
  int v87; // eax
  unsigned __int8 *v88; // rax
  __int64 v89; // rdx
  __int64 v90; // rcx
  __int64 v91; // r8
  __int64 v92; // r9
  __int64 *v93; // rdi
  __int64 v94; // r8
  __int64 v95; // r9
  __int64 v96; // r15
  __int64 v97; // rax
  __int64 v98; // r9
  __int64 v99; // r14
  __int64 v100; // r15
  __int64 v101; // rdi
  _QWORD **v102; // rdx
  unsigned int v103; // eax
  unsigned __int8 *v104; // r11
  int v105; // eax
  unsigned __int8 *v106; // rax
  __int64 v107; // rsi
  unsigned __int64 v108; // rdx
  __int64 v109; // rax
  unsigned __int16 v110; // cx
  __int64 v111; // rax
  __int64 v112; // rax
  __int64 v113; // r9
  __int64 v114; // rdx
  unsigned __int64 v115; // r8
  const char *v116; // rax
  unsigned __int64 v117; // rdx
  const char *v118; // rax
  unsigned __int64 v119; // rdx
  const char *v120; // rax
  unsigned __int64 v121; // rdx
  unsigned __int64 v122; // r12
  unsigned __int64 *v123; // r13
  unsigned __int64 *v124; // r12
  unsigned __int64 v125; // rdi
  unsigned __int64 v126; // r12
  unsigned __int64 *v127; // r13
  unsigned __int64 *v128; // r12
  unsigned __int64 v129; // rdi
  __int64 v130; // rdx
  __int64 v131; // rcx
  __int64 v132; // r8
  __int64 v133; // r9
  __int64 v134; // rdx
  __int64 v135; // rcx
  __int64 v136; // r8
  __int64 v137; // r9
  unsigned __int64 v138; // r15
  __int64 v139; // r15
  __int64 *v140; // r13
  unsigned __int64 v141; // rdi
  __int64 v142; // rax
  __int64 v143; // r15
  unsigned __int64 v144; // r13
  unsigned __int64 v145; // rdi
  __int64 *v146; // rax
  __int64 v147; // rax
  __int64 v148; // rax
  unsigned __int64 *v149; // r15
  unsigned __int64 *v150; // r13
  unsigned __int64 v151; // rdi
  unsigned __int64 *v152; // r15
  unsigned __int64 *v153; // r13
  unsigned __int64 v154; // rdi
  __int64 *v155; // r13
  __int64 *v156; // r12
  unsigned __int64 v157; // rdi
  char v158; // bl
  unsigned __int64 *v159; // rbx
  unsigned __int64 v160; // r12
  unsigned __int64 v161; // rdi
  __int64 v162; // rbx
  unsigned __int64 v163; // r12
  unsigned __int64 v164; // rdi
  unsigned __int64 v165; // rdi
  __int64 v166; // r13
  unsigned __int64 v167; // rdi
  unsigned __int64 *v168; // r13
  unsigned __int64 v169; // r12
  unsigned __int64 v170; // rdi
  __int64 v171; // rsi
  __int64 v172; // r12
  __int64 v173; // rcx
  __int64 v174; // r8
  __int64 v175; // r9
  const char *v176; // rax
  __int64 *v177; // rdx
  unsigned __int64 *v178; // rbx
  unsigned __int64 v179; // r12
  unsigned __int64 v180; // rdi
  __int64 v181; // rbx
  unsigned __int64 v182; // r12
  unsigned __int64 v183; // rdi
  __int64 v184; // [rsp+48h] [rbp-CC8h]
  __int64 v185; // [rsp+48h] [rbp-CC8h]
  __int64 v186; // [rsp+50h] [rbp-CC0h]
  __int64 v187; // [rsp+50h] [rbp-CC0h]
  __int64 **v188; // [rsp+50h] [rbp-CC0h]
  unsigned __int8 *v189; // [rsp+50h] [rbp-CC0h]
  unsigned __int8 *v190; // [rsp+50h] [rbp-CC0h]
  __int64 v191; // [rsp+50h] [rbp-CC0h]
  __int64 v192; // [rsp+50h] [rbp-CC0h]
  unsigned int v193; // [rsp+58h] [rbp-CB8h]
  unsigned int v194; // [rsp+60h] [rbp-CB0h]
  unsigned int v195; // [rsp+60h] [rbp-CB0h]
  __int64 *v196; // [rsp+60h] [rbp-CB0h]
  char v198; // [rsp+87h] [rbp-C89h]
  __int64 *v199; // [rsp+90h] [rbp-C80h]
  __int64 **v200; // [rsp+98h] [rbp-C78h]
  __int64 i; // [rsp+A0h] [rbp-C70h]
  __int64 **v202; // [rsp+A8h] [rbp-C68h]
  __int64 v203; // [rsp+B0h] [rbp-C60h] BYREF
  __int64 v204; // [rsp+B8h] [rbp-C58h]
  __int64 v205; // [rsp+C0h] [rbp-C50h]
  unsigned int v206; // [rsp+C8h] [rbp-C48h]
  __int64 v207; // [rsp+D0h] [rbp-C40h]
  __int64 v208; // [rsp+D8h] [rbp-C38h]
  __int64 v209; // [rsp+E0h] [rbp-C30h] BYREF
  __int64 v210; // [rsp+E8h] [rbp-C28h]
  __int64 v211; // [rsp+F0h] [rbp-C20h]
  unsigned int v212; // [rsp+F8h] [rbp-C18h]
  unsigned __int64 *v213; // [rsp+100h] [rbp-C10h]
  __int64 v214; // [rsp+108h] [rbp-C08h]
  __int64 v215; // [rsp+110h] [rbp-C00h] BYREF
  void *src; // [rsp+118h] [rbp-BF8h]
  __int64 v217; // [rsp+120h] [rbp-BF0h]
  unsigned int v218; // [rsp+128h] [rbp-BE8h]
  __int64 *v219; // [rsp+130h] [rbp-BE0h] BYREF
  __int64 v220; // [rsp+138h] [rbp-BD8h]
  __int64 v221[5]; // [rsp+140h] [rbp-BD0h] BYREF
  __int64 *v222; // [rsp+168h] [rbp-BA8h]
  const char *v223; // [rsp+170h] [rbp-BA0h]
  _BYTE *v224; // [rsp+178h] [rbp-B98h]
  __int64 v225; // [rsp+180h] [rbp-B90h]
  _BYTE v226[32]; // [rsp+188h] [rbp-B88h] BYREF
  __int64 v227; // [rsp+1A8h] [rbp-B68h]
  __int64 v228; // [rsp+1B0h] [rbp-B60h]
  __int16 v229; // [rsp+1B8h] [rbp-B58h]
  __int64 v230; // [rsp+1C0h] [rbp-B50h]
  void **v231; // [rsp+1C8h] [rbp-B48h]
  _QWORD *v232; // [rsp+1D0h] [rbp-B40h]
  __int64 v233; // [rsp+1D8h] [rbp-B38h]
  int v234; // [rsp+1E0h] [rbp-B30h]
  __int16 v235; // [rsp+1E4h] [rbp-B2Ch]
  char v236; // [rsp+1E6h] [rbp-B2Ah]
  __int64 v237; // [rsp+1E8h] [rbp-B28h]
  __int64 v238; // [rsp+1F0h] [rbp-B20h]
  void *v239; // [rsp+1F8h] [rbp-B18h] BYREF
  _QWORD v240[2]; // [rsp+200h] [rbp-B10h] BYREF
  __int64 v241; // [rsp+210h] [rbp-B00h]
  __int64 v242; // [rsp+218h] [rbp-AF8h]
  unsigned int v243; // [rsp+220h] [rbp-AF0h]
  int v244; // [rsp+228h] [rbp-AE8h]
  char v245; // [rsp+22Ch] [rbp-AE4h]
  __int64 v246; // [rsp+230h] [rbp-AE0h]
  __int64 v247; // [rsp+238h] [rbp-AD8h]
  __int64 v248; // [rsp+240h] [rbp-AD0h]
  unsigned int v249; // [rsp+248h] [rbp-AC8h]
  int v250; // [rsp+258h] [rbp-AB8h] BYREF
  unsigned __int64 v251; // [rsp+260h] [rbp-AB0h]
  int *v252; // [rsp+268h] [rbp-AA8h]
  int *v253; // [rsp+270h] [rbp-AA0h]
  __int64 v254; // [rsp+278h] [rbp-A98h]
  __int64 v255; // [rsp+280h] [rbp-A90h]
  __int64 v256; // [rsp+288h] [rbp-A88h]
  __int64 v257; // [rsp+290h] [rbp-A80h]
  unsigned int v258; // [rsp+298h] [rbp-A78h]
  __int64 v259; // [rsp+2A0h] [rbp-A70h]
  __int64 v260; // [rsp+2A8h] [rbp-A68h]
  __int64 v261; // [rsp+2B0h] [rbp-A60h]
  __int64 v262; // [rsp+2B8h] [rbp-A58h]
  __int64 v263[54]; // [rsp+2C0h] [rbp-A50h] BYREF
  __int64 v264; // [rsp+470h] [rbp-8A0h] BYREF
  _QWORD **v265; // [rsp+478h] [rbp-898h]
  __int64 v266; // [rsp+480h] [rbp-890h]
  __int64 v267; // [rsp+488h] [rbp-888h]
  __int64 *v268; // [rsp+490h] [rbp-880h] BYREF
  __int64 v269; // [rsp+498h] [rbp-878h]
  __int64 v270; // [rsp+4A0h] [rbp-870h] BYREF
  void *v271; // [rsp+4A8h] [rbp-868h]
  __int64 v272; // [rsp+4B0h] [rbp-860h]
  __int64 v273; // [rsp+4B8h] [rbp-858h]
  unsigned __int64 *v274; // [rsp+4C0h] [rbp-850h] BYREF
  __int64 v275; // [rsp+4C8h] [rbp-848h]
  __int64 *v276; // [rsp+4D0h] [rbp-840h] BYREF
  __int64 v277; // [rsp+4D8h] [rbp-838h]
  __int64 v278; // [rsp+4E0h] [rbp-830h] BYREF
  int v279; // [rsp+4E8h] [rbp-828h]
  __int64 v280; // [rsp+4F0h] [rbp-820h]
  int v281; // [rsp+4F8h] [rbp-818h]
  __int64 *v282; // [rsp+500h] [rbp-810h]
  const char *v283; // [rsp+620h] [rbp-6F0h] BYREF
  _BYTE *v284; // [rsp+628h] [rbp-6E8h]
  __int64 v285; // [rsp+630h] [rbp-6E0h]
  _BYTE v286[4]; // [rsp+638h] [rbp-6D8h] BYREF
  char v287; // [rsp+63Ch] [rbp-6D4h]
  __int16 v288; // [rsp+640h] [rbp-6D0h] BYREF
  _BYTE *v289; // [rsp+648h] [rbp-6C8h] BYREF
  __int64 v290; // [rsp+650h] [rbp-6C0h]
  _BYTE v291[32]; // [rsp+658h] [rbp-6B8h] BYREF
  __int64 v292; // [rsp+678h] [rbp-698h]
  __int64 **v293; // [rsp+680h] [rbp-690h] BYREF
  __int64 v294; // [rsp+688h] [rbp-688h]
  _BYTE v295[320]; // [rsp+690h] [rbp-680h] BYREF
  _BYTE *v296; // [rsp+7D0h] [rbp-540h] BYREF
  unsigned __int64 v297; // [rsp+7D8h] [rbp-538h]
  _BYTE v298[12]; // [rsp+7E0h] [rbp-530h] BYREF
  char v299; // [rsp+7ECh] [rbp-524h]
  char v300[16]; // [rsp+7F0h] [rbp-520h] BYREF
  __int64 v301; // [rsp+800h] [rbp-510h]
  __int64 v302; // [rsp+808h] [rbp-508h]
  __int16 v303; // [rsp+810h] [rbp-500h]
  _QWORD *v304; // [rsp+818h] [rbp-4F8h]
  void **v305; // [rsp+820h] [rbp-4F0h]
  _QWORD *v306; // [rsp+828h] [rbp-4E8h]
  __int64 *v307; // [rsp+830h] [rbp-4E0h] BYREF
  __int64 v308; // [rsp+838h] [rbp-4D8h]
  __int64 v309; // [rsp+840h] [rbp-4D0h] BYREF
  __int64 v310; // [rsp+848h] [rbp-4C8h]
  void *v311; // [rsp+850h] [rbp-4C0h] BYREF
  _QWORD v312[37]; // [rsp+858h] [rbp-4B8h] BYREF
  unsigned __int64 v313; // [rsp+980h] [rbp-390h] BYREF
  unsigned __int64 v314; // [rsp+988h] [rbp-388h]
  __int64 v315; // [rsp+990h] [rbp-380h] BYREF
  int v316; // [rsp+998h] [rbp-378h] BYREF
  char v317; // [rsp+99Ch] [rbp-374h]
  void *v318; // [rsp+9A0h] [rbp-370h] BYREF
  __int64 **v319; // [rsp+9A8h] [rbp-368h] BYREF
  __int64 v320; // [rsp+9B0h] [rbp-360h] BYREF
  unsigned __int64 v321[2]; // [rsp+9B8h] [rbp-358h] BYREF
  int v322; // [rsp+9C8h] [rbp-348h]
  char v323; // [rsp+9CCh] [rbp-344h]
  char v324; // [rsp+9D0h] [rbp-340h] BYREF
  __int64 v325; // [rsp+9D8h] [rbp-338h]
  __int64 *v326; // [rsp+9E0h] [rbp-330h] BYREF
  unsigned int v327; // [rsp+9E8h] [rbp-328h]
  int v328; // [rsp+9ECh] [rbp-324h]
  unsigned __int64 *v329; // [rsp+9F0h] [rbp-320h] BYREF
  unsigned __int64 *v330; // [rsp+9F8h] [rbp-318h]
  __int64 v331; // [rsp+A00h] [rbp-310h]
  unsigned __int64 *v332; // [rsp+A08h] [rbp-308h] BYREF
  unsigned __int64 *v333; // [rsp+A10h] [rbp-300h]
  __int64 v334; // [rsp+A18h] [rbp-2F8h]
  __int64 v335; // [rsp+A20h] [rbp-2F0h]
  __int64 v336; // [rsp+A28h] [rbp-2E8h]
  char v337; // [rsp+A38h] [rbp-2D8h]
  char v338[8]; // [rsp+B30h] [rbp-1E0h] BYREF
  unsigned __int64 v339; // [rsp+B38h] [rbp-1D8h]
  char v340; // [rsp+B4Ch] [rbp-1C4h]
  char *v341; // [rsp+B90h] [rbp-180h] BYREF
  unsigned int v342; // [rsp+B98h] [rbp-178h]
  char v343; // [rsp+BA0h] [rbp-170h] BYREF

  v198 = sub_B2D610(a3, 30);
  if ( v198 )
  {
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  v7 = sub_BC1CD0(a4, &unk_4F86540, a3);
  v8 = sub_BC1CD0(a4, &unk_4F81450, a3) + 8;
  v202 = (__int64 **)sub_BC1CD0(a4, &unk_4F881D0, a3);
  v9 = sub_BC1CD0(a4, &unk_4F89C30, a3);
  v221[1] = v7 + 8;
  v221[3] = v8;
  v221[2] = sub_BC1CD0(a4, &unk_4F86630, a3) + 8;
  v221[4] = (__int64)(v202 + 1);
  v10 = *(_QWORD *)(a3 + 40);
  v222 = (__int64 *)(v9 + 8);
  v221[0] = a3;
  v223 = (const char *)(v10 + 312);
  v230 = sub_B2BE50((__int64)v202[1]);
  v231 = &v239;
  v232 = v240;
  v224 = v226;
  v239 = &unk_49DA100;
  v225 = 0x200000000LL;
  v235 = 512;
  v240[0] = &unk_49DA0B0;
  v233 = 0;
  v234 = 0;
  v236 = 7;
  v237 = 0;
  v238 = 0;
  v227 = 0;
  v228 = 0;
  v229 = 0;
  v240[1] = 0;
  v241 = 0;
  v242 = 0;
  v243 = 0;
  v244 = 3;
  v245 = 0;
  v246 = 0;
  v247 = 0;
  v248 = 0;
  v249 = 0;
  v250 = 0;
  v251 = 0;
  v252 = &v250;
  v253 = &v250;
  v254 = 0;
  v255 = 0;
  v256 = 0;
  v257 = 0;
  v258 = 0;
  v259 = 0;
  v260 = 0;
  v261 = 0;
  v262 = 0;
  if ( (_BYTE)qword_500EF48 )
  {
    v11 = *(__int64 ***)(v221[0] + 80);
    v202 = (__int64 **)(v221[0] + 72);
    if ( v11 != (__int64 **)(v221[0] + 72) )
    {
      while ( 1 )
      {
        v12 = v11;
        v11 = (__int64 **)v11[1];
        v13 = (__int64)v12[4];
        v14 = v12 + 3;
        if ( (__int64 **)v13 != v12 + 3 )
          break;
LABEL_46:
        if ( v202 == v11 )
          goto LABEL_47;
      }
      while ( 1 )
      {
        v15 = v13 == 0;
        v16 = v13 - 24;
        v13 = *(_QWORD *)(v13 + 8);
        if ( v15 )
          v16 = 0;
        if ( !(unsigned __int8)sub_B46420(v16) && !(unsigned __int8)sub_B46490(v16) )
          goto LABEL_45;
        if ( *(_BYTE *)v16 == 61 )
        {
          if ( sub_B46500((unsigned __int8 *)v16) )
            goto LABEL_45;
          if ( (*(_BYTE *)(v16 + 2) & 1) != 0 )
            goto LABEL_45;
          if ( !(unsigned __int8)sub_DFE180((__int64)v222) )
            goto LABEL_45;
          v17 = *(_QWORD *)(v16 + 8);
          if ( (unsigned int)*(unsigned __int8 *)(v17 + 8) - 15 > 1 )
            goto LABEL_45;
          if ( !(unsigned __int8)sub_2AF7640(*(_QWORD *)(v16 + 8)) )
            goto LABEL_45;
          v313 = sub_9208B0((__int64)v223, v17);
          v314 = v18;
          i = (__int64)&v313;
          if ( (unsigned int)sub_CA1930(&v313) <= 7 )
            goto LABEL_45;
          v292 = 0;
          v283 = v223;
          v284 = v286;
          v289 = v291;
          v285 = 0x400000000LL;
          v290 = 0x400000000LL;
          v293 = 0;
          LOBYTE(v294) = 0;
          HIDWORD(v294) = 0;
          v304 = (_QWORD *)sub_BD5C60(v16);
          v305 = &v311;
          v306 = v312;
          v296 = v298;
          v311 = &unk_49DA100;
          WORD2(v308) = 512;
          v303 = 0;
          v297 = 0x200000000LL;
          v199 = (__int64 *)&v296;
          v312[0] = &unk_49DA0B0;
          v307 = 0;
          LODWORD(v308) = 0;
          BYTE6(v308) = 7;
          v309 = 0;
          v310 = 0;
          v301 = 0;
          v302 = 0;
          sub_D5F1F0((__int64)&v296, v16);
          v200 = *(__int64 ***)(v16 + 8);
          v263[0] = sub_ACA8A0(v200);
          v292 = *(_QWORD *)(v16 - 32);
          v293 = v200;
          v19 = *(_WORD *)(v16 + 2);
          HIDWORD(v294) = 0;
          _BitScanReverse64(&v20, 1LL << (v19 >> 1));
          LOBYTE(v294) = 63 - (v20 ^ 0x3F);
          v21 = sub_BCB2D0(v304);
          v22 = sub_ACD640(v21, 0, 0);
          v24 = (unsigned int)v290;
          v25 = (unsigned int)v290 + 1LL;
          if ( v25 > HIDWORD(v290) )
          {
            v191 = v22;
            sub_C8D5F0((__int64)&v289, v291, (unsigned int)v290 + 1LL, 8u, v25, v23);
            v24 = (unsigned int)v290;
            v22 = v191;
          }
          *(_QWORD *)&v289[8 * v24] = v22;
          LODWORD(v290) = v290 + 1;
          v186 = *(_QWORD *)(v16 + 24);
          v26 = sub_BD5D20(v16);
          v314 = v27;
          LOWORD(v318) = 261;
          v313 = (unsigned __int64)v26;
          sub_2AF9E50((__int64)&v283, v199, (__int64)v200, v263, i, v28);
          sub_BD84D0(v16, v263[0]);
          if ( sub_CE8520(v16) && *((_BYTE *)v200 + 8) == 16 )
          {
            v30 = v16 + 24;
            v199 = &v315;
            v313 = (unsigned __int64)&v315;
            v314 = 0x1000000000LL;
            v31 = (v186 & 0xFFFFFFFFFFFFFFF8LL) - 24;
            if ( (v186 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
              v31 = 0;
            v32 = v31 + 24;
            if ( v30 != v32 )
            {
              v187 = v16;
              v33 = v32;
              v184 = v13;
              v34 = v30;
              while ( 1 )
              {
                if ( *(_BYTE *)(v33 - 24) == 61 )
                {
                  v35 = (unsigned int)v314;
                  v36 = (unsigned int)v314 + 1LL;
                  if ( v36 > HIDWORD(v314) )
                  {
                    sub_C8D5F0(i, v199, v36, 8u, v30, v29);
                    v35 = (unsigned int)v314;
                  }
                  *(_QWORD *)(v313 + 8 * v35) = v33 - 24;
                  LODWORD(v314) = v314 + 1;
                }
                v33 = *(_QWORD *)(v33 + 8);
                if ( v34 == v33 )
                  break;
                if ( !v33 )
                  BUG();
              }
              v16 = v187;
              v13 = v184;
            }
            v37 = sub_9208B0((__int64)v283, *v200[2]);
            v265 = v38;
            v264 = v37;
            v39 = (__int64 **)((unsigned __int64)sub_CA1930(&v264) >> 3);
            LODWORD(i) = (_DWORD)v39;
            v200 = v39;
            v40 = sub_CE8560(v16);
            v41 = (char)v200;
            if ( (_DWORD)v314 )
            {
              v200 = v11;
              v188 = v14;
              v42 = 0;
              v185 = v13;
              v43 = v40;
              v44 = ~(-1 << v41);
              do
              {
                v45 = *(_QWORD *)(v313 + 8 * v42++);
                sub_CE85E0(v45, v43 & v44);
                v43 >>= i;
              }
              while ( (unsigned int)v314 > v42 );
              v11 = v200;
              v14 = v188;
              v13 = v185;
            }
            if ( (__int64 *)v313 != v199 )
              _libc_free(v313);
          }
          LODWORD(v285) = 0;
          LODWORD(v290) = 0;
          nullsub_61();
          v311 = &unk_49DA100;
          nullsub_63();
          if ( v296 != v298 )
            _libc_free((unsigned __int64)v296);
          if ( v289 != v291 )
            _libc_free((unsigned __int64)v289);
          v46 = (unsigned __int64)v284;
          if ( v284 == v286 )
            goto LABEL_45;
        }
        else
        {
          if ( *(_BYTE *)v16 != 62 )
            goto LABEL_45;
          if ( sub_B46500((unsigned __int8 *)v16) )
            goto LABEL_45;
          if ( (*(_BYTE *)(v16 + 2) & 1) != 0 )
            goto LABEL_45;
          if ( !(unsigned __int8)sub_DFE1B0((__int64)v222) )
            goto LABEL_45;
          v107 = *(_QWORD *)(*(_QWORD *)(v16 - 64) + 8LL);
          if ( (unsigned int)*(unsigned __int8 *)(v107 + 8) - 15 > 1 )
            goto LABEL_45;
          if ( !(unsigned __int8)sub_2AF7640(*(_QWORD *)(*(_QWORD *)(v16 - 64) + 8LL)) )
            goto LABEL_45;
          v313 = sub_9208B0((__int64)v223, v107);
          v314 = v108;
          i = (__int64)&v313;
          if ( (unsigned int)sub_CA1930(&v313) <= 7 )
            goto LABEL_45;
          v325 = 0;
          v200 = (__int64 **)v321;
          v313 = (unsigned __int64)v223;
          v314 = (unsigned __int64)&v316;
          v319 = (__int64 **)v321;
          v315 = 0x400000000LL;
          v320 = 0x400000000LL;
          v326 = 0;
          LOBYTE(v327) = 0;
          v328 = 0;
          v329 = 0;
          v330 = 0;
          v331 = 0;
          v332 = 0;
          v333 = 0;
          v334 = 0;
          v335 = 4;
          v336 = 2;
          v337 = 0;
          v304 = (_QWORD *)sub_BD5C60(v16);
          v305 = &v311;
          v306 = v312;
          WORD2(v308) = 512;
          v303 = 0;
          v296 = v298;
          v311 = &unk_49DA100;
          v199 = (__int64 *)&v296;
          v297 = 0x200000000LL;
          v312[0] = &unk_49DA0B0;
          v307 = 0;
          LODWORD(v308) = 0;
          BYTE6(v308) = 7;
          v309 = 0;
          v310 = 0;
          v301 = 0;
          v302 = 0;
          sub_D5F1F0((__int64)&v296, v16);
          v264 = *(_QWORD *)(v16 - 64);
          v196 = *(__int64 **)(v264 + 8);
          v109 = *(_QWORD *)(v16 - 32);
          v326 = v196;
          v325 = v109;
          v110 = *(_WORD *)(v16 + 2);
          v328 = 0;
          _BitScanReverse64((unsigned __int64 *)&v109, 1LL << (v110 >> 1));
          LOBYTE(v327) = 63 - (v109 ^ 0x3F);
          v111 = sub_BCB2D0(v304);
          v112 = sub_ACD640(v111, 0, 0);
          v114 = (unsigned int)v320;
          v115 = (unsigned int)v320 + 1LL;
          if ( v115 > HIDWORD(v320) )
          {
            v192 = v112;
            sub_C8D5F0((__int64)&v319, v200, (unsigned int)v320 + 1LL, 8u, v115, v113);
            v114 = (unsigned int)v320;
            v112 = v192;
          }
          v319[v114] = (__int64 *)v112;
          LODWORD(v320) = v320 + 1;
          v116 = sub_BD5D20(v16);
          v284 = (_BYTE *)v117;
          v288 = 261;
          v283 = v116;
          sub_2AFB7E0(i, (__int64)v199, (__int64)v196, &v264, (__int64)&v283);
          v118 = sub_BD5D20(v16);
          v288 = 261;
          v284 = (_BYTE *)v119;
          v283 = v118;
          sub_2AFB210(i, (__int64)v199, &v264, (__int64)&v283, (__int64 *)&v329, 0);
          v120 = sub_BD5D20(v16);
          v288 = 261;
          v284 = (_BYTE *)v121;
          v283 = v120;
          sub_2AFB210(i, (__int64)v199, &v264, (__int64)&v283, (__int64 *)&v332, 1);
          sub_B43D60((_QWORD *)v16);
          LODWORD(v315) = 0;
          LODWORD(v320) = 0;
          nullsub_61();
          v311 = &unk_49DA100;
          nullsub_63();
          if ( v296 != v298 )
            _libc_free((unsigned __int64)v296);
          v122 = (unsigned __int64)v332;
          if ( v333 != v332 )
          {
            i = v13;
            v123 = v332;
            v124 = v333;
            do
            {
              v125 = v123[4];
              if ( (unsigned __int64 *)v125 != v123 + 6 )
                _libc_free(v125);
              if ( (unsigned __int64 *)*v123 != v123 + 2 )
                _libc_free(*v123);
              v123 += 11;
            }
            while ( v124 != v123 );
            v13 = i;
            v122 = (unsigned __int64)v332;
          }
          if ( v122 )
            j_j___libc_free_0(v122);
          v126 = (unsigned __int64)v329;
          if ( v330 != v329 )
          {
            i = v13;
            v127 = v329;
            v128 = v330;
            do
            {
              v129 = v127[4];
              if ( (unsigned __int64 *)v129 != v127 + 6 )
                _libc_free(v129);
              if ( (unsigned __int64 *)*v127 != v127 + 2 )
                _libc_free(*v127);
              v127 += 11;
            }
            while ( v128 != v127 );
            v13 = i;
            v126 = (unsigned __int64)v329;
          }
          if ( v126 )
            j_j___libc_free_0(v126);
          if ( v319 != v200 )
            _libc_free((unsigned __int64)v319);
          v46 = v314;
          if ( (int *)v314 == &v316 )
            goto LABEL_45;
        }
        _libc_free(v46);
LABEL_45:
        if ( v14 == (__int64 **)v13 )
          goto LABEL_46;
      }
    }
  }
LABEL_47:
  v47 = sub_C52410();
  v48 = v47 + 1;
  v49 = sub_C959E0();
  v50 = (_QWORD *)v47[2];
  if ( v50 )
  {
    v51 = v47 + 1;
    do
    {
      while ( 1 )
      {
        v52 = v50[2];
        v53 = v50[3];
        if ( v49 <= v50[4] )
          break;
        v50 = (_QWORD *)v50[3];
        if ( !v53 )
          goto LABEL_52;
      }
      v51 = v50;
      v50 = (_QWORD *)v50[2];
    }
    while ( v52 );
LABEL_52:
    if ( v51 != v48 && v49 >= v51[4] )
      v48 = v51;
  }
  if ( v48 == (_QWORD *)((char *)sub_C52410() + 8) )
    goto LABEL_63;
  v54 = v48[7];
  if ( !v54 )
    goto LABEL_63;
  v55 = v48 + 6;
  do
  {
    while ( 1 )
    {
      v56 = *(_QWORD *)(v54 + 16);
      v57 = *(_QWORD *)(v54 + 24);
      if ( *(_DWORD *)(v54 + 32) >= dword_500F168 )
        break;
      v54 = *(_QWORD *)(v54 + 24);
      if ( !v57 )
        goto LABEL_61;
    }
    v55 = (_QWORD *)v54;
    v54 = *(_QWORD *)(v54 + 16);
  }
  while ( v56 );
LABEL_61:
  if ( v55 == v48 + 6
    || dword_500F168 < *((_DWORD *)v55 + 8)
    || *((int *)v55 + 9) <= 0
    || (v176 = sub_BD5D20(v221[0]), v76 = v177, (__int64 *)qword_500F1F0 == v177)
    && (!qword_500F1F0 || !memcmp(qword_500F1E8, v176, qword_500F1F0)) )
  {
LABEL_63:
    memset(v263, 0, sizeof(v263));
    LODWORD(v263[2]) = 8;
    v263[1] = (__int64)&v263[4];
    BYTE4(v263[3]) = 1;
    v263[12] = (__int64)&v263[14];
    HIDWORD(v263[13]) = 8;
    v58 = *(__int64 **)(v221[0] + 80);
    LODWORD(v266) = 8;
    if ( v58 )
      v58 -= 3;
    LODWORD(v267) = 0;
    v265 = &v268;
    BYTE4(v267) = 1;
    v276 = &v278;
    v277 = 0x800000000LL;
    HIDWORD(v266) = 1;
    v268 = v58;
    v264 = 1;
    v59 = v58[6] & 0xFFFFFFFFFFFFFFF8LL;
    if ( (__int64 *)v59 == v58 + 6 )
      goto LABEL_307;
    if ( !v59 )
      BUG();
    v60 = v59 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v59 - 24) - 30 > 0xA )
    {
LABEL_307:
      v61 = 0;
      v62 = 0;
      v60 = 0;
    }
    else
    {
      v61 = sub_B46E30(v60);
      v62 = v60;
    }
    v278 = v62;
    v279 = v61;
    v280 = v60;
    v282 = v58;
    v281 = 0;
    LODWORD(v277) = 1;
    sub_CE27D0((__int64)&v264);
    sub_CE35F0((__int64)&v296, (__int64)v263);
    sub_CE35F0((__int64)&v283, (__int64)&v264);
    sub_CE35F0((__int64)&v313, (__int64)&v283);
    sub_CE35F0((__int64)v338, (__int64)&v296);
    if ( v293 != (__int64 **)v295 )
      _libc_free((unsigned __int64)v293);
    if ( !v287 )
      _libc_free((unsigned __int64)v284);
    if ( v307 != &v309 )
      _libc_free((unsigned __int64)v307);
    if ( !v299 )
      _libc_free(v297);
    if ( v276 != &v278 )
      _libc_free((unsigned __int64)v276);
    if ( !BYTE4(v267) )
      _libc_free((unsigned __int64)v265);
    if ( (__int64 *)v263[12] != &v263[14] )
      _libc_free(v263[12]);
    if ( !BYTE4(v263[3]) )
      _libc_free(v263[1]);
    sub_C8CD80((__int64)&v283, (__int64)&v288, (__int64)&v313, v63, v64, v65);
    v69 = v327;
    v293 = (__int64 **)v295;
    v294 = 0x800000000LL;
    if ( v327 )
      sub_2AFF1F0((__int64)&v293, (__int64 *)&v326, v66, v67, v327, v68);
    sub_C8CD80((__int64)&v296, (__int64)v300, (__int64)v338, v67, v69, v68);
    v73 = v342;
    v307 = &v309;
    v308 = 0x800000000LL;
    if ( v342 )
    {
      sub_2AFF1F0((__int64)&v307, (__int64 *)&v341, v70, v342, v71, v72);
      v73 = (unsigned int)v308;
    }
    v74 = (unsigned int)v294;
    LODWORD(v199) = 1;
    v202 = (__int64 **)&v209;
    while ( 1 )
    {
      v75 = 5 * v74;
      if ( v74 == v73 )
      {
        v76 = v307;
        if ( &v293[v75] == v293 )
          goto LABEL_255;
        v73 = (__int64)v307;
        v77 = (unsigned __int64)v293;
        while ( *(_QWORD *)(v77 + 32) == *(_QWORD *)(v73 + 32)
             && *(_DWORD *)(v77 + 24) == *(_DWORD *)(v73 + 24)
             && *(_DWORD *)(v77 + 8) == *(_DWORD *)(v73 + 8) )
        {
          v77 += 40LL;
          v73 += 40;
          if ( &v293[v75] == (__int64 **)v77 )
            goto LABEL_255;
        }
      }
      v78 = v293[v75 - 1];
      memset(v263, 0, 28);
      v207 = (__int64)v202;
      v79 = v78 + 6;
      v200 = (__int64 **)&v215;
      v213 = (unsigned __int64 *)&v215;
      v263[4] = (__int64)&v263[6];
      v263[5] = 0;
      v80 = (__int64 *)v79[1];
      v203 = 0;
      v204 = 0;
      v205 = 0;
      v206 = 0;
      v208 = 0;
      v209 = 0;
      v210 = 0;
      v211 = 0;
      v212 = 0;
      v214 = 0;
      v215 = 0;
      src = 0;
      v217 = 0;
      v218 = 0;
      v219 = v221;
      v220 = 0;
      for ( i = (__int64)&v263[6]; v79 != v80; v80 = (__int64 *)v80[1] )
      {
        v98 = (__int64)(v80 - 3);
        if ( !v80 )
          v98 = 0;
        v99 = v98;
        if ( !(unsigned __int8)sub_B46420(v98) && !(unsigned __int8)sub_B46490(v99) )
          continue;
        if ( *(_BYTE *)v99 == 61 )
        {
          if ( sub_B46500((unsigned __int8 *)v99)
            || (*(_BYTE *)(v99 + 2) & 1) != 0
            || !(unsigned __int8)sub_DFE180((__int64)v222) )
          {
            continue;
          }
          v81 = *(_QWORD *)(v99 + 8);
          v82 = v81;
          if ( (unsigned int)*(unsigned __int8 *)(v81 + 8) - 17 <= 1 )
            v82 = **(_QWORD **)(v81 + 16);
          if ( !(unsigned __int8)sub_BCBCB0(v82) )
            continue;
          v264 = sub_9208B0((__int64)v223, v81);
          v265 = v83;
          v84 = sub_CA1930(&v264);
          if ( (v84 & 7) != 0
            || (unsigned __int8)(*(_BYTE *)(v81 + 8) - 17) <= 1u && *(_BYTE *)(**(_QWORD **)(v81 + 16) + 8LL) == 14 )
          {
            continue;
          }
          v189 = *(unsigned __int8 **)(v99 - 32);
          v193 = v84;
          v194 = v84;
          v85 = sub_DFE150((__int64)v222);
          v86 = v189;
          if ( (unsigned int)*(unsigned __int8 *)(v81 + 8) - 17 > 1 )
          {
            if ( v194 <= v85 >> 1 )
              goto LABEL_107;
          }
          else if ( v194 <= v85 >> 1 )
          {
            v87 = sub_DFE2B0(v222, v85 / v193);
            v86 = v189;
            if ( v87 )
            {
LABEL_107:
              v88 = sub_98ACB0(v86, 8u);
              if ( *v88 == 86 )
                v88 = (unsigned __int8 *)*((_QWORD *)v88 - 12);
              v264 = (__int64)v88;
              v93 = (__int64 *)v200;
              goto LABEL_110;
            }
          }
        }
        else
        {
          if ( *(_BYTE *)v99 != 62
            || sub_B46500((unsigned __int8 *)v99)
            || (*(_BYTE *)(v99 + 2) & 1) != 0
            || !(unsigned __int8)sub_DFE1B0((__int64)v222) )
          {
            continue;
          }
          v100 = *(_QWORD *)(*(_QWORD *)(v99 - 64) + 8LL);
          v101 = v100;
          if ( (unsigned int)*(unsigned __int8 *)(v100 + 8) - 17 <= 1 )
            v101 = **(_QWORD **)(v100 + 16);
          if ( !(unsigned __int8)sub_BCBCB0(v101)
            || (unsigned __int8)(*(_BYTE *)(v100 + 8) - 17) <= 1u && *(_BYTE *)(**(_QWORD **)(v100 + 16) + 8LL) == 14 )
          {
            continue;
          }
          v264 = sub_9208B0((__int64)v223, v100);
          v265 = v102;
          v195 = sub_CA1930(&v264);
          if ( (v195 & 7) != 0 )
            continue;
          v190 = *(unsigned __int8 **)(v99 - 32);
          v103 = sub_DFE150((__int64)v222);
          v104 = v190;
          if ( (unsigned int)*(unsigned __int8 *)(v100 + 8) - 17 > 1 )
          {
            if ( v195 <= v103 >> 1 )
              goto LABEL_131;
          }
          else if ( v195 <= v103 >> 1 )
          {
            v105 = sub_DFE2E0(v222, v103 / v195);
            v104 = v190;
            if ( v105 )
            {
LABEL_131:
              v106 = sub_98ACB0(v104, 8u);
              if ( *v106 == 86 )
                v106 = (unsigned __int8 *)*((_QWORD *)v106 - 12);
              v264 = (__int64)v106;
              v93 = v263;
LABEL_110:
              v96 = sub_2B05CF0((__int64)v93, &v264, v89, v90, v91, v92);
              v97 = *(unsigned int *)(v96 + 8);
              if ( v97 + 1 > (unsigned __int64)*(unsigned int *)(v96 + 12) )
              {
                sub_C8D5F0(v96, (const void *)(v96 + 16), v97 + 1, 8u, v94, v95);
                v97 = *(unsigned int *)(v96 + 8);
              }
              *(_QWORD *)(*(_QWORD *)v96 + 8 * v97) = v99;
              ++*(_DWORD *)(v96 + 8);
            }
          }
        }
      }
      v264 = 0;
      v265 = 0;
      v266 = 0;
      v267 = 0;
      sub_C7D6A0(0, 0, 8);
      LODWORD(v267) = v218;
      if ( v218 )
      {
        v265 = (_QWORD **)sub_C7D670(16LL * v218, 8);
        v266 = v217;
        memcpy(v265, src, 16LL * (unsigned int)v267);
      }
      else
      {
        v265 = 0;
        v266 = 0;
      }
      v269 = 0;
      v268 = &v270;
      if ( (_DWORD)v220 )
        sub_2AFF480((__int64)&v268, (__int64 *)&v219, v130, v131, v132, v133);
      v270 = 0;
      v271 = 0;
      v272 = 0;
      v273 = 0;
      sub_C7D6A0(0, 0, 8);
      LODWORD(v273) = v263[3];
      if ( LODWORD(v263[3]) )
      {
        v271 = (void *)sub_C7D670(16LL * LODWORD(v263[3]), 8);
        v272 = v263[2];
        memcpy(v271, (const void *)v263[1], 16LL * (unsigned int)v273);
      }
      else
      {
        v271 = 0;
        v272 = 0;
      }
      v275 = 0;
      v274 = (unsigned __int64 *)&v276;
      if ( !LODWORD(v263[5]) )
        goto LABEL_176;
      sub_2AFF480((__int64)&v274, &v263[4], v134, v135, v136, v137);
      v138 = v263[4];
      v166 = v263[4] + 88LL * LODWORD(v263[5]);
      if ( v263[4] != v166 )
        break;
LABEL_177:
      if ( v138 != i )
        _libc_free(v138);
      sub_C7D6A0(v263[1], 16LL * LODWORD(v263[3]), 8);
      v139 = (__int64)v219;
      v140 = &v219[11 * (unsigned int)v220];
      if ( v219 != v140 )
      {
        do
        {
          v140 -= 11;
          v141 = v140[1];
          if ( (__int64 *)v141 != v140 + 3 )
            _libc_free(v141);
        }
        while ( (__int64 *)v139 != v140 );
        v140 = v219;
      }
      if ( v140 != v221 )
        _libc_free((unsigned __int64)v140);
      sub_C7D6A0((__int64)src, 16LL * v218, 8);
      sub_C7D6A0(v204, 16LL * v206, 8);
      v142 = (__int64)v265;
      v143 = v207;
      v265 = 0;
      ++v203;
      v204 = v142;
      ++v264;
      v205 = v266;
      v266 = 0;
      v206 = v267;
      LODWORD(v267) = 0;
      v144 = v207 + 88LL * (unsigned int)v208;
      if ( (_DWORD)v269 )
      {
        if ( v207 != v144 )
        {
          do
          {
            v144 -= 88LL;
            v145 = *(_QWORD *)(v144 + 8);
            if ( v145 != v144 + 24 )
              _libc_free(v145);
          }
          while ( v143 != v144 );
          v144 = v207;
        }
        if ( (__int64 **)v144 != v202 )
          _libc_free(v144);
        v146 = v268;
        v268 = &v270;
        v207 = (__int64)v146;
        v147 = v269;
        v269 = 0;
        v208 = v147;
      }
      else
      {
        while ( v143 != v144 )
        {
          while ( 1 )
          {
            v144 -= 88LL;
            v165 = *(_QWORD *)(v144 + 8);
            if ( v165 == v144 + 24 )
              break;
            _libc_free(v165);
            if ( v143 == v144 )
              goto LABEL_236;
          }
        }
LABEL_236:
        LODWORD(v208) = 0;
      }
      sub_C7D6A0(v210, 16LL * v212, 8);
      v148 = (__int64)v271;
      v271 = 0;
      ++v209;
      v210 = v148;
      ++v270;
      v211 = v272;
      v272 = 0;
      v212 = v273;
      LODWORD(v273) = 0;
      if ( (_DWORD)v275 )
      {
        v168 = v213;
        v169 = (unsigned __int64)&v213[11 * (unsigned int)v214];
        if ( v213 != (unsigned __int64 *)v169 )
        {
          do
          {
            v169 -= 88LL;
            v170 = *(_QWORD *)(v169 + 8);
            if ( v170 != v169 + 24 )
              _libc_free(v170);
          }
          while ( v168 != (unsigned __int64 *)v169 );
          v169 = (unsigned __int64)v213;
        }
        if ( (__int64 **)v169 != v200 )
          _libc_free(v169);
        v213 = v274;
        v214 = v275;
      }
      else
      {
        v149 = v213;
        v150 = &v213[11 * (unsigned int)v214];
        if ( v213 == v150 )
        {
          LODWORD(v214) = 0;
          v153 = v274;
        }
        else
        {
          do
          {
            v150 -= 11;
            v151 = v150[1];
            if ( (unsigned __int64 *)v151 != v150 + 3 )
              _libc_free(v151);
          }
          while ( v149 != v150 );
          v152 = v274;
          LODWORD(v214) = 0;
          v153 = &v274[11 * (unsigned int)v275];
          if ( v274 != v153 )
          {
            do
            {
              v153 -= 11;
              v154 = v153[1];
              if ( (unsigned __int64 *)v154 != v153 + 3 )
                _libc_free(v154);
            }
            while ( v153 != v152 );
            v153 = v274;
          }
        }
        if ( v153 != (unsigned __int64 *)&v276 )
          _libc_free((unsigned __int64)v153);
      }
      sub_C7D6A0((__int64)v271, 16LL * (unsigned int)v273, 8);
      v155 = v268;
      v156 = &v268[11 * (unsigned int)v269];
      if ( v268 != v156 )
      {
        do
        {
          v156 -= 11;
          v157 = v156[1];
          if ( (__int64 *)v157 != v156 + 3 )
            _libc_free(v157);
        }
        while ( v155 != v156 );
        v156 = v268;
      }
      if ( v156 != &v270 )
        _libc_free((unsigned __int64)v156);
      sub_C7D6A0((__int64)v265, 16LL * (unsigned int)v267, 8);
      v158 = sub_2B03A80((__int64)v221, (__int64)&v203);
      v198 |= sub_2B03A80((__int64)v221, (__int64)v202) | v158;
      if ( dword_500F2E8 <= (unsigned int)v199 && dword_500F2E8 )
      {
        v178 = v213;
        v179 = (unsigned __int64)&v213[11 * (unsigned int)v214];
        if ( v213 != (unsigned __int64 *)v179 )
        {
          do
          {
            v179 -= 88LL;
            v180 = *(_QWORD *)(v179 + 8);
            if ( v180 != v179 + 24 )
              _libc_free(v180);
          }
          while ( v178 != (unsigned __int64 *)v179 );
          v179 = (unsigned __int64)v213;
        }
        if ( (__int64 **)v179 != v200 )
          _libc_free(v179);
        sub_C7D6A0(v210, 16LL * v212, 8);
        v181 = v207;
        v182 = v207 + 88LL * (unsigned int)v208;
        if ( v207 != v182 )
        {
          do
          {
            v182 -= 88LL;
            v183 = *(_QWORD *)(v182 + 8);
            if ( v183 != v182 + 24 )
              _libc_free(v183);
          }
          while ( v181 != v182 );
          v182 = v207;
        }
        if ( (__int64 **)v182 != v202 )
          _libc_free(v182);
        sub_C7D6A0(v204, 16LL * v206, 8);
        v76 = v307;
LABEL_255:
        if ( v76 != &v309 )
          _libc_free((unsigned __int64)v76);
        if ( !v299 )
          _libc_free(v297);
        if ( v293 != (__int64 **)v295 )
          _libc_free((unsigned __int64)v293);
        if ( !v287 )
          _libc_free((unsigned __int64)v284);
        if ( v341 != &v343 )
          _libc_free((unsigned __int64)v341);
        if ( !v340 )
          _libc_free(v339);
        if ( v326 != (__int64 *)&v329 )
          _libc_free((unsigned __int64)v326);
        if ( !v317 )
          _libc_free(v314);
        goto LABEL_271;
      }
      v159 = v213;
      LODWORD(v199) = (_DWORD)v199 + 1;
      v160 = (unsigned __int64)&v213[11 * (unsigned int)v214];
      if ( v213 != (unsigned __int64 *)v160 )
      {
        do
        {
          v160 -= 88LL;
          v161 = *(_QWORD *)(v160 + 8);
          if ( v161 != v160 + 24 )
            _libc_free(v161);
        }
        while ( v159 != (unsigned __int64 *)v160 );
        v160 = (unsigned __int64)v213;
      }
      if ( (__int64 **)v160 != v200 )
        _libc_free(v160);
      sub_C7D6A0(v210, 16LL * v212, 8);
      v162 = v207;
      v163 = v207 + 88LL * (unsigned int)v208;
      if ( v207 != v163 )
      {
        do
        {
          v163 -= 88LL;
          v164 = *(_QWORD *)(v163 + 8);
          if ( v164 != v163 + 24 )
            _libc_free(v164);
        }
        while ( v162 != v163 );
        v163 = v207;
      }
      if ( (__int64 **)v163 != v202 )
        _libc_free(v163);
      sub_C7D6A0(v204, 16LL * v206, 8);
      v15 = (_DWORD)v294 == 1;
      v74 = (unsigned int)(v294 - 1);
      LODWORD(v294) = v294 - 1;
      if ( !v15 )
      {
        sub_CE27D0((__int64)&v283);
        v74 = (unsigned int)v294;
      }
      v73 = (unsigned int)v308;
    }
    do
    {
      v166 -= 88;
      v167 = *(_QWORD *)(v166 + 8);
      if ( v167 != v166 + 24 )
        _libc_free(v167);
    }
    while ( v138 != v166 );
LABEL_176:
    v138 = v263[4];
    goto LABEL_177;
  }
LABEL_271:
  v315 = 0x100000002LL;
  v314 = (unsigned __int64)&v318;
  v321[0] = (unsigned __int64)&v324;
  v171 = a1 + 32;
  v316 = 0;
  v172 = a1 + 80;
  v317 = 1;
  v320 = 0;
  v321[1] = 2;
  v322 = 0;
  v323 = 1;
  v318 = &unk_4F82408;
  v313 = 1;
  if ( v198 )
  {
    sub_C8CD80(a1, v171, (__int64)&v313, v73, (__int64)v76, v72);
    sub_C8CD80(a1 + 48, v172, (__int64)&v320, v173, v174, v175);
    if ( !v323 )
      _libc_free(v321[0]);
    if ( !v317 )
      _libc_free(v314);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 8) = v171;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = v172;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)a1 = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
  }
  sub_C7D6A0(v260, 8LL * (unsigned int)v262, 8);
  sub_C7D6A0(v256, 16LL * v258, 8);
  sub_2AF7E80(v251);
  sub_C7D6A0(v247, 16LL * v249, 8);
  sub_C7D6A0(v241, 16LL * v243, 8);
  nullsub_61();
  v239 = &unk_49DA100;
  nullsub_63();
  if ( v224 != v226 )
    _libc_free((unsigned __int64)v224);
  return a1;
}
