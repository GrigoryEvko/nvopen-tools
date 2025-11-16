// Function: sub_D6FF50
// Address: 0xd6ff50
//
__int64 __fastcall sub_D6FF50(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 **v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rbx
  __int64 *v10; // r14
  __int64 *v11; // r12
  int v12; // edx
  _BYTE *v13; // r15
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 *v16; // rdx
  unsigned int v17; // eax
  __int64 v18; // rsi
  __int64 v19; // rcx
  _QWORD *v20; // rdi
  _QWORD *v21; // rsi
  __int64 v22; // rax
  __int64 *v23; // rcx
  __int64 v24; // r12
  __int64 *v25; // r15
  __int64 *v26; // r14
  __int64 *v27; // rbx
  unsigned int v28; // esi
  int v29; // eax
  int v30; // eax
  __int64 *v31; // rax
  __int64 v32; // rdx
  __int64 *v33; // rax
  int v34; // r13d
  __int64 **v35; // rdx
  char v36; // r10
  __int64 v37; // rsi
  __int64 **v38; // rax
  __int64 *v39; // r15
  __int64 *v40; // rcx
  __int64 v41; // rsi
  __int64 *v42; // r13
  _BYTE **v43; // rsi
  __int64 *v44; // rbx
  __int64 *v45; // r14
  _QWORD *v46; // rdi
  _QWORD *v47; // rsi
  __int64 v48; // r8
  __int64 *v49; // rax
  int v50; // eax
  __int64 v51; // rsi
  int v52; // edx
  unsigned int v53; // eax
  __int64 v54; // rdi
  int i; // r9d
  _QWORD *v56; // rdi
  _QWORD *v57; // rsi
  __int64 v58; // rax
  __int64 **v59; // r12
  __int64 *v60; // rax
  __int64 **v61; // rbx
  __int64 v62; // r14
  __int64 **v63; // r8
  int v64; // edi
  unsigned int v65; // eax
  __int64 *v66; // rdx
  __int64 v67; // r15
  unsigned __int64 v68; // rbx
  char v69; // r12
  __int64 **v70; // rdi
  __int64 v71; // rsi
  unsigned int v72; // edx
  __int64 **v73; // r15
  __int64 *v74; // rcx
  __int64 **v75; // rdi
  __int64 v76; // rsi
  __int64 **v77; // rdi
  __int64 **v78; // rdx
  __int64 **v79; // rdx
  unsigned int v80; // esi
  int v81; // eax
  int v82; // eax
  __int64 *v83; // r14
  __int64 *v84; // r13
  __int64 *v85; // rdx
  unsigned int v86; // esi
  int v87; // eax
  int v88; // eax
  __int64 v89; // rax
  __int64 v90; // rax
  __int64 v91; // rdx
  __int64 v92; // rcx
  __int64 v93; // r8
  __int64 v94; // r9
  __int64 *v95; // r15
  __int64 *v96; // rdx
  __int64 **v97; // rax
  __int64 v98; // rsi
  __int64 v99; // rdx
  __int64 v100; // rax
  __int64 *v101; // rax
  __int64 **v102; // r12
  __int64 **v103; // rbx
  __int64 *v104; // r8
  int v105; // esi
  unsigned int v106; // edx
  __int64 *v107; // rax
  _QWORD *v108; // r9
  unsigned __int64 v109; // r14
  unsigned int v110; // esi
  unsigned int v111; // eax
  unsigned int v112; // edx
  unsigned int v113; // edi
  __int64 *v114; // rdx
  unsigned __int64 *v115; // rax
  __int64 v116; // r14
  __int64 **v117; // rbx
  __int64 *v118; // rax
  __int64 *v119; // r8
  int v120; // esi
  unsigned int v121; // ecx
  __int64 *v122; // rdx
  _QWORD *v123; // r9
  __int64 v124; // r15
  int v125; // r12d
  int v126; // r13d
  __int64 v127; // rbx
  int v128; // r14d
  int v129; // esi
  __int64 v130; // rax
  __int64 v131; // rdx
  __int64 v132; // rdx
  __int64 *v133; // r13
  unsigned int v134; // esi
  unsigned int v135; // esi
  __int64 *v136; // rsi
  __int64 *v137; // r8
  __int64 v138; // rbx
  unsigned int v139; // edi
  __int64 *v140; // rsi
  __int64 j; // rcx
  __int64 v142; // rax
  __int64 v143; // rax
  __int64 v144; // rdx
  __int64 v145; // r10
  unsigned int v146; // ecx
  __int64 v147; // rdx
  __int64 v148; // rcx
  unsigned int v149; // esi
  __int64 v150; // rdx
  unsigned int v151; // eax
  __int64 v152; // r12
  __int64 v153; // r8
  __int64 v154; // r9
  __int64 v155; // rdx
  unsigned int v156; // eax
  __int64 v157; // rax
  unsigned __int64 v158; // rdx
  unsigned int v159; // edx
  unsigned int v160; // ecx
  unsigned int v161; // r8d
  _QWORD *v162; // rdx
  unsigned __int64 v163; // r12
  char v164; // r9
  __int64 *v165; // rdi
  __int64 *v166; // rcx
  __int64 *v167; // rax
  __int64 *v168; // rdx
  __int64 v169; // rsi
  __int64 v170; // rsi
  __int64 *v171; // rsi
  __int64 v172; // rcx
  __int64 v173; // r8
  __int64 v174; // r9
  __int64 v175; // r8
  __int64 v176; // r9
  __int64 v177; // r13
  _BYTE *v178; // rbx
  __int64 v179; // rax
  __int64 *v180; // rcx
  __int64 v181; // r13
  __int64 v182; // r15
  __int64 v183; // r12
  __int64 v184; // rbx
  __int64 v185; // r15
  int v186; // edi
  __int64 v187; // r9
  int v188; // edi
  unsigned int v189; // esi
  __int64 *v190; // rax
  __int64 v191; // r11
  __int64 v192; // rsi
  bool v193; // zf
  __int64 v194; // rax
  unsigned __int64 v195; // rax
  int v196; // eax
  __int64 v197; // rdx
  unsigned int v198; // eax
  unsigned __int64 v199; // rax
  int v200; // r11d
  __int64 *v201; // r10
  int v202; // edx
  int v203; // r11d
  __int64 *v204; // r10
  _QWORD **v205; // rdx
  _QWORD **v206; // r13
  __int64 *v207; // rax
  unsigned int v208; // edx
  unsigned int v209; // esi
  _QWORD *v210; // rdx
  __int64 v211; // r12
  int v212; // r15d
  int v213; // ebx
  __int64 *v214; // rdx
  int v215; // r15d
  _QWORD *v216; // r13
  int v217; // ebx
  __int64 v218; // rsi
  __int64 v219; // rbx
  _QWORD *v220; // r12
  __int64 v221; // rax
  __int64 *v222; // r12
  __int64 *v223; // rbx
  __int64 result; // rax
  __int64 *v225; // rdi
  __int64 v226; // rsi
  __int64 *v227; // rdi
  __int64 *v228; // rax
  __int64 v229; // rsi
  int v230; // r8d
  int v231; // r9d
  __int64 v232; // rax
  __int64 **v233; // r12
  __int64 **v234; // rbx
  __int64 *v235; // rdx
  __int64 *v236; // r13
  __int64 *v237; // rbx
  __int64 v238; // r15
  __int64 v239; // r14
  __int64 v240; // rsi
  __int64 v241; // rdx
  __int64 v242; // rcx
  __int64 v243; // r8
  __int64 v244; // r9
  __int64 v245; // rdx
  __int64 v246; // rcx
  __int64 v247; // r8
  __int64 v248; // r9
  __int64 v249; // r12
  __int64 v250; // rcx
  __int64 v251; // r8
  __int64 v252; // r9
  __int64 v253; // r14
  _QWORD *v254; // rax
  __int64 v255; // rdx
  _QWORD *v256; // rdi
  __int64 v257; // rdx
  _QWORD *v258; // rdx
  __int64 v259; // r15
  __int64 v260; // rbx
  __int64 v261; // rdi
  __int64 v262; // rsi
  __int64 **v263; // rdx
  __int64 v264; // rax
  unsigned int v265; // eax
  __int64 *v266; // r8
  __int64 *v267; // rbx
  __int64 *v268; // r15
  __int64 v269; // r13
  __int64 *v270; // [rsp+8h] [rbp-7F8h]
  unsigned __int64 v272; // [rsp+10h] [rbp-7F0h]
  __int64 v273; // [rsp+18h] [rbp-7E8h]
  __int64 *v274; // [rsp+20h] [rbp-7E0h]
  __int64 *v275; // [rsp+20h] [rbp-7E0h]
  __int64 *v276; // [rsp+30h] [rbp-7D0h]
  __int64 v278; // [rsp+38h] [rbp-7C8h]
  __int64 *v279; // [rsp+48h] [rbp-7B8h]
  __int64 *v280; // [rsp+48h] [rbp-7B8h]
  _QWORD **v281; // [rsp+48h] [rbp-7B8h]
  __int64 *v282; // [rsp+50h] [rbp-7B0h]
  __int64 *v283; // [rsp+50h] [rbp-7B0h]
  __int64 *v284; // [rsp+50h] [rbp-7B0h]
  __int64 **v285; // [rsp+58h] [rbp-7A8h]
  __int64 *v286; // [rsp+58h] [rbp-7A8h]
  __int64 *v287; // [rsp+58h] [rbp-7A8h]
  __int64 v288; // [rsp+58h] [rbp-7A8h]
  __int64 *v289; // [rsp+60h] [rbp-7A0h]
  __int64 v291; // [rsp+70h] [rbp-790h]
  __int64 v292; // [rsp+70h] [rbp-790h]
  __int64 v293; // [rsp+70h] [rbp-790h]
  __int64 v294; // [rsp+78h] [rbp-788h]
  __int64 **v295; // [rsp+78h] [rbp-788h]
  __int64 *v296; // [rsp+78h] [rbp-788h]
  __int64 v298; // [rsp+88h] [rbp-778h]
  __int64 v299; // [rsp+88h] [rbp-778h]
  __int64 v300; // [rsp+88h] [rbp-778h]
  __int64 *v301; // [rsp+88h] [rbp-778h]
  __int64 *v302; // [rsp+90h] [rbp-770h]
  __int64 v303; // [rsp+90h] [rbp-770h]
  __int64 v304; // [rsp+90h] [rbp-770h]
  __int64 v305; // [rsp+90h] [rbp-770h]
  __int64 *v306; // [rsp+90h] [rbp-770h]
  _BYTE **v307; // [rsp+98h] [rbp-768h] BYREF
  __int64 v308; // [rsp+A8h] [rbp-758h] BYREF
  _QWORD v309[4]; // [rsp+B0h] [rbp-750h] BYREF
  _QWORD v310[2]; // [rsp+D0h] [rbp-730h] BYREF
  char v311; // [rsp+E0h] [rbp-720h]
  _QWORD **v312; // [rsp+F0h] [rbp-710h]
  __int64 v313; // [rsp+100h] [rbp-700h] BYREF
  __int64 **v314; // [rsp+108h] [rbp-6F8h]
  __int64 v315; // [rsp+110h] [rbp-6F0h]
  int v316; // [rsp+118h] [rbp-6E8h]
  char v317; // [rsp+11Ch] [rbp-6E4h]
  _BYTE v318[16]; // [rsp+120h] [rbp-6E0h] BYREF
  __int64 **v319; // [rsp+130h] [rbp-6D0h] BYREF
  __int64 v320; // [rsp+138h] [rbp-6C8h]
  _BYTE v321[64]; // [rsp+140h] [rbp-6C0h] BYREF
  _QWORD *v322; // [rsp+180h] [rbp-680h] BYREF
  __int64 v323; // [rsp+188h] [rbp-678h]
  __int64 v324; // [rsp+190h] [rbp-670h]
  __int64 v325; // [rsp+198h] [rbp-668h]
  _BYTE *v326; // [rsp+1A0h] [rbp-660h]
  __int64 v327; // [rsp+1A8h] [rbp-658h]
  _BYTE v328[32]; // [rsp+1B0h] [rbp-650h] BYREF
  __int64 *v329; // [rsp+1D0h] [rbp-630h] BYREF
  int v330; // [rsp+1D8h] [rbp-628h]
  char v331; // [rsp+1E0h] [rbp-620h] BYREF
  __int64 v332; // [rsp+220h] [rbp-5E0h] BYREF
  __int64 v333; // [rsp+228h] [rbp-5D8h]
  __int64 v334; // [rsp+230h] [rbp-5D0h] BYREF
  unsigned int v335; // [rsp+238h] [rbp-5C8h]
  __int64 *v336; // [rsp+290h] [rbp-570h] BYREF
  __int64 v337; // [rsp+298h] [rbp-568h]
  _BYTE v338[128]; // [rsp+2A0h] [rbp-560h] BYREF
  _QWORD *v339; // [rsp+320h] [rbp-4E0h] BYREF
  char *v340; // [rsp+328h] [rbp-4D8h]
  __int64 v341; // [rsp+330h] [rbp-4D0h]
  int v342; // [rsp+338h] [rbp-4C8h]
  char v343; // [rsp+33Ch] [rbp-4C4h]
  char v344; // [rsp+340h] [rbp-4C0h] BYREF
  _BYTE *v345; // [rsp+3C0h] [rbp-440h] BYREF
  __int64 v346; // [rsp+3C8h] [rbp-438h]
  _BYTE v347[192]; // [rsp+3D0h] [rbp-430h] BYREF
  __int64 *v348; // [rsp+490h] [rbp-370h] BYREF
  __int64 v349; // [rsp+498h] [rbp-368h]
  __int64 *v350; // [rsp+4A0h] [rbp-360h] BYREF
  unsigned int v351; // [rsp+4A8h] [rbp-358h]
  char v352; // [rsp+4B0h] [rbp-350h]
  _BYTE v353[192]; // [rsp+4E0h] [rbp-320h] BYREF
  __int64 v354; // [rsp+5A0h] [rbp-260h] BYREF
  __int64 v355; // [rsp+5A8h] [rbp-258h]
  __int64 *v356; // [rsp+5B0h] [rbp-250h] BYREF
  unsigned int v357; // [rsp+5B8h] [rbp-248h]
  _BYTE v358[48]; // [rsp+7D0h] [rbp-30h] BYREF

  v6 = a4;
  v309[1] = &v307;
  v7 = &v356;
  v307 = (_BYTE **)a5;
  v309[0] = a1;
  v309[2] = a4;
  v354 = 0;
  v355 = 1;
  do
  {
    *v7 = (__int64 *)-4096LL;
    v7 += 17;
  }
  while ( v7 != (__int64 **)v358 );
  v8 = 16 * a3;
  v298 = a2 + v8;
  if ( a2 != a2 + v8 )
  {
    v9 = a2;
    v10 = (__int64 *)&v336;
    v11 = &v354;
    while ( 1 )
    {
      v336 = (__int64 *)(*(_QWORD *)(v9 + 8) & 0xFFFFFFFFFFFFFFF8LL);
      if ( !(unsigned __int8)sub_D67860((__int64)v11, v10, &v345) )
        break;
      v12 = *((_DWORD *)v345 + 6);
      v13 = v345 + 8;
      v339 = *(_QWORD **)v9;
      if ( v12 )
      {
        v291 = (__int64)v345;
        sub_D6CB10((__int64)&v348, (__int64)(v345 + 8), (__int64 *)&v339);
        if ( v352 )
          sub_B1A4E0(v291 + 40, (__int64)v339);
        goto LABEL_8;
      }
LABEL_16:
      v20 = (_QWORD *)*((_QWORD *)v13 + 4);
      v21 = &v20[*((unsigned int *)v13 + 10)];
      if ( v21 == sub_D67AA0(v20, (__int64)v21, (__int64 *)&v339)
        && (sub_B1A4E0((__int64)(v13 + 32), a6), v22 = *((unsigned int *)v13 + 10), (unsigned int)v22 > 2) )
      {
        v23 = (__int64 *)*((_QWORD *)v13 + 4);
        v292 = v9;
        v282 = v11;
        v24 = (__int64)v13;
        v25 = v10;
        v26 = v23;
        v27 = &v23[v22];
        do
        {
          while ( (unsigned __int8)sub_D6B660(v24, v26, &v345) )
          {
            if ( v27 == ++v26 )
              goto LABEL_26;
          }
          v348 = (__int64 *)v345;
          v28 = *(_DWORD *)(v24 + 24);
          v29 = *(_DWORD *)(v24 + 16);
          ++*(_QWORD *)v24;
          v30 = v29 + 1;
          a6 = (unsigned int)(4 * v30);
          if ( (unsigned int)a6 >= 3 * v28 )
          {
            v28 *= 2;
          }
          else
          {
            a5 = v28 >> 3;
            if ( v28 - *(_DWORD *)(v24 + 20) - v30 > (unsigned int)a5 )
              goto LABEL_23;
          }
          sub_CF28B0(v24, v28);
          sub_D6B660(v24, v26, &v348);
          v30 = *(_DWORD *)(v24 + 16) + 1;
LABEL_23:
          *(_DWORD *)(v24 + 16) = v30;
          v31 = v348;
          if ( *v348 != -4096 )
            --*(_DWORD *)(v24 + 20);
          v32 = *v26++;
          *v31 = v32;
        }
        while ( v27 != v26 );
LABEL_26:
        v11 = v282;
        v10 = v25;
        v9 = v292 + 16;
        if ( v298 == v292 + 16 )
        {
LABEL_27:
          v6 = a4;
          goto LABEL_28;
        }
      }
      else
      {
LABEL_8:
        v9 += 16;
        if ( v298 == v9 )
          goto LABEL_27;
      }
    }
    v16 = (__int64 *)v345;
    ++v354;
    v348 = (__int64 *)v345;
    v17 = ((unsigned int)v355 >> 1) + 1;
    if ( (v355 & 1) != 0 )
    {
      v19 = 12;
      v18 = 4;
      if ( 4 * v17 < 0xC )
      {
LABEL_12:
        v19 = (unsigned int)v18 >> 3;
        if ( (unsigned int)v18 - (v17 + HIDWORD(v355)) > (unsigned int)v19 )
        {
LABEL_13:
          LODWORD(v355) = v355 & 1 | (2 * v17);
          if ( *v16 != -4096 )
            --HIDWORD(v355);
          v13 = v16 + 1;
          *v16 = (__int64)v336;
          memset(v16 + 1, 0, 0x80u);
          v16[5] = (__int64)(v16 + 7);
          v16[6] = 0x200000000LL;
          v16[13] = (__int64)(v16 + 15);
          v16[14] = 0x200000000LL;
          v339 = *(_QWORD **)v9;
          goto LABEL_16;
        }
LABEL_56:
        sub_D686A0((__int64)v11, v18, (__int64)v345, v19, v14, v15);
        sub_D67860((__int64)v11, v10, &v348);
        v16 = v348;
        v17 = ((unsigned int)v355 >> 1) + 1;
        goto LABEL_13;
      }
    }
    else
    {
      v18 = v357;
      v19 = 3 * v357;
      if ( 4 * v17 < (unsigned int)v19 )
        goto LABEL_12;
    }
    v18 = (unsigned int)(2 * v18);
    goto LABEL_56;
  }
LABEL_28:
  v33 = &v334;
  v332 = 0;
  v333 = 1;
  do
  {
    *v33 = -4096;
    v33 += 3;
    *(v33 - 2) = -4096;
  }
  while ( v33 != (__int64 *)&v336 );
  LOBYTE(v34) = v355;
  v35 = (__int64 **)v318;
  v313 = 0;
  v314 = (__int64 **)v318;
  v317 = 1;
  v315 = 2;
  v36 = v355 & 1;
  v316 = 0;
  if ( (unsigned int)v355 >> 1 )
  {
    if ( v36 )
    {
      v40 = (__int64 *)v358;
      v39 = (__int64 *)&v356;
      v283 = (__int64 *)v358;
    }
    else
    {
      v37 = v357;
      v38 = (__int64 **)v356;
      v39 = v356;
      v40 = &v356[17 * v357];
      v283 = v40;
      if ( v40 == v356 )
        goto LABEL_38;
    }
    do
    {
      if ( *v39 != -8192 && *v39 != -4096 )
        break;
      v39 += 17;
    }
    while ( v39 != v40 );
  }
  else
  {
    if ( v36 )
    {
      v40 = (__int64 *)&v356;
      v232 = 68;
    }
    else
    {
      v40 = v356;
      v232 = 17LL * v357;
    }
    v39 = &v40[v232];
    v283 = &v40[v232];
  }
  if ( v36 )
  {
    v38 = &v356;
    v41 = 68;
    goto LABEL_39;
  }
  v38 = (__int64 **)v356;
  v37 = v357;
LABEL_38:
  v41 = 17 * v37;
LABEL_39:
  v279 = (__int64 *)&v38[v41];
  if ( v39 == (__int64 *)&v38[v41] )
    goto LABEL_360;
  v273 = v6;
  do
  {
    v42 = (__int64 *)*v39;
    v43 = v307;
    sub_D69DD0((__int64)&v348, (__int64)v307, *v39, (__int64)v40, a5, a6);
    v44 = v348;
    v302 = &v348[(unsigned int)v349];
    if ( v348 == v302 )
      goto LABEL_59;
    v45 = &v332;
    v294 = (__int64)(v39 + 9);
    do
    {
      v48 = *v44;
      v336 = (__int64 *)*v44;
      if ( !*((_DWORD *)v39 + 6) )
      {
        v46 = (_QWORD *)v39[5];
        v47 = &v46[*((unsigned int *)v39 + 12)];
        if ( v47 != sub_D67930(v46, (__int64)v47, (__int64 *)&v336) )
          goto LABEL_44;
LABEL_50:
        if ( !*((_DWORD *)v39 + 22) )
        {
          v56 = (_QWORD *)v39[13];
          v57 = &v56[*((unsigned int *)v39 + 28)];
          if ( v57 != sub_D67AA0(v56, (__int64)v57, (__int64 *)&v336) )
            goto LABEL_44;
          sub_B1A4E0((__int64)(v39 + 13), v48);
          v58 = *((unsigned int *)v39 + 28);
          if ( (unsigned int)v58 <= 2 )
            goto LABEL_53;
          v276 = v42;
          v274 = v45;
          v83 = (__int64 *)v39[13];
          v84 = &v83[v58];
          while ( 1 )
          {
            while ( (unsigned __int8)sub_D6B660(v294, v83, &v339) )
            {
              if ( v84 == ++v83 )
                goto LABEL_111;
            }
            v85 = v339;
            v345 = v339;
            v86 = *((_DWORD *)v39 + 24);
            v87 = *((_DWORD *)v39 + 22);
            ++v39[9];
            v88 = v87 + 1;
            if ( 4 * v88 >= 3 * v86 )
              break;
            if ( v86 - *((_DWORD *)v39 + 23) - v88 <= v86 >> 3 )
              goto LABEL_349;
LABEL_108:
            *((_DWORD *)v39 + 22) = v88;
            if ( *v85 != -4096 )
              --*((_DWORD *)v39 + 23);
            v89 = *v83++;
            *v85 = v89;
            if ( v84 == v83 )
            {
LABEL_111:
              v42 = v276;
              v45 = v274;
              v48 = (__int64)v336;
              goto LABEL_44;
            }
          }
          v86 *= 2;
LABEL_349:
          sub_CF28B0(v294, v86);
          sub_D6B660(v294, v83, &v345);
          v85 = (__int64 *)v345;
          v88 = *((_DWORD *)v39 + 22) + 1;
          goto LABEL_108;
        }
        if ( (unsigned __int8)sub_D6B660(v294, (__int64 *)&v336, &v339) )
        {
LABEL_53:
          v48 = (__int64)v336;
          goto LABEL_44;
        }
        v79 = (__int64 **)v339;
        v345 = v339;
        v80 = *((_DWORD *)v39 + 24);
        v81 = *((_DWORD *)v39 + 22);
        ++v39[9];
        v82 = v81 + 1;
        if ( 4 * v82 >= 3 * v80 )
        {
          v80 *= 2;
        }
        else if ( v80 - *((_DWORD *)v39 + 23) - v82 > v80 >> 3 )
        {
LABEL_98:
          *((_DWORD *)v39 + 22) = v82;
          if ( *v79 != (__int64 *)-4096LL )
            --*((_DWORD *)v39 + 23);
          *v79 = v336;
          sub_B1A4E0((__int64)(v39 + 13), (__int64)v336);
          goto LABEL_53;
        }
        sub_CF28B0(v294, v80);
        sub_D6B660(v294, (__int64 *)&v336, &v345);
        v79 = (__int64 **)v345;
        v82 = *((_DWORD *)v39 + 22) + 1;
        goto LABEL_98;
      }
      v50 = *((_DWORD *)v39 + 8);
      v51 = v39[2];
      if ( !v50 )
        goto LABEL_50;
      v52 = v50 - 1;
      v53 = (v50 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
      v54 = *(_QWORD *)(v51 + 8LL * v53);
      if ( v48 == v54 )
        goto LABEL_44;
      for ( i = 1; ; ++i )
      {
        if ( v54 == -4096 )
          goto LABEL_50;
        v53 = v52 & (i + v53);
        v54 = *(_QWORD *)(v51 + 8LL * v53);
        if ( v48 == v54 )
          break;
      }
LABEL_44:
      v43 = &v345;
      ++v44;
      v345 = (_BYTE *)v48;
      v346 = (__int64)v42;
      v49 = sub_B1DDD0((__int64)v45, (__int64 *)&v345);
      ++*(_DWORD *)v49;
    }
    while ( v302 != v44 );
    v302 = v348;
LABEL_59:
    v40 = (__int64 *)&v350;
    if ( v302 != (__int64 *)&v350 )
      _libc_free(v302, v43);
    if ( !*((_DWORD *)v39 + 28) )
      sub_D695C0((__int64)&v348, (__int64)&v313, v42, (__int64)v40, a5, a6);
    for ( v39 += 17; v283 != v39; v39 += 17 )
    {
      if ( *v39 != -8192 && *v39 != -4096 )
        break;
    }
  }
  while ( v279 != v39 );
  LOBYTE(v34) = v355;
  v6 = v273;
  v35 = v314;
  v36 = v355 & 1;
  if ( !v317 )
  {
    v59 = &v314[(unsigned int)v315];
    goto LABEL_70;
  }
LABEL_360:
  v59 = &v35[HIDWORD(v315)];
LABEL_70:
  if ( v59 != v35 )
  {
    while ( 1 )
    {
      v60 = *v35;
      v61 = v35;
      if ( (unsigned __int64)*v35 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v59 == ++v35 )
        goto LABEL_73;
    }
LABEL_82:
    if ( v59 == v61 )
      goto LABEL_73;
    v36 = v34 & 1;
    if ( (v34 & 1) != 0 )
    {
      v70 = &v356;
      v71 = 3;
LABEL_85:
      v72 = v71 & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
      v73 = &v70[17 * v72];
      v74 = *v73;
      if ( v60 == *v73 )
      {
LABEL_86:
        v75 = (__int64 **)v73[13];
        if ( v75 != v73 + 15 )
          _libc_free(v75, v71);
        v76 = 8LL * *((unsigned int *)v73 + 24);
        sub_C7D6A0((__int64)v73[10], v76, 8);
        v77 = (__int64 **)v73[5];
        if ( v77 != v73 + 7 )
          _libc_free(v77, v76);
        sub_C7D6A0((__int64)v73[2], 8LL * *((unsigned int *)v73 + 8), 8);
        *v73 = (__int64 *)-8192LL;
        ++HIDWORD(v355);
        v34 = (2 * ((unsigned int)v355 >> 1) - 2) | v355 & 1;
        LODWORD(v355) = v34;
        v36 = v34 & 1;
      }
      else
      {
        v230 = 1;
        while ( v74 != (__int64 *)-4096LL )
        {
          v72 = v71 & (v72 + v230);
          v73 = &v70[17 * v72];
          v74 = *v73;
          if ( v60 == *v73 )
            goto LABEL_86;
          ++v230;
        }
      }
    }
    else
    {
      v70 = (__int64 **)v356;
      if ( v357 )
      {
        v71 = v357 - 1;
        goto LABEL_85;
      }
    }
    v78 = v61 + 1;
    if ( v61 + 1 == v59 )
      goto LABEL_73;
    do
    {
      v60 = *v78;
      v61 = v78;
      if ( (unsigned __int64)*v78 < 0xFFFFFFFFFFFFFFFELL )
        goto LABEL_82;
      ++v78;
    }
    while ( v59 != v78 );
  }
LABEL_73:
  v336 = (__int64 *)v338;
  v337 = 0x1000000000LL;
  v345 = v347;
  v346 = 0x800000000LL;
  if ( a2 == v298 )
    goto LABEL_114;
  v303 = v6;
  v62 = a2;
  while ( 2 )
  {
    while ( 2 )
    {
      v68 = *(_QWORD *)(v62 + 8) & 0xFFFFFFFFFFFFFFF8LL;
      v69 = v34 & 1;
      if ( (v34 & 1) != 0 )
      {
        v63 = &v356;
        v64 = 3;
        goto LABEL_76;
      }
      v63 = (__int64 **)v356;
      if ( !v357 )
      {
LABEL_78:
        v62 += 16;
        if ( v62 == v298 )
          goto LABEL_113;
        continue;
      }
      break;
    }
    v64 = v357 - 1;
LABEL_76:
    v65 = v64 & (((unsigned int)v68 >> 9) ^ ((unsigned int)v68 >> 4));
    v66 = v63[17 * v65];
    if ( (__int64 *)v68 != v66 )
    {
      v231 = 1;
      while ( v66 != (__int64 *)-4096LL )
      {
        v65 = v64 & (v231 + v65);
        v66 = v63[17 * v65];
        if ( (__int64 *)v68 == v66 )
          goto LABEL_77;
        ++v231;
      }
      goto LABEL_78;
    }
LABEL_77:
    v67 = *a1;
    if ( sub_D68B40(*a1, *(_QWORD *)(v62 + 8) & 0xFFFFFFFFFFFFFFF8LL) )
      goto LABEL_78;
    v62 += 16;
    v90 = sub_10420D0(v67, v68);
    sub_D68D20((__int64)&v348, 2u, v90);
    sub_D6B260((__int64)&v345, (char *)&v348, v91, v92, v93, v94);
    sub_D68D70(&v348);
    LOBYTE(v34) = v355;
    v69 = v355 & 1;
    if ( v62 != v298 )
      continue;
    break;
  }
LABEL_113:
  v6 = v303;
  v36 = v69;
LABEL_114:
  if ( !((unsigned int)v355 >> 1) )
  {
    if ( v36 )
    {
      v263 = &v356;
      v264 = 68;
    }
    else
    {
      v263 = (__int64 **)v356;
      v264 = 17LL * v357;
    }
    v95 = (__int64 *)&v263[v264];
    v284 = (__int64 *)&v263[v264];
    goto LABEL_120;
  }
  if ( v36 )
  {
    v95 = (__int64 *)&v356;
    v284 = (__int64 *)v358;
    v96 = (__int64 *)v358;
    do
    {
LABEL_117:
      if ( *v95 != -4096 && *v95 != -8192 )
        break;
      v95 += 17;
    }
    while ( v96 != v95 );
LABEL_120:
    if ( !v36 )
    {
      v97 = (__int64 **)v356;
      v98 = v357;
      goto LABEL_122;
    }
    v97 = &v356;
    v99 = 68;
  }
  else
  {
    v98 = v357;
    v97 = (__int64 **)v356;
    v95 = v356;
    v284 = &v356[17 * v357];
    if ( v284 != v356 )
    {
      v96 = &v356[17 * v357];
      goto LABEL_117;
    }
LABEL_122:
    v99 = 17 * v98;
  }
  v275 = (__int64 *)&v97[v99];
  if ( &v97[v99] == (__int64 **)v95 )
    goto LABEL_234;
  v278 = v6;
  while ( 2 )
  {
    v100 = *v95;
    v348 = 0;
    v349 = 1;
    v304 = v100;
    v101 = (__int64 *)&v350;
    do
    {
      *v101 = -4096;
      v101 += 2;
    }
    while ( v101 != (__int64 *)v353 );
    v102 = (__int64 **)v95[5];
    v103 = &v102[*((unsigned int *)v95 + 12)];
    if ( v103 != v102 )
    {
      while ( 2 )
      {
        while ( 2 )
        {
          v329 = *v102;
          v109 = sub_D6A020((__int64)v309, (__int64)v329);
          if ( (v349 & 1) != 0 )
          {
            v104 = (__int64 *)&v350;
            v105 = 3;
            goto LABEL_130;
          }
          v110 = v351;
          v104 = v350;
          if ( !v351 )
          {
            v111 = v349;
            v348 = (__int64 *)((char *)v348 + 1);
            v339 = 0;
            v112 = ((unsigned int)v349 >> 1) + 1;
            goto LABEL_136;
          }
          v105 = v351 - 1;
LABEL_130:
          v106 = v105 & (((unsigned int)v329 >> 9) ^ ((unsigned int)v329 >> 4));
          v107 = &v104[2 * v106];
          v108 = (_QWORD *)*v107;
          if ( (__int64 *)*v107 == v329 )
          {
LABEL_131:
            ++v102;
            v107[1] = v109;
            if ( v103 == v102 )
              goto LABEL_142;
            continue;
          }
          break;
        }
        v200 = 1;
        v201 = 0;
        while ( v108 != (_QWORD *)-4096LL )
        {
          if ( !v201 && v108 == (_QWORD *)-8192LL )
            v201 = v107;
          v106 = v105 & (v200 + v106);
          v107 = &v104[2 * v106];
          v108 = (_QWORD *)*v107;
          if ( v329 == (__int64 *)*v107 )
            goto LABEL_131;
          ++v200;
        }
        v113 = 12;
        v110 = 4;
        if ( !v201 )
          v201 = v107;
        v111 = v349;
        v348 = (__int64 *)((char *)v348 + 1);
        v339 = v201;
        v112 = ((unsigned int)v349 >> 1) + 1;
        if ( (v349 & 1) == 0 )
        {
          v110 = v351;
LABEL_136:
          v113 = 3 * v110;
        }
        if ( v113 <= 4 * v112 )
        {
          v110 *= 2;
        }
        else if ( v110 - HIDWORD(v349) - v112 > v110 >> 3 )
        {
          goto LABEL_139;
        }
        sub_D6C020((__int64)&v348, v110);
        sub_D6AFA0((__int64)&v348, (__int64 *)&v329, &v339);
        v111 = v349;
LABEL_139:
        LODWORD(v349) = (2 * (v111 >> 1) + 2) | v111 & 1;
        if ( *v339 != -4096 )
          --HIDWORD(v349);
        v114 = v329;
        v115 = v339 + 1;
        v339[1] = 0;
        ++v102;
        *(v115 - 1) = (unsigned __int64)v114;
        *v115 = v109;
        if ( v103 == v102 )
          break;
        continue;
      }
    }
LABEL_142:
    v116 = sub_D68B40(*a1, v304);
    if ( (*(_DWORD *)(v116 + 4) & 0x7FFFFFF) != 0 )
    {
      v117 = (__int64 **)v95[5];
      v285 = &v117[*((unsigned int *)v95 + 12)];
      if ( v285 == v117 )
        goto LABEL_166;
      v280 = v95;
      while ( 2 )
      {
        v118 = *v117;
        v329 = *v117;
        if ( (v349 & 1) != 0 )
        {
          v119 = (__int64 *)&v350;
          v120 = 3;
          goto LABEL_147;
        }
        v149 = v351;
        v119 = v350;
        if ( !v351 )
        {
          v159 = v349;
          v348 = (__int64 *)((char *)v348 + 1);
          v339 = 0;
          v160 = ((unsigned int)v349 >> 1) + 1;
          goto LABEL_203;
        }
        v120 = v351 - 1;
LABEL_147:
        v121 = v120 & (((unsigned int)v118 >> 9) ^ ((unsigned int)v118 >> 4));
        v122 = &v119[2 * v121];
        v123 = (_QWORD *)*v122;
        if ( v118 == (__int64 *)*v122 )
        {
LABEL_148:
          v124 = v122[1];
          goto LABEL_149;
        }
        v203 = 1;
        v204 = 0;
        while ( v123 != (_QWORD *)-4096LL )
        {
          if ( !v204 && v123 == (_QWORD *)-8192LL )
            v204 = v122;
          v121 = v120 & (v203 + v121);
          v122 = &v119[2 * v121];
          v123 = (_QWORD *)*v122;
          if ( v118 == (__int64 *)*v122 )
            goto LABEL_148;
          ++v203;
        }
        v161 = 12;
        v149 = 4;
        if ( !v204 )
          v204 = v122;
        v159 = v349;
        v348 = (__int64 *)((char *)v348 + 1);
        v339 = v204;
        v160 = ((unsigned int)v349 >> 1) + 1;
        if ( (v349 & 1) == 0 )
        {
          v149 = v351;
LABEL_203:
          v161 = 3 * v149;
        }
        if ( 4 * v160 >= v161 )
        {
          v149 *= 2;
        }
        else if ( v149 - HIDWORD(v349) - v160 > v149 >> 3 )
        {
LABEL_206:
          LODWORD(v349) = (2 * (v159 >> 1) + 2) | v159 & 1;
          v162 = v339;
          if ( *v339 != -4096 )
            --HIDWORD(v349);
          *v339 = v118;
          v124 = 0;
          v162[1] = 0;
          v118 = v329;
LABEL_149:
          v339 = v118;
          v125 = 0;
          v340 = (char *)v304;
          v126 = *(_DWORD *)sub_B1DDD0((__int64)&v332, (__int64 *)&v339);
          if ( v126 > 0 )
          {
            v295 = v117;
            v127 = v116;
            v128 = v126;
            do
            {
              v133 = v329;
              v134 = *(_DWORD *)(v127 + 4) & 0x7FFFFFF;
              if ( v134 == *(_DWORD *)(v127 + 76) )
              {
                v135 = (v134 >> 1) + v134;
                if ( v135 < 2 )
                  v135 = 2;
                *(_DWORD *)(v127 + 76) = v135;
                sub_BD2A80(v127, v135, 1);
                v134 = *(_DWORD *)(v127 + 4) & 0x7FFFFFF;
              }
              v129 = (v134 + 1) & 0x7FFFFFF;
              *(_DWORD *)(v127 + 4) = v129 | *(_DWORD *)(v127 + 4) & 0xF8000000;
              v130 = *(_QWORD *)(v127 - 8) + 32LL * (unsigned int)(v129 - 1);
              if ( *(_QWORD *)v130 )
              {
                v131 = *(_QWORD *)(v130 + 8);
                **(_QWORD **)(v130 + 16) = v131;
                if ( v131 )
                  *(_QWORD *)(v131 + 16) = *(_QWORD *)(v130 + 16);
              }
              *(_QWORD *)v130 = v124;
              if ( v124 )
              {
                v132 = *(_QWORD *)(v124 + 16);
                *(_QWORD *)(v130 + 8) = v132;
                if ( v132 )
                  *(_QWORD *)(v132 + 16) = v130 + 8;
                *(_QWORD *)(v130 + 16) = v124 + 16;
                *(_QWORD *)(v124 + 16) = v130;
              }
              ++v125;
              *(_QWORD *)(*(_QWORD *)(v127 - 8)
                        + 32LL * *(unsigned int *)(v127 + 76)
                        + 8LL * ((*(_DWORD *)(v127 + 4) & 0x7FFFFFFu) - 1)) = v133;
            }
            while ( v128 != v125 );
            v116 = v127;
            v117 = v295;
          }
          if ( v285 == ++v117 )
            goto LABEL_165;
          continue;
        }
        break;
      }
      sub_D6C020((__int64)&v348, v149);
      sub_D6AFA0((__int64)&v348, (__int64 *)&v329, &v339);
      v118 = v329;
      v159 = v349;
      goto LABEL_206;
    }
    v163 = sub_D6A020((__int64)v309, *(_QWORD *)v95[13]);
    if ( !((unsigned int)v349 >> 1) )
    {
      v164 = v349 & 1;
      if ( (v349 & 1) != 0 )
      {
        v228 = (__int64 *)&v350;
        v229 = 8;
      }
      else
      {
        v228 = v350;
        v229 = 2LL * v351;
      }
      v167 = &v228[v229];
      v166 = v167;
      goto LABEL_216;
    }
    v164 = v349 & 1;
    if ( (v349 & 1) != 0 )
    {
      v165 = (__int64 *)v353;
      v166 = (__int64 *)&v350;
LABEL_212:
      while ( *v166 == -8192 || *v166 == -4096 )
      {
        v166 += 2;
        if ( v165 == v166 )
        {
          v167 = v166;
          goto LABEL_216;
        }
      }
      v167 = v166;
      v166 = v165;
LABEL_216:
      if ( !v164 )
      {
        v168 = v350;
        v169 = v351;
        goto LABEL_218;
      }
      v168 = (__int64 *)&v350;
      v170 = 8;
    }
    else
    {
      v169 = v351;
      v168 = v350;
      v166 = v350;
      v165 = &v350[2 * v351];
      if ( v165 != v350 )
        goto LABEL_212;
      v167 = v350;
LABEL_218:
      v170 = 2 * v169;
    }
    v171 = &v168[v170];
    if ( v167 == v171 )
    {
LABEL_225:
      sub_BD84D0(v116, v163);
      sub_D6E4B0(a1, v116, 0, v172, v173, v174);
      goto LABEL_226;
    }
    while ( v167[1] == v163 )
    {
      do
        v167 += 2;
      while ( v166 != v167 && (*v167 == -8192 || *v167 == -4096) );
      if ( v167 == v171 )
        goto LABEL_225;
    }
    v205 = (_QWORD **)v95[5];
    v281 = &v205[*((unsigned int *)v95 + 12)];
    if ( v281 == v205 )
      goto LABEL_302;
    v272 = v163;
    v206 = (_QWORD **)v95[5];
    v270 = v95;
    while ( 2 )
    {
      v322 = *v206;
      v193 = (unsigned __int8)sub_D6AFA0((__int64)&v348, (__int64 *)&v322, &v329) == 0;
      v207 = v329;
      if ( v193 )
      {
        v348 = (__int64 *)((char *)v348 + 1);
        v339 = v329;
        v208 = ((unsigned int)v349 >> 1) + 1;
        if ( (v349 & 1) == 0 )
        {
          v209 = v351;
          if ( 3 * v351 > 4 * v208 )
            goto LABEL_294;
LABEL_309:
          v209 *= 2;
          goto LABEL_310;
        }
        v209 = 4;
        if ( 4 * v208 >= 0xC )
          goto LABEL_309;
LABEL_294:
        if ( v209 - (v208 + HIDWORD(v349)) <= v209 >> 3 )
        {
LABEL_310:
          sub_D6C020((__int64)&v348, v209);
          sub_D6AFA0((__int64)&v348, (__int64 *)&v322, &v339);
          v207 = v339;
          v208 = ((unsigned int)v349 >> 1) + 1;
        }
        LODWORD(v349) = v349 & 1 | (2 * v208);
        if ( *v207 != -4096 )
          --HIDWORD(v349);
        v210 = v322;
        v207[1] = 0;
        *v207 = (__int64)v210;
      }
      v211 = v207[1];
      v212 = 0;
      v339 = v322;
      v340 = (char *)v304;
      v213 = *(_DWORD *)sub_B1DDD0((__int64)&v332, (__int64 *)&v339);
      if ( v213 > 0 )
      {
        do
        {
          ++v212;
          sub_D689D0(v116, v211, (__int64)v322);
        }
        while ( v213 != v212 );
      }
      if ( v281 != ++v206 )
        continue;
      break;
    }
    v163 = v272;
    v95 = v270;
LABEL_302:
    v214 = (__int64 *)v95[13];
    v287 = &v214[*((unsigned int *)v95 + 28)];
    if ( v287 == v214 )
    {
      v138 = *v287;
      v139 = *(_DWORD *)(v278 + 32);
    }
    else
    {
      v296 = (__int64 *)v95[13];
      v280 = v95;
      do
      {
        v215 = 0;
        v216 = (_QWORD *)*v296;
        v339 = (_QWORD *)*v296;
        v340 = (char *)v304;
        v217 = *(_DWORD *)sub_B1DDD0((__int64)&v332, (__int64 *)&v339);
        if ( v217 > 0 )
        {
          do
          {
            ++v215;
            sub_D689D0(v116, v163, (__int64)v216);
          }
          while ( v217 != v215 );
        }
        ++v296;
      }
      while ( v287 != v296 );
LABEL_165:
      v95 = v280;
LABEL_166:
      v136 = (__int64 *)v95[13];
      v137 = &v136[*((unsigned int *)v95 + 28)];
      v138 = *v136;
      v139 = *(_DWORD *)(v278 + 32);
      if ( v136 != v137 )
      {
        v140 = v136 + 1;
        for ( j = v138; ; j = *v140++ )
        {
          v142 = *(_QWORD *)(*(_QWORD *)(v138 + 72) + 80LL);
          if ( v142 )
            v142 -= 24;
          if ( v138 == v142 || v142 == j )
          {
            v138 = v142;
            if ( v137 == v140 )
              break;
          }
          else
          {
            v143 = 0;
            v144 = (unsigned int)(*(_DWORD *)(v138 + 44) + 1);
            if ( (unsigned int)v144 < v139 )
              v143 = *(_QWORD *)(*(_QWORD *)(v278 + 24) + 8 * v144);
            if ( j )
            {
              v145 = (unsigned int)(*(_DWORD *)(j + 44) + 1);
              v146 = *(_DWORD *)(j + 44) + 1;
            }
            else
            {
              v145 = 0;
              v146 = 0;
            }
            v147 = 0;
            if ( v146 < v139 )
              v147 = *(_QWORD *)(*(_QWORD *)(v278 + 24) + 8 * v145);
            for ( ; v143 != v147; v143 = *(_QWORD *)(v143 + 8) )
            {
              if ( *(_DWORD *)(v143 + 16) < *(_DWORD *)(v147 + 16) )
              {
                v148 = v143;
                v143 = v147;
                v147 = v148;
              }
            }
            v138 = *(_QWORD *)v147;
            if ( v137 == v140 )
              break;
          }
        }
      }
    }
    if ( v304 )
    {
      v150 = (unsigned int)(*(_DWORD *)(v304 + 44) + 1);
      v151 = *(_DWORD *)(v304 + 44) + 1;
    }
    else
    {
      v150 = 0;
      v151 = 0;
    }
    if ( v151 >= v139 )
      BUG();
    v152 = **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v278 + 24) + 8 * v150) + 8LL);
    if ( v152 != v138 )
    {
      sub_B1A4E0((__int64)&v336, v138);
      while ( 1 )
      {
        if ( v138 )
        {
          v155 = (unsigned int)(*(_DWORD *)(v138 + 44) + 1);
          v156 = *(_DWORD *)(v138 + 44) + 1;
        }
        else
        {
          v155 = 0;
          v156 = 0;
        }
        if ( v156 >= *(_DWORD *)(v278 + 32) )
          BUG();
        v138 = **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v278 + 24) + 8 * v155) + 8LL);
        if ( v152 == v138 || !v138 )
          break;
        v157 = (unsigned int)v337;
        v158 = (unsigned int)v337 + 1LL;
        if ( v158 > HIDWORD(v337) )
        {
          sub_C8D5F0((__int64)&v336, v338, v158, 8u, v153, v154);
          v157 = (unsigned int)v337;
        }
        v336[v157] = v138;
        LODWORD(v337) = v337 + 1;
      }
    }
LABEL_226:
    if ( (v349 & 1) == 0 )
      sub_C7D6A0((__int64)v350, 16LL * v351, 8);
    for ( v95 += 17; v95 != v284; v95 += 17 )
    {
      if ( *v95 != -8192 && *v95 != -4096 )
        break;
    }
    if ( v95 != v275 )
      continue;
    break;
  }
  v6 = v278;
LABEL_234:
  sub_D6FF00((__int64)a1, (__int64)v345, (unsigned int)v346);
  v177 = (__int64)v345;
  v319 = (__int64 **)v321;
  v320 = 0x800000000LL;
  v178 = &v345[24 * (unsigned int)v346];
  if ( v178 == v345 )
  {
    v348 = (__int64 *)&v350;
    v349 = 0x2000000000LL;
    goto LABEL_239;
  }
  do
  {
    v179 = *(_QWORD *)(v177 + 16);
    if ( v179 )
      sub_B1A4E0((__int64)&v319, *(_QWORD *)(v179 + 64));
    v177 += 24;
  }
  while ( (_BYTE *)v177 != v178 );
  v180 = (__int64 *)&v350;
  v349 = 0x2000000000LL;
  v348 = (__int64 *)&v350;
  if ( (_DWORD)v320 )
  {
    v233 = v319;
    v310[0] = v6;
    v234 = &v319[(unsigned int)v320];
    v310[1] = v307;
    v311 = 0;
    v339 = 0;
    v340 = &v344;
    v341 = 16;
    v342 = 0;
    v343 = 1;
    do
    {
      v235 = *v233++;
      sub_D695C0((__int64)&v329, (__int64)&v339, v235, (__int64)v180, v175, v176);
    }
    while ( v234 != v233 );
    v312 = &v339;
    sub_D6A180((__int64)v310, (__int64)&v348);
    v236 = v348;
    v322 = 0;
    v326 = v328;
    v327 = 0x400000000LL;
    v323 = 0;
    v324 = 0;
    v237 = &v348[(unsigned int)v349];
    v325 = 0;
    if ( v237 != v348 )
    {
      v300 = v6;
      do
      {
        while ( 1 )
        {
          v238 = *v236;
          v239 = *a1;
          v240 = *v236;
          if ( !sub_D68B40(*a1, *v236) )
            break;
          if ( v237 == ++v236 )
            goto LABEL_392;
        }
        ++v236;
        v308 = sub_10420D0(v239, v238);
        sub_D68D20((__int64)&v329, 2u, v308);
        sub_D6B260((__int64)&v345, (char *)&v329, v241, v242, v243, v244);
        sub_D68D70(&v329);
        v240 = (__int64)&v308;
        sub_D6CE50((__int64)&v322, &v308, v245, v246, v247, v248);
      }
      while ( v237 != v236 );
LABEL_392:
      v6 = v300;
      v301 = &v348[(unsigned int)v349];
      if ( v301 != v348 )
      {
        v306 = v348;
        v288 = v6;
        while ( 1 )
        {
          v249 = *v306;
          v240 = *v306;
          v253 = sub_D68B40(*a1, *v306);
          if ( !(_DWORD)v324 )
            break;
          if ( (_DWORD)v325 )
          {
            v265 = (v325 - 1) & (((unsigned int)v253 >> 9) ^ ((unsigned int)v253 >> 4));
            v240 = *(_QWORD *)(v323 + 8LL * v265);
            if ( v253 == v240 )
            {
LABEL_417:
              v240 = (__int64)v307;
              sub_D69DD0((__int64)&v329, (__int64)v307, v249, v250, v251, v252);
              v266 = v329;
              v267 = &v329[v330];
              if ( v267 != v329 )
              {
                v268 = v329;
                do
                {
                  v269 = *v268++;
                  v240 = sub_D6A020((__int64)v309, v269);
                  sub_D689D0(v253, v240, v269);
                }
                while ( v267 != v268 );
                v266 = v329;
              }
              if ( v266 != (__int64 *)&v331 )
                _libc_free(v266, v240);
              goto LABEL_406;
            }
            v251 = 1;
            while ( v240 != -4096 )
            {
              v250 = (unsigned int)(v251 + 1);
              v265 = (v325 - 1) & (v251 + v265);
              v240 = *(_QWORD *)(v323 + 8LL * v265);
              if ( v253 == v240 )
                goto LABEL_417;
              v251 = (unsigned int)v250;
            }
          }
LABEL_403:
          if ( (*(_DWORD *)(v253 + 4) & 0x7FFFFFF) != 0 )
          {
            v259 = 0;
            v260 = 8LL * (*(_DWORD *)(v253 + 4) & 0x7FFFFFF);
            do
            {
              v240 = sub_D6A020(
                       (__int64)v309,
                       *(_QWORD *)(*(_QWORD *)(v253 - 8) + 32LL * *(unsigned int *)(v253 + 76) + v259));
              v261 = *(_QWORD *)(v253 - 8) + 4 * v259;
              v259 += 8;
              sub_AC2B30(v261, v240);
            }
            while ( v259 != v260 );
          }
LABEL_406:
          if ( v301 == ++v306 )
          {
            v6 = v288;
            goto LABEL_408;
          }
        }
        v254 = v326;
        v255 = 8LL * (unsigned int)v327;
        v256 = &v326[v255];
        v240 = v255 >> 3;
        v257 = v255 >> 5;
        if ( v257 )
        {
          v258 = &v326[32 * v257];
          while ( v253 != *v254 )
          {
            if ( v253 == v254[1] )
            {
              ++v254;
              goto LABEL_402;
            }
            if ( v253 == v254[2] )
            {
              v254 += 2;
              goto LABEL_402;
            }
            if ( v253 == v254[3] )
            {
              v254 += 3;
              goto LABEL_402;
            }
            v254 += 4;
            if ( v258 == v254 )
            {
              v240 = v256 - v254;
              goto LABEL_424;
            }
          }
          goto LABEL_402;
        }
LABEL_424:
        if ( v240 != 2 )
        {
          if ( v240 != 3 )
          {
            if ( v240 != 1 || v253 != *v254 )
              goto LABEL_403;
            goto LABEL_402;
          }
          if ( v253 == *v254 )
          {
LABEL_402:
            if ( v256 != v254 )
              goto LABEL_417;
            goto LABEL_403;
          }
          ++v254;
        }
        if ( v253 != *v254 && v253 != *++v254 )
          goto LABEL_403;
        goto LABEL_402;
      }
LABEL_408:
      if ( v326 != v328 )
        _libc_free(v326, v240);
    }
    v262 = 8LL * (unsigned int)v325;
    sub_C7D6A0(v323, v262, 8);
    if ( !v343 )
      _libc_free(v340, v262);
  }
LABEL_239:
  v289 = v336;
  v286 = &v336[(unsigned int)v337];
  if ( v286 != v336 )
  {
    while ( 1 )
    {
      v293 = sub_D68C20(*a1, *v289);
      if ( v293 )
      {
        v299 = *(_QWORD *)(v293 + 8);
        if ( v293 != v299 )
          break;
      }
LABEL_241:
      if ( v286 == ++v289 )
        goto LABEL_314;
    }
    while ( 1 )
    {
      if ( !v299 )
        BUG();
      v305 = *(_QWORD *)(v299 + 16);
      v181 = *(_QWORD *)(v299 - 32);
      if ( v181 )
        break;
LABEL_260:
      v299 = *(_QWORD *)(v299 + 8);
      if ( v293 == v299 )
        goto LABEL_241;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        v183 = v181;
        v181 = *(_QWORD *)(v181 + 8);
        v184 = *(_QWORD *)(v183 + 24);
        if ( *(_BYTE *)v184 == 28 )
        {
          v182 = *(_QWORD *)(*(_QWORD *)(v184 - 8)
                           + 32LL * *(unsigned int *)(v184 + 76)
                           + 8LL * (unsigned int)((v183 - *(_QWORD *)(v184 - 8)) >> 5));
          if ( !(unsigned __int8)sub_B19720(v6, v305, v182) )
          {
            v195 = sub_D6A020((__int64)v309, v182);
            sub_AC2B30(v183, v195);
          }
          goto LABEL_249;
        }
        v185 = *(_QWORD *)(v184 + 64);
        if ( !(unsigned __int8)sub_B19720(v6, v305, v185) )
          break;
LABEL_249:
        if ( !v181 )
          goto LABEL_260;
      }
      v186 = *(_DWORD *)(*a1 + 56);
      v187 = *(_QWORD *)(*a1 + 40);
      if ( !v186 )
        goto LABEL_264;
      v188 = v186 - 1;
      v189 = v188 & (((unsigned int)v185 >> 9) ^ ((unsigned int)v185 >> 4));
      v190 = (__int64 *)(v187 + 16LL * v189);
      v191 = *v190;
      if ( v185 != *v190 )
        break;
LABEL_254:
      v192 = v190[1];
      if ( !v192 )
        goto LABEL_264;
      sub_AC2B30(v183, v192);
      if ( *(_BYTE *)v184 != 27 )
        goto LABEL_268;
LABEL_256:
      v193 = *(_QWORD *)(v184 - 32) == 0;
      *(_DWORD *)(v184 + 84) = -1;
      if ( !v193 )
      {
        v194 = *(_QWORD *)(v184 - 24);
        **(_QWORD **)(v184 - 16) = v194;
        if ( v194 )
          *(_QWORD *)(v194 + 16) = *(_QWORD *)(v184 - 16);
      }
      *(_QWORD *)(v184 - 32) = 0;
      if ( !v181 )
        goto LABEL_260;
    }
    v196 = 1;
    while ( v191 != -4096 )
    {
      v202 = v196 + 1;
      v189 = v188 & (v196 + v189);
      v190 = (__int64 *)(v187 + 16LL * v189);
      v191 = *v190;
      if ( v185 == *v190 )
        goto LABEL_254;
      v196 = v202;
    }
LABEL_264:
    if ( v185 )
    {
      v197 = (unsigned int)(*(_DWORD *)(v185 + 44) + 1);
      v198 = *(_DWORD *)(v185 + 44) + 1;
    }
    else
    {
      v197 = 0;
      v198 = 0;
    }
    if ( v198 >= *(_DWORD *)(v6 + 32) )
      BUG();
    v199 = sub_D6A020((__int64)v309, **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v6 + 24) + 8 * v197) + 8LL));
    sub_AC2B30(v183, v199);
    if ( *(_BYTE *)v184 == 27 )
      goto LABEL_256;
LABEL_268:
    *(_DWORD *)(v184 + 80) = -1;
    goto LABEL_249;
  }
LABEL_314:
  v218 = (__int64)v345;
  sub_D6FF00((__int64)a1, (__int64)v345, (unsigned int)v346);
  if ( v348 != (__int64 *)&v350 )
    _libc_free(v348, v218);
  if ( v319 != (__int64 **)v321 )
    _libc_free(v319, v218);
  v219 = (__int64)v345;
  v220 = &v345[24 * (unsigned int)v346];
  if ( v345 != (_BYTE *)v220 )
  {
    do
    {
      v221 = *(v220 - 1);
      v220 -= 3;
      if ( v221 != -4096 && v221 != 0 && v221 != -8192 )
        sub_BD60C0(v220);
    }
    while ( (_QWORD *)v219 != v220 );
    v220 = v345;
  }
  if ( v220 != (_QWORD *)v347 )
    _libc_free(v220, v218);
  if ( v336 != (__int64 *)v338 )
    _libc_free(v336, v218);
  if ( !v317 )
    _libc_free(v314, v218);
  if ( (v333 & 1) == 0 )
  {
    v218 = 24LL * v335;
    sub_C7D6A0(v334, v218, 8);
  }
  if ( (v355 & 1) != 0 )
  {
    v223 = (__int64 *)v358;
    v222 = (__int64 *)&v356;
    goto LABEL_335;
  }
  v222 = v356;
  v218 = 136LL * v357;
  if ( !v357 )
    return sub_C7D6A0((__int64)v222, v218, 8);
  v223 = (__int64 *)((char *)v356 + v218);
  if ( (__int64 *)((char *)v356 + v218) == v356 )
    return sub_C7D6A0((__int64)v222, v218, 8);
  do
  {
LABEL_335:
    result = *v222;
    if ( *v222 != -8192 && result != -4096 )
    {
      v225 = (__int64 *)v222[13];
      if ( v225 != v222 + 15 )
        _libc_free(v225, v218);
      v226 = 8LL * *((unsigned int *)v222 + 24);
      sub_C7D6A0(v222[10], v226, 8);
      v227 = (__int64 *)v222[5];
      if ( v227 != v222 + 7 )
        _libc_free(v227, v226);
      v218 = 8LL * *((unsigned int *)v222 + 8);
      result = sub_C7D6A0(v222[2], v218, 8);
    }
    v222 += 17;
  }
  while ( v222 != v223 );
  if ( (v355 & 1) == 0 )
  {
    v222 = v356;
    v218 = 136LL * v357;
    return sub_C7D6A0((__int64)v222, v218, 8);
  }
  return result;
}
