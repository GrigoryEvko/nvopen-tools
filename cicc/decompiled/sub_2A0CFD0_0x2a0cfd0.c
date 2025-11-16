// Function: sub_2A0CFD0
// Address: 0x2a0cfd0
//
_BOOL8 __fastcall sub_2A0CFD0(unsigned int *a1, __int64 a2, char a3)
{
  __int64 *v3; // rax
  _QWORD *v4; // rbx
  unsigned __int64 v5; // r15
  int v6; // edi
  __int64 *v7; // r12
  __int64 v8; // r13
  unsigned int v9; // r15d
  __int64 v10; // rsi
  _QWORD *v11; // rax
  _QWORD *v12; // rdx
  unsigned __int64 v14; // r15
  __int64 *v15; // r12
  unsigned int v16; // r15d
  int v17; // r14d
  __int64 v18; // rsi
  _QWORD *v19; // rax
  _QWORD *v20; // rdx
  __int64 v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // r8
  __int64 v24; // rdi
  __int64 v25; // rbx
  __int64 *v26; // rax
  __int64 *v27; // rdx
  _QWORD *v28; // rax
  _QWORD *v29; // rdx
  char v30; // cl
  _QWORD *v31; // rax
  __int64 v32; // r8
  __int64 v33; // r9
  _QWORD *v34; // rdx
  char v35; // cl
  __int64 j; // r12
  __int64 v37; // rsi
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  unsigned __int64 v41; // rax
  int v42; // edx
  unsigned __int64 v43; // rax
  bool v44; // cf
  _QWORD *v45; // rdx
  void **v46; // rax
  _QWORD *v47; // rax
  _QWORD *v48; // rbx
  _QWORD *v49; // r13
  __int64 v50; // rdi
  __int64 v51; // rdx
  __int64 v52; // r13
  __int64 v53; // r12
  __int64 v54; // r15
  __int64 v55; // rax
  __int64 v56; // rbx
  __int64 v57; // rax
  __int64 v58; // rax
  unsigned __int64 v59; // rdx
  __int64 v60; // rax
  __int64 v61; // rdx
  __int64 v62; // rcx
  __int64 v63; // rdi
  __int64 v64; // rax
  __int64 *v65; // r15
  __int64 *v66; // r14
  __int64 v67; // rbx
  _QWORD *v68; // r13
  __int64 v69; // rsi
  int v70; // edx
  __int64 v71; // r12
  __int64 v72; // rax
  __int64 v73; // rdx
  __int64 v74; // rcx
  __int64 v75; // r8
  __int64 v76; // r9
  unsigned __int64 v77; // rdx
  __int64 v78; // rcx
  __int64 *v79; // r8
  __int64 *v80; // r9
  __int64 v81; // rax
  __int64 v82; // rdx
  __int64 v83; // r9
  __int64 v84; // rdi
  __int64 v85; // rax
  int v86; // esi
  __int64 v87; // r8
  int v88; // esi
  unsigned int v89; // ecx
  __int64 *v90; // rax
  __int64 v91; // r11
  _QWORD *v92; // rcx
  unsigned int v93; // r9d
  __int64 *v94; // rax
  __int64 v95; // r11
  _QWORD *v96; // rax
  const char *v97; // rax
  unsigned __int64 v98; // rdx
  __int64 v99; // r8
  __int64 *v100; // r9
  unsigned __int64 v101; // rax
  __int64 v102; // r12
  unsigned int kk; // r15d
  __int64 mm; // rbx
  __int64 v105; // rax
  unsigned int v106; // edi
  __int64 v107; // rcx
  int v108; // edx
  __int64 v109; // rax
  __int64 v110; // rsi
  int v111; // edx
  __int64 v112; // rax
  __int64 v113; // rdx
  __int64 v114; // rdx
  _QWORD *v115; // rcx
  __int64 v116; // rax
  _QWORD *v117; // rdx
  __int64 v118; // r12
  __int64 *v119; // rdi
  unsigned __int64 v120; // rdi
  unsigned __int64 v121; // rbx
  __int64 v122; // rax
  char v123; // cl
  unsigned __int64 v124; // rax
  unsigned __int64 v125; // rdx
  unsigned __int64 v126; // rcx
  unsigned __int64 v127; // rax
  int v128; // edx
  __int64 v129; // r13
  __int64 v130; // rax
  unsigned int nn; // r12d
  unsigned __int8 *v132; // r12
  unsigned __int64 v133; // rdx
  __int64 v134; // r8
  __int64 v135; // r9
  __int64 v136; // r12
  __int64 v137; // rcx
  __int64 v138; // rax
  signed __int64 v139; // r13
  __int64 *v140; // rcx
  __int64 v141; // rdx
  __int64 *v142; // rdi
  __int64 *v143; // r13
  __int64 v144; // rax
  __int64 v145; // rbx
  int v146; // esi
  __int64 v147; // rdi
  int v148; // esi
  unsigned int v149; // ecx
  __int64 *v150; // rax
  __int64 v151; // r8
  __int64 v152; // rcx
  _QWORD *v153; // rax
  _QWORD *v154; // rcx
  __int64 v155; // rax
  __int64 **v156; // rsi
  __int64 v157; // r12
  char v158; // al
  _BYTE *v159; // rdx
  __int64 v160; // rcx
  __int64 v161; // r9
  __int64 v162; // r8
  __int64 v163; // rdx
  __int64 v164; // rcx
  __int64 v165; // r8
  __int64 v166; // r9
  _QWORD *v167; // r12
  _QWORD *v168; // r14
  void (__fastcall *v169)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v170; // rax
  unsigned int v171; // eax
  _QWORD *v172; // r12
  __int64 v173; // rax
  _QWORD *v174; // r13
  __int64 v175; // rdx
  __int64 v176; // rax
  unsigned int v177; // eax
  _QWORD *v178; // r12
  __int64 v179; // rax
  _QWORD *v180; // r13
  __int64 v181; // rdx
  __int64 v182; // rax
  __int64 v183; // rsi
  __int64 v184; // rax
  unsigned int v185; // eax
  unsigned __int64 v186; // r8
  __int64 v187; // rcx
  unsigned __int64 v188; // rax
  __int64 v189; // rdi
  __int64 v190; // rdx
  __int64 v191; // r13
  __int64 v192; // rbx
  __int64 v193; // r13
  __int64 v194; // r15
  __int64 v195; // rax
  unsigned int v196; // r13d
  int v197; // eax
  bool v198; // al
  _QWORD *v199; // rax
  _QWORD *v200; // r13
  __int64 *v201; // rsi
  void **v202; // r15
  __int64 v203; // rdi
  __int64 *v204; // rdi
  int v205; // eax
  __int64 v206; // rax
  unsigned int v207; // eax
  __int64 v208; // rcx
  __int64 v209; // rsi
  __int64 v210; // rdi
  int v211; // r10d
  unsigned int n; // eax
  unsigned int v213; // eax
  _QWORD *v214; // rax
  __int64 v215; // rdx
  __int64 v216; // r14
  __int64 v217; // rbx
  __int64 *v218; // r15
  _QWORD *v219; // rcx
  unsigned __int64 v220; // r14
  __int64 ii; // rbx
  unsigned __int64 v222; // r12
  __int64 v223; // r13
  __int64 v224; // rax
  __int64 v225; // rsi
  __int64 *v226; // r9
  int v227; // edi
  int v228; // r11d
  unsigned int jj; // eax
  __int64 *v230; // rdx
  unsigned int v231; // eax
  unsigned __int64 v232; // rax
  __int64 v233; // rax
  __int64 v234; // rcx
  __int64 v235; // rdi
  unsigned __int64 v236; // rax
  int v237; // edx
  __int64 v238; // rdi
  unsigned __int64 v239; // rax
  __int64 v240; // rbx
  unsigned int i1; // r15d
  __int64 v242; // rax
  __int64 v243; // rax
  __int64 v244; // rcx
  __int64 v245; // rsi
  unsigned __int8 *v246; // rsi
  unsigned int v247; // eax
  _QWORD *v248; // rbx
  _QWORD *v249; // r12
  __int64 v250; // rsi
  unsigned int v251; // eax
  _QWORD *v252; // rbx
  _QWORD *v253; // r12
  __int64 v254; // rsi
  int v255; // eax
  int v256; // r10d
  int v257; // eax
  int v258; // edx
  unsigned __int64 *v259; // rbx
  char v260; // dh
  char v261; // al
  char v262; // dl
  __int64 v263; // rax
  _BYTE **v264; // r15
  _BYTE **v265; // r12
  __int64 v266; // r14
  _BYTE *v267; // rdi
  _QWORD *v268; // rax
  __int64 v269; // rax
  __int64 v270; // r9
  __int64 *v271; // r12
  _BYTE *v272; // r13
  _BYTE *v273; // rbx
  __int64 *v274; // rdx
  __int64 v275; // r8
  __int64 v276; // rax
  unsigned __int64 v277; // rdx
  unsigned int v278; // edx
  __int64 *v279; // rdi
  __int64 v280; // rsi
  unsigned __int64 v281; // rax
  __int64 v282; // r9
  __int64 v283; // r13
  unsigned __int64 v284; // rax
  __int64 v285; // rcx
  __int64 v286; // rax
  __int64 v287; // rax
  __int64 v288; // rdi
  __int64 *v289; // rdx
  __int64 v290; // rax
  int v291; // r10d
  __int64 v292; // rax
  __int64 v293; // rdx
  _QWORD *v294; // rbx
  __int64 *v295; // rdx
  _QWORD *v296; // rbx
  __int64 v297; // rdx
  __int64 v298; // r12
  __int64 *v299; // r15
  _QWORD *v300; // rcx
  unsigned __int64 v301; // r14
  __int64 k; // rbx
  __int64 v303; // r12
  __int64 v304; // rax
  __int64 v305; // rsi
  __int64 *v306; // rdi
  int v307; // edx
  int v308; // r9d
  unsigned int m; // eax
  __int64 *v310; // r8
  unsigned int v311; // eax
  __int64 v312; // [rsp-8h] [rbp-638h]
  __int64 v313; // [rsp+8h] [rbp-628h]
  _QWORD *v314; // [rsp+30h] [rbp-600h]
  __int64 v315; // [rsp+38h] [rbp-5F8h]
  __int64 v316; // [rsp+40h] [rbp-5F0h]
  __int64 v317; // [rsp+48h] [rbp-5E8h]
  __int64 v318; // [rsp+50h] [rbp-5E0h]
  __int64 v319; // [rsp+60h] [rbp-5D0h]
  char v320; // [rsp+6Dh] [rbp-5C3h]
  char v321; // [rsp+6Eh] [rbp-5C2h]
  _QWORD *v323; // [rsp+78h] [rbp-5B8h]
  __int64 *v324; // [rsp+80h] [rbp-5B0h]
  unsigned __int64 v325; // [rsp+80h] [rbp-5B0h]
  _QWORD *v326; // [rsp+88h] [rbp-5A8h]
  _QWORD *v327; // [rsp+88h] [rbp-5A8h]
  _QWORD *v328; // [rsp+90h] [rbp-5A0h]
  bool v329; // [rsp+98h] [rbp-598h]
  _QWORD *v330; // [rsp+98h] [rbp-598h]
  unsigned __int64 v331; // [rsp+A0h] [rbp-590h]
  __int64 v332; // [rsp+A8h] [rbp-588h]
  __int64 v333; // [rsp+B0h] [rbp-580h]
  __int64 v335; // [rsp+C8h] [rbp-568h]
  unsigned __int64 v337; // [rsp+D8h] [rbp-558h]
  __int64 v338; // [rsp+D8h] [rbp-558h]
  __int64 v339; // [rsp+D8h] [rbp-558h]
  __int64 v340; // [rsp+E8h] [rbp-548h]
  unsigned __int64 v341; // [rsp+E8h] [rbp-548h]
  int v342; // [rsp+E8h] [rbp-548h]
  __int64 *v343; // [rsp+E8h] [rbp-548h]
  unsigned __int64 v344; // [rsp+E8h] [rbp-548h]
  __int64 v345; // [rsp+E8h] [rbp-548h]
  __int64 v346; // [rsp+E8h] [rbp-548h]
  __int64 *v347; // [rsp+E8h] [rbp-548h]
  __int64 *v348; // [rsp+100h] [rbp-530h] BYREF
  __int64 v349; // [rsp+108h] [rbp-528h]
  _BYTE v350[16]; // [rsp+110h] [rbp-520h] BYREF
  __int64 v351[4]; // [rsp+120h] [rbp-510h] BYREF
  int v352; // [rsp+140h] [rbp-4F0h]
  char v353; // [rsp+144h] [rbp-4ECh]
  _BYTE *v354; // [rsp+150h] [rbp-4E0h] BYREF
  __int64 v355; // [rsp+158h] [rbp-4D8h]
  _BYTE v356[48]; // [rsp+160h] [rbp-4D0h] BYREF
  __m128i v357; // [rsp+190h] [rbp-4A0h] BYREF
  unsigned __int64 v358; // [rsp+1A0h] [rbp-490h] BYREF
  unsigned __int64 v359; // [rsp+1A8h] [rbp-488h]
  __int64 *v360; // [rsp+1B0h] [rbp-480h]
  unsigned __int64 v361; // [rsp+1B8h] [rbp-478h]
  unsigned __int64 v362; // [rsp+1C0h] [rbp-470h]
  __int64 v363; // [rsp+1C8h] [rbp-468h]
  __int64 v364; // [rsp+1D0h] [rbp-460h] BYREF
  _QWORD *v365; // [rsp+1D8h] [rbp-458h]
  __int64 v366; // [rsp+1E0h] [rbp-450h]
  unsigned int v367; // [rsp+1E8h] [rbp-448h]
  _QWORD *v368; // [rsp+1F8h] [rbp-438h]
  unsigned int v369; // [rsp+208h] [rbp-428h]
  char v370; // [rsp+210h] [rbp-420h]
  __int64 v371; // [rsp+220h] [rbp-410h] BYREF
  _QWORD *v372; // [rsp+228h] [rbp-408h]
  __int64 v373; // [rsp+230h] [rbp-400h]
  unsigned int v374; // [rsp+238h] [rbp-3F8h]
  _QWORD *v375; // [rsp+248h] [rbp-3E8h]
  unsigned int v376; // [rsp+258h] [rbp-3D8h]
  char v377; // [rsp+260h] [rbp-3D0h]
  void *v378; // [rsp+270h] [rbp-3C0h] BYREF
  __int64 v379; // [rsp+278h] [rbp-3B8h] BYREF
  __int64 *v380; // [rsp+280h] [rbp-3B0h] BYREF
  __int64 v381; // [rsp+288h] [rbp-3A8h]
  __int64 v382; // [rsp+290h] [rbp-3A0h]
  int v383; // [rsp+298h] [rbp-398h]
  int v384; // [rsp+2A0h] [rbp-390h]
  __int64 v385; // [rsp+2A8h] [rbp-388h]
  __int64 v386; // [rsp+2B0h] [rbp-380h]
  __int64 v387; // [rsp+2B8h] [rbp-378h]
  unsigned int v388; // [rsp+2C0h] [rbp-370h]
  __int64 v389; // [rsp+2C8h] [rbp-368h]
  __int64 v390; // [rsp+2D0h] [rbp-360h]
  __int64 v391; // [rsp+2D8h] [rbp-358h]
  __int64 v392; // [rsp+2E0h] [rbp-350h]
  __int64 v393; // [rsp+2E8h] [rbp-348h]
  __int64 v394; // [rsp+2F0h] [rbp-340h]
  __int64 *v395; // [rsp+340h] [rbp-2F0h] BYREF
  __int64 v396; // [rsp+348h] [rbp-2E8h] BYREF
  __int64 v397; // [rsp+350h] [rbp-2E0h] BYREF
  __int64 v398; // [rsp+358h] [rbp-2D8h]
  __int64 i; // [rsp+360h] [rbp-2D0h] BYREF
  __int64 *v400; // [rsp+368h] [rbp-2C8h]
  __int64 v401; // [rsp+550h] [rbp-E0h]
  __int64 v402; // [rsp+558h] [rbp-D8h]
  __int64 v403; // [rsp+560h] [rbp-D0h]
  __int64 v404; // [rsp+568h] [rbp-C8h]
  char v405; // [rsp+570h] [rbp-C0h]
  __int64 v406; // [rsp+578h] [rbp-B8h]
  char *v407; // [rsp+580h] [rbp-B0h]
  __int64 v408; // [rsp+588h] [rbp-A8h]
  int v409; // [rsp+590h] [rbp-A0h]
  char v410; // [rsp+594h] [rbp-9Ch]
  char v411; // [rsp+598h] [rbp-98h] BYREF
  __int16 v412; // [rsp+5D8h] [rbp-58h]
  _QWORD *v413; // [rsp+5E0h] [rbp-50h]
  _QWORD *v414; // [rsp+5E8h] [rbp-48h]
  __int64 v415; // [rsp+5F0h] [rbp-40h]

  v3 = *(__int64 **)(a2 + 32);
  v329 = 0;
  if ( *(_QWORD *)(a2 + 40) - (_QWORD)v3 != 8 )
  {
    while ( 2 )
    {
      v4 = (_QWORD *)*v3;
      v332 = *v3;
      v340 = sub_D47930(a2);
      v335 = (__int64)(v4 + 6);
      v5 = v4[6] & 0xFFFFFFFFFFFFFFF8LL;
      if ( v4 + 6 == (_QWORD *)v5 )
        goto LABEL_524;
      if ( !v5 )
        goto LABEL_524;
      v317 = v5 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v5 - 24) - 30 > 0xA )
        goto LABEL_524;
      if ( *(_BYTE *)(v5 - 24) != 31 )
        return v329;
      if ( (*(_DWORD *)(v5 - 20) & 0x7FFFFFF) == 1 )
        return v329;
      v378 = v4;
      sub_FDC9F0((__int64)&v395, &v378);
      v6 = v398;
      v7 = v395;
      if ( (_DWORD)v396 == (_DWORD)v398 )
        return v329;
      v337 = v5;
      v8 = a2 + 56;
      v9 = v396;
      while ( 1 )
      {
        v10 = sub_B46EC0((__int64)v7, v9);
        if ( !*(_BYTE *)(a2 + 84) )
          break;
        v11 = *(_QWORD **)(a2 + 64);
        v12 = &v11[*(unsigned int *)(a2 + 76)];
        if ( v11 == v12 )
          goto LABEL_17;
        while ( v10 != *v11 )
        {
          if ( v12 == ++v11 )
            goto LABEL_17;
        }
LABEL_14:
        if ( v6 == ++v9 )
          return v329;
      }
      if ( sub_C8CA60(v8, v10) )
        goto LABEL_14;
LABEL_17:
      v14 = v337;
      if ( !v340 )
        return v329;
      v378 = (void *)v340;
      sub_FDC9F0((__int64)&v395, &v378);
      v15 = v395;
      if ( (_DWORD)v396 == (_DWORD)v398 )
        goto LABEL_27;
      v16 = v396;
      v17 = v398;
      while ( 2 )
      {
        v18 = sub_B46EC0((__int64)v15, v16);
        if ( *(_BYTE *)(a2 + 84) )
        {
          v19 = *(_QWORD **)(a2 + 64);
          v20 = &v19[*(unsigned int *)(a2 + 76)];
          if ( v19 == v20 )
            goto LABEL_65;
          while ( v18 != *v19 )
          {
            if ( v20 == ++v19 )
              goto LABEL_65;
          }
LABEL_25:
          if ( v17 == ++v16 )
          {
            v14 = v337;
            goto LABEL_27;
          }
          continue;
        }
        break;
      }
      if ( sub_C8CA60(v8, v18) )
        goto LABEL_25;
LABEL_65:
      v14 = v337;
      if ( !a3 && !*((_BYTE *)a1 + 65) && !(unsigned __int8)sub_2A0B590(a2) && !sub_2A0B6A0(a2, v18) )
        return v329;
LABEL_27:
      v396 = (__int64)&i;
      v21 = *((_QWORD *)a1 + 3);
      v395 = 0;
      v397 = 32;
      LODWORD(v398) = 0;
      BYTE4(v398) = 1;
      sub_30AB790(a2, v21, &v395);
      v22 = *((_QWORD *)a1 + 2);
      LOWORD(v378) = 0;
      v23 = *((unsigned __int8 *)a1 + 66);
      BYTE2(v378) = 0;
      HIDWORD(v378) = 0;
      LOBYTE(v379) = 0;
      v380 = 0;
      LODWORD(v381) = 0;
      v382 = 0;
      v383 = 0;
      v384 = 0;
      v385 = 0;
      v386 = 0;
      v387 = 0;
      v388 = 0;
      v389 = 0;
      v390 = 0;
      v391 = 0;
      v392 = 0;
      v393 = 0;
      v394 = 0;
      sub_30ABD80(&v378, v332, v22, &v395, v23, 0);
      a3 = BYTE2(v378);
      if ( !BYTE2(v378)
        && !(HIDWORD(v378) | (unsigned int)v381)
        && *a1 >= (__int64)v380
        && (!*((_BYTE *)a1 + 66) || !HIDWORD(v393)) )
      {
        sub_C7D6A0(v386, 24LL * v388, 8);
        if ( !BYTE4(v398) )
          _libc_free(v396);
        v331 = sub_D4B130(a2);
        if ( !v331 )
          return v329;
        v320 = sub_D474B0(a2);
        if ( !v320 )
          return v329;
        v24 = *((_QWORD *)a1 + 5);
        if ( v24 )
        {
          sub_DAC8B0(v24, (_QWORD *)a2);
          sub_D9D700(*((_QWORD *)a1 + 5), 0);
        }
        if ( *((_QWORD *)a1 + 6) && byte_4F8F8E8[0] )
          nullsub_390();
        v25 = *(_QWORD *)(v14 - 56);
        v318 = *(_QWORD *)(v14 - 88);
        v321 = *(_BYTE *)(a2 + 84);
        if ( v321 )
        {
          v26 = *(__int64 **)(a2 + 64);
          v27 = &v26[*(unsigned int *)(a2 + 76)];
          if ( v26 == v27 )
          {
LABEL_72:
            v321 = 0;
LABEL_73:
            v40 = v318;
            v318 = v25;
            v319 = v40;
            sub_F34590(v40, 0);
          }
          else
          {
            while ( v25 != *v26 )
            {
              if ( v27 == ++v26 )
                goto LABEL_72;
            }
            v319 = *v26;
            sub_F34590(*v26, 0);
          }
        }
        else
        {
          if ( !sub_C8CA60(v8, v25) )
            goto LABEL_73;
          v319 = v25;
          v321 = v320;
          sub_F34590(v25, 0);
        }
        v364 = 0;
        v367 = 128;
        v338 = *(_QWORD *)(v332 + 56);
        v28 = (_QWORD *)sub_C7D670(0x2000, 8);
        v366 = 0;
        v365 = v28;
        v396 = 2;
        v29 = &v28[8 * (unsigned __int64)v367];
        v395 = (__int64 *)&unk_49DD7B0;
        v397 = 0;
        v398 = -4096;
        for ( i = 0; v29 != v28; v28 += 8 )
        {
          if ( v28 )
          {
            v30 = v396;
            v28[2] = 0;
            v28[3] = -4096;
            *v28 = &unk_49DD7B0;
            v28[1] = v30 & 6;
            v28[4] = i;
          }
        }
        v370 = 0;
        v371 = 0;
        v374 = 128;
        v31 = (_QWORD *)sub_C7D670(0x2000, 8);
        v373 = 0;
        v372 = v31;
        v396 = 2;
        v34 = &v31[8 * (unsigned __int64)v374];
        v395 = (__int64 *)&unk_49DD7B0;
        v397 = 0;
        v398 = -4096;
        for ( i = 0; v34 != v31; v31 += 8 )
        {
          if ( v31 )
          {
            v35 = v396;
            v31[2] = 0;
            v31[3] = -4096;
            *v31 = &unk_49DD7B0;
            v31[1] = v35 & 6;
            v31[4] = i;
          }
        }
        v377 = 0;
        for ( j = v338; ; j = *(_QWORD *)(j + 8) )
        {
          if ( !j )
            goto LABEL_524;
          if ( *(_BYTE *)(j - 24) != 84 )
            break;
          v37 = *(_QWORD *)(j - 32);
          v38 = 0x1FFFFFFFE0LL;
          if ( (*(_DWORD *)(j - 20) & 0x7FFFFFF) != 0 )
          {
            v39 = 0;
            do
            {
              if ( v331 == *(_QWORD *)(v37 + 32LL * *(unsigned int *)(j + 48) + 8 * v39) )
              {
                v38 = 32 * v39;
                goto LABEL_63;
              }
              ++v39;
            }
            while ( (*(_DWORD *)(j - 20) & 0x7FFFFFF) != (_DWORD)v39 );
            v38 = 0x1FFFFFFFE0LL;
          }
LABEL_63:
          sub_2A0CF40((__int64)&v364, j - 24, *(_QWORD *)(v37 + v38));
        }
        v339 = j;
        v330 = (_QWORD *)(v331 + 48);
        v41 = *(_QWORD *)(v331 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v331 + 48 == v41 )
        {
          v323 = 0;
          goto LABEL_79;
        }
        if ( !v41 )
          goto LABEL_524;
        v42 = *(unsigned __int8 *)(v41 - 24);
        v43 = v41 - 24;
        v44 = (unsigned int)(v42 - 30) < 0xB;
        v45 = 0;
        if ( v44 )
          v45 = (_QWORD *)v43;
        v323 = v45;
LABEL_79:
        v46 = (void **)&v380;
        v378 = 0;
        v379 = 1;
        do
        {
          *v46 = (void *)-1LL;
          v46 += 3;
          *(v46 - 2) = (void *)-4096LL;
          *(v46 - 1) = (void *)-4096LL;
        }
        while ( v46 != (void **)&v395 );
        v47 = (_QWORD *)(*(_QWORD *)(v331 + 48) & 0xFFFFFFFFFFFFFFF8LL);
        v48 = (_QWORD *)(*v47 & 0xFFFFFFFFFFFFFFF8LL);
        if ( v330 != v48 )
        {
          do
          {
            if ( !v48 )
              goto LABEL_524;
            v49 = v48 - 3;
            if ( *((_BYTE *)v48 - 24) != 85 )
              break;
            v184 = *(v48 - 7);
            if ( !v184 || *(_BYTE *)v184 || *(_QWORD *)(v184 + 24) != v48[7] || (*(_BYTE *)(v184 + 33) & 0x20) == 0 )
              break;
            v185 = *(_DWORD *)(v184 + 36);
            if ( v185 > 0x45 )
            {
              if ( v185 != 71 )
                break;
            }
            else if ( v185 <= 0x43 )
            {
              break;
            }
            sub_B58E30(&v395, (__int64)(v48 - 3));
            v348 = (__int64 *)v396;
            v354 = (_BYTE *)v396;
            v351[0] = (__int64)v395;
            v357.m128i_i64[0] = (__int64)v395;
            v186 = sub_2A0C360(v357.m128i_i64, &v354);
            v187 = *((_DWORD *)v48 - 5) & 0x7FFFFFF;
            v188 = *(_QWORD *)(v49[4 * (2 - v187)] + 24LL);
            v357.m128i_i64[1] = *(_QWORD *)(v49[4 * (1 - v187)] + 24LL);
            v357.m128i_i64[0] = v186;
            v358 = v188;
            sub_2A0CDB0((__int64)&v395, (__int64)&v378, v357.m128i_i64);
            v189 = v48[5];
            if ( v189 )
            {
              v191 = sub_B14240(v189);
              if ( v190 != v191 )
              {
                while ( *(_BYTE *)(v191 + 32) )
                {
                  v191 = *(_QWORD *)(v191 + 8);
                  if ( v190 == v191 )
                    goto LABEL_309;
                }
                if ( v190 != v191 )
                {
                  v328 = v48;
                  v192 = v191;
                  v193 = v190;
LABEL_305:
                  sub_B129C0(&v395, v192);
                  v348 = (__int64 *)v396;
                  v354 = (_BYTE *)v396;
                  v351[0] = (__int64)v395;
                  v357.m128i_i64[0] = (__int64)v395;
                  v344 = sub_2A0BE90(v357.m128i_i64, &v354);
                  v194 = sub_B12000(v192 + 72);
                  v195 = sub_B11F60(v192 + 80);
                  v357.m128i_i64[1] = v194;
                  v357.m128i_i64[0] = v344;
                  v358 = v195;
                  sub_2A0CDB0((__int64)&v395, (__int64)&v378, v357.m128i_i64);
                  while ( 1 )
                  {
                    v192 = *(_QWORD *)(v192 + 8);
                    if ( v193 == v192 )
                      break;
                    if ( !*(_BYTE *)(v192 + 32) )
                    {
                      if ( v193 != v192 )
                        goto LABEL_305;
                      break;
                    }
                  }
                  v48 = v328;
                }
              }
            }
LABEL_309:
            v48 = (_QWORD *)(*v48 & 0xFFFFFFFFFFFFFFF8LL);
          }
          while ( v48 != v330 );
          v47 = (_QWORD *)(*(_QWORD *)(v331 + 48) & 0xFFFFFFFFFFFFFFF8LL);
        }
        if ( v330 == v47 )
          goto LABEL_525;
        if ( !v47 )
          goto LABEL_524;
        if ( (unsigned int)*((unsigned __int8 *)v47 - 24) - 30 > 0xA )
LABEL_525:
          BUG();
        v50 = v47[5];
        if ( v50 )
        {
          v52 = sub_B14240(v50);
          if ( v52 != v51 )
          {
            while ( *(_BYTE *)(v52 + 32) )
            {
              v52 = *(_QWORD *)(v52 + 8);
              if ( v52 == v51 )
                goto LABEL_98;
            }
            if ( v52 != v51 )
            {
              v53 = v51;
LABEL_95:
              sub_B129C0(&v395, v52);
              v348 = (__int64 *)v396;
              v354 = (_BYTE *)v396;
              v351[0] = (__int64)v395;
              v357.m128i_i64[0] = (__int64)v395;
              v341 = sub_2A0BE90(v357.m128i_i64, &v354);
              v54 = sub_B12000(v52 + 72);
              v55 = sub_B11F60(v52 + 80);
              v357.m128i_i64[1] = v54;
              v357.m128i_i64[0] = v341;
              v358 = v55;
              sub_2A0CDB0((__int64)&v395, (__int64)&v378, v357.m128i_i64);
              while ( 1 )
              {
                v52 = *(_QWORD *)(v52 + 8);
                if ( v52 == v53 )
                  break;
                if ( !*(_BYTE *)(v52 + 32) )
                {
                  if ( v52 != v53 )
                    goto LABEL_95;
                  break;
                }
              }
            }
          }
        }
LABEL_98:
        v354 = v356;
        v355 = 0x600000000LL;
        v56 = *(_QWORD *)(v332 + 56);
        if ( v56 != v335 )
        {
          while ( v56 )
          {
            if ( *(_BYTE *)(v56 - 24) == 85
              && (v57 = *(_QWORD *)(v56 - 56)) != 0
              && !*(_BYTE *)v57
              && *(_QWORD *)(v57 + 24) == *(_QWORD *)(v56 + 56)
              && (*(_BYTE *)(v57 + 33) & 0x20) != 0
              && *(_DWORD *)(v57 + 36) == 155 )
            {
              v58 = (unsigned int)v355;
              v59 = (unsigned int)v355 + 1LL;
              if ( v59 > HIDWORD(v355) )
              {
                sub_C8D5F0((__int64)&v354, v356, v59, 8u, v32, v33);
                v58 = (unsigned int)v355;
              }
              *(_QWORD *)&v354[8 * v58] = v56 - 24;
              LODWORD(v355) = v355 + 1;
              v56 = *(_QWORD *)(v56 + 8);
              if ( v56 == v335 )
                goto LABEL_111;
            }
            else
            {
              v56 = *(_QWORD *)(v56 + 8);
              if ( v56 == v335 )
                goto LABEL_111;
            }
          }
LABEL_524:
          BUG();
        }
LABEL_111:
        v60 = sub_AA4B30(v332);
        v62 = v335;
        v316 = v60;
        if ( v339 != v335 )
        {
          v63 = *(_QWORD *)(v339 + 40);
          if ( v63 )
          {
            v64 = sub_B14240(v63);
            v65 = (__int64 *)v61;
            v66 = (__int64 *)v64;
          }
          else
          {
            v65 = &qword_4F81430[1];
            v66 = &qword_4F81430[1];
          }
          while ( 1 )
          {
            v67 = v339;
            v68 = (_QWORD *)(v339 - 24);
            v69 = v339 - 24;
            v339 = *(_QWORD *)(v339 + 8);
            if ( !sub_D484B0(a2, v69, v61, v62) )
              break;
            if ( (unsigned __int8)sub_B46420((__int64)v68) )
              break;
            if ( (unsigned __int8)sub_B46490((__int64)v68) )
              break;
            v70 = *(unsigned __int8 *)(v67 - 24);
            if ( (unsigned int)(v70 - 30) <= 0xA )
              break;
            if ( (_BYTE)v70 == 85 )
            {
              v286 = *(_QWORD *)(v67 - 56);
              if ( v286
                && !*(_BYTE *)v286
                && *(_QWORD *)(v286 + 24) == *(_QWORD *)(v67 + 56)
                && (*(_BYTE *)(v286 + 33) & 0x20) != 0
                && (unsigned int)(*(_DWORD *)(v286 + 36) - 68) <= 3 )
              {
                break;
              }
            }
            else if ( (_BYTE)v70 == 60 )
            {
              break;
            }
            v287 = sub_B43CB0((__int64)v68);
            if ( (unsigned __int8)sub_B2D610(v287, 49) )
              break;
            if ( *(_BYTE *)(v323[5] + 40LL) && v65 != v66 )
            {
              v395 = v66;
              LOBYTE(v396) = 1;
              v296 = sub_B43F50((__int64)v323, (__int64)v68, (__int64)v66, 1, 0);
              v298 = v297;
              sub_FC75A0((__int64 *)&v395, (__int64)&v364, 3, 0, 0, 0);
              sub_FCD310((__int64 *)&v395, v316, (__int64)v296, v298);
              sub_FC7680((__int64 *)&v395, v316);
              for ( ; (_QWORD *)v298 != v296; v296 = (_QWORD *)v296[1] )
              {
                if ( !*((_BYTE *)v296 + 32) )
                  break;
              }
              v357.m128i_i64[0] = (__int64)v296;
              v357.m128i_i64[1] = v298;
              BYTE1(v359) = 1;
              v360 = (__int64 *)v298;
              v361 = v298;
              BYTE1(v363) = 1;
              sub_2A0B3A0((__int64)&v395, &v357);
              v299 = v395;
              v327 = v300;
              v301 = v396;
              v347 = v400;
              if ( v400 != v395 )
              {
                while ( 1 )
                {
                  for ( k = v299[1]; v301 != k; k = *(_QWORD *)(k + 8) )
                  {
                    if ( !*(_BYTE *)(k + 32) )
                      break;
                  }
                  sub_B129C0(&v357, (__int64)v299);
                  v348 = (__int64 *)v357.m128i_i64[1];
                  v351[0] = v357.m128i_i64[0];
                  v325 = sub_2A0BE90(v351, v327);
                  v303 = sub_B12000((__int64)(v299 + 9));
                  v304 = sub_B11F60((__int64)(v299 + 10));
                  v305 = v304;
                  if ( (v379 & 1) != 0 )
                  {
                    v306 = (__int64 *)&v380;
                    v307 = 7;
                  }
                  else
                  {
                    v306 = v380;
                    if ( !(_DWORD)v381 )
                      goto LABEL_517;
                    v307 = v381 - 1;
                  }
                  v308 = 1;
                  for ( m = v307
                          & (((0xBF58476D1CE4E5B9LL
                             * (((((0xBF58476D1CE4E5B9LL
                                  * ((v325 << 32) | ((unsigned int)v303 >> 9) ^ ((unsigned int)v303 >> 4))) >> 31)
                                ^ (0xBF58476D1CE4E5B9LL
                                 * ((v325 << 32) | ((unsigned int)v303 >> 9) ^ ((unsigned int)v303 >> 4)))) << 32)
                              | ((unsigned int)v304 >> 9) ^ ((unsigned int)v304 >> 4))) >> 31)
                           ^ (484763065 * (((unsigned int)v304 >> 9) ^ ((unsigned int)v304 >> 4)))); ; m = v307 & v311 )
                  {
                    v310 = &v306[3 * m];
                    if ( *v310 == v325 && v303 == v310[1] && v305 == v310[2] )
                      break;
                    if ( *v310 == -1 && v310[1] == -4096 && v310[2] == -4096 )
                      goto LABEL_517;
                    v311 = v308 + m;
                    ++v308;
                  }
                  sub_B14290(v299);
LABEL_517:
                  if ( v347 == (__int64 *)k )
                    break;
                  v299 = (__int64 *)k;
                }
              }
            }
            if ( !v339 )
              goto LABEL_525;
            v288 = *(_QWORD *)(v339 + 40);
            if ( v288 )
            {
              v66 = (__int64 *)sub_B14240(v288);
              v65 = v289;
            }
            else
            {
              v65 = &qword_4F81430[1];
              v66 = &qword_4F81430[1];
            }
            v290 = v313;
            LOWORD(v290) = 0;
            v313 = v290;
            sub_B444E0(v68, (__int64)(v323 + 3), v290);
LABEL_137:
            v62 = v335;
            if ( v339 == v335 )
              goto LABEL_138;
          }
          v71 = sub_B47F80(v68);
          v72 = v333;
          LOWORD(v72) = 0;
          v333 = v72;
          sub_B44220((_QWORD *)v71, (__int64)(v323 + 3), v72);
          if ( !*(_BYTE *)(v323[5] + 40LL) || v65 == v66 )
          {
LABEL_122:
            sub_FC75A0((__int64 *)&v395, (__int64)&v364, 3, 0, 0, 0);
            sub_FCD280((__int64 *)&v395, (unsigned __int8 *)v71, v73, v74, v75, v76);
            sub_FC7680((__int64 *)&v395, v71);
            if ( *(_BYTE *)v71 == 85 )
            {
              v206 = *(_QWORD *)(v71 - 32);
              if ( v206 )
              {
                if ( !*(_BYTE *)v206 )
                {
                  v78 = *(_QWORD *)(v71 + 80);
                  if ( *(_QWORD *)(v206 + 24) == v78 && (*(_BYTE *)(v206 + 33) & 0x20) != 0 )
                  {
                    v207 = *(_DWORD *)(v206 + 36);
                    if ( v207 > 0x45 )
                    {
                      if ( v207 != 71 )
                        goto LABEL_123;
                    }
                    else if ( v207 <= 0x43 )
                    {
                      goto LABEL_123;
                    }
                    sub_B58E30(&v395, v71);
                    v348 = v395;
                    v357.m128i_i64[0] = (__int64)v395;
                    v351[0] = v396;
                    v77 = sub_2A0C360(v357.m128i_i64, v351);
                    v208 = *(_DWORD *)(v71 + 4) & 0x7FFFFFF;
                    v209 = *(_QWORD *)(*(_QWORD *)(v71 + 32 * (1 - v208)) + 24LL);
                    v210 = *(_QWORD *)(*(_QWORD *)(v71 + 32 * (2 - v208)) + 24LL);
                    if ( (v379 & 1) != 0 )
                    {
                      v79 = (__int64 *)&v380;
                      v78 = 7;
                      goto LABEL_349;
                    }
                    v78 = (unsigned int)v381;
                    v79 = v380;
                    if ( (_DWORD)v381 )
                    {
                      v78 = (unsigned int)(v381 - 1);
LABEL_349:
                      v211 = 1;
                      for ( n = v78
                              & (((0xBF58476D1CE4E5B9LL
                                 * (((unsigned int)v210 >> 9) ^ ((unsigned int)v210 >> 4)
                                  | ((((0xBF58476D1CE4E5B9LL
                                      * ((v77 << 32) | ((unsigned int)v209 >> 9) ^ ((unsigned int)v209 >> 4))) >> 31)
                                    ^ (0xBF58476D1CE4E5B9LL
                                     * ((v77 << 32) | ((unsigned int)v209 >> 9) ^ ((unsigned int)v209 >> 4)))) << 32))) >> 31)
                               ^ (484763065 * (((unsigned int)v210 >> 9) ^ ((unsigned int)v210 >> 4)))); ; n = v78 & v213 )
                      {
                        v80 = &v79[3 * n];
                        if ( v77 == *v80 && v209 == v80[1] && v210 == v80[2] )
                          break;
                        if ( *v80 == -1 && v80[1] == -4096 && v80[2] == -4096 )
                          goto LABEL_123;
                        v213 = v211 + n;
                        ++v211;
                      }
LABEL_338:
                      sub_B43D60((_QWORD *)v71);
                      goto LABEL_137;
                    }
                  }
                }
              }
            }
LABEL_123:
            v81 = sub_1020E10(v71, *((const __m128i **)a1 + 7), v77, v78, v79, (__int64)v80);
            v82 = v81;
            if ( !v81 )
              goto LABEL_333;
            if ( *(_BYTE *)v81 > 0x1Cu )
            {
              v83 = *(_QWORD *)(v81 + 40);
              v84 = *(_QWORD *)(v71 + 40);
              if ( v83 != v84 )
              {
                v85 = *((_QWORD *)a1 + 1);
                v86 = *(_DWORD *)(v85 + 24);
                v87 = *(_QWORD *)(v85 + 8);
                if ( v86 )
                {
                  v88 = v86 - 1;
                  v89 = v88 & (((unsigned int)v83 >> 9) ^ ((unsigned int)v83 >> 4));
                  v90 = (__int64 *)(v87 + 16LL * v89);
                  v91 = *v90;
                  if ( v83 == *v90 )
                  {
LABEL_128:
                    v92 = (_QWORD *)v90[1];
                    if ( v92 )
                    {
                      v93 = v88 & (((unsigned int)v84 >> 9) ^ ((unsigned int)v84 >> 4));
                      v94 = (__int64 *)(v87 + 16LL * v93);
                      v95 = *v94;
                      if ( v84 != *v94 )
                      {
                        v205 = 1;
                        while ( v95 != -4096 )
                        {
                          v291 = v205 + 1;
                          v93 = v88 & (v205 + v93);
                          v94 = (__int64 *)(v87 + 16LL * v93);
                          v95 = *v94;
                          if ( v84 == *v94 )
                            goto LABEL_130;
                          v205 = v291;
                        }
                        goto LABEL_333;
                      }
LABEL_130:
                      v96 = (_QWORD *)v94[1];
                      if ( v92 != v96 )
                      {
                        while ( v96 )
                        {
                          v96 = (_QWORD *)*v96;
                          if ( v92 == v96 )
                            goto LABEL_133;
                        }
LABEL_333:
                        sub_2A0CF40((__int64)&v364, (__int64)v68, v71);
LABEL_134:
                        v97 = sub_BD5D20((__int64)v68);
                        LOWORD(i) = 261;
                        v395 = (__int64 *)v97;
                        v396 = v98;
                        sub_BD6B50((unsigned __int8 *)v71, (const char **)&v395);
                        if ( *(_BYTE *)v71 == 85 )
                        {
                          v243 = *(_QWORD *)(v71 - 32);
                          if ( v243 )
                          {
                            if ( !*(_BYTE *)v243 )
                            {
                              v244 = *(_QWORD *)(v71 + 80);
                              if ( *(_QWORD *)(v243 + 24) == v244
                                && (*(_BYTE *)(v243 + 33) & 0x20) != 0
                                && *(_DWORD *)(v243 + 36) == 11 )
                              {
                                sub_CFEAE0(*((_QWORD *)a1 + 3), v71, v61, v244, v99, v100);
                              }
                            }
                          }
                        }
                        if ( *((_QWORD *)a1 + 6) )
                          sub_2A0CF40((__int64)&v371, (__int64)v68, v71);
                        goto LABEL_137;
                      }
                    }
                  }
                  else
                  {
                    v255 = 1;
                    while ( v91 != -4096 )
                    {
                      v256 = v255 + 1;
                      v89 = v88 & (v255 + v89);
                      v90 = (__int64 *)(v87 + 16LL * v89);
                      v91 = *v90;
                      if ( v83 == *v90 )
                        goto LABEL_128;
                      v255 = v256;
                    }
                  }
                }
              }
            }
LABEL_133:
            sub_2A0CF40((__int64)&v364, (__int64)v68, v82);
            if ( (unsigned __int8)sub_B46970((unsigned __int8 *)v71) )
              goto LABEL_134;
            goto LABEL_338;
          }
          v395 = v66;
          LOBYTE(v396) = 1;
          v214 = sub_B43F50(v71, (__int64)v68, (__int64)v66, 1, 0);
          v216 = v215;
          v217 = (__int64)v214;
          sub_FC75A0((__int64 *)&v395, (__int64)&v364, 3, 0, 0, 0);
          sub_FCD310((__int64 *)&v395, v316, v217, v216);
          sub_FC7680((__int64 *)&v395, v316);
          for ( ; v216 != v217; v217 = *(_QWORD *)(v217 + 8) )
          {
            if ( !*(_BYTE *)(v217 + 32) )
              break;
          }
          v357.m128i_i64[1] = v216;
          v360 = (__int64 *)v216;
          v361 = v216;
          v357.m128i_i64[0] = v217;
          BYTE1(v359) = 1;
          BYTE1(v363) = 1;
          sub_2A0B3A0((__int64)&v395, &v357);
          v218 = v395;
          v326 = v219;
          v220 = v396;
          v324 = v400;
          if ( v400 == v395 )
          {
LABEL_381:
            v65 = &qword_4F81430[1];
            v66 = &qword_4F81430[1];
            goto LABEL_122;
          }
          v315 = v71;
          v314 = v68;
          while ( 1 )
          {
            for ( ii = v218[1]; v220 != ii; ii = *(_QWORD *)(ii + 8) )
            {
              if ( !*(_BYTE *)(ii + 32) )
                break;
            }
            sub_B129C0(&v357, (__int64)v218);
            v348 = (__int64 *)v357.m128i_i64[1];
            v351[0] = v357.m128i_i64[0];
            v222 = sub_2A0BE90(v351, v326);
            v223 = sub_B12000((__int64)(v218 + 9));
            v224 = sub_B11F60((__int64)(v218 + 10));
            v225 = v224;
            if ( (v379 & 1) != 0 )
            {
              v226 = (__int64 *)&v380;
              v227 = 7;
            }
            else
            {
              v226 = v380;
              if ( !(_DWORD)v381 )
                goto LABEL_378;
              v227 = v381 - 1;
            }
            v228 = 1;
            for ( jj = v227
                     & (((0xBF58476D1CE4E5B9LL
                        * (((((0xBF58476D1CE4E5B9LL
                             * ((v222 << 32) | ((unsigned int)v223 >> 9) ^ ((unsigned int)v223 >> 4))) >> 31)
                           ^ (0xBF58476D1CE4E5B9LL
                            * ((v222 << 32) | ((unsigned int)v223 >> 9) ^ ((unsigned int)v223 >> 4)))) << 32)
                         | ((unsigned int)v224 >> 9) ^ ((unsigned int)v224 >> 4))) >> 31)
                      ^ (484763065 * (((unsigned int)v224 >> 9) ^ ((unsigned int)v224 >> 4)))); ; jj = v227 & v231 )
            {
              v230 = &v226[3 * jj];
              if ( *v230 == v222 && v223 == v230[1] && v225 == v230[2] )
                break;
              if ( *v230 == -1 && v230[1] == -4096 && v230[2] == -4096 )
                goto LABEL_378;
              v231 = v228 + jj;
              ++v228;
            }
            sub_B14290(v218);
LABEL_378:
            if ( v324 == (__int64 *)ii )
            {
              v71 = v315;
              v68 = v314;
              goto LABEL_381;
            }
            v218 = (__int64 *)ii;
          }
        }
LABEL_138:
        if ( (_DWORD)v355 )
        {
          v259 = (unsigned __int64 *)sub_AA4FF0(v319);
          v261 = v260;
          if ( !v259 )
            v261 = 0;
          v262 = v261;
          v263 = 1;
          BYTE1(v263) = v262;
          v264 = (_BYTE **)v354;
          v265 = (_BYTE **)&v354[8 * (unsigned int)v355];
          if ( v265 != (_BYTE **)v354 )
          {
            v266 = v263;
            do
            {
              v267 = *v264++;
              v268 = (_QWORD *)sub_B47F80(v267);
              sub_B44150(v268, v319, v259, v266);
            }
            while ( v265 != v264 );
          }
          v269 = sub_AA48A0(v319);
          v395 = &v397;
          v271 = (__int64 *)v269;
          v396 = 0x800000000LL;
          v272 = &v354[8 * (unsigned int)v355];
          if ( v272 == v354 )
          {
            v280 = 0;
            v279 = &v397;
          }
          else
          {
            v273 = v354 + 8;
            v274 = &v397;
            v275 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v354 - 32LL * (*(_DWORD *)(*(_QWORD *)v354 + 4LL) & 0x7FFFFFF))
                             + 24LL);
            v276 = 0;
            while ( 1 )
            {
              v274[v276] = v275;
              v278 = v396 + 1;
              LODWORD(v396) = v396 + 1;
              if ( v272 == v273 )
                break;
              v275 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v273 - 32LL * (*(_DWORD *)(*(_QWORD *)v273 + 4LL) & 0x7FFFFFF))
                               + 24LL);
              v276 = v278;
              v277 = v278 + 1LL;
              if ( v277 > HIDWORD(v396) )
              {
                v346 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v273 - 32LL * (*(_DWORD *)(*(_QWORD *)v273 + 4LL) & 0x7FFFFFF))
                                 + 24LL);
                sub_C8D5F0((__int64)&v395, &v397, v277, 8u, v275, v270);
                v276 = (unsigned int)v396;
                v275 = v346;
              }
              v274 = v395;
              v273 += 8;
            }
            v279 = v395;
            v280 = v278;
          }
          v357.m128i_i64[0] = v332;
          sub_F4CD20(v279, v280, (__int64)&v357, 1, v271, v270, "h.rot", 5u);
          v281 = *(_QWORD *)v354;
          v357.m128i_i64[1] = 2;
          v358 = 0;
          v359 = v281;
          if ( v281 != -4096 && v281 != 0 && v281 != -8192 )
            sub_BD73F0((__int64)&v357.m128i_i64[1]);
          v360 = &v364;
          v357.m128i_i64[0] = (__int64)&unk_49DD7B0;
          if ( (unsigned __int8)sub_F9E960((__int64)&v364, (__int64)&v357, v351) )
          {
            v283 = v351[0] + 40;
            v284 = v359;
          }
          else
          {
            v292 = sub_256DFC0((__int64)&v364, (__int64)&v357, v351[0]);
            v293 = *(_QWORD *)(v292 + 24);
            v294 = (_QWORD *)v292;
            v284 = v359;
            if ( v293 != v359 )
            {
              if ( v293 != -4096 && v293 != 0 && v293 != -8192 )
              {
                sub_BD60C0(v294 + 1);
                v284 = v359;
              }
              v294[3] = v284;
              if ( v284 != 0 && v284 != -4096 && v284 != -8192 )
                sub_BD6050(v294 + 1, v357.m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL);
              v284 = v359;
            }
            v295 = v360;
            v283 = (__int64)(v294 + 5);
            v294[5] = 6;
            v294[6] = 0;
            v294[4] = v295;
            v294[7] = 0;
          }
          v357.m128i_i64[0] = (__int64)&unk_49DB368;
          if ( v284 != 0 && v284 != -4096 && v284 != -8192 )
            sub_BD60C0(&v357.m128i_i64[1]);
          v345 = *(_QWORD *)(v331 + 48);
          v285 = (v345 & 0xFFFFFFFFFFFFFFF8LL) - 24;
          if ( (v345 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
            v285 = 0;
          sub_F4CE10(v395, (unsigned int)v396, *(_QWORD *)(v283 + 16), v285, v271, v282, "pre.rot", 7u);
          if ( v395 != &v397 )
            _libc_free((unsigned __int64)v395);
        }
        v101 = *(_QWORD *)(v332 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v335 != v101 )
        {
          if ( !v101 )
            BUG();
          v102 = v101 - 24;
          if ( (unsigned int)*(unsigned __int8 *)(v101 - 24) - 30 <= 0xA )
          {
            v342 = sub_B46E30(v102);
            if ( v342 )
            {
              for ( kk = 0; v342 != kk; ++kk )
              {
                for ( mm = *(_QWORD *)(sub_B46EC0(v102, kk) + 56); ; mm = *(_QWORD *)(mm + 8) )
                {
                  if ( !mm )
                    goto LABEL_524;
                  if ( *(_BYTE *)(mm - 24) != 84 )
                    break;
                  v105 = 0x1FFFFFFFE0LL;
                  v106 = *(_DWORD *)(mm + 48);
                  v107 = *(_QWORD *)(mm - 32);
                  v108 = *(_DWORD *)(mm - 20) & 0x7FFFFFF;
                  if ( v108 )
                  {
                    v109 = 0;
                    do
                    {
                      if ( v332 == *(_QWORD *)(v107 + 32LL * v106 + 8 * v109) )
                      {
                        v105 = 32 * v109;
                        goto LABEL_152;
                      }
                      ++v109;
                    }
                    while ( v108 != (_DWORD)v109 );
                    v110 = *(_QWORD *)(v107 + 0x1FFFFFFFE0LL);
                    if ( v106 == v108 )
                    {
LABEL_290:
                      sub_B48D90(mm - 24);
                      v107 = *(_QWORD *)(mm - 32);
                      v108 = *(_DWORD *)(mm - 20) & 0x7FFFFFF;
                    }
                  }
                  else
                  {
LABEL_152:
                    v110 = *(_QWORD *)(v107 + v105);
                    if ( v106 == v108 )
                      goto LABEL_290;
                  }
                  v111 = (v108 + 1) & 0x7FFFFFF;
                  *(_DWORD *)(mm - 20) = v111 | *(_DWORD *)(mm - 20) & 0xF8000000;
                  v112 = v107 + 32LL * (unsigned int)(v111 - 1);
                  if ( *(_QWORD *)v112 )
                  {
                    v113 = *(_QWORD *)(v112 + 8);
                    **(_QWORD **)(v112 + 16) = v113;
                    if ( v113 )
                      *(_QWORD *)(v113 + 16) = *(_QWORD *)(v112 + 16);
                  }
                  *(_QWORD *)v112 = v110;
                  if ( v110 )
                  {
                    v114 = *(_QWORD *)(v110 + 16);
                    *(_QWORD *)(v112 + 8) = v114;
                    if ( v114 )
                      *(_QWORD *)(v114 + 16) = v112 + 8;
                    *(_QWORD *)(v112 + 16) = v110 + 16;
                    *(_QWORD *)(v110 + 16) = v112;
                  }
                  *(_QWORD *)(*(_QWORD *)(mm - 32)
                            + 32LL * *(unsigned int *)(mm + 48)
                            + 8LL * ((*(_DWORD *)(mm - 20) & 0x7FFFFFFu) - 1)) = v331;
                }
              }
            }
          }
        }
        sub_B43D60(v323);
        sub_AA6320(v331);
        if ( *((_QWORD *)a1 + 6) )
        {
          sub_2A0CF40((__int64)&v371, v332, v331);
          sub_D6BD90(*((__int64 **)a1 + 6), v332, v331, (__int64)&v371);
        }
        v348 = (__int64 *)v350;
        v349 = 0x200000000LL;
        sub_2A0B8D0(v332, v331, (__int64)&v364, *((_QWORD *)a1 + 5), (__int64)&v348);
        if ( (_DWORD)v349 )
          sub_F657D0(v332, &v348);
        v115 = *(_QWORD **)(a2 + 32);
        LODWORD(v116) = 0;
        if ( v319 != *v115 )
        {
          do
          {
            v116 = (unsigned int)(v116 + 1);
            v117 = &v115[v116];
          }
          while ( v319 != *v117 );
          *v117 = *v115;
          **(_QWORD **)(a2 + 32) = v319;
        }
        v118 = *((_QWORD *)a1 + 4);
        if ( !v118 )
          goto LABEL_177;
        v357.m128i_i64[0] = (__int64)&v358;
        v358 = v331;
        v359 = v318 & 0xFFFFFFFFFFFFFFFBLL;
        v360 = (__int64 *)v331;
        v362 = v331;
        v119 = (__int64 *)*((_QWORD *)a1 + 6);
        v361 = v319 & 0xFFFFFFFFFFFFFFFBLL;
        v363 = v332 | 4;
        v357.m128i_i64[1] = 0x300000003LL;
        if ( v119 )
        {
          sub_D75690(v119, &v358, 3, v118, 1);
          if ( byte_4F8F8E8[0] )
            nullsub_390();
          v120 = v357.m128i_i64[0];
          if ( (unsigned __int64 *)v357.m128i_i64[0] == &v358 )
            goto LABEL_177;
        }
        else
        {
          sub_B26290((__int64)&v395, &v358, 3, 1u);
          sub_B24D40(v118, (__int64)&v395, 0);
          sub_B1A8B0((__int64)&v395, (__int64)&v395);
          v120 = v357.m128i_i64[0];
          if ( (unsigned __int64 *)v357.m128i_i64[0] == &v358 )
            goto LABEL_177;
        }
        _libc_free(v120);
LABEL_177:
        v121 = *(_QWORD *)(v331 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v330 == (_QWORD *)v121 )
          goto LABEL_523;
        if ( !v121 )
          goto LABEL_524;
        if ( (unsigned int)*(unsigned __int8 *)(v121 - 24) - 30 > 0xA )
LABEL_523:
          BUG();
        v122 = *(_QWORD *)(v121 - 120);
        v123 = v321;
        if ( *(_BYTE *)v122 == 17 )
        {
          v196 = *(_DWORD *)(v122 + 32);
          if ( v196 <= 0x40 )
          {
            v198 = *(_QWORD *)(v122 + 24) == 0;
          }
          else
          {
            v197 = sub_C444A0(v122 + 24);
            v123 = v321;
            v198 = v196 == v197;
          }
          if ( *(_QWORD *)(v121 - 32LL * v198 - 56) == v319 )
          {
            sub_2A0B200(v121 - 24, v317, 0, v123);
            sub_AA5980(v318, v331, 1u);
            v199 = sub_BD2C40(72, 1u);
            v200 = v199;
            if ( v199 )
              sub_B4C8F0((__int64)v199, v319, 1u, v121, 0);
            v201 = *(__int64 **)(v121 + 24);
            v202 = (void **)(v200 + 6);
            v395 = v201;
            if ( v201 )
            {
              sub_B96E90((__int64)&v395, (__int64)v201, 1);
              if ( v202 == (void **)&v395 )
              {
                if ( v395 )
                  sub_B91220((__int64)&v395, (__int64)v395);
                goto LABEL_320;
              }
              v245 = v200[6];
              if ( !v245 )
              {
LABEL_405:
                v246 = (unsigned __int8 *)v395;
                v200[6] = v395;
                if ( v246 )
                  sub_B976B0((__int64)&v395, v246, (__int64)(v200 + 6));
                goto LABEL_320;
              }
            }
            else if ( v202 == (void **)&v395 || (v245 = v200[6]) == 0 )
            {
LABEL_320:
              sub_B43D60((_QWORD *)(v121 - 24));
              v203 = *((_QWORD *)a1 + 4);
              if ( v203 )
                sub_B20B50(v203, v331, v318);
              v204 = (__int64 *)*((_QWORD *)a1 + 6);
              if ( !v204 )
                goto LABEL_219;
              sub_D6D7F0(v204, v331, v318);
LABEL_216:
              if ( *((_QWORD *)a1 + 6) && byte_4F8F8E8[0] )
                nullsub_390();
LABEL_219:
              v396 = 0x1000000000LL;
              v155 = *((_QWORD *)a1 + 4);
              v412 = 0;
              v395 = &v397;
              v403 = v155;
              v401 = 0;
              v402 = 0;
              v404 = 0;
              v405 = 0;
              v406 = 0;
              v407 = &v411;
              v408 = 8;
              v409 = 0;
              v410 = 1;
              v413 = 0;
              v414 = 0;
              v415 = 0;
              v156 = &v395;
              v157 = sub_AA5510(v332);
              v158 = sub_F39690(v332, (__int64)&v395, *((_QWORD *)a1 + 1), *((__int64 **)a1 + 6), 0, 0, 0);
              v162 = v312;
              if ( v158 )
                sub_F3F2F0(v157, (__int64)&v395);
              if ( *((_QWORD *)a1 + 6) )
              {
                v159 = byte_4F8F8E8;
                if ( byte_4F8F8E8[0] )
                {
                  v156 = 0;
                  nullsub_390();
                }
              }
              sub_FFCE90((__int64)&v395, (__int64)v156, (__int64)v159, v160, v162, v161);
              sub_FFD870((__int64)&v395, (__int64)v156, v163, v164, v165, v166);
              sub_FFBC40((__int64)&v395, (__int64)v156);
              v167 = v414;
              v168 = v413;
              if ( v414 != v413 )
              {
                do
                {
                  v169 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v168[7];
                  *v168 = &unk_49E5048;
                  if ( v169 )
                    v169(v168 + 5, v168 + 5, 3);
                  *v168 = &unk_49DB368;
                  v170 = v168[3];
                  if ( v170 != 0 && v170 != -4096 && v170 != -8192 )
                    sub_BD60C0(v168 + 1);
                  v168 += 9;
                }
                while ( v167 != v168 );
                v168 = v413;
              }
              if ( v168 )
                j_j___libc_free_0((unsigned __int64)v168);
              if ( !v410 )
                _libc_free((unsigned __int64)v407);
              if ( v395 != &v397 )
                _libc_free((unsigned __int64)v395);
              if ( v348 != (__int64 *)v350 )
                _libc_free((unsigned __int64)v348);
              if ( v354 != v356 )
                _libc_free((unsigned __int64)v354);
              if ( (v379 & 1) == 0 )
                sub_C7D6A0((__int64)v380, 24LL * (unsigned int)v381, 8);
              if ( v377 )
              {
                v251 = v376;
                v377 = 0;
                if ( v376 )
                {
                  v252 = v375;
                  v253 = &v375[2 * v376];
                  do
                  {
                    if ( *v252 != -8192 && *v252 != -4096 )
                    {
                      v254 = v252[1];
                      if ( v254 )
                        sub_B91220((__int64)(v252 + 1), v254);
                    }
                    v252 += 2;
                  }
                  while ( v253 != v252 );
                  v251 = v376;
                }
                sub_C7D6A0((__int64)v375, 16LL * v251, 8);
              }
              v171 = v374;
              if ( v374 )
              {
                v172 = v372;
                v379 = 2;
                v380 = 0;
                v173 = -4096;
                v174 = &v372[8 * (unsigned __int64)v374];
                v381 = -4096;
                v378 = &unk_49DD7B0;
                v382 = 0;
                v396 = 2;
                v397 = 0;
                v398 = -8192;
                v395 = (__int64 *)&unk_49DD7B0;
                i = 0;
                while ( 1 )
                {
                  v175 = v172[3];
                  if ( v173 != v175 )
                  {
                    v173 = v398;
                    if ( v175 != v398 )
                    {
                      v176 = v172[7];
                      if ( v176 != 0 && v176 != -4096 && v176 != -8192 )
                      {
                        sub_BD60C0(v172 + 5);
                        v175 = v172[3];
                      }
                      v173 = v175;
                    }
                  }
                  *v172 = &unk_49DB368;
                  if ( v173 != 0 && v173 != -4096 && v173 != -8192 )
                    sub_BD60C0(v172 + 1);
                  v172 += 8;
                  if ( v174 == v172 )
                    break;
                  v173 = v381;
                }
                v395 = (__int64 *)&unk_49DB368;
                if ( v398 != 0 && v398 != -4096 && v398 != -8192 )
                  sub_BD60C0(&v396);
                v378 = &unk_49DB368;
                if ( v381 != 0 && v381 != -4096 && v381 != -8192 )
                  sub_BD60C0(&v379);
                v171 = v374;
              }
              sub_C7D6A0((__int64)v372, (unsigned __int64)v171 << 6, 8);
              if ( v370 )
              {
                v247 = v369;
                v370 = 0;
                if ( v369 )
                {
                  v248 = v368;
                  v249 = &v368[2 * v369];
                  do
                  {
                    if ( *v248 != -8192 && *v248 != -4096 )
                    {
                      v250 = v248[1];
                      if ( v250 )
                        sub_B91220((__int64)(v248 + 1), v250);
                    }
                    v248 += 2;
                  }
                  while ( v249 != v248 );
                  v247 = v369;
                }
                sub_C7D6A0((__int64)v368, 16LL * v247, 8);
              }
              v177 = v367;
              if ( v367 )
              {
                v178 = v365;
                v379 = 2;
                v380 = 0;
                v179 = -4096;
                v180 = &v365[8 * (unsigned __int64)v367];
                v381 = -4096;
                v378 = &unk_49DD7B0;
                v382 = 0;
                v396 = 2;
                v397 = 0;
                v398 = -8192;
                v395 = (__int64 *)&unk_49DD7B0;
                i = 0;
                while ( 1 )
                {
                  v181 = v178[3];
                  if ( v179 != v181 )
                  {
                    v179 = v398;
                    if ( v181 != v398 )
                    {
                      v182 = v178[7];
                      if ( v182 != 0 && v182 != -4096 && v182 != -8192 )
                      {
                        sub_BD60C0(v178 + 5);
                        v181 = v178[3];
                      }
                      v179 = v181;
                    }
                  }
                  *v178 = &unk_49DB368;
                  if ( v179 != 0 && v179 != -4096 && v179 != -8192 )
                    sub_BD60C0(v178 + 1);
                  v178 += 8;
                  if ( v180 == v178 )
                    break;
                  v179 = v381;
                }
                v395 = (__int64 *)&unk_49DB368;
                if ( v398 != -4096 && v398 != 0 && v398 != -8192 )
                  sub_BD60C0(&v396);
                v378 = &unk_49DB368;
                if ( v381 != 0 && v381 != -4096 && v381 != -8192 )
                  sub_BD60C0(&v379);
                v177 = v367;
              }
              v183 = (unsigned __int64)v177 << 6;
              sub_C7D6A0((__int64)v365, v183, 8);
              if ( !(_BYTE)qword_5009F88 || !(v329 = sub_2A0B6A0(a2, v183)) )
                return v320;
              v3 = *(__int64 **)(a2 + 32);
              continue;
            }
            sub_B91220((__int64)(v200 + 6), v245);
            goto LABEL_405;
          }
        }
        sub_2A0B200(v121 - 24, v317, 1, v123);
        LODWORD(v360) = 0x10000;
        v124 = *((_QWORD *)a1 + 6);
        v125 = *((_QWORD *)a1 + 1);
        BYTE4(v360) = 1;
        v126 = *((_QWORD *)a1 + 4);
        v359 = v124;
        v357 = (__m128i)v126;
        v358 = v125;
        v127 = *(_QWORD *)(v331 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v330 == (_QWORD *)v127 )
        {
          v129 = 0;
        }
        else
        {
          if ( !v127 )
            goto LABEL_524;
          v128 = *(unsigned __int8 *)(v127 - 24);
          v129 = 0;
          v130 = v127 - 24;
          if ( (unsigned int)(v128 - 30) < 0xB )
            v129 = v130;
        }
        for ( nn = 0; v319 != sub_B46EC0(v129, nn); ++nn )
          ;
        LOWORD(i) = 257;
        v132 = (unsigned __int8 *)sub_F451F0(v129, nn, (__int64)&v357, (void **)&v395);
        v395 = (__int64 *)sub_BD5D20(v319);
        v397 = (__int64)".lr.ph";
        LOWORD(i) = 773;
        v396 = v133;
        sub_BD6B50(v132, (const char **)&v395);
        v136 = *(_QWORD *)(v318 + 16);
        if ( !v136 )
          goto LABEL_216;
        while ( (unsigned __int8)(**(_BYTE **)(v136 + 24) - 30) > 0xAu )
        {
          v136 = *(_QWORD *)(v136 + 8);
          if ( !v136 )
            goto LABEL_216;
        }
        v137 = 0;
        v395 = &v397;
        v396 = 0x400000000LL;
        v138 = v136;
        while ( 1 )
        {
          v138 = *(_QWORD *)(v138 + 8);
          if ( !v138 )
            break;
          while ( (unsigned __int8)(**(_BYTE **)(v138 + 24) - 30) <= 0xAu )
          {
            v138 = *(_QWORD *)(v138 + 8);
            ++v137;
            if ( !v138 )
              goto LABEL_194;
          }
        }
LABEL_194:
        v139 = v137 + 1;
        v140 = &v397;
        if ( v139 > 4 )
        {
          sub_C8D5F0((__int64)&v395, &v397, v139, 8u, v134, v135);
          v140 = &v395[(unsigned int)v396];
        }
        v141 = *(_QWORD *)(v136 + 24);
LABEL_199:
        if ( v140 )
          *v140 = *(_QWORD *)(v141 + 40);
        while ( 1 )
        {
          v136 = *(_QWORD *)(v136 + 8);
          if ( !v136 )
            break;
          v141 = *(_QWORD *)(v136 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v141 - 30) <= 0xAu )
          {
            ++v140;
            goto LABEL_199;
          }
        }
        v142 = v395;
        LODWORD(v396) = v139 + v396;
        v343 = &v395[(unsigned int)v396];
        if ( v343 == v395 )
        {
LABEL_214:
          if ( v142 != &v397 )
            _libc_free((unsigned __int64)v142);
          goto LABEL_216;
        }
        v143 = v395;
        while ( 1 )
        {
          v144 = *((_QWORD *)a1 + 1);
          v145 = *v143;
          v146 = *(_DWORD *)(v144 + 24);
          v147 = *(_QWORD *)(v144 + 8);
          if ( v146 )
          {
            v148 = v146 - 1;
            v149 = v148 & (((unsigned int)v145 >> 9) ^ ((unsigned int)v145 >> 4));
            v150 = (__int64 *)(v147 + 16LL * v149);
            v151 = *v150;
            if ( v145 == *v150 )
            {
LABEL_206:
              v152 = v150[1];
              if ( v152 )
              {
                if ( *(_BYTE *)(v152 + 84) )
                {
                  v153 = *(_QWORD **)(v152 + 64);
                  v154 = &v153[*(unsigned int *)(v152 + 76)];
                  if ( v153 == v154 )
                    goto LABEL_383;
                  while ( v318 != *v153 )
                  {
                    if ( v154 == ++v153 )
                      goto LABEL_383;
                  }
                }
                else if ( !sub_C8CA60(v152 + 56, v318) )
                {
LABEL_383:
                  v232 = *(_QWORD *)(v145 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                  if ( v145 + 48 == v232 || !v232 || (unsigned int)*(unsigned __int8 *)(v232 - 24) - 30 > 0xA )
                    goto LABEL_524;
                  if ( *(_BYTE *)(v232 - 24) != 33 )
                  {
                    sub_D47930(a2);
                    v233 = *((_QWORD *)a1 + 6);
                    v234 = *((_QWORD *)a1 + 1);
                    v351[1] = 0;
                    v235 = *((_QWORD *)a1 + 4);
                    v353 = 1;
                    v351[2] = v234;
                    v351[0] = v235;
                    v351[3] = v233;
                    v352 = 0x10000;
                    v236 = *(_QWORD *)(v145 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                    if ( v145 + 48 == v236 )
                    {
                      v240 = 0;
                    }
                    else
                    {
                      if ( !v236 )
                        goto LABEL_524;
                      v237 = *(unsigned __int8 *)(v236 - 24);
                      v238 = 0;
                      v239 = v236 - 24;
                      if ( (unsigned int)(v237 - 30) < 0xB )
                        v238 = v239;
                      v240 = v238;
                    }
                    for ( i1 = 0; v318 != sub_B46EC0(v240, i1); ++i1 )
                      ;
                    LOWORD(v360) = 257;
                    v242 = sub_F451F0(v240, i1, (__int64)v351, (void **)&v357);
                    sub_AA4AC0(v242, v318 + 24);
                  }
                }
              }
            }
            else
            {
              v257 = 1;
              while ( v151 != -4096 )
              {
                v258 = v257 + 1;
                v149 = v148 & (v257 + v149);
                v150 = (__int64 *)(v147 + 16LL * v149);
                v151 = *v150;
                if ( v145 == *v150 )
                  goto LABEL_206;
                v257 = v258;
              }
            }
          }
          if ( v343 == ++v143 )
          {
            v142 = v395;
            goto LABEL_214;
          }
        }
      }
      break;
    }
    sub_C7D6A0(v386, 24LL * v388, 8);
    if ( !BYTE4(v398) )
      _libc_free(v396);
  }
  return v329;
}
