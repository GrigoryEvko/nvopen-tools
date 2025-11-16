// Function: sub_1D89B50
// Address: 0x1d89b50
//
__int64 __fastcall sub_1D89B50(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 (*v15)(); // rax
  __int64 v16; // rdi
  __int64 (*v17)(); // rax
  __int64 *v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 *v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // r14
  __int64 v27; // rsi
  unsigned __int64 v28; // rbx
  __int64 v29; // rax
  unsigned __int64 v30; // rdi
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  _QWORD *v34; // r8
  unsigned __int8 v35; // bl
  __int64 v36; // r13
  __int64 v37; // r12
  char v38; // al
  _QWORD *v39; // r8
  __int64 v40; // rax
  __int64 v41; // r8
  unsigned __int64 v42; // rdi
  _QWORD *v43; // rax
  _DWORD *v44; // r9
  __int64 v45; // rcx
  __int64 v46; // rdx
  __int64 (*v47)(); // rdx
  __int64 v48; // rax
  _QWORD *v49; // rax
  unsigned __int64 v50; // r8
  __int64 v51; // rax
  unsigned __int8 *v52; // rsi
  _DWORD *v53; // rsi
  unsigned __int64 v54; // rbx
  unsigned __int64 v55; // rax
  unsigned __int64 v56; // rsi
  __int64 v57; // rcx
  __int64 v58; // r10
  __int64 v59; // r9
  unsigned __int64 v60; // r15
  unsigned __int64 v61; // r14
  _QWORD *v62; // r15
  _QWORD *v63; // rdi
  __int64 v64; // rax
  __int64 v65; // rdx
  unsigned __int8 *v66; // rsi
  unsigned __int8 *v67; // rsi
  _QWORD *v68; // rax
  __int64 v69; // rax
  __int64 v70; // r13
  __int64 v71; // rax
  __int64 v72; // r10
  __int64 v73; // rbx
  __int64 *v74; // r13
  __int64 v75; // rcx
  __int64 v76; // rax
  __int64 v77; // r10
  __int64 v78; // rsi
  __int64 v79; // r13
  unsigned __int8 *v80; // rsi
  __int64 v81; // r13
  _QWORD *v82; // rax
  __int64 v83; // rbx
  char v84; // dl
  unsigned int i; // ebx
  __int64 v86; // rsi
  unsigned int v87; // eax
  __int64 v88; // r15
  __int64 v89; // r14
  _QWORD *v90; // rax
  __int64 v91; // r13
  __int64 v92; // rbx
  unsigned __int64 v93; // r13
  unsigned __int64 v94; // r12
  _QWORD *v95; // rcx
  unsigned __int64 v96; // rax
  unsigned int v97; // ebx
  unsigned int v98; // esi
  unsigned int v99; // ecx
  double v100; // xmm4_8
  double v101; // xmm5_8
  __int64 v102; // r13
  __int64 v103; // rax
  _DWORD *v104; // rdi
  _DWORD *v105; // r9
  __int64 v106; // rcx
  __int64 v107; // rdx
  unsigned __int64 v108; // rax
  unsigned __int8 *v109; // rsi
  unsigned __int8 *v110; // rsi
  char v111; // si
  unsigned int v112; // ebx
  unsigned __int64 v113; // rax
  __int64 v114; // rdx
  __int64 v115; // r15
  _QWORD *v116; // rax
  _QWORD *v117; // r14
  __int64 *v118; // r15
  __int64 v119; // rax
  __int64 v120; // rsi
  __int64 v121; // rsi
  __int64 v122; // rdx
  unsigned __int8 *v123; // rsi
  unsigned int v124; // esi
  __int64 v125; // r13
  __int64 v126; // rcx
  __int64 *v127; // rax
  char v128; // al
  _QWORD *v129; // rax
  __int64 v130; // rax
  __int64 v131; // rax
  __int64 v132; // r8
  __int64 v133; // r9
  __int64 v134; // rdx
  __int64 v135; // rcx
  __int64 v136; // r13
  int v137; // eax
  __int64 v138; // rax
  int v139; // esi
  __int64 v140; // rsi
  __int64 *v141; // rax
  __int64 v142; // rdi
  unsigned __int64 v143; // r8
  __int64 v144; // rdi
  __int64 v145; // rax
  __int64 v146; // rsi
  __int64 v147; // rax
  _QWORD *v148; // rax
  __int64 **v149; // rdx
  unsigned int v150; // r13d
  _QWORD *v151; // rax
  __int64 v152; // r13
  __int64 v153; // rax
  __int64 v154; // r10
  __int64 v155; // rbx
  __int64 v156; // rax
  __int64 v157; // rsi
  __int64 v158; // r10
  __int64 v159; // rsi
  __int64 v160; // rdx
  unsigned __int8 *v161; // rsi
  __int64 v162; // rax
  __int64 v163; // r10
  __int64 v164; // rbx
  __int64 *v165; // r13
  __int64 v166; // rcx
  __int64 v167; // rax
  __int64 v168; // r10
  __int64 v169; // rsi
  __int64 v170; // r13
  unsigned __int8 *v171; // rsi
  unsigned int v172; // r12d
  _QWORD *v173; // rax
  __int64 v174; // r13
  __int64 **v175; // rbx
  __int64 **v176; // rdx
  __int64 **v177; // rdx
  _QWORD *v178; // rax
  __int64 v179; // r10
  __int64 *v180; // rbx
  __int64 v181; // rcx
  __int64 v182; // rax
  __int64 v183; // r10
  __int64 v184; // rsi
  unsigned __int8 *v185; // rsi
  __int64 *v186; // r13
  _QWORD *v187; // rax
  __int64 v188; // rbx
  __int64 *v189; // r13
  __int64 v190; // rax
  __int64 v191; // rcx
  __int64 v192; // rsi
  unsigned __int8 *v193; // rsi
  __int64 *v194; // rax
  __int64 v195; // rax
  __int64 v196; // r13
  __int64 ***v197; // r13
  __int64 **v198; // rdx
  __int64 **v199; // rdx
  __int64 *v200; // rax
  __int64 **v201; // rax
  _BYTE *v202; // r13
  __int64 **v203; // rax
  __int64 *v204; // rsi
  __int64 v205; // rdx
  __int64 v206; // rax
  __int64 *v207; // rbx
  __int64 v208; // rax
  __int64 v209; // rcx
  __int64 v210; // rsi
  unsigned __int8 *v211; // rsi
  __int64 *v212; // rbx
  __int64 v213; // rcx
  __int64 v214; // rax
  __int64 v215; // rsi
  __int64 v216; // rdx
  unsigned __int8 *v217; // rsi
  __int64 *v218; // rbx
  __int64 v219; // rcx
  __int64 v220; // rax
  __int64 v221; // rsi
  __int64 v222; // rdx
  unsigned __int8 *v223; // rsi
  unsigned __int64 *v224; // rbx
  __int64 **v225; // rax
  unsigned __int64 v226; // rcx
  __int64 v227; // rsi
  __int64 v228; // rdx
  unsigned __int8 *v229; // rsi
  __int64 *v230; // rbx
  __int64 v231; // rax
  __int64 v232; // rcx
  __int64 v233; // rsi
  __int64 v234; // rdx
  unsigned __int8 *v235; // rsi
  __int64 *v236; // rbx
  __int64 v237; // rax
  __int64 v238; // rcx
  __int64 v239; // rsi
  __int64 v240; // rdx
  unsigned __int8 *v241; // rsi
  unsigned __int64 *v242; // rbx
  __int64 **v243; // rax
  unsigned __int64 v244; // rcx
  __int64 v245; // rsi
  __int64 v246; // rdx
  unsigned __int8 *v247; // rsi
  __int64 v248; // [rsp+10h] [rbp-290h]
  _QWORD *v249; // [rsp+18h] [rbp-288h]
  __int64 *v250; // [rsp+20h] [rbp-280h]
  _DWORD *v251; // [rsp+28h] [rbp-278h]
  __int64 v252; // [rsp+30h] [rbp-270h]
  unsigned __int64 v253; // [rsp+38h] [rbp-268h]
  __int64 v254; // [rsp+40h] [rbp-260h]
  unsigned int v255; // [rsp+58h] [rbp-248h]
  _QWORD *v256; // [rsp+58h] [rbp-248h]
  __int64 v258; // [rsp+68h] [rbp-238h]
  __int64 v259; // [rsp+68h] [rbp-238h]
  __int64 v260; // [rsp+68h] [rbp-238h]
  __int64 v261; // [rsp+68h] [rbp-238h]
  __int64 v262; // [rsp+68h] [rbp-238h]
  __int64 *v263; // [rsp+68h] [rbp-238h]
  __int64 v264; // [rsp+68h] [rbp-238h]
  __int64 v265; // [rsp+68h] [rbp-238h]
  char v266; // [rsp+70h] [rbp-230h]
  __int64 v267; // [rsp+70h] [rbp-230h]
  unsigned int v268; // [rsp+78h] [rbp-228h]
  __int64 v269; // [rsp+78h] [rbp-228h]
  __int64 v270; // [rsp+78h] [rbp-228h]
  _QWORD *v271; // [rsp+78h] [rbp-228h]
  __int64 v272; // [rsp+78h] [rbp-228h]
  __int64 v273; // [rsp+78h] [rbp-228h]
  _QWORD *v274; // [rsp+80h] [rbp-220h]
  __int64 v275; // [rsp+80h] [rbp-220h]
  unsigned __int64 v276; // [rsp+80h] [rbp-220h]
  unsigned __int64 v277; // [rsp+80h] [rbp-220h]
  unsigned __int64 v278; // [rsp+80h] [rbp-220h]
  __int64 v279; // [rsp+80h] [rbp-220h]
  __int64 v280; // [rsp+80h] [rbp-220h]
  __int64 v281; // [rsp+80h] [rbp-220h]
  __int64 v282; // [rsp+80h] [rbp-220h]
  __int64 v283; // [rsp+80h] [rbp-220h]
  __int64 v284; // [rsp+80h] [rbp-220h]
  __int64 v285; // [rsp+80h] [rbp-220h]
  int v286; // [rsp+80h] [rbp-220h]
  __int64 v287; // [rsp+80h] [rbp-220h]
  __int64 v288; // [rsp+80h] [rbp-220h]
  __int64 v289; // [rsp+80h] [rbp-220h]
  __int64 v290; // [rsp+80h] [rbp-220h]
  int v291; // [rsp+80h] [rbp-220h]
  __int64 v292; // [rsp+80h] [rbp-220h]
  __int64 v293; // [rsp+80h] [rbp-220h]
  __int64 v294; // [rsp+80h] [rbp-220h]
  __int64 v295; // [rsp+80h] [rbp-220h]
  __int64 v296; // [rsp+80h] [rbp-220h]
  __int64 v297; // [rsp+80h] [rbp-220h]
  char v298; // [rsp+95h] [rbp-20Bh]
  unsigned __int8 v299; // [rsp+96h] [rbp-20Ah]
  char v300; // [rsp+97h] [rbp-209h]
  __int64 v301; // [rsp+98h] [rbp-208h]
  __int64 v302; // [rsp+A0h] [rbp-200h]
  __int64 v303; // [rsp+A8h] [rbp-1F8h]
  unsigned int v304; // [rsp+BCh] [rbp-1E4h] BYREF
  __int64 *v305; // [rsp+C0h] [rbp-1E0h] BYREF
  __int64 ***v306; // [rsp+C8h] [rbp-1D8h] BYREF
  __int64 ***v307; // [rsp+D0h] [rbp-1D0h] BYREF
  unsigned __int8 *v308; // [rsp+D8h] [rbp-1C8h] BYREF
  __int64 v309[2]; // [rsp+E0h] [rbp-1C0h] BYREF
  __int16 v310; // [rsp+F0h] [rbp-1B0h]
  unsigned __int8 *v311[2]; // [rsp+100h] [rbp-1A0h] BYREF
  __int16 v312; // [rsp+110h] [rbp-190h]
  __int64 v313[14]; // [rsp+120h] [rbp-180h] BYREF
  _BYTE *v314; // [rsp+190h] [rbp-110h]
  unsigned __int8 *v315; // [rsp+198h] [rbp-108h] BYREF
  __int64 v316; // [rsp+1A0h] [rbp-100h]
  __int64 v317; // [rsp+1A8h] [rbp-F8h]
  _QWORD *v318; // [rsp+1B0h] [rbp-F0h]
  __int64 v319; // [rsp+1B8h] [rbp-E8h]
  int v320; // [rsp+1C0h] [rbp-E0h]
  __int64 v321; // [rsp+1C8h] [rbp-D8h]
  __int64 v322; // [rsp+1D0h] [rbp-D0h]
  _BYTE *v323; // [rsp+1E0h] [rbp-C0h] BYREF
  __int64 v324; // [rsp+1E8h] [rbp-B8h]
  _BYTE v325[176]; // [rsp+1F0h] [rbp-B0h] BYREF

  v299 = sub_1636880(a1, a2);
  if ( v299 )
    return 0;
  v12 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4FCBA30, 1u);
  if ( v12 )
  {
    v13 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v12 + 104LL))(v12, &unk_4FCBA30);
    if ( v13 )
    {
      v14 = *(_QWORD *)(v13 + 208);
      v15 = *(__int64 (**)())(*(_QWORD *)v14 + 16LL);
      if ( v15 == sub_16FF750 )
        goto LABEL_327;
      v251 = 0;
      v16 = ((__int64 (__fastcall *)(__int64, __int64))v15)(v14, a2);
      v17 = *(__int64 (**)())(*(_QWORD *)v16 + 56LL);
      if ( v17 != sub_1D12D20 )
        v251 = (_DWORD *)((__int64 (__fastcall *)(__int64))v17)(v16);
      v18 = *(__int64 **)(a1 + 8);
      v19 = *v18;
      v20 = v18[1];
      if ( v19 == v20 )
        goto LABEL_327;
      while ( *(_UNKNOWN **)v19 != &unk_4F9B6E8 )
      {
        v19 += 16;
        if ( v20 == v19 )
          goto LABEL_327;
      }
      v21 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v19 + 8) + 104LL))(
              *(_QWORD *)(v19 + 8),
              &unk_4F9B6E8);
      v22 = *(__int64 **)(a1 + 8);
      v301 = v21;
      v23 = *v22;
      v24 = v22[1];
      if ( v23 == v24 )
LABEL_327:
        BUG();
      while ( *(_UNKNOWN **)v23 != &unk_4F9D3C0 )
      {
        v23 += 16;
        if ( v24 == v23 )
          goto LABEL_327;
      }
      v25 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v23 + 8) + 104LL))(
              *(_QWORD *)(v23 + 8),
              &unk_4F9D3C0);
      v250 = (__int64 *)sub_14A4050(v25, a2);
      v298 = 0;
      v248 = sub_1632FA0(*(_QWORD *)(a2 + 40));
      v302 = *(_QWORD *)(a2 + 80);
      if ( v302 == a2 + 72 )
        goto LABEL_169;
LABEL_18:
      if ( !v302 )
        BUG();
      v26 = *(_QWORD *)(v302 + 24);
      if ( v26 == v302 + 16 )
      {
LABEL_35:
        v302 = *(_QWORD *)(v302 + 8);
        goto LABEL_36;
      }
      v303 = v302 + 16;
      while ( 1 )
      {
        if ( !v26 )
          BUG();
        if ( *(_BYTE *)(v26 - 8) != 78 )
          goto LABEL_27;
        v28 = (v26 - 24) & 0xFFFFFFFFFFFFFFF8LL;
        if ( !(unsigned __int8)sub_1560260((_QWORD *)(v28 + 56), -1, 21) )
        {
          v27 = *(_QWORD *)(v28 - 24);
          if ( *(_BYTE *)(v27 + 16) )
            goto LABEL_24;
          v313[0] = *(_QWORD *)(v27 + 112);
          if ( !(unsigned __int8)sub_1560260(v313, -1, 21) )
            goto LABEL_23;
        }
        if ( (unsigned __int8)sub_1560260((_QWORD *)(v28 + 56), -1, 5) )
          goto LABEL_23;
        v29 = *(_QWORD *)(v28 - 24);
        if ( *(_BYTE *)(v29 + 16) )
          goto LABEL_27;
        v313[0] = *(_QWORD *)(v29 + 112);
        if ( (unsigned __int8)sub_1560260(v313, -1, 5) )
        {
LABEL_23:
          v27 = *(_QWORD *)(v28 - 24);
LABEL_24:
          if ( *(_BYTE *)(v27 + 16) )
            goto LABEL_27;
          v300 = sub_149CB50(*(_QWORD *)(v301 + 360), v27, &v304);
          if ( !v300 )
            goto LABEL_27;
          if ( v304 != 289 )
            goto LABEL_27;
          v32 = sub_15F2060(v26 - 24);
          v266 = sub_1560180(v32 + 112, 17);
          if ( v266 )
            goto LABEL_27;
          v33 = *(_QWORD *)(v26 - 24 + 24 * (2LL - (*(_DWORD *)(v26 - 4) & 0xFFFFFFF)));
          if ( *(_BYTE *)(v33 + 16) != 13 )
            goto LABEL_27;
          v34 = *(_QWORD **)(v33 + 24);
          if ( *(_DWORD *)(v33 + 32) > 0x40u )
            v34 = (_QWORD *)*v34;
          v274 = v34;
          if ( !v34 )
            goto LABEL_27;
          v35 = sub_14AA530(v26 - 24);
          v36 = sub_14A2F00(v250, v35);
          if ( !v36 )
            goto LABEL_27;
          v37 = sub_15F2060(v26 - 24) + 112;
          v38 = sub_1560180(v37, 34);
          v39 = v274;
          if ( v38 || (v128 = sub_1560180(v37, 17), v39 = v274, v128) )
            v268 = v251[20381];
          else
            v268 = v251[20380];
          v275 = (__int64)v39;
          v40 = sub_16D5D50();
          v41 = v275;
          v42 = v40;
          v43 = *(_QWORD **)&dword_4FA0208[2];
          if ( !*(_QWORD *)&dword_4FA0208[2] )
            goto LABEL_60;
          v44 = dword_4FA0208;
          do
          {
            while ( 1 )
            {
              v45 = v43[2];
              v46 = v43[3];
              if ( v42 <= v43[4] )
                break;
              v43 = (_QWORD *)v43[3];
              if ( !v46 )
                goto LABEL_58;
            }
            v44 = v43;
            v43 = (_QWORD *)v43[2];
          }
          while ( v45 );
LABEL_58:
          if ( v44 == dword_4FA0208 )
            goto LABEL_60;
          if ( v42 < *((_QWORD *)v44 + 4) )
            goto LABEL_60;
          v103 = *((_QWORD *)v44 + 7);
          v104 = v44 + 12;
          if ( !v103 )
            goto LABEL_60;
          v105 = v44 + 12;
          do
          {
            while ( 1 )
            {
              v106 = *(_QWORD *)(v103 + 16);
              v107 = *(_QWORD *)(v103 + 24);
              if ( *(_DWORD *)(v103 + 32) >= dword_4FC3528 )
                break;
              v103 = *(_QWORD *)(v103 + 24);
              if ( !v107 )
                goto LABEL_138;
            }
            v105 = (_DWORD *)v103;
            v103 = *(_QWORD *)(v103 + 16);
          }
          while ( v106 );
LABEL_138:
          if ( v104 == v105 || dword_4FC3528 < v105[8] || (v48 = (unsigned int)dword_4FC35C0, !v105[9]) )
          {
LABEL_60:
            v47 = *(__int64 (**)())(*(_QWORD *)v251 + 440LL);
            v48 = 1;
            if ( v47 != sub_1D86260 )
            {
              LODWORD(v48) = ((__int64 (__fastcall *)(_DWORD *))v47)(v251);
              v41 = v275;
              v48 = (unsigned int)v48;
            }
          }
          v313[7] = v48;
          v313[4] = v41;
          v276 = v41;
          v313[0] = v26 - 24;
          memset(&v313[1], 0, 24);
          LODWORD(v313[5]) = 0;
          v313[6] = 0;
          memset(&v313[8], 0, 24);
          LOBYTE(v313[13]) = v35;
          v314 = (_BYTE *)v248;
          v49 = (_QWORD *)sub_16498A0(v26 - 24);
          v50 = v276;
          v317 = 0;
          v316 = 0;
          v315 = 0;
          v318 = v49;
          v319 = 0;
          v320 = 0;
          v321 = 0;
          v322 = 0;
          v51 = *(_QWORD *)(v26 + 16);
          v317 = v26;
          v316 = v51;
          v311[0] = *(unsigned __int8 **)(v26 + 24);
          if ( v311[0] )
          {
            sub_1623A60((__int64)v311, (__int64)v311[0], 2);
            v50 = v276;
            if ( v315 )
            {
              sub_161E7C0((__int64)&v315, (__int64)v315);
              v52 = v311[0];
              v50 = v276;
            }
            else
            {
              v52 = v311[0];
            }
            v315 = v52;
            if ( v52 )
            {
              v277 = v50;
              sub_1623210((__int64)v311, v52, (__int64)&v315);
              v50 = v277;
            }
          }
          v323 = v325;
          v324 = 0x800000000LL;
          if ( *(_DWORD *)(v36 + 8) )
          {
            v53 = *(_DWORD **)v36;
            v54 = 0;
            while ( 1 )
            {
              v55 = (unsigned int)v53[v54];
              if ( v50 >= v55 )
                break;
              if ( ++v54 == *(_DWORD *)(v36 + 8) )
              {
                LODWORD(v55) = v53[v54];
                break;
              }
            }
          }
          else
          {
            v54 = 0;
            LODWORD(v55) = **(_DWORD **)v36;
          }
          LODWORD(v313[5]) = v55;
          v56 = *(unsigned int *)(v36 + 8);
          v57 = 0;
          v58 = v36;
          v258 = 0;
          v59 = v26;
          v249 = (_QWORD *)(v26 - 24);
          v60 = v54;
          while ( 2 )
          {
            if ( v56 > v60 )
            {
              while ( 1 )
              {
                v61 = *(unsigned int *)(*(_QWORD *)v58 + 4 * v60);
                v278 = v50 % v61;
                if ( v50 / v61 + (unsigned int)v57 > v268 )
                {
                  v26 = v59;
                  goto LABEL_125;
                }
                if ( *(unsigned int *)(*(_QWORD *)v58 + 4 * v60) <= v50 )
                  break;
                if ( v56 <= ++v60 )
                  goto LABEL_76;
              }
              v92 = v258;
              v255 = *(_DWORD *)(*(_QWORD *)v58 + 4 * v60);
              v93 = 0;
              v94 = v50 / v255;
              do
              {
                if ( HIDWORD(v324) <= (unsigned int)v57 )
                {
                  v252 = v58;
                  v253 = v50;
                  v254 = v59;
                  sub_16CD150((__int64)&v323, v325, 0, 16, v50, v59);
                  v57 = (unsigned int)v324;
                  v58 = v252;
                  v50 = v253;
                  v59 = v254;
                }
                v95 = &v323[16 * v57];
                ++v93;
                v95[1] = v92;
                v92 += v255;
                *v95 = v61;
                v57 = (unsigned int)(v324 + 1);
                LODWORD(v324) = v324 + 1;
              }
              while ( v94 > v93 );
              v96 = v94 - 1;
              if ( v255 > v50 )
                v96 = 0;
              v258 += v255 + v255 * v96;
              if ( v255 > 1 )
                ++v313[6];
              v50 = v278;
              ++v60;
              if ( v278 )
              {
                v56 = *(unsigned int *)(v58 + 8);
                continue;
              }
            }
            break;
          }
LABEL_76:
          v62 = v249;
          v26 = v59;
          if ( !(_DWORD)v57 )
            goto LABEL_125;
          if ( LOBYTE(v313[13]) )
            LODWORD(v57) = (unsigned __int64)(unsigned int)v57 / v313[7]
                         - (((unsigned __int64)(unsigned int)v57 % v313[7] == 0)
                          - 1);
          if ( (_DWORD)v57 != 1 )
          {
            v63 = *(_QWORD **)(v313[0] + 40);
            v311[0] = "endblock";
            v312 = 259;
            v313[11] = sub_157FBF0(v63, (__int64 *)(v313[0] + 24), (__int64)v311);
            v64 = *(_QWORD *)(v313[11] + 48);
            if ( !v64 )
              BUG();
            v65 = *(_QWORD *)(v64 + 16);
            v317 = *(_QWORD *)(v313[11] + 48);
            v316 = v65;
            v66 = *(unsigned __int8 **)(v64 + 24);
            v311[0] = v66;
            if ( v66 )
            {
              sub_1623A60((__int64)v311, (__int64)v66, 2);
              v67 = v315;
              if ( !v315 )
              {
LABEL_83:
                v315 = v311[0];
                if ( v311[0] )
                  sub_1623210((__int64)v311, v311[0], (__int64)&v315);
                goto LABEL_85;
              }
            }
            else
            {
              v67 = v315;
              if ( !v315 )
              {
LABEL_85:
                v309[0] = (__int64)"phi.res";
                v310 = 259;
                v68 = (_QWORD *)sub_16498A0(v313[0]);
                v69 = sub_1643350(v68);
                v312 = 257;
                v70 = v69;
                v71 = sub_1648B60(64);
                v72 = v71;
                if ( v71 )
                {
                  v279 = v71;
                  v73 = v71;
                  sub_15F1EA0(v71, v70, 53, 0, 0, 0);
                  *(_DWORD *)(v279 + 56) = 2;
                  sub_164B780(v279, (__int64 *)v311);
                  sub_1648880(v279, *(_DWORD *)(v279 + 56), 1);
                  v72 = v279;
                }
                else
                {
                  v73 = 0;
                }
                if ( v316 )
                {
                  v74 = (__int64 *)v317;
                  v280 = v72;
                  sub_157E9D0(v316 + 40, v72);
                  v72 = v280;
                  v75 = *v74;
                  v76 = *(_QWORD *)(v280 + 24);
                  *(_QWORD *)(v280 + 32) = v74;
                  v75 &= 0xFFFFFFFFFFFFFFF8LL;
                  *(_QWORD *)(v280 + 24) = v75 | v76 & 7;
                  *(_QWORD *)(v75 + 8) = v280 + 24;
                  *v74 = *v74 & 7 | (v280 + 24);
                }
                v281 = v72;
                sub_164B780(v73, v309);
                v77 = v281;
                if ( v315 )
                {
                  v308 = v315;
                  sub_1623A60((__int64)&v308, (__int64)v315, 2);
                  v77 = v281;
                  v78 = *(_QWORD *)(v281 + 48);
                  v79 = v281 + 48;
                  if ( v78 )
                  {
                    sub_161E7C0(v281 + 48, v78);
                    v77 = v281;
                  }
                  v80 = v308;
                  *(_QWORD *)(v77 + 48) = v308;
                  if ( v80 )
                  {
                    v282 = v77;
                    sub_1623210((__int64)&v308, v80, v79);
                    v77 = v282;
                  }
                }
                v313[12] = v77;
                v259 = v313[11];
                v283 = *(_QWORD *)(v313[11] + 56);
                v311[0] = "res_block";
                v312 = 259;
                v81 = sub_16498A0(v313[0]);
                v82 = (_QWORD *)sub_22077B0(64);
                v83 = (__int64)v82;
                if ( v82 )
                  sub_157FB60(v82, v81, (__int64)v311, v283, v259);
                v84 = v313[13];
                v313[1] = v83;
                if ( !LOBYTE(v313[13]) )
                {
                  v150 = 8 * LODWORD(v313[5]);
                  v151 = (_QWORD *)sub_16498A0(v313[0]);
                  v152 = sub_1644900(v151, v150);
                  v310 = 259;
                  v312 = 257;
                  v316 = v313[1];
                  v317 = v313[1] + 40;
                  v309[0] = (__int64)"phi.src1";
                  v286 = v313[6];
                  v153 = sub_1648B60(64);
                  v154 = v153;
                  if ( v153 )
                  {
                    v262 = v153;
                    v155 = v153;
                    sub_15F1EA0(v153, v152, 53, 0, 0, 0);
                    *(_DWORD *)(v262 + 56) = v286;
                    sub_164B780(v262, (__int64 *)v311);
                    sub_1648880(v262, *(_DWORD *)(v262 + 56), 1);
                    v154 = v262;
                  }
                  else
                  {
                    v155 = 0;
                  }
                  if ( v316 )
                  {
                    v287 = v154;
                    v263 = (__int64 *)v317;
                    sub_157E9D0(v316 + 40, v154);
                    v154 = v287;
                    v156 = *(_QWORD *)(v287 + 24);
                    v157 = *v263;
                    *(_QWORD *)(v287 + 32) = v263;
                    v157 &= 0xFFFFFFFFFFFFFFF8LL;
                    *(_QWORD *)(v287 + 24) = v157 | v156 & 7;
                    *(_QWORD *)(v157 + 8) = v287 + 24;
                    *v263 = *v263 & 7 | (v287 + 24);
                  }
                  v288 = v154;
                  sub_164B780(v155, v309);
                  v158 = v288;
                  if ( v315 )
                  {
                    v308 = v315;
                    sub_1623A60((__int64)&v308, (__int64)v315, 2);
                    v158 = v288;
                    v159 = *(_QWORD *)(v288 + 48);
                    v160 = v288 + 48;
                    if ( v159 )
                    {
                      v264 = v288;
                      v289 = v288 + 48;
                      sub_161E7C0(v289, v159);
                      v158 = v264;
                      v160 = v289;
                    }
                    v161 = v308;
                    *(_QWORD *)(v158 + 48) = v308;
                    if ( v161 )
                    {
                      v290 = v158;
                      sub_1623210((__int64)&v308, v161, v160);
                      v158 = v290;
                    }
                  }
                  v313[2] = v158;
                  v309[0] = (__int64)"phi.src2";
                  v310 = 259;
                  v291 = v313[6];
                  v312 = 257;
                  v162 = sub_1648B60(64);
                  v163 = v162;
                  if ( v162 )
                  {
                    v265 = v162;
                    v164 = v162;
                    sub_15F1EA0(v162, v152, 53, 0, 0, 0);
                    *(_DWORD *)(v265 + 56) = v291;
                    sub_164B780(v265, (__int64 *)v311);
                    sub_1648880(v265, *(_DWORD *)(v265 + 56), 1);
                    v163 = v265;
                  }
                  else
                  {
                    v164 = 0;
                  }
                  if ( v316 )
                  {
                    v165 = (__int64 *)v317;
                    v292 = v163;
                    sub_157E9D0(v316 + 40, v163);
                    v163 = v292;
                    v166 = *v165;
                    v167 = *(_QWORD *)(v292 + 24);
                    *(_QWORD *)(v292 + 32) = v165;
                    v166 &= 0xFFFFFFFFFFFFFFF8LL;
                    *(_QWORD *)(v292 + 24) = v166 | v167 & 7;
                    *(_QWORD *)(v166 + 8) = v292 + 24;
                    *v165 = *v165 & 7 | (v292 + 24);
                  }
                  v293 = v163;
                  sub_164B780(v164, v309);
                  v168 = v293;
                  if ( v315 )
                  {
                    v308 = v315;
                    sub_1623A60((__int64)&v308, (__int64)v315, 2);
                    v168 = v293;
                    v169 = *(_QWORD *)(v293 + 48);
                    v170 = v293 + 48;
                    if ( v169 )
                    {
                      sub_161E7C0(v293 + 48, v169);
                      v168 = v293;
                    }
                    v171 = v308;
                    *(_QWORD *)(v168 + 48) = v308;
                    if ( v171 )
                    {
                      v294 = v168;
                      sub_1623210((__int64)&v308, v171, v170);
                      v168 = v294;
                    }
                  }
                  v313[3] = v168;
                  v84 = v313[13];
                }
                v260 = v26;
                for ( i = 0; ; ++i )
                {
                  v87 = v324;
                  if ( v84 )
                    v87 = (unsigned __int64)(unsigned int)v324 / v313[7]
                        - (((unsigned __int64)(unsigned int)v324 % v313[7] == 0)
                         - 1);
                  if ( i >= v87 )
                    break;
                  v88 = *(_QWORD *)(v313[11] + 56);
                  v284 = v313[11];
                  v311[0] = "loadbb";
                  v312 = 259;
                  v89 = sub_16498A0(v313[0]);
                  v90 = (_QWORD *)sub_22077B0(64);
                  v91 = (__int64)v90;
                  if ( v90 )
                    sub_157FB60(v90, v89, (__int64)v311, v88, v284);
                  v309[0] = v91;
                  v86 = v313[9];
                  if ( v313[9] == v313[10] )
                  {
                    sub_1292090((__int64)&v313[8], (_BYTE *)v313[9], v309);
                  }
                  else
                  {
                    if ( v313[9] )
                    {
                      *(_QWORD *)v313[9] = v91;
                      v86 = v313[9];
                    }
                    v313[9] = v86 + 8;
                  }
                  v84 = v313[13];
                }
                v26 = v260;
                v62 = v249;
                v108 = sub_157EBA0((__int64)v63);
                sub_15F4ED0(v108, 0, *(_QWORD *)v313[8]);
                goto LABEL_143;
              }
            }
            sub_161E7C0((__int64)&v315, (__int64)v67);
            goto LABEL_83;
          }
LABEL_143:
          v109 = *(unsigned __int8 **)(v313[0] + 48);
          v311[0] = v109;
          if ( v109 )
          {
            sub_1623A60((__int64)v311, (__int64)v109, 2);
            v110 = v315;
            if ( !v315 )
              goto LABEL_146;
          }
          else
          {
            v110 = v315;
            if ( !v315 )
              goto LABEL_148;
          }
          sub_161E7C0((__int64)&v315, (__int64)v110);
LABEL_146:
          v315 = v311[0];
          if ( v311[0] )
            sub_1623210((__int64)v311, v311[0], (__int64)&v315);
LABEL_148:
          v111 = v313[13];
          v99 = v324;
          if ( !LOBYTE(v313[13]) )
          {
            v97 = 0;
            if ( (_DWORD)v324 != 1 )
            {
              while ( v99 > v97 )
              {
                while ( 1 )
                {
                  v98 = v97++;
                  sub_1D88020((__int64)v313, v98, *(double *)a3.m128_u64, a4, a5);
                  v99 = v324;
                  if ( !LOBYTE(v313[13]) )
                    break;
                  if ( (unsigned int)((unsigned __int64)(unsigned int)v324 / v313[7])
                     - (((unsigned __int64)(unsigned int)v324 % v313[7] == 0)
                      - 1) <= v97 )
                    goto LABEL_123;
                }
              }
LABEL_123:
              sub_1D865C0((__int64)v313);
              v102 = v313[12];
              goto LABEL_124;
            }
            v172 = 8 * LODWORD(v313[4]);
            v173 = (_QWORD *)sub_16498A0(v313[0]);
            v305 = (__int64 *)sub_1644900(v173, v172);
            v174 = *(_QWORD *)(v313[0] - 24LL * (*(_DWORD *)(v313[0] + 20) & 0xFFFFFFF));
            v175 = *(__int64 ***)v174;
            v296 = *(_QWORD *)(v313[0] + 24 * (1LL - (*(_DWORD *)(v313[0] + 20) & 0xFFFFFFF)));
            if ( v305 != *(__int64 **)v174 )
            {
              v310 = 257;
              v176 = (__int64 **)sub_1647190(v305, 0);
              if ( v176 != *(__int64 ***)v174 )
              {
                if ( *(_BYTE *)(v174 + 16) > 0x10u )
                {
                  v312 = 257;
                  v174 = sub_15FDBD0(47, v174, (__int64)v176, (__int64)v311, 0);
                  if ( v316 )
                  {
                    v230 = (__int64 *)v317;
                    sub_157E9D0(v316 + 40, v174);
                    v231 = *(_QWORD *)(v174 + 24);
                    v232 = *v230;
                    *(_QWORD *)(v174 + 32) = v230;
                    v232 &= 0xFFFFFFFFFFFFFFF8LL;
                    *(_QWORD *)(v174 + 24) = v232 | v231 & 7;
                    *(_QWORD *)(v232 + 8) = v174 + 24;
                    *v230 = *v230 & 7 | (v174 + 24);
                  }
                  sub_164B780(v174, v309);
                  if ( v315 )
                  {
                    v308 = v315;
                    sub_1623A60((__int64)&v308, (__int64)v315, 2);
                    v233 = *(_QWORD *)(v174 + 48);
                    v234 = v174 + 48;
                    if ( v233 )
                    {
                      sub_161E7C0(v174 + 48, v233);
                      v234 = v174 + 48;
                    }
                    v235 = v308;
                    *(_QWORD *)(v174 + 48) = v308;
                    if ( v235 )
                      sub_1623210((__int64)&v308, v235, v234);
                  }
                }
                else
                {
                  v174 = sub_15A46C0(47, (__int64 ***)v174, v176, 0);
                }
              }
              v175 = (__int64 **)v305;
            }
            if ( *(__int64 ***)v296 != v175 )
            {
              v310 = 257;
              v177 = (__int64 **)sub_1647190((__int64 *)v175, 0);
              if ( v177 != *(__int64 ***)v296 )
              {
                if ( *(_BYTE *)(v296 + 16) > 0x10u )
                {
                  v312 = 257;
                  v296 = sub_15FDBD0(47, v296, (__int64)v177, (__int64)v311, 0);
                  if ( v316 )
                  {
                    v212 = (__int64 *)v317;
                    sub_157E9D0(v316 + 40, v296);
                    v213 = *v212;
                    v214 = *(_QWORD *)(v296 + 24);
                    *(_QWORD *)(v296 + 32) = v212;
                    v213 &= 0xFFFFFFFFFFFFFFF8LL;
                    *(_QWORD *)(v296 + 24) = v213 | v214 & 7;
                    *(_QWORD *)(v213 + 8) = v296 + 24;
                    *v212 = *v212 & 7 | (v296 + 24);
                  }
                  sub_164B780(v296, v309);
                  if ( v315 )
                  {
                    v308 = v315;
                    sub_1623A60((__int64)&v308, (__int64)v315, 2);
                    v215 = *(_QWORD *)(v296 + 48);
                    v216 = v296 + 48;
                    if ( v215 )
                    {
                      sub_161E7C0(v296 + 48, v215);
                      v216 = v296 + 48;
                    }
                    v217 = v308;
                    *(_QWORD *)(v296 + 48) = v308;
                    if ( v217 )
                      sub_1623210((__int64)&v308, v217, v216);
                  }
                }
                else
                {
                  v296 = sub_15A46C0(47, (__int64 ***)v296, v177, 0);
                }
              }
              v175 = (__int64 **)v305;
            }
            v312 = 257;
            v178 = sub_1648A60(64, 1u);
            v179 = (__int64)v178;
            if ( v178 )
            {
              v271 = v178;
              sub_15F9210((__int64)v178, (__int64)v175, v174, 0, 0, 0);
              v179 = (__int64)v271;
            }
            if ( v316 )
            {
              v180 = (__int64 *)v317;
              v272 = v179;
              sub_157E9D0(v316 + 40, v179);
              v179 = v272;
              v181 = *v180;
              v182 = *(_QWORD *)(v272 + 24);
              *(_QWORD *)(v272 + 32) = v180;
              v181 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v272 + 24) = v181 | v182 & 7;
              *(_QWORD *)(v181 + 8) = v272 + 24;
              *v180 = *v180 & 7 | (v272 + 24);
            }
            v273 = v179;
            sub_164B780(v179, (__int64 *)v311);
            v183 = v273;
            if ( v315 )
            {
              v309[0] = (__int64)v315;
              sub_1623A60((__int64)v309, (__int64)v315, 2);
              v183 = v273;
              v184 = *(_QWORD *)(v273 + 48);
              if ( v184 )
              {
                sub_161E7C0(v273 + 48, v184);
                v183 = v273;
              }
              v185 = (unsigned __int8 *)v309[0];
              *(_QWORD *)(v183 + 48) = v309[0];
              if ( v185 )
              {
                v267 = v183;
                sub_1623210((__int64)v309, v185, v273 + 48);
                v183 = v267;
              }
            }
            v306 = (__int64 ***)v183;
            v312 = 257;
            v186 = v305;
            v187 = sub_1648A60(64, 1u);
            v188 = (__int64)v187;
            if ( v187 )
              sub_15F9210((__int64)v187, (__int64)v186, v296, 0, 0, 0);
            if ( v316 )
            {
              v189 = (__int64 *)v317;
              sub_157E9D0(v316 + 40, v188);
              v190 = *(_QWORD *)(v188 + 24);
              v191 = *v189;
              *(_QWORD *)(v188 + 32) = v189;
              v191 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v188 + 24) = v191 | v190 & 7;
              *(_QWORD *)(v191 + 8) = v188 + 24;
              *v189 = *v189 & 7 | (v188 + 24);
            }
            sub_164B780(v188, (__int64 *)v311);
            if ( v315 )
            {
              v309[0] = (__int64)v315;
              sub_1623A60((__int64)v309, (__int64)v315, 2);
              v192 = *(_QWORD *)(v188 + 48);
              if ( v192 )
                sub_161E7C0(v188 + 48, v192);
              v193 = (unsigned __int8 *)v309[0];
              *(_QWORD *)(v188 + 48) = v309[0];
              if ( v193 )
                sub_1623210((__int64)v309, v193, v188 + 48);
            }
            v307 = (__int64 ***)v188;
            if ( *v314 )
            {
LABEL_252:
              if ( v313[4] <= 3uLL )
                goto LABEL_262;
              v312 = 257;
              v297 = sub_12AA0C0((__int64 *)&v315, 0x22u, v306, (__int64)v307, (__int64)v311);
              v312 = 257;
              v197 = (__int64 ***)sub_12AA0C0((__int64 *)&v315, 0x24u, v306, (__int64)v307, (__int64)v311);
              v310 = 257;
              v198 = (__int64 **)sub_1643350(v318);
              if ( v198 != *(__int64 ***)v297 )
              {
                if ( *(_BYTE *)(v297 + 16) > 0x10u )
                {
                  v312 = 257;
                  v297 = sub_15FDBD0(37, v297, (__int64)v198, (__int64)v311, 0);
                  if ( v316 )
                  {
                    v218 = (__int64 *)v317;
                    sub_157E9D0(v316 + 40, v297);
                    v219 = *v218;
                    v220 = *(_QWORD *)(v297 + 24);
                    *(_QWORD *)(v297 + 32) = v218;
                    v219 &= 0xFFFFFFFFFFFFFFF8LL;
                    *(_QWORD *)(v297 + 24) = v219 | v220 & 7;
                    *(_QWORD *)(v219 + 8) = v297 + 24;
                    *v218 = *v218 & 7 | (v297 + 24);
                  }
                  sub_164B780(v297, v309);
                  if ( v315 )
                  {
                    v308 = v315;
                    sub_1623A60((__int64)&v308, (__int64)v315, 2);
                    v221 = *(_QWORD *)(v297 + 48);
                    v222 = v297 + 48;
                    if ( v221 )
                    {
                      sub_161E7C0(v297 + 48, v221);
                      v222 = v297 + 48;
                    }
                    v223 = v308;
                    *(_QWORD *)(v297 + 48) = v308;
                    if ( v223 )
                      sub_1623210((__int64)&v308, v223, v222);
                  }
                }
                else
                {
                  v297 = sub_15A46C0(37, (__int64 ***)v297, v198, 0);
                }
              }
              v310 = 257;
              v199 = (__int64 **)sub_1643350(v318);
              if ( v199 != *v197 )
              {
                if ( *((_BYTE *)v197 + 16) > 0x10u )
                {
                  v312 = 257;
                  v197 = (__int64 ***)sub_15FDBD0(37, (__int64)v197, (__int64)v199, (__int64)v311, 0);
                  if ( v316 )
                  {
                    v224 = (unsigned __int64 *)v317;
                    sub_157E9D0(v316 + 40, (__int64)v197);
                    v225 = v197[3];
                    v226 = *v224;
                    v197[4] = (__int64 **)v224;
                    v226 &= 0xFFFFFFFFFFFFFFF8LL;
                    v197[3] = (__int64 **)(v226 | (unsigned __int8)v225 & 7);
                    *(_QWORD *)(v226 + 8) = v197 + 3;
                    *v224 = *v224 & 7 | (unsigned __int64)(v197 + 3);
                  }
                  sub_164B780((__int64)v197, v309);
                  if ( v315 )
                  {
                    v308 = v315;
                    sub_1623A60((__int64)&v308, (__int64)v315, 2);
                    v227 = (__int64)v197[6];
                    v228 = (__int64)(v197 + 6);
                    if ( v227 )
                    {
                      sub_161E7C0((__int64)(v197 + 6), v227);
                      v228 = (__int64)(v197 + 6);
                    }
                    v229 = v308;
                    v197[6] = (__int64 **)v308;
                    if ( v229 )
                      sub_1623210((__int64)&v308, v229, v228);
                  }
                }
                else
                {
                  v197 = (__int64 ***)sub_15A46C0(37, v197, v199, 0);
                }
              }
              v200 = (__int64 *)v297;
              v310 = 257;
              if ( *(_BYTE *)(v297 + 16) > 0x10u || *((_BYTE *)v197 + 16) > 0x10u )
              {
                v205 = (__int64)v197;
                v312 = 257;
                v204 = (__int64 *)v297;
LABEL_271:
                v102 = sub_15FB440(13, v204, v205, (__int64)v311, 0);
                v206 = v316;
                if ( v316 )
                {
LABEL_272:
                  v207 = (__int64 *)v317;
                  sub_157E9D0(v206 + 40, v102);
                  v208 = *(_QWORD *)(v102 + 24);
                  v209 = *v207;
                  *(_QWORD *)(v102 + 32) = v207;
                  v209 &= 0xFFFFFFFFFFFFFFF8LL;
                  *(_QWORD *)(v102 + 24) = v209 | v208 & 7;
                  *(_QWORD *)(v209 + 8) = v102 + 24;
                  *v207 = *v207 & 7 | (v102 + 24);
                }
LABEL_273:
                sub_164B780(v102, v309);
                if ( v315 )
                {
                  v308 = v315;
                  sub_1623A60((__int64)&v308, (__int64)v315, 2);
                  v210 = *(_QWORD *)(v102 + 48);
                  if ( v210 )
                    sub_161E7C0(v102 + 48, v210);
                  v211 = v308;
                  *(_QWORD *)(v102 + 48) = v308;
                  if ( v211 )
                    sub_1623210((__int64)&v308, v211, v102 + 48);
                }
                goto LABEL_124;
              }
            }
            else
            {
              if ( v313[4] != 1 )
              {
                v194 = (__int64 *)sub_15F2050(v313[0]);
                v195 = sub_15E26F0(v194, 6, (__int64 *)&v305, 1);
                v312 = 257;
                v196 = v195;
                v306 = (__int64 ***)sub_1285290(
                                      (__int64 *)&v315,
                                      *(_QWORD *)(v195 + 24),
                                      v195,
                                      (int)&v306,
                                      1,
                                      (__int64)v311,
                                      0);
                v312 = 257;
                v307 = (__int64 ***)sub_1285290(
                                      (__int64 *)&v315,
                                      *(_QWORD *)(v196 + 24),
                                      v196,
                                      (int)&v307,
                                      1,
                                      (__int64)v311,
                                      0);
                goto LABEL_252;
              }
LABEL_262:
              v310 = 257;
              v201 = (__int64 **)sub_1643350(v318);
              v202 = v306;
              if ( v201 != *v306 )
              {
                if ( *((_BYTE *)v306 + 16) > 0x10u )
                {
                  v312 = 257;
                  v202 = (_BYTE *)sub_15FDBD0(37, (__int64)v306, (__int64)v201, (__int64)v311, 0);
                  if ( v316 )
                  {
                    v236 = (__int64 *)v317;
                    sub_157E9D0(v316 + 40, (__int64)v202);
                    v237 = *((_QWORD *)v202 + 3);
                    v238 = *v236;
                    *((_QWORD *)v202 + 4) = v236;
                    v238 &= 0xFFFFFFFFFFFFFFF8LL;
                    *((_QWORD *)v202 + 3) = v238 | v237 & 7;
                    *(_QWORD *)(v238 + 8) = v202 + 24;
                    *v236 = *v236 & 7 | (unsigned __int64)(v202 + 24);
                  }
                  sub_164B780((__int64)v202, v309);
                  if ( v315 )
                  {
                    v308 = v315;
                    sub_1623A60((__int64)&v308, (__int64)v315, 2);
                    v239 = *((_QWORD *)v202 + 6);
                    v240 = (__int64)(v202 + 48);
                    if ( v239 )
                    {
                      sub_161E7C0((__int64)(v202 + 48), v239);
                      v240 = (__int64)(v202 + 48);
                    }
                    v241 = v308;
                    *((_QWORD *)v202 + 6) = v308;
                    if ( v241 )
                      sub_1623210((__int64)&v308, v241, v240);
                  }
                }
                else
                {
                  v202 = (_BYTE *)sub_15A46C0(37, v306, v201, 0);
                }
              }
              v306 = (__int64 ***)v202;
              v310 = 257;
              v203 = (__int64 **)sub_1643350(v318);
              v197 = v307;
              if ( v203 != *v307 )
              {
                if ( *((_BYTE *)v307 + 16) > 0x10u )
                {
                  v312 = 257;
                  v197 = (__int64 ***)sub_15FDBD0(37, (__int64)v307, (__int64)v203, (__int64)v311, 0);
                  if ( v316 )
                  {
                    v242 = (unsigned __int64 *)v317;
                    sub_157E9D0(v316 + 40, (__int64)v197);
                    v243 = v197[3];
                    v244 = *v242;
                    v197[4] = (__int64 **)v242;
                    v244 &= 0xFFFFFFFFFFFFFFF8LL;
                    v197[3] = (__int64 **)(v244 | (unsigned __int8)v243 & 7);
                    *(_QWORD *)(v244 + 8) = v197 + 3;
                    *v242 = *v242 & 7 | (unsigned __int64)(v197 + 3);
                  }
                  sub_164B780((__int64)v197, v309);
                  if ( v315 )
                  {
                    v308 = v315;
                    sub_1623A60((__int64)&v308, (__int64)v315, 2);
                    v245 = (__int64)v197[6];
                    v246 = (__int64)(v197 + 6);
                    if ( v245 )
                    {
                      sub_161E7C0((__int64)(v197 + 6), v245);
                      v246 = (__int64)(v197 + 6);
                    }
                    v247 = v308;
                    v197[6] = (__int64 **)v308;
                    if ( v247 )
                      sub_1623210((__int64)&v308, v247, v246);
                  }
                }
                else
                {
                  v197 = (__int64 ***)sub_15A46C0(37, v307, v203, 0);
                }
              }
              v200 = (__int64 *)v306;
              v307 = v197;
              v310 = 257;
              if ( *((_BYTE *)v306 + 16) > 0x10u || *((_BYTE *)v197 + 16) > 0x10u )
              {
                v204 = (__int64 *)v306;
                v312 = 257;
                v205 = (__int64)v197;
                goto LABEL_271;
              }
            }
            v102 = sub_15A2B60(v200, (__int64)v197, 0, 0, *(double *)a3.m128_u64, a4, a5);
            goto LABEL_124;
          }
          if ( (unsigned int)((unsigned __int64)(unsigned int)v324 / v313[7])
             - (((unsigned __int64)(unsigned int)v324 % v313[7] == 0)
              - 1) == 1 )
          {
            LODWORD(v307) = 0;
            v147 = sub_1D87110((__int64)v313, 0, &v307, *(double *)a3.m128_u64, a4, a5);
            v310 = 257;
            v102 = v147;
            v148 = (_QWORD *)sub_16498A0(v313[0]);
            v149 = (__int64 **)sub_1643350(v148);
            if ( v149 != *(__int64 ***)v102 )
            {
              if ( *(_BYTE *)(v102 + 16) > 0x10u )
              {
                v312 = 257;
                v102 = sub_15FDBD0(37, v102, (__int64)v149, (__int64)v311, 0);
                v206 = v316;
                if ( v316 )
                  goto LABEL_272;
                goto LABEL_273;
              }
              v102 = sub_15A46C0(37, (__int64 ***)v102, v149, 0);
            }
          }
          else
          {
            v261 = v26;
            v112 = 0;
            v113 = (unsigned int)v324;
            LODWORD(v308) = 0;
            v256 = v62;
            while ( 1 )
            {
              if ( v111 )
                LODWORD(v113) = v113 / v313[7] - ((v113 % v313[7] == 0) - 1);
              if ( v112 >= (unsigned int)v113 )
                break;
              v124 = v112;
              v125 = v112++;
              v126 = sub_1D87110((__int64)v313, v124, &v308, *(double *)a3.m128_u64, a4, a5);
              if ( v125 == ((v313[9] - v313[8]) >> 3) - 1 )
                v114 = v313[11];
              else
                v114 = *(_QWORD *)(v313[8] + 8LL * v112);
              v269 = v114;
              v115 = v313[1];
              v285 = v126;
              v116 = sub_1648A60(56, 3u);
              v117 = v116;
              if ( v116 )
                sub_15F83E0((__int64)v116, v115, v269, v285, 0);
              v312 = 257;
              if ( v316 )
              {
                v118 = (__int64 *)v317;
                sub_157E9D0(v316 + 40, (__int64)v117);
                v119 = v117[3];
                v120 = *v118;
                v117[4] = v118;
                v120 &= 0xFFFFFFFFFFFFFFF8LL;
                v117[3] = v120 | v119 & 7;
                *(_QWORD *)(v120 + 8) = v117 + 3;
                *v118 = *v118 & 7 | (unsigned __int64)(v117 + 3);
              }
              sub_164B780((__int64)v117, (__int64 *)v311);
              if ( v315 )
              {
                v309[0] = (__int64)v315;
                sub_1623A60((__int64)v309, (__int64)v315, 2);
                v121 = v117[6];
                v122 = (__int64)(v117 + 6);
                if ( v121 )
                {
                  sub_161E7C0((__int64)(v117 + 6), v121);
                  v122 = (__int64)(v117 + 6);
                }
                v123 = (unsigned __int8 *)v309[0];
                v117[6] = v309[0];
                if ( v123 )
                  sub_1623210((__int64)v309, v123, v122);
              }
              if ( v125 == ((v313[9] - v313[8]) >> 3) - 1 )
              {
                v129 = (_QWORD *)sub_16498A0(v313[0]);
                v130 = sub_1643350(v129);
                v131 = sub_159C470(v130, 0, 0);
                v134 = v313[12];
                v135 = v131;
                v136 = *(_QWORD *)(v313[8] + 8 * v125);
                v137 = *(_DWORD *)(v313[12] + 20) & 0xFFFFFFF;
                if ( v137 == *(_DWORD *)(v313[12] + 56) )
                {
                  v270 = v135;
                  v295 = v313[12];
                  sub_15F55D0(v313[12], 0, v313[12], v135, v132, v133);
                  v134 = v295;
                  v135 = v270;
                  v137 = *(_DWORD *)(v295 + 20) & 0xFFFFFFF;
                }
                v138 = (v137 + 1) & 0xFFFFFFF;
                v139 = v138 | *(_DWORD *)(v134 + 20) & 0xF0000000;
                *(_DWORD *)(v134 + 20) = v139;
                if ( (v139 & 0x40000000) != 0 )
                  v140 = *(_QWORD *)(v134 - 8);
                else
                  v140 = v134 - 24 * v138;
                v141 = (__int64 *)(v140 + 24LL * (unsigned int)(v138 - 1));
                if ( *v141 )
                {
                  v142 = v141[1];
                  v143 = v141[2] & 0xFFFFFFFFFFFFFFFCLL;
                  *(_QWORD *)v143 = v142;
                  if ( v142 )
                    *(_QWORD *)(v142 + 16) = v143 | *(_QWORD *)(v142 + 16) & 3LL;
                }
                *v141 = v135;
                if ( v135 )
                {
                  v144 = *(_QWORD *)(v135 + 8);
                  v141[1] = v144;
                  if ( v144 )
                    *(_QWORD *)(v144 + 16) = (unsigned __int64)(v141 + 1) | *(_QWORD *)(v144 + 16) & 3LL;
                  v141[2] = (v135 + 8) | v141[2] & 3;
                  *(_QWORD *)(v135 + 8) = v141;
                }
                v145 = *(_DWORD *)(v134 + 20) & 0xFFFFFFF;
                if ( (*(_BYTE *)(v134 + 23) & 0x40) != 0 )
                  v146 = *(_QWORD *)(v134 - 8);
                else
                  v146 = v134 - 24 * v145;
                *(_QWORD *)(v146 + 8LL * (unsigned int)(v145 - 1) + 24LL * *(unsigned int *)(v134 + 56) + 8) = v136;
              }
              v111 = v313[13];
              v113 = (unsigned int)v324;
            }
            v26 = v261;
            v62 = v256;
            sub_1D865C0((__int64)v313);
            v102 = v313[12];
          }
LABEL_124:
          sub_164D160((__int64)v62, v102, a3, a4, a5, a6, v100, v101, a9, a10);
          sub_15F20C0(v62);
          v266 = v300;
LABEL_125:
          if ( v323 != v325 )
            _libc_free((unsigned __int64)v323);
          if ( v315 )
            sub_161E7C0((__int64)&v315, (__int64)v315);
          if ( v313[8] )
            j_j___libc_free_0(v313[8], v313[10] - v313[8]);
          if ( v266 )
          {
            v298 = v266;
            v302 = *(_QWORD *)(a2 + 80);
LABEL_36:
            if ( v302 == a2 + 72 )
            {
              if ( v298 )
              {
                memset(v313, 0, sizeof(v313));
                LODWORD(v313[3]) = 2;
                v313[1] = (__int64)&v313[5];
                v313[2] = (__int64)&v313[5];
                v313[8] = (__int64)&v313[12];
                v313[9] = (__int64)&v313[12];
                LODWORD(v313[10]) = 2;
                goto LABEL_39;
              }
LABEL_169:
              v127 = &v313[5];
              v313[7] = 0;
              v313[8] = (__int64)&v313[12];
              v313[9] = (__int64)&v313[12];
              v313[3] = 0x100000002LL;
              v313[5] = (__int64)&unk_4F9EE48;
              v313[1] = (__int64)&v313[5];
              v313[2] = (__int64)&v313[5];
              v313[10] = 2;
              LODWORD(v313[11]) = 0;
              LODWORD(v313[4]) = 0;
              v313[0] = 1;
              if ( &v313[6] == &v313[5] )
                goto LABEL_173;
              do
              {
                if ( (unsigned __int64)*v127 < 0xFFFFFFFFFFFFFFFELL )
                  break;
                ++v127;
              }
              while ( v127 != &v313[6] );
              if ( v127 == &v313[6] )
              {
LABEL_39:
                v299 = 1;
                v30 = v313[9];
                v31 = v313[8];
              }
              else
              {
LABEL_173:
                v30 = v313[9];
                v31 = v313[8];
              }
              if ( v30 != v31 )
                _libc_free(v30);
              if ( v313[2] != v313[1] )
                _libc_free(v313[2]);
              return v299;
            }
            goto LABEL_18;
          }
LABEL_27:
          v26 = *(_QWORD *)(v26 + 8);
          if ( v303 == v26 )
            goto LABEL_35;
        }
        else
        {
          v26 = *(_QWORD *)(v26 + 8);
          if ( v303 == v26 )
            goto LABEL_35;
        }
      }
    }
  }
  return v299;
}
