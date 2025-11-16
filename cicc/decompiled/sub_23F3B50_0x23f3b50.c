// Function: sub_23F3B50
// Address: 0x23f3b50
//
__int64 __fastcall sub_23F3B50(__int64 a1, char *a2, _QWORD *a3, __int64 a4)
{
  int v8; // eax
  char v9; // dl
  char v10; // r13
  char v11; // bl
  int v12; // eax
  char v13; // dl
  unsigned __int64 v14; // rsi
  char *v15; // rax
  char *v16; // r8
  char *v17; // rdi
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // rax
  _DWORD *v21; // rdi
  __int64 v22; // rcx
  __int64 v23; // rdx
  unsigned __int64 v24; // rsi
  char *v25; // rax
  char *v26; // r9
  char *v27; // rdi
  __int64 v28; // rcx
  __int64 v29; // rdx
  __int64 v30; // rax
  _DWORD *v31; // rdi
  __int64 v32; // rcx
  __int64 v33; // rdx
  char v34; // al
  unsigned __int64 v35; // rsi
  char *v36; // rax
  char *v37; // r9
  char *v38; // rdi
  __int64 v39; // rcx
  __int64 v40; // rdx
  __int64 v41; // rax
  _DWORD *v42; // rdi
  __int64 v43; // rcx
  __int64 v44; // rdx
  char v45; // al
  unsigned __int64 v46; // rsi
  char *v47; // rax
  char *v48; // r9
  char *v49; // rdi
  __int64 v50; // rcx
  __int64 v51; // rdx
  __int64 v52; // rax
  _DWORD *v53; // rdi
  __int64 v54; // rcx
  __int64 v55; // rdx
  bool v56; // cc
  int v57; // eax
  __int64 *v58; // rax
  unsigned int v59; // ebx
  __m128i v60; // xmm2
  char v61; // r13
  __int64 v62; // rax
  char v63; // bl
  unsigned __int64 v64; // rsi
  char *v65; // rax
  char *v66; // r8
  char *v67; // rdi
  __int64 v68; // rcx
  __int64 v69; // rdx
  __int64 v70; // rax
  _DWORD *v71; // rdi
  __int64 v72; // rcx
  __int64 v73; // rdx
  _QWORD *v74; // r13
  _QWORD *v75; // rbx
  unsigned __int64 v76; // rsi
  _QWORD *v77; // rax
  _QWORD *v78; // rdi
  __int64 v79; // rcx
  __int64 v80; // rdx
  __int64 v81; // rax
  _QWORD *v82; // rdi
  __int64 v83; // rcx
  __int64 v84; // rdx
  _QWORD *v85; // r13
  _QWORD *v86; // rbx
  unsigned __int64 v87; // rsi
  _QWORD *v88; // rax
  _QWORD *v89; // rdi
  __int64 v90; // rcx
  __int64 v91; // rdx
  __int64 v92; // rax
  _QWORD *v93; // rdi
  __int64 v94; // rcx
  __int64 v95; // rdx
  _QWORD *v96; // r13
  _QWORD *v97; // rbx
  unsigned __int64 v98; // rsi
  _QWORD *v99; // rax
  _QWORD *v100; // rdi
  __int64 v101; // rcx
  __int64 v102; // rdx
  __int64 v103; // rax
  _QWORD *v104; // rdi
  __int64 v105; // rcx
  __int64 v106; // rdx
  __int64 v107; // rax
  __int64 v108; // r15
  __int64 *v109; // rax
  unsigned __int64 v110; // rax
  __int64 v111; // rax
  __int64 v112; // rdx
  __int64 v113; // r15
  __int64 v114; // r15
  __int64 *v115; // rax
  unsigned __int64 v116; // rax
  __int64 v117; // rax
  __int64 v118; // rdx
  __int64 v119; // r15
  __int64 v120; // r15
  __int64 *v121; // rax
  unsigned __int64 v122; // rax
  __int64 v123; // rax
  __int64 v124; // rdx
  __int64 v125; // r15
  __int64 v126; // r15
  __int64 *v127; // rax
  unsigned __int64 v128; // rax
  __int64 v129; // rax
  __int64 v130; // rdx
  __int64 v131; // r15
  __int64 v132; // r15
  __int64 *v133; // rax
  unsigned __int64 v134; // rax
  __int64 v135; // rax
  __int64 v136; // rdx
  __int64 v137; // r15
  __int64 v138; // r15
  __int64 *v139; // rax
  unsigned __int64 v140; // rax
  __int64 v141; // rax
  __int64 v142; // rdx
  __int64 v143; // r15
  __int64 v144; // r15
  __int64 *v145; // rax
  unsigned __int64 v146; // rax
  __int64 v147; // rax
  __int64 v148; // rdx
  __int64 v149; // r15
  __int64 v150; // r15
  __int64 *v151; // rax
  unsigned __int64 v152; // rax
  __int64 v153; // rax
  __int64 v154; // rdx
  __int64 v155; // r8
  unsigned int v156; // ebx
  __int64 v157; // rbx
  __int64 v158; // rdx
  unsigned __int64 v159; // rax
  int v160; // edx
  __int64 v161; // rbx
  __int64 v162; // rax
  __int64 v163; // rax
  __int64 v164; // rax
  __int64 v165; // rdx
  __int64 v166; // rcx
  __int64 v167; // r12
  __int64 v168; // rax
  __int64 v169; // rdx
  __int64 v170; // rcx
  _DWORD *v171; // rax
  bool v172; // dl
  bool v173; // zf
  char v174; // r15
  __int64 v175; // rcx
  unsigned __int64 *v176; // rsi
  __int64 v178; // [rsp+50h] [rbp-660h]
  __int64 v179; // [rsp+58h] [rbp-658h]
  _QWORD *v180; // [rsp+68h] [rbp-648h]
  _QWORD *v181; // [rsp+68h] [rbp-648h]
  __int64 v182; // [rsp+70h] [rbp-640h]
  char v183; // [rsp+80h] [rbp-630h]
  _QWORD *v184; // [rsp+80h] [rbp-630h]
  _QWORD *v185; // [rsp+80h] [rbp-630h]
  int v186; // [rsp+80h] [rbp-630h]
  int v187; // [rsp+88h] [rbp-628h]
  int v188; // [rsp+88h] [rbp-628h]
  int v189; // [rsp+8Ch] [rbp-624h]
  int v190; // [rsp+8Ch] [rbp-624h]
  char v191; // [rsp+90h] [rbp-620h]
  _QWORD *v192; // [rsp+90h] [rbp-620h]
  _QWORD *v193; // [rsp+90h] [rbp-620h]
  char v194; // [rsp+90h] [rbp-620h]
  __int64 v195; // [rsp+90h] [rbp-620h]
  char v196; // [rsp+98h] [rbp-618h]
  _QWORD *v197; // [rsp+98h] [rbp-618h]
  _QWORD *v198; // [rsp+98h] [rbp-618h]
  _QWORD *v199; // [rsp+98h] [rbp-618h]
  _QWORD *v200; // [rsp+98h] [rbp-618h]
  _QWORD *i; // [rsp+98h] [rbp-618h]
  __int64 v202; // [rsp+98h] [rbp-618h]
  __int64 v203; // [rsp+98h] [rbp-618h]
  __int64 v204; // [rsp+98h] [rbp-618h]
  __int64 v205; // [rsp+98h] [rbp-618h]
  __int64 v206; // [rsp+98h] [rbp-618h]
  __int64 v207; // [rsp+98h] [rbp-618h]
  __int64 v208; // [rsp+98h] [rbp-618h]
  __int64 v209; // [rsp+98h] [rbp-618h]
  __int64 v210; // [rsp+98h] [rbp-618h]
  __int64 v211; // [rsp+98h] [rbp-618h]
  __int64 v212; // [rsp+98h] [rbp-618h]
  __int64 v213; // [rsp+98h] [rbp-618h]
  __int64 v214; // [rsp+98h] [rbp-618h]
  __int64 v215; // [rsp+98h] [rbp-618h]
  __int64 v216; // [rsp+98h] [rbp-618h]
  __int64 v217; // [rsp+98h] [rbp-618h]
  __m128i v218; // [rsp+A0h] [rbp-610h] BYREF
  __int64 v219; // [rsp+B0h] [rbp-600h]
  __int64 v220; // [rsp+C0h] [rbp-5F0h] BYREF
  __int64 *v221; // [rsp+E0h] [rbp-5D0h] BYREF
  __int64 v222; // [rsp+E8h] [rbp-5C8h]
  __int64 v223; // [rsp+F0h] [rbp-5C0h] BYREF
  __int64 v224; // [rsp+F8h] [rbp-5B8h]
  __int64 v225; // [rsp+100h] [rbp-5B0h]
  _QWORD *v226; // [rsp+110h] [rbp-5A0h] BYREF
  char v227; // [rsp+118h] [rbp-598h]
  char v228; // [rsp+119h] [rbp-597h]
  char v229; // [rsp+11Ah] [rbp-596h]
  char v230; // [rsp+11Bh] [rbp-595h]
  char v231; // [rsp+11Ch] [rbp-594h]
  char v232; // [rsp+11Dh] [rbp-593h]
  char v233; // [rsp+11Eh] [rbp-592h]
  int v234; // [rsp+120h] [rbp-590h]
  int v235; // [rsp+124h] [rbp-58Ch]
  __int64 v236; // [rsp+128h] [rbp-588h]
  __int64 v237; // [rsp+130h] [rbp-580h]
  __int64 *v238; // [rsp+138h] [rbp-578h]
  unsigned __int64 v239[2]; // [rsp+140h] [rbp-570h] BYREF
  _BYTE v240[16]; // [rsp+150h] [rbp-560h] BYREF
  __int64 v241; // [rsp+160h] [rbp-550h]
  __int64 v242; // [rsp+168h] [rbp-548h]
  __int64 v243; // [rsp+170h] [rbp-540h]
  __m128i v244; // [rsp+178h] [rbp-538h]
  __int64 v245; // [rsp+188h] [rbp-528h]
  __int64 v246; // [rsp+190h] [rbp-520h]
  __int64 v247; // [rsp+198h] [rbp-518h]
  __int64 v248; // [rsp+1A0h] [rbp-510h]
  __int64 v249; // [rsp+1A8h] [rbp-508h]
  __int64 v250; // [rsp+1B0h] [rbp-500h]
  __int64 v251; // [rsp+1B8h] [rbp-4F8h]
  __int64 v252; // [rsp+1C0h] [rbp-4F0h]
  __int64 v253; // [rsp+1C8h] [rbp-4E8h]
  __int64 v254; // [rsp+1D0h] [rbp-4E0h]
  __int64 v255; // [rsp+1D8h] [rbp-4D8h]
  __int64 v256; // [rsp+1E0h] [rbp-4D0h]
  __int64 v257; // [rsp+1E8h] [rbp-4C8h]
  __int64 v258; // [rsp+1F0h] [rbp-4C0h]
  __int64 v259; // [rsp+1F8h] [rbp-4B8h]
  __int64 v260; // [rsp+200h] [rbp-4B0h]
  __int64 v261; // [rsp+208h] [rbp-4A8h]
  __int64 v262; // [rsp+210h] [rbp-4A0h]
  __int64 v263; // [rsp+218h] [rbp-498h]
  __int64 v264; // [rsp+220h] [rbp-490h]
  unsigned __int64 v265[12]; // [rsp+230h] [rbp-480h] BYREF
  __int64 v266; // [rsp+290h] [rbp-420h]
  __int64 v267; // [rsp+298h] [rbp-418h]
  __int64 v268; // [rsp+2A0h] [rbp-410h]
  __m128i v269; // [rsp+2A8h] [rbp-408h] BYREF
  _QWORD v270[88]; // [rsp+2B8h] [rbp-3F8h] BYREF
  __int128 v271; // [rsp+578h] [rbp-138h]
  __int128 v272; // [rsp+588h] [rbp-128h]
  __int128 v273; // [rsp+598h] [rbp-118h]
  __int128 v274; // [rsp+5A8h] [rbp-108h]
  __int128 v275; // [rsp+5B8h] [rbp-F8h]
  __int128 v276; // [rsp+5C8h] [rbp-E8h]
  __int128 v277; // [rsp+5D8h] [rbp-D8h]
  __int128 v278; // [rsp+5E8h] [rbp-C8h]
  __int64 v279; // [rsp+5F8h] [rbp-B8h]
  __int64 v280; // [rsp+600h] [rbp-B0h]
  __int64 v281; // [rsp+608h] [rbp-A8h]
  __int64 v282; // [rsp+610h] [rbp-A0h]
  __int64 v283; // [rsp+618h] [rbp-98h]
  __int64 v284; // [rsp+620h] [rbp-90h]
  __int64 v285; // [rsp+628h] [rbp-88h]
  __int64 v286; // [rsp+630h] [rbp-80h]
  __int64 v287; // [rsp+638h] [rbp-78h]
  __int64 v288; // [rsp+640h] [rbp-70h]
  __int64 v289; // [rsp+648h] [rbp-68h]
  unsigned int v290; // [rsp+650h] [rbp-60h]
  __int64 v291; // [rsp+658h] [rbp-58h]
  __int64 v292; // [rsp+660h] [rbp-50h]
  __int64 v293; // [rsp+668h] [rbp-48h]
  __int64 v294; // [rsp+670h] [rbp-40h]
  int v295; // [rsp+678h] [rbp-38h]
  int v296; // [rsp+67Ch] [rbp-34h]

  if ( (unsigned __int8)sub_29F38C0(a3, "nosanitize_address", 18) )
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
  v8 = *((_DWORD *)a2 + 7);
  v9 = *a2;
  v226 = a3;
  v10 = a2[21];
  v11 = a2[20];
  v189 = v8;
  v183 = v9;
  v187 = *((_DWORD *)a2 + 6);
  v196 = a2[1];
  v191 = a2[16];
  v12 = sub_23DF0D0(&dword_4FE25E8);
  v13 = v183;
  if ( v12 > 0 )
    v13 = qword_4FE2668;
  v227 = v13;
  v184 = sub_C52410();
  v14 = sub_C959E0();
  v15 = (char *)v184[2];
  v16 = (char *)(v184 + 1);
  if ( v15 )
  {
    v17 = (char *)(v184 + 1);
    do
    {
      while ( 1 )
      {
        v18 = *((_QWORD *)v15 + 2);
        v19 = *((_QWORD *)v15 + 3);
        if ( v14 <= *((_QWORD *)v15 + 4) )
          break;
        v15 = (char *)*((_QWORD *)v15 + 3);
        if ( !v19 )
          goto LABEL_11;
      }
      v17 = v15;
      v15 = (char *)*((_QWORD *)v15 + 2);
    }
    while ( v18 );
LABEL_11:
    if ( v16 != v17 && v14 >= *((_QWORD *)v17 + 4) )
      v16 = v17;
  }
  v185 = v16;
  if ( v16 != (char *)sub_C52410() + 8 )
  {
    v20 = v185[7];
    if ( v20 )
    {
      v21 = v185 + 6;
      do
      {
        while ( 1 )
        {
          v22 = *(_QWORD *)(v20 + 16);
          v23 = *(_QWORD *)(v20 + 24);
          if ( *(_DWORD *)(v20 + 32) >= dword_4FE2428 )
            break;
          v20 = *(_QWORD *)(v20 + 24);
          if ( !v23 )
            goto LABEL_20;
        }
        v21 = (_DWORD *)v20;
        v20 = *(_QWORD *)(v20 + 16);
      }
      while ( v22 );
LABEL_20:
      if ( v21 != (_DWORD *)(v185 + 6) && dword_4FE2428 >= v21[8] && (int)v21[9] > 0 )
        v191 = qword_4FE24A8;
    }
  }
  v228 = v191;
  v192 = sub_C52410();
  v24 = sub_C959E0();
  v25 = (char *)v192[2];
  v26 = (char *)(v192 + 1);
  if ( v25 )
  {
    v27 = (char *)(v192 + 1);
    do
    {
      while ( 1 )
      {
        v28 = *((_QWORD *)v25 + 2);
        v29 = *((_QWORD *)v25 + 3);
        if ( v24 <= *((_QWORD *)v25 + 4) )
          break;
        v25 = (char *)*((_QWORD *)v25 + 3);
        if ( !v29 )
          goto LABEL_27;
      }
      v27 = v25;
      v25 = (char *)*((_QWORD *)v25 + 2);
    }
    while ( v28 );
LABEL_27:
    if ( v27 != v26 && v24 >= *((_QWORD *)v27 + 4) )
      v26 = v27;
  }
  v193 = v26;
  if ( v26 != (char *)sub_C52410() + 8 )
  {
    v30 = v193[7];
    if ( v30 )
    {
      v31 = v193 + 6;
      do
      {
        while ( 1 )
        {
          v32 = *(_QWORD *)(v30 + 16);
          v33 = *(_QWORD *)(v30 + 24);
          if ( *(_DWORD *)(v30 + 32) >= dword_4FE2508 )
            break;
          v30 = *(_QWORD *)(v30 + 24);
          if ( !v33 )
            goto LABEL_36;
        }
        v31 = (_DWORD *)v30;
        v30 = *(_QWORD *)(v30 + 16);
      }
      while ( v32 );
LABEL_36:
      if ( v31 != (_DWORD *)(v193 + 6) && dword_4FE2508 >= v31[8] && (int)v31[9] > 0 )
        v196 = byte_4FE2588;
    }
  }
  v229 = v196;
  v34 = 0;
  if ( v11 )
  {
    v34 = byte_4FDFF68;
    if ( byte_4FDFF68 )
      v34 = v227 ^ 1;
  }
  v230 = v34;
  v197 = sub_C52410();
  v35 = sub_C959E0();
  v36 = (char *)v197[2];
  v37 = (char *)(v197 + 1);
  if ( v36 )
  {
    v38 = (char *)(v197 + 1);
    do
    {
      while ( 1 )
      {
        v39 = *((_QWORD *)v36 + 2);
        v40 = *((_QWORD *)v36 + 3);
        if ( v35 <= *((_QWORD *)v36 + 4) )
          break;
        v36 = (char *)*((_QWORD *)v36 + 3);
        if ( !v40 )
          goto LABEL_46;
      }
      v38 = v36;
      v36 = (char *)*((_QWORD *)v36 + 2);
    }
    while ( v39 );
LABEL_46:
    if ( v38 != v37 && v35 >= *((_QWORD *)v38 + 4) )
      v37 = v38;
  }
  v198 = v37;
  if ( v37 == (char *)sub_C52410() + 8 || (v41 = v198[7]) == 0 )
  {
    v45 = v10;
  }
  else
  {
    v42 = v198 + 6;
    do
    {
      while ( 1 )
      {
        v43 = *(_QWORD *)(v41 + 16);
        v44 = *(_QWORD *)(v41 + 24);
        if ( *(_DWORD *)(v41 + 32) >= dword_4FE00A8 )
          break;
        v41 = *(_QWORD *)(v41 + 24);
        if ( !v44 )
          goto LABEL_55;
      }
      v42 = (_DWORD *)v41;
      v41 = *(_QWORD *)(v41 + 16);
    }
    while ( v43 );
LABEL_55:
    v45 = v10;
    if ( v42 != (_DWORD *)(v198 + 6) && dword_4FE00A8 >= v42[8] )
    {
      v45 = byte_4FE0128;
      if ( (int)v42[9] <= 0 )
        v45 = v10;
    }
  }
  v231 = v45;
  v199 = sub_C52410();
  v46 = sub_C959E0();
  v47 = (char *)v199[2];
  v48 = (char *)(v199 + 1);
  if ( v47 )
  {
    v49 = (char *)(v199 + 1);
    do
    {
      while ( 1 )
      {
        v50 = *((_QWORD *)v47 + 2);
        v51 = *((_QWORD *)v47 + 3);
        if ( v46 <= *((_QWORD *)v47 + 4) )
          break;
        v47 = (char *)*((_QWORD *)v47 + 3);
        if ( !v51 )
          goto LABEL_62;
      }
      v49 = v47;
      v47 = (char *)*((_QWORD *)v47 + 2);
    }
    while ( v50 );
LABEL_62:
    if ( v49 != v48 && v46 >= *((_QWORD *)v49 + 4) )
      v48 = v49;
  }
  v200 = v48;
  if ( v48 != (char *)sub_C52410() + 8 )
  {
    v52 = v200[7];
    if ( v52 )
    {
      v53 = v200 + 6;
      do
      {
        while ( 1 )
        {
          v54 = *(_QWORD *)(v52 + 16);
          v55 = *(_QWORD *)(v52 + 24);
          if ( *(_DWORD *)(v52 + 32) >= dword_4FDFFC8 )
            break;
          v52 = *(_QWORD *)(v52 + 24);
          if ( !v55 )
            goto LABEL_71;
        }
        v53 = (_DWORD *)v52;
        v52 = *(_QWORD *)(v52 + 16);
      }
      while ( v54 );
LABEL_71:
      if ( v53 != (_DWORD *)(v200 + 6) && dword_4FDFFC8 >= v53[8] && (int)v53[9] > 0 )
        v10 = byte_4FE0048;
    }
  }
  v232 = v10;
  if ( v11 )
  {
    v11 = byte_4FDFE88;
    if ( byte_4FDFE88 )
      v11 = v227 ^ 1;
  }
  v233 = v11;
  v234 = v187;
  v240[0] = 0;
  v239[1] = 0;
  v56 = (int)sub_23DF0D0(dword_4FE0968) <= 0;
  v57 = v189;
  if ( !v56 )
    v57 = dword_4FE09E8;
  v241 = 0;
  v242 = 0;
  v235 = v57;
  v239[0] = (unsigned __int64)v240;
  v58 = (__int64 *)*a3;
  v243 = 0;
  v238 = v58;
  v246 = 0;
  v247 = 0;
  v248 = 0;
  v249 = 0;
  v250 = 0;
  v251 = 0;
  v252 = 0;
  v253 = 0;
  v254 = 0;
  v255 = 0;
  v256 = 0;
  v257 = 0;
  v258 = 0;
  v259 = 0;
  v260 = 0;
  v261 = 0;
  v262 = 0;
  v263 = 0;
  v264 = 0;
  v59 = sub_AE2980((__int64)(a3 + 39), 0)[1];
  v236 = sub_BCD140(v238, v59);
  v237 = sub_BCE3C0(v238, 0);
  sub_2240AE0(v239, a3 + 29);
  v241 = a3[33];
  v242 = a3[34];
  v243 = a3[35];
  sub_23DF1C0((__int64)&v218, v239, v59, v227);
  v60 = _mm_loadu_si128(&v218);
  v245 = v219;
  v244 = v60;
  if ( dword_4FDFC28 != 2 )
    v234 = dword_4FDFC28;
  v179 = 0;
  v178 = *(_QWORD *)(sub_BC0510(a4, &unk_4F82418, (__int64)a3) + 8);
  if ( (_BYTE)qword_4FE2208 )
    v179 = sub_BC0510(a4, &unk_4F87F18, (__int64)a3) + 8;
  for ( i = (_QWORD *)a3[4]; a3 + 3 != i; i = (_QWORD *)i[1] )
  {
    v265[0] = (unsigned __int64)a3;
    v265[4] = 0;
    v61 = a2[1];
    v62 = (__int64)(i - 7);
    LOBYTE(v265[5]) = 0;
    v63 = a2[2];
    if ( !i )
      v62 = 0;
    memset(&v265[7], 0, 24);
    v182 = v62;
    v186 = *((_DWORD *)a2 + 1);
    v194 = *a2;
    v190 = *((_DWORD *)a2 + 3);
    v188 = *((_DWORD *)a2 + 2);
    v265[3] = (unsigned __int64)&v265[5];
    v180 = sub_C52410();
    v64 = sub_C959E0();
    v65 = (char *)v180[2];
    v66 = (char *)(v180 + 1);
    if ( v65 )
    {
      v67 = (char *)(v180 + 1);
      do
      {
        while ( 1 )
        {
          v68 = *((_QWORD *)v65 + 2);
          v69 = *((_QWORD *)v65 + 3);
          if ( v64 <= *((_QWORD *)v65 + 4) )
            break;
          v65 = (char *)*((_QWORD *)v65 + 3);
          if ( !v69 )
            goto LABEL_90;
        }
        v67 = v65;
        v65 = (char *)*((_QWORD *)v65 + 2);
      }
      while ( v68 );
LABEL_90:
      if ( v67 != v66 && v64 >= *((_QWORD *)v67 + 4) )
        v66 = v67;
    }
    v181 = v66;
    if ( v66 != (char *)sub_C52410() + 8 )
    {
      v70 = v181[7];
      if ( v70 )
      {
        v71 = v181 + 6;
        do
        {
          while ( 1 )
          {
            v72 = *(_QWORD *)(v70 + 16);
            v73 = *(_QWORD *)(v70 + 24);
            if ( *(_DWORD *)(v70 + 32) >= dword_4FE25E8 )
              break;
            v70 = *(_QWORD *)(v70 + 24);
            if ( !v73 )
              goto LABEL_99;
          }
          v71 = (_DWORD *)v70;
          v70 = *(_QWORD *)(v70 + 16);
        }
        while ( v72 );
LABEL_99:
        if ( v71 != (_DWORD *)(v181 + 6) && dword_4FE25E8 >= v71[8] && (int)v71[9] > 0 )
          v194 = qword_4FE2668;
      }
    }
    BYTE4(v265[10]) = v194;
    if ( (int)sub_23DF0D0(&dword_4FE2508) > 0 )
      v61 = byte_4FE2588;
    if ( !v63 )
      v63 = qword_4FE1608;
    BYTE5(v265[10]) = v61;
    BYTE6(v265[10]) = v63;
    v74 = sub_C52410();
    v75 = v74 + 1;
    v76 = sub_C959E0();
    v77 = (_QWORD *)v74[2];
    if ( v77 )
    {
      v78 = v74 + 1;
      do
      {
        while ( 1 )
        {
          v79 = v77[2];
          v80 = v77[3];
          if ( v76 <= v77[4] )
            break;
          v77 = (_QWORD *)v77[3];
          if ( !v80 )
            goto LABEL_112;
        }
        v78 = v77;
        v77 = (_QWORD *)v77[2];
      }
      while ( v79 );
LABEL_112:
      if ( v78 != v75 && v76 >= v78[4] )
        v75 = v78;
    }
    if ( v75 != (_QWORD *)((char *)sub_C52410() + 8) )
    {
      v81 = v75[7];
      if ( v81 )
      {
        v82 = v75 + 6;
        do
        {
          while ( 1 )
          {
            v83 = *(_QWORD *)(v81 + 16);
            v84 = *(_QWORD *)(v81 + 24);
            if ( *(_DWORD *)(v81 + 32) >= dword_4FE1748 )
              break;
            v81 = *(_QWORD *)(v81 + 24);
            if ( !v84 )
              goto LABEL_121;
          }
          v82 = (_QWORD *)v81;
          v81 = *(_QWORD *)(v81 + 16);
        }
        while ( v83 );
LABEL_121:
        if ( v82 != v75 + 6 && dword_4FE1748 >= *((_DWORD *)v82 + 8) && *((_DWORD *)v82 + 9) )
          v186 = dword_4FE17C8;
      }
    }
    memset(&v270[1], 0, 48);
    LODWORD(v265[11]) = v186;
    memset(&v270[8], 0, 640);
    v287 = 0;
    v271 = 0;
    v272 = 0;
    v273 = 0;
    v274 = 0;
    v275 = 0;
    v276 = 0;
    v277 = 0;
    v278 = 0;
    v286 = v179;
    v279 = 0;
    v280 = 0;
    v281 = 0;
    v282 = 0;
    v283 = 0;
    v284 = 0;
    v285 = 0;
    v288 = 0;
    v289 = 0;
    v290 = 0;
    v291 = 0;
    v292 = 0;
    v293 = 0;
    v294 = 0;
    v85 = sub_C52410();
    v86 = v85 + 1;
    v87 = sub_C959E0();
    v88 = (_QWORD *)v85[2];
    if ( v88 )
    {
      v89 = v85 + 1;
      do
      {
        while ( 1 )
        {
          v90 = v88[2];
          v91 = v88[3];
          if ( v87 <= v88[4] )
            break;
          v88 = (_QWORD *)v88[3];
          if ( !v91 )
            goto LABEL_130;
        }
        v89 = v88;
        v88 = (_QWORD *)v88[2];
      }
      while ( v90 );
LABEL_130:
      if ( v89 != v86 && v87 >= v89[4] )
        v86 = v89;
    }
    if ( v86 != (_QWORD *)((char *)sub_C52410() + 8) )
    {
      v92 = v86[7];
      if ( v92 )
      {
        v93 = v86 + 6;
        do
        {
          while ( 1 )
          {
            v94 = *(_QWORD *)(v92 + 16);
            v95 = *(_QWORD *)(v92 + 24);
            if ( *(_DWORD *)(v92 + 32) >= dword_4FE0F68 )
              break;
            v92 = *(_QWORD *)(v92 + 24);
            if ( !v95 )
              goto LABEL_139;
          }
          v93 = (_QWORD *)v92;
          v92 = *(_QWORD *)(v92 + 16);
        }
        while ( v94 );
LABEL_139:
        if ( v93 != v86 + 6 && dword_4FE0F68 >= *((_DWORD *)v93 + 8) && *((int *)v93 + 9) > 0 )
          v188 = qword_4FE0FE8;
      }
    }
    v295 = v188;
    v96 = sub_C52410();
    v97 = v96 + 1;
    v98 = sub_C959E0();
    v99 = (_QWORD *)v96[2];
    if ( v99 )
    {
      v100 = v96 + 1;
      do
      {
        while ( 1 )
        {
          v101 = v99[2];
          v102 = v99[3];
          if ( v98 <= v99[4] )
            break;
          v99 = (_QWORD *)v99[3];
          if ( !v102 )
            goto LABEL_148;
        }
        v100 = v99;
        v99 = (_QWORD *)v99[2];
      }
      while ( v101 );
LABEL_148:
      if ( v100 != v97 && v98 >= v100[4] )
        v97 = v100;
    }
    if ( v97 != (_QWORD *)((char *)sub_C52410() + 8) )
    {
      v103 = v97[7];
      if ( v103 )
      {
        v104 = v97 + 6;
        do
        {
          while ( 1 )
          {
            v105 = *(_QWORD *)(v103 + 16);
            v106 = *(_QWORD *)(v103 + 24);
            if ( *(_DWORD *)(v103 + 32) >= dword_4FE19A8 )
              break;
            v103 = *(_QWORD *)(v103 + 24);
            if ( !v106 )
              goto LABEL_157;
          }
          v104 = (_QWORD *)v103;
          v103 = *(_QWORD *)(v103 + 16);
        }
        while ( v105 );
LABEL_157:
        if ( v104 != v97 + 6 && dword_4FE19A8 >= *((_DWORD *)v104 + 8) && *((int *)v104 + 9) > 0 )
          v190 = qword_4FE1A28;
      }
    }
    v296 = v190;
    v265[1] = *a3;
    v265[2] = (unsigned __int64)(a3 + 39);
    LODWORD(v265[10]) = sub_AE2980((__int64)(a3 + 39), 0)[1];
    v266 = sub_BCD140((_QWORD *)v265[1], v265[10]);
    v268 = sub_BCE3C0((__int64 *)v265[1], 0);
    v267 = sub_BCB2D0((_QWORD *)v265[1]);
    sub_2240AE0(&v265[3], a3 + 29);
    v265[7] = a3[33];
    v265[8] = a3[34];
    v265[9] = a3[35];
    sub_23DF1C0((__int64)&v218, &v265[3], v265[10], SBYTE4(v265[10]));
    v269 = _mm_loadu_si128(&v218);
    v270[0] = v219;
    v107 = sub_BC1CD0(v178, &unk_4F6D3F8, v182);
    sub_23F0B80((__int64)v265, v182, (__int64 *)(v107 + 8));
    sub_C7D6A0(v288, 16LL * v290, 8);
    if ( (unsigned __int64 *)v265[3] != &v265[5] )
      j_j___libc_free_0(v265[3]);
  }
  v265[1] = 0x200000000LL;
  WORD2(v267) = 512;
  v108 = (__int64)v226;
  v265[9] = (unsigned __int64)v238;
  v202 = v236;
  v269.m128i_i64[1] = (__int64)&unk_49DA100;
  v265[0] = (unsigned __int64)&v265[2];
  v265[10] = (unsigned __int64)&v269.m128i_u64[1];
  v265[11] = (unsigned __int64)v270;
  v266 = 0;
  LODWORD(v267) = 0;
  BYTE6(v267) = 7;
  v268 = 0;
  v269.m128i_i64[0] = 0;
  memset(&v265[6], 0, 18);
  v270[0] = &unk_49DA0B0;
  v109 = (__int64 *)sub_BCB120(v238);
  v221 = &v223;
  v223 = v202;
  v222 = 0x100000001LL;
  v110 = sub_BCF480(v109, &v223, 1, 0);
  v111 = sub_BA8C10(v108, (__int64)"__asan_before_dynamic_init", 0x1Au, v110, 0);
  v113 = v112;
  if ( v221 != &v223 )
  {
    v203 = v111;
    _libc_free((unsigned __int64)v221);
    v111 = v203;
  }
  v247 = v113;
  v246 = v111;
  v114 = (__int64)v226;
  v115 = (__int64 *)sub_BCB120((_QWORD *)v265[9]);
  v221 = &v223;
  v222 = 0;
  v116 = sub_BCF480(v115, &v223, 0, 0);
  v117 = sub_BA8C10(v114, (__int64)"__asan_after_dynamic_init", 0x19u, v116, 0);
  v119 = v118;
  if ( v221 != &v223 )
  {
    v204 = v117;
    _libc_free((unsigned __int64)v221);
    v117 = v204;
  }
  v249 = v119;
  v248 = v117;
  v120 = (__int64)v226;
  v205 = v236;
  v121 = (__int64 *)sub_BCB120((_QWORD *)v265[9]);
  v221 = &v223;
  v223 = v205;
  v224 = v205;
  v222 = 0x200000002LL;
  v122 = sub_BCF480(v121, &v223, 2, 0);
  v123 = sub_BA8C10(v120, (__int64)"__asan_register_globals", 0x17u, v122, 0);
  v125 = v124;
  if ( v221 != &v223 )
  {
    v206 = v123;
    _libc_free((unsigned __int64)v221);
    v123 = v206;
  }
  v251 = v125;
  v250 = v123;
  v126 = (__int64)v226;
  v207 = v236;
  v127 = (__int64 *)sub_BCB120((_QWORD *)v265[9]);
  v221 = &v223;
  v223 = v207;
  v224 = v207;
  v222 = 0x200000002LL;
  v128 = sub_BCF480(v127, &v223, 2, 0);
  v129 = sub_BA8C10(v126, (__int64)"__asan_unregister_globals", 0x19u, v128, 0);
  v131 = v130;
  if ( v221 != &v223 )
  {
    v208 = v129;
    _libc_free((unsigned __int64)v221);
    v129 = v208;
  }
  v253 = v131;
  v252 = v129;
  v132 = (__int64)v226;
  v209 = v236;
  v133 = (__int64 *)sub_BCB120((_QWORD *)v265[9]);
  v221 = &v223;
  v223 = v209;
  v222 = 0x100000001LL;
  v134 = sub_BCF480(v133, &v223, 1, 0);
  v135 = sub_BA8C10(v132, (__int64)"__asan_register_image_globals", 0x1Du, v134, 0);
  v137 = v136;
  if ( v221 != &v223 )
  {
    v210 = v135;
    _libc_free((unsigned __int64)v221);
    v135 = v210;
  }
  v255 = v137;
  v254 = v135;
  v138 = (__int64)v226;
  v211 = v236;
  v139 = (__int64 *)sub_BCB120((_QWORD *)v265[9]);
  v221 = &v223;
  v223 = v211;
  v222 = 0x100000001LL;
  v140 = sub_BCF480(v139, &v223, 1, 0);
  v141 = sub_BA8C10(v138, (__int64)"__asan_unregister_image_globals", 0x1Fu, v140, 0);
  v143 = v142;
  if ( v221 != &v223 )
  {
    v212 = v141;
    _libc_free((unsigned __int64)v221);
    v141 = v212;
  }
  v257 = v143;
  v256 = v141;
  v144 = (__int64)v226;
  v213 = v236;
  v145 = (__int64 *)sub_BCB120((_QWORD *)v265[9]);
  v221 = &v223;
  v223 = v213;
  v224 = v213;
  v225 = v213;
  v222 = 0x300000003LL;
  v146 = sub_BCF480(v145, &v223, 3, 0);
  v147 = sub_BA8C10(v144, (__int64)"__asan_register_elf_globals", 0x1Bu, v146, 0);
  v149 = v148;
  if ( v221 != &v223 )
  {
    v214 = v147;
    _libc_free((unsigned __int64)v221);
    v147 = v214;
  }
  v259 = v149;
  v258 = v147;
  v150 = (__int64)v226;
  v215 = v236;
  v151 = (__int64 *)sub_BCB120((_QWORD *)v265[9]);
  v221 = &v223;
  v223 = v215;
  v224 = v215;
  v225 = v215;
  v222 = 0x300000003LL;
  v152 = sub_BCF480(v151, &v223, 3, 0);
  v153 = sub_BA8C10(v150, (__int64)"__asan_unregister_elf_globals", 0x1Du, v152, 0);
  if ( v221 != &v223 )
  {
    v195 = v154;
    v216 = v153;
    _libc_free((unsigned __int64)v221);
    v154 = v195;
    v153 = v216;
  }
  v260 = v153;
  v261 = v154;
  nullsub_61();
  v269.m128i_i64[1] = (__int64)&unk_49DA100;
  nullsub_63();
  if ( (unsigned __int64 *)v265[0] != &v265[2] )
    _libc_free(v265[0]);
  if ( v235 == 1 )
  {
    if ( v227 )
    {
      v262 = sub_2A41400(v226, "asan.module_ctor", 16);
      v155 = v262;
      goto LABEL_184;
    }
    v171 = sub_AE2980((__int64)(v226 + 39), 0);
    v172 = *((_DWORD *)v226 + 70) == 17;
    v173 = v171[1] == 32;
    v221 = &v223;
    v174 = (v172 && v173) + 8;
    sub_2240A50((__int64 *)&v221, 1u, 45);
    *(_BYTE *)v221 = v174 + 48;
    if ( v228 )
    {
      sub_8FD6D0((__int64)v265, "__asan_version_mismatch_check_v", &v221);
      v176 = (unsigned __int64 *)v265[0];
      v175 = v265[1];
    }
    else
    {
      v265[1] = 0;
      v175 = 0;
      LOBYTE(v265[2]) = 0;
      v265[0] = (unsigned __int64)&v265[2];
      v176 = &v265[2];
    }
    sub_2A41510(
      (unsigned int)&v220,
      (_DWORD)v226,
      (unsigned int)"asan.module_ctor",
      16,
      (unsigned int)"__asan_init",
      11,
      0,
      0,
      0,
      0,
      (__int64)v176,
      v175,
      0);
    v262 = v220;
    if ( (unsigned __int64 *)v265[0] != &v265[2] )
      j_j___libc_free_0(v265[0]);
    if ( v221 != &v223 )
      j_j___libc_free_0((unsigned __int64)v221);
  }
  v155 = v262;
LABEL_184:
  LOBYTE(v221) = 1;
  if ( byte_4FE1528 )
  {
    if ( v155 )
    {
      v158 = *(_QWORD *)(v155 + 80);
      if ( !v158 )
        BUG();
      v159 = *(_QWORD *)(v158 + 24) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v159 == v158 + 24 )
      {
        v161 = 0;
      }
      else
      {
        if ( !v159 )
          BUG();
        v160 = *(unsigned __int8 *)(v159 - 24);
        v161 = 0;
        v162 = v159 - 24;
        if ( (unsigned int)(v160 - 30) < 0xB )
          v161 = v162;
      }
      v163 = sub_BD5C60(v161);
      WORD2(v267) = 512;
      v265[9] = v163;
      v265[1] = 0x200000000LL;
      v265[0] = (unsigned __int64)&v265[2];
      v265[10] = (unsigned __int64)&v269.m128i_u64[1];
      v270[0] = &unk_49DA0B0;
      v265[11] = (unsigned __int64)v270;
      v266 = 0;
      LODWORD(v267) = 0;
      BYTE6(v267) = 7;
      v268 = 0;
      v269.m128i_i64[0] = 0;
      memset(&v265[6], 0, 18);
      v269.m128i_i64[1] = (__int64)&unk_49DA100;
      sub_D5F1F0((__int64)v265, v161);
      sub_23E7330(&v226, (__int64)v265, (bool *)&v221);
      nullsub_61();
      v269.m128i_i64[1] = (__int64)&unk_49DA100;
      nullsub_63();
      if ( (unsigned __int64 *)v265[0] != &v265[2] )
        _libc_free(v265[0]);
    }
    else
    {
      v265[0] = (unsigned __int64)&v265[2];
      v265[1] = 0x200000000LL;
      v265[9] = (unsigned __int64)v238;
      v265[10] = (unsigned __int64)&v269.m128i_u64[1];
      v265[11] = (unsigned __int64)v270;
      v269.m128i_i64[1] = (__int64)&unk_49DA100;
      v266 = 0;
      WORD2(v267) = 512;
      LODWORD(v267) = 0;
      BYTE6(v267) = 7;
      v268 = 0;
      v269.m128i_i64[0] = 0;
      memset(&v265[6], 0, 18);
      v270[0] = &unk_49DA0B0;
      sub_23E7330(&v226, (__int64)v265, (bool *)&v221);
      sub_F94A20(v265, (__int64)v265);
    }
    v155 = v262;
  }
  v156 = 50;
  if ( HIDWORD(v242) != 37 )
    v156 = 1;
  if ( v233 && HIDWORD(v243) == 3 && (_BYTE)v221 )
  {
    if ( v155 )
    {
      v217 = v155;
      v164 = sub_BAA410((__int64)v226, "asan.module_ctor", 0x10u);
      sub_B2F990(v217, v164, v165, v166);
      sub_2A3ED40(v226, v262, v156, v262);
    }
    v167 = v263;
    if ( v263 )
    {
      v168 = sub_BAA410((__int64)v226, "asan.module_dtor", 0x10u);
      sub_B2F990(v167, v168, v169, v170);
      sub_2A3ED60(v226, v263, v156, v263);
    }
  }
  else
  {
    if ( v155 )
      sub_2A3ED40(v226, v155, v156, 0);
    if ( v263 )
      sub_2A3ED60(v226, v263, v156, 0);
  }
  v157 = a1;
  memset(v265, 0, sizeof(v265));
  v265[8] = 0x100000002LL;
  v265[1] = (unsigned __int64)&v265[4];
  v265[10] = (unsigned __int64)&unk_4F86B78;
  LODWORD(v265[2]) = 2;
  BYTE4(v265[3]) = 1;
  v265[7] = (unsigned __int64)&v265[10];
  BYTE4(v265[9]) = 1;
  v265[6] = 1;
  sub_C8CF70(v157, (void *)(v157 + 32), 2, (__int64)&v265[4], (__int64)v265);
  sub_C8CF70(v157 + 48, (void *)(v157 + 80), 2, (__int64)&v265[10], (__int64)&v265[6]);
  if ( !BYTE4(v265[9]) )
    _libc_free(v265[7]);
  if ( !BYTE4(v265[3]) )
    _libc_free(v265[1]);
  if ( (_BYTE *)v239[0] != v240 )
    j_j___libc_free_0(v239[0]);
  return a1;
}
