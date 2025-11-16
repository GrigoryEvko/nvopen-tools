// Function: sub_1A87830
// Address: 0x1a87830
//
__int64 __fastcall sub_1A87830(__int64 a1, __m128i a2, __m128i a3)
{
  __int64 v3; // rdx
  __int64 *v4; // rax
  __int64 v5; // r13
  _QWORD *v6; // r14
  __int64 v7; // r14
  __int64 **v8; // rax
  char v9; // dl
  __int64 *v10; // r12
  __int64 **v11; // rax
  __int64 **v12; // rcx
  __int64 **v13; // rsi
  __int64 v15; // rdi
  __int64 *v16; // rbx
  __int64 *v17; // r13
  __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rdx
  unsigned int v22; // ebx
  unsigned __int64 v23; // rcx
  _QWORD *v24; // rax
  _DWORD *v25; // rdx
  __int64 v26; // rdi
  __int64 v27; // rsi
  unsigned int v28; // eax
  unsigned __int64 v29; // rcx
  _QWORD *v30; // rax
  _DWORD *v31; // rdx
  __int64 v32; // rdi
  __int64 v33; // rsi
  unsigned int v34; // eax
  __int64 *v35; // rbx
  _QWORD *v36; // r15
  __int64 *v37; // rbx
  char v38; // al
  __int64 v39; // r13
  __int64 v40; // rax
  __int64 v41; // r14
  unsigned __int64 v42; // rdx
  _QWORD *v43; // rax
  _DWORD *v44; // rdi
  __int64 v45; // rsi
  __int64 v46; // rcx
  unsigned int v47; // r13d
  __int64 v48; // rax
  __int64 v49; // rax
  unsigned int v50; // edx
  __int64 *v51; // rax
  __int64 v52; // rax
  unsigned __int64 v53; // r12
  _BYTE *v54; // r13
  __int64 *v55; // r15
  __int64 v56; // rbx
  __int64 v57; // rax
  __int64 v58; // rax
  unsigned int v59; // edx
  __int64 *v60; // rax
  __int64 v61; // rax
  _DWORD *v62; // r8
  _DWORD *v63; // rdi
  __int64 v64; // rcx
  __int64 v65; // rdx
  _QWORD *v66; // r12
  __int64 v67; // rax
  __int64 v68; // r13
  __int64 v69; // rax
  __int64 v70; // rax
  __int64 *v71; // r12
  int v72; // r8d
  int v73; // r9d
  __int64 v74; // rax
  _QWORD *v75; // rax
  _QWORD *v76; // rax
  __int64 v77; // rax
  __int64 v78; // r14
  __int64 v79; // rax
  __int64 *v80; // r14
  __int64 v81; // rax
  __int64 v82; // rax
  __int64 *v83; // rsi
  _QWORD *v84; // rax
  __int64 v85; // r12
  __int64 v86; // rax
  unsigned __int8 v87; // al
  __int64 v88; // rax
  __int64 v89; // r14
  __int64 v90; // rax
  __int64 *v91; // rcx
  __int64 *v92; // rdx
  int v93; // eax
  __int64 v94; // rsi
  __int64 v95; // rax
  __int64 *v96; // rcx
  __int64 v97; // r12
  __int64 *v98; // rdx
  __int64 v99; // rsi
  int v100; // r8d
  __int64 v101; // rax
  char v102; // al
  int v103; // r14d
  __int64 v104; // rax
  _DWORD *v105; // rcx
  _DWORD *v106; // rdx
  __int64 v107; // r8
  __int64 v108; // rdi
  __int64 v109; // rax
  _DWORD *v110; // rcx
  _DWORD *v111; // rdx
  __int64 v112; // r8
  __int64 v113; // rdi
  __int64 *v114; // r14
  __int64 v115; // rax
  __int64 v116; // rcx
  __int64 v117; // rsi
  __int64 v118; // rdx
  unsigned __int8 *v119; // rsi
  __int64 v120; // rax
  __int64 v121; // rsi
  unsigned int v122; // ecx
  const __m128i *v123; // r14
  __m128i *v124; // r13
  __m128i *v125; // r14
  __m128i *v126; // rdi
  _QWORD *v127; // r14
  _QWORD *v128; // r13
  __int64 v129; // rax
  __m128i *v130; // r13
  const __m128i *v131; // r12
  unsigned int v132; // ebx
  const __m128i *v133; // r13
  const __m128i *v134; // rdi
  __int64 v135; // rax
  __int64 v136; // rax
  _QWORD *v137; // rax
  __int64 *v138; // [rsp+20h] [rbp-A20h]
  __int64 **v139; // [rsp+28h] [rbp-A18h]
  __int64 *v140; // [rsp+48h] [rbp-9F8h]
  __int64 v141; // [rsp+50h] [rbp-9F0h]
  unsigned int v142; // [rsp+50h] [rbp-9F0h]
  __int64 *v143; // [rsp+50h] [rbp-9F0h]
  _QWORD *v144; // [rsp+58h] [rbp-9E8h]
  __int64 v145; // [rsp+58h] [rbp-9E8h]
  __int64 v146; // [rsp+58h] [rbp-9E8h]
  __int64 v147; // [rsp+58h] [rbp-9E8h]
  __int64 v148; // [rsp+58h] [rbp-9E8h]
  __int64 v149; // [rsp+60h] [rbp-9E0h]
  char v150; // [rsp+68h] [rbp-9D8h]
  __int64 *v151; // [rsp+68h] [rbp-9D8h]
  __int64 *v152; // [rsp+68h] [rbp-9D8h]
  __int64 v153; // [rsp+68h] [rbp-9D8h]
  __int64 *v154; // [rsp+68h] [rbp-9D8h]
  char v155; // [rsp+72h] [rbp-9CEh]
  unsigned __int8 v156; // [rsp+73h] [rbp-9CDh]
  unsigned int v157; // [rsp+74h] [rbp-9CCh]
  __int64 v158; // [rsp+78h] [rbp-9C8h]
  __int64 **v159; // [rsp+80h] [rbp-9C0h]
  __int64 *v160; // [rsp+88h] [rbp-9B8h]
  __int64 v162; // [rsp+A0h] [rbp-9A0h]
  __int64 v163; // [rsp+B0h] [rbp-990h] BYREF
  __int16 v164; // [rsp+C0h] [rbp-980h]
  __int64 *v165; // [rsp+D0h] [rbp-970h] BYREF
  __int64 v166; // [rsp+D8h] [rbp-968h]
  __int64 *v167; // [rsp+E0h] [rbp-960h]
  __int64 v168; // [rsp+E8h] [rbp-958h]
  __int64 v169; // [rsp+F0h] [rbp-950h]
  int v170; // [rsp+F8h] [rbp-948h]
  __int64 *v171; // [rsp+100h] [rbp-940h]
  __int64 v172; // [rsp+108h] [rbp-938h]
  int v173; // [rsp+120h] [rbp-920h] BYREF
  char v174; // [rsp+124h] [rbp-91Ch]
  __int64 v175; // [rsp+128h] [rbp-918h]
  __int64 v176; // [rsp+130h] [rbp-910h]
  __int64 v177; // [rsp+138h] [rbp-908h]
  __int64 v178; // [rsp+140h] [rbp-900h]
  int v179; // [rsp+148h] [rbp-8F8h]
  __int64 v180; // [rsp+150h] [rbp-8F0h]
  __int64 v181; // [rsp+158h] [rbp-8E8h]
  __int64 v182; // [rsp+160h] [rbp-8E0h]
  int v183; // [rsp+168h] [rbp-8D8h]
  __int64 v184; // [rsp+170h] [rbp-8D0h] BYREF
  __int64 **v185; // [rsp+178h] [rbp-8C8h]
  __int64 **v186; // [rsp+180h] [rbp-8C0h]
  __int64 v187; // [rsp+188h] [rbp-8B8h]
  int v188; // [rsp+190h] [rbp-8B0h]
  _QWORD v189[8]; // [rsp+198h] [rbp-8A8h] BYREF
  unsigned __int64 v190; // [rsp+1D8h] [rbp-868h] BYREF
  __int64 v191; // [rsp+1E0h] [rbp-860h]
  __int64 v192; // [rsp+1E8h] [rbp-858h]
  _QWORD v193[16]; // [rsp+1F0h] [rbp-850h] BYREF
  _BYTE *v194; // [rsp+270h] [rbp-7D0h] BYREF
  __int64 v195; // [rsp+278h] [rbp-7C8h]
  _BYTE v196[256]; // [rsp+280h] [rbp-7C0h] BYREF
  __int64 v197; // [rsp+380h] [rbp-6C0h] BYREF
  _BYTE *v198; // [rsp+388h] [rbp-6B8h]
  _BYTE *v199; // [rsp+390h] [rbp-6B0h]
  __int64 v200; // [rsp+398h] [rbp-6A8h]
  int v201; // [rsp+3A0h] [rbp-6A0h]
  _BYTE v202[264]; // [rsp+3A8h] [rbp-698h] BYREF
  _QWORD v203[4]; // [rsp+4B0h] [rbp-590h] BYREF
  _QWORD *v204; // [rsp+4D0h] [rbp-570h]
  __int64 v205; // [rsp+4D8h] [rbp-568h]
  unsigned int v206; // [rsp+4E0h] [rbp-560h]
  __int64 v207; // [rsp+4E8h] [rbp-558h]
  __int64 v208; // [rsp+4F0h] [rbp-550h]
  __int64 v209; // [rsp+4F8h] [rbp-548h]
  __int64 v210; // [rsp+500h] [rbp-540h]
  __int64 v211; // [rsp+508h] [rbp-538h]
  __int64 v212; // [rsp+510h] [rbp-530h]
  __int64 v213; // [rsp+518h] [rbp-528h]
  __int64 v214; // [rsp+520h] [rbp-520h]
  __int64 v215; // [rsp+528h] [rbp-518h]
  __int64 v216; // [rsp+530h] [rbp-510h]
  __int64 v217; // [rsp+538h] [rbp-508h]
  int v218; // [rsp+540h] [rbp-500h]
  __int64 v219; // [rsp+548h] [rbp-4F8h]
  _BYTE *v220; // [rsp+550h] [rbp-4F0h]
  _BYTE *v221; // [rsp+558h] [rbp-4E8h]
  __int64 v222; // [rsp+560h] [rbp-4E0h]
  int v223; // [rsp+568h] [rbp-4D8h]
  _BYTE v224[16]; // [rsp+570h] [rbp-4D0h] BYREF
  __int64 v225; // [rsp+580h] [rbp-4C0h]
  __int64 v226; // [rsp+588h] [rbp-4B8h]
  __int64 v227; // [rsp+590h] [rbp-4B0h]
  __int64 v228; // [rsp+598h] [rbp-4A8h]
  __int64 v229; // [rsp+5A0h] [rbp-4A0h]
  __int64 v230; // [rsp+5A8h] [rbp-498h]
  __int16 v231; // [rsp+5B0h] [rbp-490h]
  __int64 v232[5]; // [rsp+5B8h] [rbp-488h] BYREF
  int v233; // [rsp+5E0h] [rbp-460h]
  __int64 v234; // [rsp+5E8h] [rbp-458h]
  __int64 v235; // [rsp+5F0h] [rbp-450h]
  __int64 v236; // [rsp+5F8h] [rbp-448h]
  _BYTE *v237; // [rsp+600h] [rbp-440h]
  __int64 v238; // [rsp+608h] [rbp-438h]
  _BYTE v239[64]; // [rsp+610h] [rbp-430h] BYREF
  __int64 *v240; // [rsp+650h] [rbp-3F0h] BYREF
  int v241; // [rsp+658h] [rbp-3E8h]
  char v242; // [rsp+65Ch] [rbp-3E4h]
  __int64 v243; // [rsp+660h] [rbp-3E0h]
  __m128i v244; // [rsp+668h] [rbp-3D8h]
  __int64 v245; // [rsp+678h] [rbp-3C8h]
  __int64 v246; // [rsp+680h] [rbp-3C0h]
  __m128i v247; // [rsp+688h] [rbp-3B8h]
  __int64 v248; // [rsp+698h] [rbp-3A8h]
  char v249; // [rsp+6A0h] [rbp-3A0h]
  __m128i *v250; // [rsp+6A8h] [rbp-398h] BYREF
  __int64 v251; // [rsp+6B0h] [rbp-390h]
  _BYTE v252[352]; // [rsp+6B8h] [rbp-388h] BYREF
  char v253; // [rsp+818h] [rbp-228h]
  int v254; // [rsp+81Ch] [rbp-224h]
  __int64 v255; // [rsp+820h] [rbp-220h]
  __int64 *v256; // [rsp+830h] [rbp-210h] BYREF
  __int64 v257; // [rsp+838h] [rbp-208h]
  __int64 v258; // [rsp+840h] [rbp-200h] BYREF
  __m128i v259; // [rsp+848h] [rbp-1F8h] BYREF
  __int64 v260; // [rsp+858h] [rbp-1E8h]
  __int64 v261; // [rsp+860h] [rbp-1E0h]
  __m128i v262; // [rsp+868h] [rbp-1D8h] BYREF
  __int64 v263; // [rsp+878h] [rbp-1C8h]
  char v264; // [rsp+880h] [rbp-1C0h]
  const __m128i *v265; // [rsp+888h] [rbp-1B8h]
  unsigned int v266; // [rsp+890h] [rbp-1B0h]
  char v267; // [rsp+898h] [rbp-1A8h] BYREF
  char v268; // [rsp+9F8h] [rbp-48h]
  int v269; // [rsp+9FCh] [rbp-44h]
  __int64 v270; // [rsp+A00h] [rbp-40h]

  v3 = *(_QWORD *)(a1 + 8);
  v139 = *(__int64 ***)(v3 + 40);
  if ( *(__int64 ***)(v3 + 32) == v139 )
    return 0;
  v159 = *(__int64 ***)(v3 + 32);
  v156 = 0;
  do
  {
    v4 = *v159;
    v185 = (__int64 **)v189;
    v186 = (__int64 **)v189;
    v189[0] = v4;
    v256 = v4;
    v190 = 0;
    v191 = 0;
    v192 = 0;
    v187 = 0x100000008LL;
    v188 = 0;
    v184 = 1;
    LOBYTE(v258) = 0;
    sub_197E9F0(&v190, (__int64)&v256);
    v5 = v191;
    memset(v193, 0, sizeof(v193));
    LODWORD(v193[3]) = 8;
    v193[1] = &v193[5];
    v193[2] = &v193[5];
    if ( v191 == v190 )
      goto LABEL_19;
LABEL_4:
    v6 = *(_QWORD **)(v5 - 24);
    if ( v6[2] == v6[1] )
    {
      v15 = *(_QWORD *)(v5 - 24);
      v197 = 0;
      v198 = v202;
      v199 = v202;
      v200 = 32;
      v201 = 0;
      sub_14D04F0(v15, *(_QWORD *)a1, (__int64)&v197);
      v174 = 0;
      v179 = 0;
      v180 = 0;
      v181 = 0;
      v182 = 0;
      v183 = 0;
      v16 = (__int64 *)v6[5];
      v178 = 0;
      v17 = (__int64 *)v6[4];
      v173 = 0;
      v175 = 0;
      v176 = 0;
      v177 = 0;
      if ( v17 != v16 )
      {
        while ( 1 )
        {
          v18 = *v17;
          v19 = *(_QWORD *)(*v17 + 48);
          v20 = *v17 + 40;
          if ( v19 != v20 )
            break;
LABEL_38:
          ++v17;
          sub_14D0990((__int64)&v173, v18, *(__int64 **)(a1 + 24), (__int64)&v197);
          if ( v16 == v17 )
          {
            v22 = 1;
            if ( (_DWORD)v175 )
              v22 = v175;
            goto LABEL_41;
          }
        }
        while ( 1 )
        {
          if ( !v19 )
            BUG();
          if ( *(_BYTE *)(v19 - 8) == 78 )
          {
            v21 = *(_QWORD *)(v19 - 48);
            if ( !*(_BYTE *)(v21 + 16) && *(_DWORD *)(v21 + 36) == 148 )
              break;
          }
          v19 = *(_QWORD *)(v19 + 8);
          if ( v20 == v19 )
            goto LABEL_38;
        }
LABEL_88:
        j___libc_free_0(v177);
        if ( v199 != v198 )
          _libc_free((unsigned __int64)v199);
        v5 = v191;
        goto LABEL_5;
      }
      v22 = 1;
LABEL_41:
      v23 = sub_16D5D50();
      v24 = *(_QWORD **)&dword_4FA0208[2];
      if ( !*(_QWORD *)&dword_4FA0208[2] )
        goto LABEL_48;
      v25 = dword_4FA0208;
      do
      {
        while ( 1 )
        {
          v26 = v24[2];
          v27 = v24[3];
          if ( v23 <= v24[4] )
            break;
          v24 = (_QWORD *)v24[3];
          if ( !v27 )
            goto LABEL_46;
        }
        v25 = v24;
        v24 = (_QWORD *)v24[2];
      }
      while ( v26 );
LABEL_46:
      if ( v25 == dword_4FA0208 )
        goto LABEL_48;
      if ( v23 < *((_QWORD *)v25 + 4) )
        goto LABEL_48;
      v104 = *((_QWORD *)v25 + 7);
      v105 = v25 + 12;
      if ( !v104 )
        goto LABEL_48;
      v106 = v25 + 12;
      do
      {
        while ( 1 )
        {
          v107 = *(_QWORD *)(v104 + 16);
          v108 = *(_QWORD *)(v104 + 24);
          if ( *(_DWORD *)(v104 + 32) >= dword_4FB5108 )
            break;
          v104 = *(_QWORD *)(v104 + 24);
          if ( !v108 )
            goto LABEL_146;
        }
        v106 = (_DWORD *)v104;
        v104 = *(_QWORD *)(v104 + 16);
      }
      while ( v107 );
LABEL_146:
      if ( v105 == v106 || dword_4FB5108 < v106[8] || (v28 = dword_4FB51A0, (int)v106[9] <= 0) )
LABEL_48:
        v28 = sub_14A3290(*(_QWORD *)(a1 + 24));
      v157 = 1;
      if ( v28 >= v22 )
        v157 = v28 / v22;
      v29 = sub_16D5D50();
      v30 = *(_QWORD **)&dword_4FA0208[2];
      if ( !*(_QWORD *)&dword_4FA0208[2] )
        goto LABEL_58;
      v31 = dword_4FA0208;
      do
      {
        while ( 1 )
        {
          v32 = v30[2];
          v33 = v30[3];
          if ( v29 <= v30[4] )
            break;
          v30 = (_QWORD *)v30[3];
          if ( !v33 )
            goto LABEL_56;
        }
        v31 = v30;
        v30 = (_QWORD *)v30[2];
      }
      while ( v32 );
LABEL_56:
      if ( v31 == dword_4FA0208 )
        goto LABEL_58;
      if ( v29 < *((_QWORD *)v31 + 4) )
        goto LABEL_58;
      v109 = *((_QWORD *)v31 + 7);
      v110 = v31 + 12;
      if ( !v109 )
        goto LABEL_58;
      v111 = v31 + 12;
      do
      {
        while ( 1 )
        {
          v112 = *(_QWORD *)(v109 + 16);
          v113 = *(_QWORD *)(v109 + 24);
          if ( *(_DWORD *)(v109 + 32) >= dword_4FB4F48 )
            break;
          v109 = *(_QWORD *)(v109 + 24);
          if ( !v113 )
            goto LABEL_155;
        }
        v111 = (_DWORD *)v109;
        v109 = *(_QWORD *)(v109 + 16);
      }
      while ( v112 );
LABEL_155:
      if ( v111 == v110 || dword_4FB4F48 < v111[8] || (v34 = dword_4FB4FE0, (int)v111[9] <= 0) )
LABEL_58:
        v34 = sub_14A32F0(*(_QWORD *)(a1 + 24));
      if ( v157 > v34 )
        goto LABEL_88;
      v35 = (__int64 *)v6[5];
      v194 = v196;
      v195 = 0x1000000000LL;
      v138 = v35;
      if ( (__int64 *)v6[4] == v35 )
        goto LABEL_88;
      v160 = (__int64 *)v6[4];
      v150 = 0;
      v158 = (__int64)v6;
      while ( 1 )
      {
        v36 = *(_QWORD **)(*v160 + 48);
        v149 = *v160;
        v162 = *v160 + 40;
        if ( v36 != (_QWORD *)v162 )
          break;
LABEL_85:
        if ( v138 == ++v160 )
        {
          v156 |= v150;
          if ( v194 != v196 )
            _libc_free((unsigned __int64)v194);
          goto LABEL_88;
        }
      }
      v37 = (__int64 *)a1;
      while ( 1 )
      {
LABEL_64:
        if ( !v36 )
          BUG();
        v38 = *((_BYTE *)v36 - 8);
        if ( v38 == 54 )
          goto LABEL_66;
        if ( v38 != 55 )
          goto LABEL_84;
        if ( byte_4FB5280 )
        {
LABEL_66:
          v39 = *(v36 - 6);
          v40 = *(_QWORD *)v39;
          if ( *(_BYTE *)(*(_QWORD *)v39 + 8LL) == 16 )
            v40 = **(_QWORD **)(v40 + 16);
          if ( *(_DWORD *)(v40 + 8) >> 8 )
            goto LABEL_84;
          if ( sub_13FC1A0(v158, *(v36 - 6)) )
            goto LABEL_84;
          v41 = sub_146F1B0(v37[2], v39);
          if ( *(_WORD *)(v41 + 24) != 7 )
            goto LABEL_84;
          v42 = sub_16D5D50();
          v43 = *(_QWORD **)&dword_4FA0208[2];
          if ( !*(_QWORD *)&dword_4FA0208[2] )
            goto LABEL_78;
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
                goto LABEL_76;
            }
            v44 = v43;
            v43 = (_QWORD *)v43[2];
          }
          while ( v45 );
LABEL_76:
          if ( v44 == dword_4FA0208 )
            goto LABEL_78;
          if ( v42 < *((_QWORD *)v44 + 4) )
            goto LABEL_78;
          v61 = *((_QWORD *)v44 + 7);
          v62 = v44 + 12;
          if ( !v61 )
            goto LABEL_78;
          v63 = v44 + 12;
          do
          {
            while ( 1 )
            {
              v64 = *(_QWORD *)(v61 + 16);
              v65 = *(_QWORD *)(v61 + 24);
              if ( *(_DWORD *)(v61 + 32) >= dword_4FB5028 )
                break;
              v61 = *(_QWORD *)(v61 + 24);
              if ( !v65 )
                goto LABEL_108;
            }
            v63 = (_DWORD *)v61;
            v61 = *(_QWORD *)(v61 + 16);
          }
          while ( v64 );
LABEL_108:
          if ( v63 == v62 || dword_4FB5028 < v63[8] || (v47 = dword_4FB50C0, (int)v63[9] <= 0) )
LABEL_78:
            v47 = sub_14A32C0(v37[3]);
          if ( v47 > 1 )
          {
            v48 = sub_13A5BC0((_QWORD *)v41, v37[2]);
            if ( *(_WORD *)(v48 + 24) )
              goto LABEL_84;
            v49 = *(_QWORD *)(v48 + 32);
            v50 = *(_DWORD *)(v49 + 32);
            v51 = *(__int64 **)(v49 + 24);
            v52 = v50 > 0x40
                ? *v51
                : (__int64)((_QWORD)v51 << (64 - (unsigned __int8)v50)) >> (64 - (unsigned __int8)v50);
            if ( v47 > (unsigned int)abs64(v52) )
              goto LABEL_84;
          }
          v53 = (unsigned __int64)v194;
          v54 = &v194[16 * (unsigned int)v195];
          if ( v194 == v54 )
            goto LABEL_113;
          v144 = v36;
          v55 = v37;
          do
          {
            v57 = sub_14806B0(v55[2], v41, *(_QWORD *)(v53 + 8), 0, 0);
            if ( !*(_WORD *)(v57 + 24) )
            {
              v58 = *(_QWORD *)(v57 + 32);
              v59 = *(_DWORD *)(v58 + 32);
              v60 = *(__int64 **)(v58 + 24);
              v56 = v59 <= 0x40
                  ? (__int64)((_QWORD)v60 << (64 - (unsigned __int8)v59)) >> (64 - (unsigned __int8)v59)
                  : *v60;
              if ( (__int64)abs64(v56) < (unsigned int)sub_14A3260(v55[3]) )
              {
                v37 = v55;
                v36 = (_QWORD *)v144[1];
                if ( (_QWORD *)v162 == v36 )
                  goto LABEL_85;
                goto LABEL_64;
              }
            }
            v53 += 16LL;
          }
          while ( v54 != (_BYTE *)v53 );
          v37 = v55;
          v36 = v144;
LABEL_113:
          v66 = (_QWORD *)v37[2];
          v67 = sub_13A5BC0((_QWORD *)v41, (__int64)v66);
          v68 = v37[2];
          v145 = v67;
          v69 = sub_1456040(**(_QWORD **)(v41 + 32));
          v258 = sub_145CF80(v68, v69, v157, 0);
          v256 = &v258;
          v259.m128i_i64[0] = v145;
          v257 = 0x200000002LL;
          v70 = sub_147EE30(v66, &v256, 0, 0, a2, a3);
          if ( v256 != &v258 )
          {
            v146 = v70;
            _libc_free((unsigned __int64)v256);
            v70 = v146;
          }
          v259.m128i_i64[0] = v70;
          v256 = &v258;
          v258 = v41;
          v257 = 0x200000002LL;
          v71 = sub_147DD40((__int64)v66, (__int64 *)&v256, 0, 0, a2, a3);
          if ( v256 != &v258 )
            _libc_free((unsigned __int64)v256);
          v155 = sub_3870AF0(v71, v37[2]);
          if ( !v155 )
          {
LABEL_84:
            v36 = (_QWORD *)v36[1];
            if ( (_QWORD *)v162 == v36 )
              goto LABEL_85;
          }
          else
          {
            v74 = (unsigned int)v195;
            if ( (unsigned int)v195 >= HIDWORD(v195) )
            {
              sub_16CD150((__int64)&v194, v196, 0, 16, v72, v73);
              v74 = (unsigned int)v195;
            }
            v75 = &v194[16 * v74];
            v75[1] = v41;
            *v75 = v36 - 3;
            LODWORD(v195) = v195 + 1;
            v76 = (_QWORD *)sub_157E9C0(v149);
            v147 = sub_16471D0(v76, 0);
            v77 = sub_15F2050((__int64)(v36 - 3));
            v78 = sub_1632FA0(v77);
            v79 = v37[2];
            v220 = v224;
            v221 = v224;
            v203[0] = v79;
            v203[1] = v78;
            v203[2] = "prefaddr";
            v203[3] = 0;
            v204 = 0;
            v205 = 0;
            v206 = 0;
            v207 = 0;
            v208 = 0;
            v209 = 0;
            v210 = 0;
            v211 = 0;
            v212 = 0;
            v213 = 0;
            v214 = 0;
            v215 = 0;
            v216 = 0;
            v217 = 0;
            v218 = 0;
            v219 = 0;
            v222 = 2;
            v223 = 0;
            v225 = 0;
            v226 = 0;
            v227 = 0;
            v228 = 0;
            v229 = 0;
            v230 = 0;
            v231 = 1;
            v232[3] = sub_15E0530(*(_QWORD *)(v79 + 24));
            v237 = v239;
            v236 = v78;
            memset(v232, 0, 24);
            v232[4] = 0;
            v233 = 0;
            v234 = 0;
            v235 = 0;
            v238 = 0x800000000LL;
            v80 = (__int64 *)sub_38767A0(v203, v71, v147, v36 - 3);
            v81 = sub_16498A0((__int64)(v36 - 3));
            v169 = 0;
            v168 = v81;
            v170 = 0;
            v171 = 0;
            v172 = 0;
            v82 = v36[2];
            v83 = (__int64 *)v36[3];
            v167 = v36;
            v165 = 0;
            v166 = v82;
            v256 = v83;
            if ( v83 )
            {
              sub_1623A60((__int64)&v256, (__int64)v83, 2);
              if ( v165 )
                sub_161E7C0((__int64)&v165, (__int64)v165);
              v165 = v256;
              if ( v256 )
                sub_1623210((__int64)&v256, (unsigned __int8 *)v256, (__int64)&v165);
            }
            v151 = *(__int64 **)(*(_QWORD *)(v149 + 56) + 40LL);
            v84 = (_QWORD *)sub_157E9C0(v149);
            v85 = sub_1643350(v84);
            v86 = sub_15E26F0(v151, 148, 0, 0);
            v256 = v80;
            v164 = 257;
            v148 = v86;
            v87 = sub_15F2ED0((__int64)(v36 - 3));
            v257 = sub_15A0680(v85, v87 ^ 1u, 0);
            v258 = sub_15A0680(v85, 3, 0);
            v88 = sub_15A0680(v85, 1, 0);
            v89 = v172;
            v259.m128i_i64[0] = v88;
            v140 = v171;
            v90 = *(_QWORD *)(*(_QWORD *)v148 + 24LL);
            LOWORD(v243) = 257;
            v141 = v90;
            v91 = &v171[7 * v172];
            if ( v91 == v171 )
            {
              v137 = sub_1648AB0(72, 5u, 16 * (int)v172);
              v97 = (__int64)v137;
              if ( v137 )
              {
                v153 = (__int64)v137;
                v101 = -120;
                v100 = 5;
                goto LABEL_132;
              }
LABEL_228:
              v153 = 0;
              v97 = 0;
              goto LABEL_133;
            }
            v92 = v171;
            v93 = 0;
            do
            {
              v94 = v92[5] - v92[4];
              v92 += 7;
              v93 += v94 >> 3;
            }
            while ( v91 != v92 );
            v152 = &v171[7 * v172];
            v95 = (__int64)sub_1648AB0(72, v93 + 5, 16 * (int)v172);
            v96 = v152;
            v97 = v95;
            if ( !v95 )
              goto LABEL_228;
            v98 = v140;
            v153 = v95;
            LODWORD(v95) = 0;
            do
            {
              v99 = v98[5] - v98[4];
              v98 += 7;
              v95 = (unsigned int)(v99 >> 3) + (unsigned int)v95;
            }
            while ( v96 != v98 );
            v100 = v95 + 5;
            v101 = -24 - 8 * (3 * v95 + 12);
LABEL_132:
            sub_15F1EA0(v97, **(_QWORD **)(v141 + 16), 54, v97 + v101, v100, 0);
            *(_QWORD *)(v97 + 56) = 0;
            sub_15F5B40(v97, v141, v148, (__int64 *)&v256, 4, (__int64)&v240, v140, v89);
LABEL_133:
            v102 = *(_BYTE *)(*(_QWORD *)v97 + 8LL);
            if ( v102 == 16 )
              v102 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)v97 + 16LL) + 8LL);
            if ( (unsigned __int8)(v102 - 1) <= 5u || *(_BYTE *)(v97 + 16) == 76 )
            {
              v103 = v170;
              if ( v169 )
                sub_1625C10(v97, 3, v169);
              sub_15F2440(v97, v103);
            }
            if ( v166 )
            {
              v114 = v167;
              sub_157E9D0(v166 + 40, v97);
              v115 = *(_QWORD *)(v97 + 24);
              v116 = *v114;
              *(_QWORD *)(v97 + 32) = v114;
              v116 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v97 + 24) = v116 | v115 & 7;
              *(_QWORD *)(v116 + 8) = v97 + 24;
              *v114 = *v114 & 7 | (v97 + 24);
            }
            sub_164B780(v153, &v163);
            if ( v165 )
            {
              v240 = v165;
              sub_1623A60((__int64)&v240, (__int64)v165, 2);
              v117 = *(_QWORD *)(v97 + 48);
              v118 = v97 + 48;
              if ( v117 )
              {
                sub_161E7C0(v97 + 48, v117);
                v118 = v97 + 48;
              }
              v119 = (unsigned __int8 *)v240;
              *(_QWORD *)(v97 + 48) = v240;
              if ( v119 )
                sub_1623210((__int64)&v240, v119, v118);
            }
            v154 = (__int64 *)v37[4];
            v120 = sub_15E0530(*v154);
            if ( sub_1602790(v120)
              || (v135 = sub_15E0530(*v154),
                  v136 = sub_16033E0(v135),
                  (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v136 + 48LL))(v136)) )
            {
              sub_15CA3B0((__int64)&v256, (__int64)"loop-data-prefetch", (__int64)"Prefetched", 10, (__int64)(v36 - 3));
              sub_15CAB20((__int64)&v256, "prefetched memory access", 0x18u);
              a2 = _mm_loadu_si128(&v259);
              a3 = _mm_loadu_si128(&v262);
              v241 = v257;
              v244 = a2;
              v242 = BYTE4(v257);
              v247 = a3;
              v243 = v258;
              v245 = v260;
              v240 = (__int64 *)&unk_49ECF68;
              v246 = v261;
              v249 = v264;
              if ( v264 )
                v248 = v263;
              v121 = v266;
              v250 = (__m128i *)v252;
              v122 = v266;
              v251 = 0x400000000LL;
              if ( !v266 )
              {
                v123 = v265;
                goto LABEL_174;
              }
              if ( v266 > 4uLL )
              {
                v142 = v266;
                sub_14B3F20((__int64)&v250, v266);
                v130 = v250;
                v121 = v266;
                v122 = v142;
              }
              else
              {
                v130 = (__m128i *)v252;
              }
              v131 = v265;
              v123 = (const __m128i *)((char *)v265 + 88 * v121);
              if ( v265 == v123 )
              {
                LODWORD(v251) = v122;
LABEL_174:
                v253 = v268;
                v254 = v269;
                v255 = v270;
                v240 = (__int64 *)&unk_49ECF98;
              }
              else
              {
                v143 = v37;
                v132 = v122;
                do
                {
                  if ( v130 )
                  {
                    v130->m128i_i64[0] = (__int64)v130[1].m128i_i64;
                    sub_1A87450(v130->m128i_i64, v131->m128i_i64[0], v131->m128i_i64[0] + v131->m128i_i64[1]);
                    v130[2].m128i_i64[0] = (__int64)v130[3].m128i_i64;
                    sub_1A87450(
                      v130[2].m128i_i64,
                      (_BYTE *)v131[2].m128i_i64[0],
                      v131[2].m128i_i64[0] + v131[2].m128i_i64[1]);
                    v130[4] = _mm_loadu_si128(v131 + 4);
                    v130[5].m128i_i64[0] = v131[5].m128i_i64[0];
                  }
                  v131 = (const __m128i *)((char *)v131 + 88);
                  v130 = (__m128i *)((char *)v130 + 88);
                }
                while ( v123 != v131 );
                v133 = v265;
                LODWORD(v251) = v132;
                v37 = v143;
                v123 = (const __m128i *)((char *)v265 + 88 * v266);
                v253 = v268;
                v254 = v269;
                v255 = v270;
                v240 = (__int64 *)&unk_49ECF98;
                v256 = (__int64 *)&unk_49ECF68;
                if ( v265 != v123 )
                {
                  do
                  {
                    v123 = (const __m128i *)((char *)v123 - 88);
                    v134 = (const __m128i *)v123[2].m128i_i64[0];
                    if ( v134 != &v123[3] )
                      j_j___libc_free_0(v134, v123[3].m128i_i64[0] + 1);
                    if ( (const __m128i *)v123->m128i_i64[0] != &v123[1] )
                      j_j___libc_free_0(v123->m128i_i64[0], v123[1].m128i_i64[0] + 1);
                  }
                  while ( v133 != v123 );
                  v123 = v265;
                }
              }
              if ( v123 != (const __m128i *)&v267 )
                _libc_free((unsigned __int64)v123);
              sub_143AA50(v154, (__int64)&v240);
              v124 = v250;
              v240 = (__int64 *)&unk_49ECF68;
              v125 = (__m128i *)((char *)v250 + 88 * (unsigned int)v251);
              if ( v250 != v125 )
              {
                do
                {
                  v125 = (__m128i *)((char *)v125 - 88);
                  v126 = (__m128i *)v125[2].m128i_i64[0];
                  if ( v126 != &v125[3] )
                    j_j___libc_free_0(v126, v125[3].m128i_i64[0] + 1);
                  if ( (__m128i *)v125->m128i_i64[0] != &v125[1] )
                    j_j___libc_free_0(v125->m128i_i64[0], v125[1].m128i_i64[0] + 1);
                }
                while ( v124 != v125 );
                v125 = v250;
              }
              if ( v125 != (__m128i *)v252 )
                _libc_free((unsigned __int64)v125);
            }
            if ( v165 )
              sub_161E7C0((__int64)&v165, (__int64)v165);
            if ( v237 != v239 )
              _libc_free((unsigned __int64)v237);
            if ( v232[0] )
              sub_161E7C0((__int64)v232, v232[0]);
            j___libc_free_0(v228);
            if ( v221 != v220 )
              _libc_free((unsigned __int64)v221);
            j___libc_free_0(v216);
            j___libc_free_0(v212);
            j___libc_free_0(v208);
            if ( v206 )
            {
              v127 = v204;
              v128 = &v204[5 * v206];
              do
              {
                while ( *v127 == -8 )
                {
                  if ( v127[1] != -8 )
                    goto LABEL_197;
                  v127 += 5;
                  if ( v128 == v127 )
                    goto LABEL_204;
                }
                if ( *v127 != -16 || v127[1] != -16 )
                {
LABEL_197:
                  v129 = v127[4];
                  if ( v129 != 0 && v129 != -8 && v129 != -16 )
                    sub_1649B30(v127 + 2);
                }
                v127 += 5;
              }
              while ( v128 != v127 );
            }
LABEL_204:
            j___libc_free_0(v204);
            v36 = (_QWORD *)v36[1];
            v150 = v155;
            if ( (_QWORD *)v162 == v36 )
              goto LABEL_85;
          }
        }
        else
        {
          v36 = (_QWORD *)v36[1];
          if ( (_QWORD *)v162 == v36 )
            goto LABEL_85;
        }
      }
    }
LABEL_5:
    while ( 2 )
    {
      v7 = *(_QWORD *)(v5 - 24);
      if ( !*(_BYTE *)(v5 - 8) )
      {
        v8 = *(__int64 ***)(v7 + 8);
        *(_BYTE *)(v5 - 8) = 1;
        *(_QWORD *)(v5 - 16) = v8;
        goto LABEL_9;
      }
      while ( 1 )
      {
        v8 = *(__int64 ***)(v5 - 16);
LABEL_9:
        if ( *(__int64 ***)(v7 + 16) == v8 )
          break;
        *(_QWORD *)(v5 - 16) = v8 + 1;
        v10 = *v8;
        v11 = v185;
        if ( v186 != v185 )
          goto LABEL_7;
        v12 = &v185[HIDWORD(v187)];
        if ( v185 == v12 )
        {
LABEL_27:
          if ( HIDWORD(v187) < (unsigned int)v187 )
          {
            ++HIDWORD(v187);
            *v12 = v10;
            ++v184;
LABEL_18:
            v256 = v10;
            LOBYTE(v258) = 0;
            sub_197E9F0(&v190, (__int64)&v256);
            v5 = v191;
            if ( v191 != v190 )
              goto LABEL_4;
            goto LABEL_19;
          }
LABEL_7:
          sub_16CCBA0((__int64)&v184, (__int64)v10);
          if ( v9 )
            goto LABEL_18;
        }
        else
        {
          v13 = 0;
          while ( v10 != *v11 )
          {
            if ( *v11 == (__int64 *)-2LL )
            {
              v13 = v11;
              if ( v12 == v11 + 1 )
                goto LABEL_17;
              ++v11;
            }
            else if ( v12 == ++v11 )
            {
              if ( !v13 )
                goto LABEL_27;
LABEL_17:
              *v13 = v10;
              --v188;
              ++v184;
              goto LABEL_18;
            }
          }
        }
      }
      v191 -= 24;
      v5 = v191;
      if ( v191 != v190 )
        continue;
      break;
    }
LABEL_19:
    if ( v5 )
      j_j___libc_free_0(v5, v192 - v5);
    if ( v186 != v185 )
      _libc_free((unsigned __int64)v186);
    ++v159;
  }
  while ( v139 != v159 );
  return v156;
}
