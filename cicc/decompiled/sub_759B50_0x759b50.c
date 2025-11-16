// Function: sub_759B50
// Address: 0x759b50
//
_DWORD *__fastcall sub_759B50(
        __int64 (__fastcall *a1)(_QWORD, _QWORD),
        __int64 (__fastcall *a2)(_QWORD, _QWORD, _QWORD),
        __int64 (__fastcall *a3)(_QWORD, _QWORD),
        __int64 (__fastcall *a4)(_QWORD, _QWORD),
        __int64 (__fastcall *a5)(_QWORD, _QWORD),
        int a6)
{
  __int64 (__fastcall *v6)(_QWORD, _QWORD); // rax
  int v7; // r15d
  __int64 (__fastcall *v8)(_QWORD, _QWORD, _QWORD); // rax
  __int64 (__fastcall *v9)(_QWORD, _QWORD); // rax
  __int64 (__fastcall *v10)(_QWORD, _QWORD); // rax
  __int64 (__fastcall *v11)(_QWORD, _QWORD); // rax
  int v12; // r14d
  __int64 *v13; // rdi
  __int64 *v14; // rbx
  __int64 i; // rdi
  __int64 v16; // rax
  __int64 *v17; // rdi
  const char *v18; // rbx
  void (__fastcall *v19)(const char *, __int64, size_t); // r13
  size_t v20; // rax
  const char *v21; // rdi
  void (__fastcall *v22)(const char *, __int64, size_t); // rbx
  size_t v23; // rax
  __int64 v24; // rdi
  __int64 *j; // rbx
  __int64 v26; // rax
  __int64 *v27; // rdi
  __int64 v28; // rdi
  __int64 *v29; // r13
  __int64 v30; // rax
  __int64 *v31; // rdi
  __int64 v32; // rdi
  __int64 *v33; // r13
  __int64 v34; // rax
  __int64 *v35; // rdi
  __int64 v36; // rdi
  __int64 *v37; // r13
  __int64 v38; // rax
  __int64 *v39; // rdi
  __int64 v40; // rdi
  __int64 *v41; // r13
  __int64 v42; // rax
  __int64 *v43; // rdi
  __int64 v44; // rdi
  __int64 *v45; // r13
  __int64 v46; // rax
  __int64 *v47; // rdi
  __int64 v48; // rdi
  __int64 *v49; // r13
  __int64 v50; // rax
  __int64 *v51; // rdi
  __int64 v52; // rdi
  __int64 *v53; // r13
  __int64 v54; // rax
  __int64 *v55; // rdi
  __int64 v56; // rdi
  __int64 *v57; // r13
  __int64 v58; // rax
  __int64 *v59; // rdi
  __int64 v60; // rdi
  __int64 *v61; // r13
  __int64 v62; // rax
  __int64 *v63; // rdi
  __int64 v64; // rdi
  __int64 *v65; // r13
  __int64 v66; // rax
  __int64 *v67; // rdi
  __int64 v68; // rdi
  __int64 *v69; // r13
  __int64 v70; // rax
  __int64 *v71; // rdi
  __int64 v72; // rdi
  __int64 *v73; // r13
  __int64 v74; // rax
  __int64 *v75; // rdi
  __int64 v76; // rdi
  __int64 *v77; // r13
  __int64 v78; // rax
  __int64 *v79; // rdi
  __int64 v80; // rdi
  __int64 *v81; // r13
  __int64 v82; // rax
  __int64 *v83; // rdi
  __int64 v84; // rdi
  __int64 *v85; // r13
  __int64 v86; // rax
  __int64 *v87; // rdi
  __int64 v88; // rdi
  __int64 *v89; // r13
  __int64 v90; // rax
  __int64 *v91; // rdi
  __int64 v92; // rdi
  __int64 *v93; // r13
  __int64 v94; // rax
  __int64 *v95; // rdi
  __int64 v96; // rdi
  __int64 *v97; // r13
  __int64 v98; // rax
  __int64 *v99; // rdi
  __int64 v100; // rdi
  __int64 *v101; // r13
  __int64 v102; // rax
  __int64 *v103; // rdi
  __int64 v104; // rdi
  __int64 *v105; // r13
  __int64 v106; // rax
  __int64 *v107; // rdi
  __int64 v108; // rdi
  __int64 *v109; // r13
  __int64 v110; // rax
  __int64 *v111; // rdi
  __int64 v112; // rdi
  __int64 *v113; // r13
  __int64 v114; // rax
  __int64 *v115; // rdi
  __int64 v116; // rdi
  __int64 *v117; // r13
  __int64 v118; // rax
  __int64 *v119; // rdi
  __int64 v120; // rdi
  __int64 *v121; // r13
  __int64 v122; // rax
  __int64 *v123; // rdi
  __int64 v124; // rdi
  __int64 *v125; // r13
  __int64 v126; // rax
  __int64 *v127; // rdi
  __int64 v128; // rdi
  __int64 *v129; // r13
  __int64 v130; // rax
  __int64 *v131; // rdi
  __int64 v132; // rdi
  __int64 *v133; // r13
  __int64 v134; // rax
  __int64 *v135; // rdi
  __int64 v136; // rdi
  __int64 *v137; // r13
  __int64 v138; // rax
  __int64 *v139; // rdi
  __int64 v140; // rdi
  __int64 *v141; // r13
  __int64 v142; // rax
  __int64 *v143; // rdi
  __int64 v144; // rdi
  __int64 *v145; // r13
  __int64 v146; // rax
  __int64 *v147; // rdi
  __int64 v148; // rdi
  __int64 *v149; // r13
  __int64 v150; // rax
  __int64 *v151; // rdi
  __int64 v152; // rdi
  __int64 *v153; // r13
  __int64 v154; // rax
  __int64 *v155; // rdi
  __int64 v156; // rdi
  __int64 *v157; // r13
  __int64 v158; // rax
  __int64 *v159; // rdi
  __int64 v160; // rdi
  __int64 *v161; // r13
  __int64 v162; // rax
  __int64 *v163; // rdi
  __int64 v164; // rdi
  __int64 *v165; // r13
  __int64 v166; // rax
  __int64 *v167; // rdi
  __int64 v168; // rdi
  __int64 *v169; // r13
  __int64 v170; // rax
  __int64 *v171; // rdi
  __int64 v172; // rdi
  __int64 *v173; // r13
  __int64 v174; // rax
  __int64 *v175; // rdi
  __int64 v176; // rdi
  __int64 *v177; // r13
  __int64 v178; // rax
  __int64 *v179; // rdi
  __int64 v180; // rdi
  __int64 *v181; // r13
  __int64 v182; // rax
  __int64 *v183; // rdi
  __int64 v184; // rdi
  __int64 *v185; // r13
  __int64 v186; // rax
  __int64 *v187; // rdi
  __int64 v188; // rdi
  __int64 *v189; // r13
  __int64 v190; // rax
  __int64 *v191; // rdi
  __int64 v192; // rdi
  __int64 *v193; // r13
  __int64 v194; // rax
  __int64 *v195; // rdi
  __int64 v196; // rdi
  __int64 *v197; // r13
  __int64 v198; // rax
  __int64 *v199; // rdi
  __int64 v200; // rdi
  __int64 *v201; // r13
  __int64 v202; // rax
  __int64 *v203; // rdi
  __int64 v204; // rdi
  __int64 *v205; // rbx
  __int64 v206; // rax
  __int64 *v207; // rdi
  __int64 v208; // rdi
  __int64 *v209; // rbx
  __int64 v210; // rax
  __int64 *v211; // rdi
  __int64 v212; // rdi
  __int64 *v213; // rbx
  __int64 v214; // rax
  __int64 *v215; // rdi
  __int64 v216; // rdi
  __int64 *k; // rbx
  __int64 v218; // rax
  __int64 *v219; // rdi
  int v221; // [rsp+10h] [rbp-60h]
  int v222; // [rsp+14h] [rbp-5Ch]
  __int64 (__fastcall *v223)(_QWORD, _QWORD); // [rsp+18h] [rbp-58h]
  __int64 (__fastcall *v224)(_QWORD, _QWORD); // [rsp+20h] [rbp-50h]
  __int64 (__fastcall *v225)(_QWORD, _QWORD); // [rsp+28h] [rbp-48h]
  __int64 (__fastcall *v226)(_QWORD, _QWORD, _QWORD); // [rsp+30h] [rbp-40h]
  __int64 (__fastcall *v227)(_QWORD, _QWORD); // [rsp+38h] [rbp-38h]

  v6 = qword_4F08040;
  qword_4F08040 = a1;
  v7 = dword_4F08018;
  dword_4F08018 = a6;
  v227 = v6;
  v8 = qword_4F08038;
  qword_4F08038 = a2;
  v226 = v8;
  v9 = qword_4F08030;
  qword_4F08030 = a5;
  v225 = v9;
  v10 = qword_4F08028;
  qword_4F08028 = a3;
  v224 = v10;
  v11 = qword_4F08020;
  qword_4F08020 = a4;
  v223 = v11;
  LODWORD(v11) = dword_4F08014;
  dword_4F08014 = 1;
  v222 = (int)v11;
  v221 = dword_4F08010;
  v12 = dword_4D03B64;
  v13 = (__int64 *)unk_4F07288;
  if ( a3 )
  {
    unk_4F07288 = a3(unk_4F07288, 23);
    v13 = (__int64 *)unk_4F07288;
  }
  v14 = &qword_4F07280;
  dword_4D03B64 = ((*((_BYTE *)v13 - 8) >> 2) ^ 1) & 1;
  dword_4F08010 = (*(_BYTE *)(v13 - 1) & 2) != 0;
  sub_7506E0(v13, 23);
  for ( i = qword_4F07280; i; i = v17[7] )
  {
    if ( qword_4F08020 )
    {
      v16 = qword_4F08020(i, 1);
      *v14 = v16;
      v17 = (__int64 *)v16;
    }
    else
    {
      v17 = (__int64 *)*v14;
    }
    if ( v17 )
    {
      sub_7506E0(v17, 1);
      v17 = (__int64 *)*v14;
    }
    v14 = v17 + 7;
  }
  if ( !qword_4F08028 || (unk_4F07290 = qword_4F08028(unk_4F07290, 11), !qword_4F08028) )
  {
    v18 = (const char *)unk_4F07298;
    if ( !unk_4F07298 )
    {
      v21 = (const char *)unk_4F072A0;
      goto LABEL_18;
    }
    goto LABEL_14;
  }
  unk_4F07298 = qword_4F08028(unk_4F07298, 26);
  v18 = (const char *)unk_4F07298;
  if ( unk_4F07298 )
  {
LABEL_14:
    v19 = (void (__fastcall *)(const char *, __int64, size_t))qword_4F08038;
    if ( qword_4F08038 )
    {
      v20 = strlen(v18);
      v19(v18, 26, v20 + 1);
    }
  }
  v21 = (const char *)unk_4F072A0;
  if ( qword_4F08028 )
  {
    unk_4F072A0 = qword_4F08028(unk_4F072A0, 26);
    v21 = (const char *)unk_4F072A0;
  }
LABEL_18:
  if ( v21 )
  {
    v22 = (void (__fastcall *)(const char *, __int64, size_t))qword_4F08038;
    if ( qword_4F08038 )
    {
      v23 = strlen(v21);
      v22(v21, 26, v23 + 1);
    }
  }
  v24 = qword_4F072C0;
  for ( j = &qword_4F072C0; v24; v24 = *v27 )
  {
    if ( qword_4F08020 )
    {
      v26 = qword_4F08020(v24, 57);
      *j = v26;
      v27 = (__int64 *)v26;
    }
    else
    {
      v27 = (__int64 *)*j;
    }
    if ( v27 )
    {
      sub_7506E0(v27, 57);
      v27 = (__int64 *)*j;
    }
    j = v27;
  }
  v28 = qword_4F06D10[0];
  if ( qword_4F06D10[0] )
  {
    v29 = qword_4F06D10;
    do
    {
      if ( qword_4F08028 )
      {
        v30 = qword_4F08028(v28, 1);
        *v29 = v30;
        v31 = (__int64 *)v30;
      }
      else
      {
        v31 = (__int64 *)*v29;
      }
      if ( v31 )
      {
        sub_7506E0(v31, 1);
        v31 = (__int64 *)*v29;
      }
      v29 = v31 - 2;
      v28 = *(v31 - 2);
    }
    while ( v28 );
  }
  v32 = qword_4F06D10[2];
  if ( qword_4F06D10[2] )
  {
    v33 = &qword_4F06D10[2];
    do
    {
      if ( qword_4F08028 )
      {
        v34 = qword_4F08028(v32, 2);
        *v33 = v34;
        v35 = (__int64 *)v34;
      }
      else
      {
        v35 = (__int64 *)*v33;
      }
      if ( v35 )
      {
        sub_7506E0(v35, 2);
        v35 = (__int64 *)*v33;
      }
      v33 = v35 - 2;
      v32 = *(v35 - 2);
    }
    while ( v32 );
  }
  v36 = qword_4F06D10[4];
  if ( qword_4F06D10[4] )
  {
    v37 = &qword_4F06D10[4];
    do
    {
      if ( qword_4F08028 )
      {
        v38 = qword_4F08028(v36, 3);
        *v37 = v38;
        v39 = (__int64 *)v38;
      }
      else
      {
        v39 = (__int64 *)*v37;
      }
      if ( v39 )
      {
        sub_7506E0(v39, 3);
        v39 = (__int64 *)*v37;
      }
      v37 = v39 - 2;
      v36 = *(v39 - 2);
    }
    while ( v36 );
  }
  v40 = qword_4F06D10[6];
  if ( qword_4F06D10[6] )
  {
    v41 = &qword_4F06D10[6];
    do
    {
      if ( qword_4F08028 )
      {
        v42 = qword_4F08028(v40, 4);
        *v41 = v42;
        v43 = (__int64 *)v42;
      }
      else
      {
        v43 = (__int64 *)*v41;
      }
      if ( v43 )
      {
        sub_7506E0(v43, 4);
        v43 = (__int64 *)*v41;
      }
      v41 = v43 - 2;
      v40 = *(v43 - 2);
    }
    while ( v40 );
  }
  v44 = qword_4F06D10[8];
  if ( qword_4F06D10[8] )
  {
    v45 = &qword_4F06D10[8];
    do
    {
      if ( qword_4F08028 )
      {
        v46 = qword_4F08028(v44, 5);
        *v45 = v46;
        v47 = (__int64 *)v46;
      }
      else
      {
        v47 = (__int64 *)*v45;
      }
      if ( v47 )
      {
        sub_7506E0(v47, 5);
        v47 = (__int64 *)*v45;
      }
      v45 = v47 - 2;
      v44 = *(v47 - 2);
    }
    while ( v44 );
  }
  v48 = qword_4F06D10[10];
  if ( qword_4F06D10[10] )
  {
    v49 = &qword_4F06D10[10];
    do
    {
      if ( qword_4F08028 )
      {
        v50 = qword_4F08028(v48, 6);
        *v49 = v50;
        v51 = (__int64 *)v50;
      }
      else
      {
        v51 = (__int64 *)*v49;
      }
      if ( v51 )
      {
        sub_7506E0(v51, 6);
        v51 = (__int64 *)*v49;
      }
      v49 = v51 - 2;
      v48 = *(v51 - 2);
    }
    while ( v48 );
  }
  v52 = qword_4F06D10[12];
  if ( qword_4F06D10[12] )
  {
    v53 = &qword_4F06D10[12];
    do
    {
      if ( qword_4F08028 )
      {
        v54 = qword_4F08028(v52, 7);
        *v53 = v54;
        v55 = (__int64 *)v54;
      }
      else
      {
        v55 = (__int64 *)*v53;
      }
      if ( v55 )
      {
        sub_7506E0(v55, 7);
        v55 = (__int64 *)*v53;
      }
      v53 = v55 - 2;
      v52 = *(v55 - 2);
    }
    while ( v52 );
  }
  v56 = qword_4F06D10[14];
  if ( qword_4F06D10[14] )
  {
    v57 = &qword_4F06D10[14];
    do
    {
      if ( qword_4F08028 )
      {
        v58 = qword_4F08028(v56, 8);
        *v57 = v58;
        v59 = (__int64 *)v58;
      }
      else
      {
        v59 = (__int64 *)*v57;
      }
      if ( v59 )
      {
        sub_7506E0(v59, 8);
        v59 = (__int64 *)*v57;
      }
      v57 = v59 - 2;
      v56 = *(v59 - 2);
    }
    while ( v56 );
  }
  v60 = qword_4F06D10[16];
  if ( qword_4F06D10[16] )
  {
    v61 = &qword_4F06D10[16];
    do
    {
      if ( qword_4F08028 )
      {
        v62 = qword_4F08028(v60, 9);
        *v61 = v62;
        v63 = (__int64 *)v62;
      }
      else
      {
        v63 = (__int64 *)*v61;
      }
      if ( v63 )
      {
        sub_7506E0(v63, 9);
        v63 = (__int64 *)*v61;
      }
      v61 = v63 - 2;
      v60 = *(v63 - 2);
    }
    while ( v60 );
  }
  v64 = qword_4F06D10[18];
  if ( qword_4F06D10[18] )
  {
    v65 = &qword_4F06D10[18];
    do
    {
      if ( qword_4F08028 )
      {
        v66 = qword_4F08028(v64, 10);
        *v65 = v66;
        v67 = (__int64 *)v66;
      }
      else
      {
        v67 = (__int64 *)*v65;
      }
      if ( v67 )
      {
        sub_7506E0(v67, 10);
        v67 = (__int64 *)*v65;
      }
      v65 = v67 - 2;
      v64 = *(v67 - 2);
    }
    while ( v64 );
  }
  v68 = qword_4F06D10[20];
  if ( qword_4F06D10[20] )
  {
    v69 = &qword_4F06D10[20];
    do
    {
      if ( qword_4F08028 )
      {
        v70 = qword_4F08028(v68, 11);
        *v69 = v70;
        v71 = (__int64 *)v70;
      }
      else
      {
        v71 = (__int64 *)*v69;
      }
      if ( v71 )
      {
        sub_7506E0(v71, 11);
        v71 = (__int64 *)*v69;
      }
      v69 = v71 - 2;
      v68 = *(v71 - 2);
    }
    while ( v68 );
  }
  v72 = qword_4F06D10[22];
  if ( qword_4F06D10[22] )
  {
    v73 = &qword_4F06D10[22];
    do
    {
      if ( qword_4F08028 )
      {
        v74 = qword_4F08028(v72, 12);
        *v73 = v74;
        v75 = (__int64 *)v74;
      }
      else
      {
        v75 = (__int64 *)*v73;
      }
      if ( v75 )
      {
        sub_7506E0(v75, 12);
        v75 = (__int64 *)*v73;
      }
      v73 = v75 - 2;
      v72 = *(v75 - 2);
    }
    while ( v72 );
  }
  v76 = qword_4F06D10[24];
  if ( qword_4F06D10[24] )
  {
    v77 = &qword_4F06D10[24];
    do
    {
      if ( qword_4F08028 )
      {
        v78 = qword_4F08028(v76, 13);
        *v77 = v78;
        v79 = (__int64 *)v78;
      }
      else
      {
        v79 = (__int64 *)*v77;
      }
      if ( v79 )
      {
        sub_7506E0(v79, 13);
        v79 = (__int64 *)*v77;
      }
      v77 = v79 - 2;
      v76 = *(v79 - 2);
    }
    while ( v76 );
  }
  v80 = qword_4F06D10[26];
  if ( qword_4F06D10[26] )
  {
    v81 = &qword_4F06D10[26];
    do
    {
      if ( qword_4F08028 )
      {
        v82 = qword_4F08028(v80, 14);
        *v81 = v82;
        v83 = (__int64 *)v82;
      }
      else
      {
        v83 = (__int64 *)*v81;
      }
      if ( v83 )
      {
        sub_7506E0(v83, 14);
        v83 = (__int64 *)*v81;
      }
      v81 = v83 - 2;
      v80 = *(v83 - 2);
    }
    while ( v80 );
  }
  v84 = qword_4F06D10[28];
  if ( qword_4F06D10[28] )
  {
    v85 = &qword_4F06D10[28];
    do
    {
      if ( qword_4F08028 )
      {
        v86 = qword_4F08028(v84, 15);
        *v85 = v86;
        v87 = (__int64 *)v86;
      }
      else
      {
        v87 = (__int64 *)*v85;
      }
      if ( v87 )
      {
        sub_7506E0(v87, 15);
        v87 = (__int64 *)*v85;
      }
      v85 = v87 - 2;
      v84 = *(v87 - 2);
    }
    while ( v84 );
  }
  v88 = qword_4F06D10[30];
  if ( qword_4F06D10[30] )
  {
    v89 = &qword_4F06D10[30];
    do
    {
      if ( qword_4F08028 )
      {
        v90 = qword_4F08028(v88, 16);
        *v89 = v90;
        v91 = (__int64 *)v90;
      }
      else
      {
        v91 = (__int64 *)*v89;
      }
      if ( v91 )
      {
        sub_7506E0(v91, 16);
        v91 = (__int64 *)*v89;
      }
      v89 = v91 - 2;
      v88 = *(v91 - 2);
    }
    while ( v88 );
  }
  v92 = qword_4F06D10[32];
  if ( qword_4F06D10[32] )
  {
    v93 = &qword_4F06D10[32];
    do
    {
      if ( qword_4F08028 )
      {
        v94 = qword_4F08028(v92, 17);
        *v93 = v94;
        v95 = (__int64 *)v94;
      }
      else
      {
        v95 = (__int64 *)*v93;
      }
      if ( v95 )
      {
        sub_7506E0(v95, 17);
        v95 = (__int64 *)*v93;
      }
      v93 = v95 - 2;
      v92 = *(v95 - 2);
    }
    while ( v92 );
  }
  v96 = qword_4F06D10[34];
  if ( qword_4F06D10[34] )
  {
    v97 = &qword_4F06D10[34];
    do
    {
      if ( qword_4F08028 )
      {
        v98 = qword_4F08028(v96, 18);
        *v97 = v98;
        v99 = (__int64 *)v98;
      }
      else
      {
        v99 = (__int64 *)*v97;
      }
      if ( v99 )
      {
        sub_7506E0(v99, 18);
        v99 = (__int64 *)*v97;
      }
      v97 = v99 - 2;
      v96 = *(v99 - 2);
    }
    while ( v96 );
  }
  v100 = qword_4F06D10[36];
  if ( qword_4F06D10[36] )
  {
    v101 = &qword_4F06D10[36];
    do
    {
      if ( qword_4F08028 )
      {
        v102 = qword_4F08028(v100, 19);
        *v101 = v102;
        v103 = (__int64 *)v102;
      }
      else
      {
        v103 = (__int64 *)*v101;
      }
      if ( v103 )
      {
        sub_7506E0(v103, 19);
        v103 = (__int64 *)*v101;
      }
      v101 = v103 - 2;
      v100 = *(v103 - 2);
    }
    while ( v100 );
  }
  v104 = qword_4F06D10[38];
  if ( qword_4F06D10[38] )
  {
    v105 = &qword_4F06D10[38];
    do
    {
      if ( qword_4F08028 )
      {
        v106 = qword_4F08028(v104, 20);
        *v105 = v106;
        v107 = (__int64 *)v106;
      }
      else
      {
        v107 = (__int64 *)*v105;
      }
      if ( v107 )
      {
        sub_7506E0(v107, 20);
        v107 = (__int64 *)*v105;
      }
      v105 = v107 - 2;
      v104 = *(v107 - 2);
    }
    while ( v104 );
  }
  v108 = qword_4F06D10[40];
  if ( qword_4F06D10[40] )
  {
    v109 = &qword_4F06D10[40];
    do
    {
      if ( qword_4F08028 )
      {
        v110 = qword_4F08028(v108, 21);
        *v109 = v110;
        v111 = (__int64 *)v110;
      }
      else
      {
        v111 = (__int64 *)*v109;
      }
      if ( v111 )
      {
        sub_7506E0(v111, 21);
        v111 = (__int64 *)*v109;
      }
      v109 = v111 - 2;
      v108 = *(v111 - 2);
    }
    while ( v108 );
  }
  v112 = qword_4F06D10[42];
  if ( qword_4F06D10[42] )
  {
    v113 = &qword_4F06D10[42];
    do
    {
      if ( qword_4F08028 )
      {
        v114 = qword_4F08028(v112, 22);
        *v113 = v114;
        v115 = (__int64 *)v114;
      }
      else
      {
        v115 = (__int64 *)*v113;
      }
      if ( v115 )
      {
        sub_7506E0(v115, 22);
        v115 = (__int64 *)*v113;
      }
      v113 = v115 - 2;
      v112 = *(v115 - 2);
    }
    while ( v112 );
  }
  v116 = qword_4F06D10[44];
  if ( qword_4F06D10[44] )
  {
    v117 = &qword_4F06D10[44];
    do
    {
      if ( qword_4F08028 )
      {
        v118 = qword_4F08028(v116, 23);
        *v117 = v118;
        v119 = (__int64 *)v118;
      }
      else
      {
        v119 = (__int64 *)*v117;
      }
      if ( v119 )
      {
        sub_7506E0(v119, 23);
        v119 = (__int64 *)*v117;
      }
      v117 = v119 - 2;
      v116 = *(v119 - 2);
    }
    while ( v116 );
  }
  v120 = qword_4F06D10[52];
  if ( qword_4F06D10[52] )
  {
    v121 = &qword_4F06D10[52];
    do
    {
      if ( qword_4F08028 )
      {
        v122 = qword_4F08028(v120, 27);
        *v121 = v122;
        v123 = (__int64 *)v122;
      }
      else
      {
        v123 = (__int64 *)*v121;
      }
      if ( v123 )
      {
        sub_7506E0(v123, 27);
        v123 = (__int64 *)*v121;
      }
      v121 = v123 - 2;
      v120 = *(v123 - 2);
    }
    while ( v120 );
  }
  v124 = qword_4F06D10[54];
  if ( qword_4F06D10[54] )
  {
    v125 = &qword_4F06D10[54];
    do
    {
      if ( qword_4F08028 )
      {
        v126 = qword_4F08028(v124, 28);
        *v125 = v126;
        v127 = (__int64 *)v126;
      }
      else
      {
        v127 = (__int64 *)*v125;
      }
      if ( v127 )
      {
        sub_7506E0(v127, 28);
        v127 = (__int64 *)*v125;
      }
      v125 = v127 - 2;
      v124 = *(v127 - 2);
    }
    while ( v124 );
  }
  v128 = qword_4F06D10[56];
  if ( qword_4F06D10[56] )
  {
    v129 = &qword_4F06D10[56];
    do
    {
      if ( qword_4F08028 )
      {
        v130 = qword_4F08028(v128, 29);
        *v129 = v130;
        v131 = (__int64 *)v130;
      }
      else
      {
        v131 = (__int64 *)*v129;
      }
      if ( v131 )
      {
        sub_7506E0(v131, 29);
        v131 = (__int64 *)*v129;
      }
      v129 = v131 - 2;
      v128 = *(v131 - 2);
    }
    while ( v128 );
  }
  v132 = qword_4F06D10[58];
  if ( qword_4F06D10[58] )
  {
    v133 = &qword_4F06D10[58];
    do
    {
      if ( qword_4F08028 )
      {
        v134 = qword_4F08028(v132, 30);
        *v133 = v134;
        v135 = (__int64 *)v134;
      }
      else
      {
        v135 = (__int64 *)*v133;
      }
      if ( v135 )
      {
        sub_7506E0(v135, 30);
        v135 = (__int64 *)*v133;
      }
      v133 = v135 - 2;
      v132 = *(v135 - 2);
    }
    while ( v132 );
  }
  v136 = qword_4F06D10[66];
  if ( qword_4F06D10[66] )
  {
    v137 = &qword_4F06D10[66];
    do
    {
      if ( qword_4F08028 )
      {
        v138 = qword_4F08028(v136, 34);
        *v137 = v138;
        v139 = (__int64 *)v138;
      }
      else
      {
        v139 = (__int64 *)*v137;
      }
      if ( v139 )
      {
        sub_7506E0(v139, 34);
        v139 = (__int64 *)*v137;
      }
      v137 = v139 - 2;
      v136 = *(v139 - 2);
    }
    while ( v136 );
  }
  v140 = qword_4F06D10[68];
  if ( qword_4F06D10[68] )
  {
    v141 = &qword_4F06D10[68];
    do
    {
      if ( qword_4F08028 )
      {
        v142 = qword_4F08028(v140, 35);
        *v141 = v142;
        v143 = (__int64 *)v142;
      }
      else
      {
        v143 = (__int64 *)*v141;
      }
      if ( v143 )
      {
        sub_7506E0(v143, 35);
        v143 = (__int64 *)*v141;
      }
      v141 = v143 - 2;
      v140 = *(v143 - 2);
    }
    while ( v140 );
  }
  v144 = qword_4F06D10[70];
  if ( qword_4F06D10[70] )
  {
    v145 = &qword_4F06D10[70];
    do
    {
      if ( qword_4F08028 )
      {
        v146 = qword_4F08028(v144, 36);
        *v145 = v146;
        v147 = (__int64 *)v146;
      }
      else
      {
        v147 = (__int64 *)*v145;
      }
      if ( v147 )
      {
        sub_7506E0(v147, 36);
        v147 = (__int64 *)*v145;
      }
      v145 = v147 - 2;
      v144 = *(v147 - 2);
    }
    while ( v144 );
  }
  v148 = qword_4F06D10[72];
  if ( qword_4F06D10[72] )
  {
    v149 = &qword_4F06D10[72];
    do
    {
      if ( qword_4F08028 )
      {
        v150 = qword_4F08028(v148, 37);
        *v149 = v150;
        v151 = (__int64 *)v150;
      }
      else
      {
        v151 = (__int64 *)*v149;
      }
      if ( v151 )
      {
        sub_7506E0(v151, 37);
        v151 = (__int64 *)*v149;
      }
      v149 = v151 - 2;
      v148 = *(v151 - 2);
    }
    while ( v148 );
  }
  v152 = qword_4F06D10[74];
  if ( qword_4F06D10[74] )
  {
    v153 = &qword_4F06D10[74];
    do
    {
      if ( qword_4F08028 )
      {
        v154 = qword_4F08028(v152, 38);
        *v153 = v154;
        v155 = (__int64 *)v154;
      }
      else
      {
        v155 = (__int64 *)*v153;
      }
      if ( v155 )
      {
        sub_7506E0(v155, 38);
        v155 = (__int64 *)*v153;
      }
      v153 = v155 - 2;
      v152 = *(v155 - 2);
    }
    while ( v152 );
  }
  v156 = qword_4F06D10[76];
  if ( qword_4F06D10[76] )
  {
    v157 = &qword_4F06D10[76];
    do
    {
      if ( qword_4F08028 )
      {
        v158 = qword_4F08028(v156, 39);
        *v157 = v158;
        v159 = (__int64 *)v158;
      }
      else
      {
        v159 = (__int64 *)*v157;
      }
      if ( v159 )
      {
        sub_7506E0(v159, 39);
        v159 = (__int64 *)*v157;
      }
      v157 = v159 - 2;
      v156 = *(v159 - 2);
    }
    while ( v156 );
  }
  v160 = qword_4F06D10[78];
  if ( qword_4F06D10[78] )
  {
    v161 = &qword_4F06D10[78];
    do
    {
      if ( qword_4F08028 )
      {
        v162 = qword_4F08028(v160, 40);
        *v161 = v162;
        v163 = (__int64 *)v162;
      }
      else
      {
        v163 = (__int64 *)*v161;
      }
      if ( v163 )
      {
        sub_7506E0(v163, 40);
        v163 = (__int64 *)*v161;
      }
      v161 = v163 - 2;
      v160 = *(v163 - 2);
    }
    while ( v160 );
  }
  v164 = qword_4F06D10[80];
  if ( qword_4F06D10[80] )
  {
    v165 = &qword_4F06D10[80];
    do
    {
      if ( qword_4F08028 )
      {
        v166 = qword_4F08028(v164, 41);
        *v165 = v166;
        v167 = (__int64 *)v166;
      }
      else
      {
        v167 = (__int64 *)*v165;
      }
      if ( v167 )
      {
        sub_7506E0(v167, 41);
        v167 = (__int64 *)*v165;
      }
      v165 = v167 - 2;
      v164 = *(v167 - 2);
    }
    while ( v164 );
  }
  v168 = qword_4F06D10[82];
  if ( qword_4F06D10[82] )
  {
    v169 = &qword_4F06D10[82];
    do
    {
      if ( qword_4F08028 )
      {
        v170 = qword_4F08028(v168, 42);
        *v169 = v170;
        v171 = (__int64 *)v170;
      }
      else
      {
        v171 = (__int64 *)*v169;
      }
      if ( v171 )
      {
        sub_7506E0(v171, 42);
        v171 = (__int64 *)*v169;
      }
      v169 = v171 - 2;
      v168 = *(v171 - 2);
    }
    while ( v168 );
  }
  v172 = qword_4F06D10[84];
  if ( qword_4F06D10[84] )
  {
    v173 = &qword_4F06D10[84];
    do
    {
      if ( qword_4F08028 )
      {
        v174 = qword_4F08028(v172, 43);
        *v173 = v174;
        v175 = (__int64 *)v174;
      }
      else
      {
        v175 = (__int64 *)*v173;
      }
      if ( v175 )
      {
        sub_7506E0(v175, 43);
        v175 = (__int64 *)*v173;
      }
      v173 = v175 - 2;
      v172 = *(v175 - 2);
    }
    while ( v172 );
  }
  v176 = qword_4F06D10[94];
  if ( qword_4F06D10[94] )
  {
    v177 = &qword_4F06D10[94];
    do
    {
      if ( qword_4F08028 )
      {
        v178 = qword_4F08028(v176, 48);
        *v177 = v178;
        v179 = (__int64 *)v178;
      }
      else
      {
        v179 = (__int64 *)*v177;
      }
      if ( v179 )
      {
        sub_7506E0(v179, 48);
        v179 = (__int64 *)*v177;
      }
      v177 = v179 - 2;
      v176 = *(v179 - 2);
    }
    while ( v176 );
  }
  v180 = qword_4F06D10[96];
  if ( qword_4F06D10[96] )
  {
    v181 = &qword_4F06D10[96];
    do
    {
      if ( qword_4F08028 )
      {
        v182 = qword_4F08028(v180, 49);
        *v181 = v182;
        v183 = (__int64 *)v182;
      }
      else
      {
        v183 = (__int64 *)*v181;
      }
      if ( v183 )
      {
        sub_7506E0(v183, 49);
        v183 = (__int64 *)*v181;
      }
      v181 = v183 - 2;
      v180 = *(v183 - 2);
    }
    while ( v180 );
  }
  v184 = qword_4F06D10[98];
  if ( qword_4F06D10[98] )
  {
    v185 = &qword_4F06D10[98];
    do
    {
      if ( qword_4F08028 )
      {
        v186 = qword_4F08028(v184, 50);
        *v185 = v186;
        v187 = (__int64 *)v186;
      }
      else
      {
        v187 = (__int64 *)*v185;
      }
      if ( v187 )
      {
        sub_7506E0(v187, 50);
        v187 = (__int64 *)*v185;
      }
      v185 = v187 - 2;
      v184 = *(v187 - 2);
    }
    while ( v184 );
  }
  v188 = qword_4F06D10[126];
  if ( qword_4F06D10[126] )
  {
    v189 = &qword_4F06D10[126];
    do
    {
      if ( qword_4F08028 )
      {
        v190 = qword_4F08028(v188, 64);
        *v189 = v190;
        v191 = (__int64 *)v190;
      }
      else
      {
        v191 = (__int64 *)*v189;
      }
      if ( v191 )
      {
        sub_7506E0(v191, 64);
        v191 = (__int64 *)*v189;
      }
      v189 = v191 - 2;
      v188 = *(v191 - 2);
    }
    while ( v188 );
  }
  v192 = qword_4F06D10[122];
  if ( qword_4F06D10[122] )
  {
    v193 = &qword_4F06D10[122];
    do
    {
      if ( qword_4F08028 )
      {
        v194 = qword_4F08028(v192, 62);
        *v193 = v194;
        v195 = (__int64 *)v194;
      }
      else
      {
        v195 = (__int64 *)*v193;
      }
      if ( v195 )
      {
        sub_7506E0(v195, 62);
        v195 = (__int64 *)*v193;
      }
      v193 = v195 - 2;
      v192 = *(v195 - 2);
    }
    while ( v192 );
  }
  v196 = qword_4F06D10[128];
  if ( qword_4F06D10[128] )
  {
    v197 = &qword_4F06D10[128];
    do
    {
      if ( qword_4F08028 )
      {
        v198 = qword_4F08028(v196, 65);
        *v197 = v198;
        v199 = (__int64 *)v198;
      }
      else
      {
        v199 = (__int64 *)*v197;
      }
      if ( v199 )
      {
        sub_7506E0(v199, 65);
        v199 = (__int64 *)*v197;
      }
      v197 = v199 - 2;
      v196 = *(v199 - 2);
    }
    while ( v196 );
  }
  v200 = qword_4F06D10[130];
  if ( qword_4F06D10[130] )
  {
    v201 = &qword_4F06D10[130];
    do
    {
      if ( qword_4F08028 )
      {
        v202 = qword_4F08028(v200, 66);
        *v201 = v202;
        v203 = (__int64 *)v202;
      }
      else
      {
        v203 = (__int64 *)*v201;
      }
      if ( v203 )
      {
        sub_7506E0(v203, 66);
        v203 = (__int64 *)*v201;
      }
      v201 = v203 - 2;
      v200 = *(v203 - 2);
    }
    while ( v200 );
  }
  v204 = qword_4F06D10[148];
  if ( qword_4F06D10[148] )
  {
    v205 = &qword_4F06D10[148];
    do
    {
      if ( qword_4F08028 )
      {
        v206 = qword_4F08028(v204, 75);
        *v205 = v206;
        v207 = (__int64 *)v206;
      }
      else
      {
        v207 = (__int64 *)*v205;
      }
      if ( v207 )
      {
        sub_7506E0(v207, 75);
        v207 = (__int64 *)*v205;
      }
      v205 = v207 - 2;
      v204 = *(v207 - 2);
    }
    while ( v204 );
  }
  v208 = qword_4F07308;
  if ( qword_4F07308 )
  {
    v209 = &qword_4F07308;
    do
    {
      if ( qword_4F08020 )
      {
        v210 = qword_4F08020(v208, 67);
        *v209 = v210;
        v211 = (__int64 *)v210;
      }
      else
      {
        v211 = (__int64 *)*v209;
      }
      if ( v211 )
      {
        sub_7506E0(v211, 67);
        v211 = (__int64 *)*v209;
      }
      v209 = v211;
      v208 = *v211;
    }
    while ( v208 );
  }
  v212 = qword_4F07300;
  if ( qword_4F07300 )
  {
    v213 = &qword_4F07300;
    do
    {
      if ( qword_4F08020 )
      {
        v214 = qword_4F08020(v212, 6);
        *v213 = v214;
        v215 = (__int64 *)v214;
      }
      else
      {
        v215 = (__int64 *)*v213;
      }
      if ( v215 )
      {
        sub_7506E0(v215, 6);
        v215 = (__int64 *)*v213;
      }
      v213 = v215 + 14;
      v212 = v215[14];
    }
    while ( v212 );
  }
  v216 = qword_4F07320[0];
  for ( k = qword_4F07320; v216; v216 = *v219 )
  {
    if ( qword_4F08020 )
    {
      v218 = qword_4F08020(v216, 86);
      *k = v218;
      v219 = (__int64 *)v218;
    }
    else
    {
      v219 = (__int64 *)*k;
    }
    if ( v219 )
    {
      sub_7506E0(v219, 86);
      v219 = (__int64 *)*k;
    }
    k = v219;
  }
  dword_4F08018 = v7;
  qword_4F08040 = v227;
  qword_4F08038 = v226;
  qword_4F08030 = v225;
  qword_4F08028 = v224;
  qword_4F08020 = v223;
  dword_4F08014 = v222;
  dword_4F08010 = v221;
  dword_4D03B64 = v12;
  return &dword_4D03B64;
}
