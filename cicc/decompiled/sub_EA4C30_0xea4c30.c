// Function: sub_EA4C30
// Address: 0xea4c30
//
__int64 __fastcall sub_EA4C30(__int64 a1)
{
  __int64 *v1; // r12
  int v2; // eax
  unsigned int v3; // r13d
  __int64 *v4; // r14
  __int64 v5; // rdx
  int v6; // eax
  unsigned int v7; // r13d
  __int64 *v8; // r14
  __int64 v9; // rdx
  int v10; // eax
  unsigned int v11; // r13d
  __int64 *v12; // r14
  __int64 v13; // rdx
  int v14; // eax
  unsigned int v15; // r13d
  __int64 *v16; // r14
  __int64 v17; // rdx
  int v18; // eax
  unsigned int v19; // r13d
  __int64 *v20; // r14
  __int64 v21; // rdx
  int v22; // eax
  unsigned int v23; // r13d
  __int64 *v24; // r14
  __int64 v25; // rdx
  int v26; // eax
  unsigned int v27; // r13d
  __int64 *v28; // r14
  __int64 v29; // rdx
  int v30; // eax
  unsigned int v31; // r13d
  __int64 *v32; // r14
  __int64 v33; // rdx
  int v34; // eax
  unsigned int v35; // r13d
  __int64 *v36; // r14
  __int64 v37; // rdx
  int v38; // eax
  unsigned int v39; // r13d
  __int64 *v40; // r14
  __int64 v41; // rdx
  int v42; // eax
  unsigned int v43; // r13d
  __int64 *v44; // r14
  __int64 v45; // rdx
  int v46; // eax
  unsigned int v47; // r13d
  __int64 *v48; // r14
  __int64 v49; // rdx
  int v50; // eax
  unsigned int v51; // r13d
  __int64 *v52; // r14
  __int64 v53; // rdx
  int v54; // eax
  unsigned int v55; // r13d
  __int64 *v56; // r14
  __int64 v57; // rdx
  int v58; // eax
  unsigned int v59; // r13d
  __int64 *v60; // r14
  __int64 v61; // rdx
  int v62; // eax
  unsigned int v63; // r13d
  __int64 *v64; // r14
  __int64 v65; // rdx
  int v66; // eax
  unsigned int v67; // r13d
  __int64 *v68; // r14
  __int64 v69; // rdx
  int v70; // eax
  unsigned int v71; // r13d
  __int64 *v72; // r14
  __int64 v73; // rdx
  int v74; // eax
  unsigned int v75; // r13d
  __int64 *v76; // r14
  __int64 v77; // rax
  int v78; // eax
  unsigned int v79; // r13d
  __int64 *v80; // r14
  __int64 v81; // rax
  int v82; // eax
  unsigned int v83; // r13d
  __int64 *v84; // r14
  __int64 v85; // rax
  int v86; // eax
  unsigned int v87; // r13d
  __int64 *v88; // r14
  __int64 v89; // rax
  int v90; // eax
  unsigned int v91; // r13d
  __int64 *v92; // r14
  __int64 v93; // rax
  int v94; // eax
  unsigned int v95; // r13d
  __int64 *v96; // r14
  __int64 v97; // rax
  __int64 result; // rax
  __int64 v99; // rax
  __int64 *v100; // rdx
  __int64 v101; // rax
  __int64 *v102; // rdx
  __int64 v103; // rax
  __int64 *v104; // rdx
  __int64 v105; // rax
  __int64 *v106; // rdx
  __int64 v107; // rax
  __int64 *v108; // rdx
  __int64 v109; // rax
  __int64 *v110; // rdx
  __int64 v111; // rax
  __int64 *v112; // rax
  __int64 *v113; // rax
  __int64 v114; // rax
  __int64 *v115; // rax
  __int64 *v116; // rax
  __int64 v117; // rax
  __m128i si128; // xmm0
  __int64 *v119; // rax
  __int64 *v120; // rax
  __int64 v121; // rax
  __int64 *v122; // rax
  __int64 *v123; // rax
  __int64 v124; // rax
  __int64 *v125; // rax
  __int64 *v126; // rax
  __int64 v127; // rax
  __int64 *v128; // rax
  __int64 *v129; // rax
  __int64 v130; // rax
  __int64 *v131; // rax
  __int64 *v132; // rax
  __int64 v133; // rax
  __int64 *v134; // rax
  __int64 *v135; // rax
  __int64 v136; // rax
  __int64 *v137; // rax
  __int64 *v138; // rax
  __int64 v139; // rax
  __int64 *v140; // rax
  __int64 *v141; // rax
  __int64 v142; // rax
  __int64 *v143; // rax
  __int64 *v144; // rax
  __int64 v145; // rax
  __int64 *v146; // rax
  __int64 *v147; // rax
  __int64 v148; // rax
  __int64 *v149; // rax
  __int64 *v150; // rax
  __int64 v151; // rax
  __int64 *v152; // rax
  __int64 *v153; // rax
  __int64 v154; // rax
  __int64 *v155; // rax
  __int64 *v156; // rax
  __int64 v157; // rax
  __int64 *v158; // rax
  __int64 *v159; // rax
  __int64 v160; // rax
  __int64 *v161; // rax
  __int64 *v162; // rax
  __int64 v163; // rax
  __int64 *v164; // rax
  __int64 *v165; // rax

  v1 = (__int64 *)(a1 + 872);
  *(_DWORD *)(*sub_EA2DC0(a1 + 872, ".set", 4u) + 8) = 1;
  *(_DWORD *)(*sub_EA2DC0(a1 + 872, ".equ", 4u) + 8) = 2;
  *(_DWORD *)(*sub_EA2DC0(a1 + 872, ".equiv", 6u) + 8) = 3;
  v2 = sub_C92610();
  v3 = sub_C92740(a1 + 872, ".ascii", 6u, v2);
  v4 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * v3);
  v5 = *v4;
  if ( *v4 )
  {
    if ( v5 != -8 )
      goto LABEL_3;
    --*(_DWORD *)(a1 + 888);
  }
  v163 = sub_C7D670(23, 8);
  strcpy((char *)(v163 + 16), ".ascii");
  *(_QWORD *)v163 = 6;
  *(_DWORD *)(v163 + 8) = 0;
  *v4 = v163;
  ++*(_DWORD *)(a1 + 884);
  v164 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * (unsigned int)sub_C929D0(v1, v3));
  v5 = *v164;
  if ( !*v164 || v5 == -8 )
  {
    v165 = v164 + 1;
    do
    {
      do
        v5 = *v165++;
      while ( !v5 );
    }
    while ( v5 == -8 );
  }
LABEL_3:
  *(_DWORD *)(v5 + 8) = 4;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".asciz", 6u) + 8) = 5;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".string", 7u) + 8) = 6;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".byte", 5u) + 8) = 7;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".short", 6u) + 8) = 8;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".value", 6u) + 8) = 10;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".2byte", 6u) + 8) = 11;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".long", 5u) + 8) = 12;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".int", 4u) + 8) = 13;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".4byte", 6u) + 8) = 14;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".quad", 5u) + 8) = 15;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".8byte", 6u) + 8) = 16;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".octa", 5u) + 8) = 17;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".single", 7u) + 8) = 41;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".float", 6u) + 8) = 42;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".double", 7u) + 8) = 43;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".align", 6u) + 8) = 44;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".align32", 8u) + 8) = 45;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".balign", 7u) + 8) = 46;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".balignw", 8u) + 8) = 47;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".balignl", 8u) + 8) = 48;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".p2align", 8u) + 8) = 49;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".p2alignw", 9u) + 8) = 50;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".p2alignl", 9u) + 8) = 51;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".org", 4u) + 8) = 52;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".fill", 5u) + 8) = 53;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".zero", 5u) + 8) = 58;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".extern", 7u) + 8) = 59;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".globl", 6u) + 8) = 60;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".global", 7u) + 8) = 61;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".lazy_reference", 0xFu) + 8) = 62;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".no_dead_strip", 0xEu) + 8) = 63;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".symbol_resolver", 0x10u) + 8) = 64;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".private_extern", 0xFu) + 8) = 65;
  v6 = sub_C92610();
  v7 = sub_C92740((__int64)v1, ".reference", 0xAu, v6);
  v8 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * v7);
  v9 = *v8;
  if ( *v8 )
  {
    if ( v9 != -8 )
      goto LABEL_5;
    --*(_DWORD *)(a1 + 888);
  }
  v160 = sub_C7D670(27, 8);
  strcpy((char *)(v160 + 16), ".reference");
  *(_QWORD *)v160 = 10;
  *(_DWORD *)(v160 + 8) = 0;
  *v8 = v160;
  ++*(_DWORD *)(a1 + 884);
  v161 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * (unsigned int)sub_C929D0(v1, v7));
  v9 = *v161;
  if ( !*v161 || v9 == -8 )
  {
    v162 = v161 + 1;
    do
    {
      do
        v9 = *v162++;
      while ( !v9 );
    }
    while ( v9 == -8 );
  }
LABEL_5:
  *(_DWORD *)(v9 + 8) = 66;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".weak_definition", 0x10u) + 8) = 67;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".weak_reference", 0xFu) + 8) = 68;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".weak_def_can_be_hidden", 0x17u) + 8) = 69;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".cold", 5u) + 8) = 70;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".comm", 5u) + 8) = 71;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".common", 7u) + 8) = 72;
  v10 = sub_C92610();
  v11 = sub_C92740((__int64)v1, ".lcomm", 6u, v10);
  v12 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * v11);
  v13 = *v12;
  if ( *v12 )
  {
    if ( v13 != -8 )
      goto LABEL_7;
    --*(_DWORD *)(a1 + 888);
  }
  v157 = sub_C7D670(23, 8);
  strcpy((char *)(v157 + 16), ".lcomm");
  *(_QWORD *)v157 = 6;
  *(_DWORD *)(v157 + 8) = 0;
  *v12 = v157;
  ++*(_DWORD *)(a1 + 884);
  v158 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * (unsigned int)sub_C929D0(v1, v11));
  v13 = *v158;
  if ( !*v158 || v13 == -8 )
  {
    v159 = v158 + 1;
    do
    {
      do
        v13 = *v159++;
      while ( !v13 );
    }
    while ( v13 == -8 );
  }
LABEL_7:
  *(_DWORD *)(v13 + 8) = 73;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".abort", 6u) + 8) = 74;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".include", 8u) + 8) = 75;
  v14 = sub_C92610();
  v15 = sub_C92740((__int64)v1, ".incbin", 7u, v14);
  v16 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * v15);
  v17 = *v16;
  if ( *v16 )
  {
    if ( v17 != -8 )
      goto LABEL_9;
    --*(_DWORD *)(a1 + 888);
  }
  v154 = sub_C7D670(24, 8);
  *(_WORD *)(v154 + 20) = 26978;
  *(_DWORD *)(v154 + 16) = 1668180270;
  *(_BYTE *)(v154 + 22) = 110;
  *(_BYTE *)(v154 + 23) = 0;
  *(_QWORD *)v154 = 7;
  *(_DWORD *)(v154 + 8) = 0;
  *v16 = v154;
  ++*(_DWORD *)(a1 + 884);
  v155 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * (unsigned int)sub_C929D0(v1, v15));
  v17 = *v155;
  if ( !*v155 || v17 == -8 )
  {
    v156 = v155 + 1;
    do
    {
      do
        v17 = *v156++;
      while ( !v17 );
    }
    while ( v17 == -8 );
  }
LABEL_9:
  *(_DWORD *)(v17 + 8) = 76;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".code16", 7u) + 8) = 77;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".code16gcc", 0xAu) + 8) = 78;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".rept", 5u) + 8) = 79;
  v18 = sub_C92610();
  v19 = sub_C92740((__int64)v1, ".rep", 4u, v18);
  v20 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * v19);
  v21 = *v20;
  if ( *v20 )
  {
    if ( v21 != -8 )
      goto LABEL_11;
    --*(_DWORD *)(a1 + 888);
  }
  v151 = sub_C7D670(21, 8);
  strcpy((char *)(v151 + 16), ".rep");
  *(_QWORD *)v151 = 4;
  *(_DWORD *)(v151 + 8) = 0;
  *v20 = v151;
  ++*(_DWORD *)(a1 + 884);
  v152 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * (unsigned int)sub_C929D0(v1, v19));
  v21 = *v152;
  if ( !*v152 || v21 == -8 )
  {
    v153 = v152 + 1;
    do
    {
      do
        v21 = *v153++;
      while ( !v21 );
    }
    while ( v21 == -8 );
  }
LABEL_11:
  *(_DWORD *)(v21 + 8) = 79;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".irp", 4u) + 8) = 80;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".irpc", 5u) + 8) = 81;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".endr", 5u) + 8) = 54;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".bundle_align_mode", 0x12u) + 8) = 55;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".bundle_lock", 0xCu) + 8) = 56;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".bundle_unlock", 0xEu) + 8) = 57;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".if", 3u) + 8) = 82;
  v22 = sub_C92610();
  v23 = sub_C92740((__int64)v1, ".ifeq", 5u, v22);
  v24 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * v23);
  v25 = *v24;
  if ( *v24 )
  {
    if ( v25 != -8 )
      goto LABEL_13;
    --*(_DWORD *)(a1 + 888);
  }
  v148 = sub_C7D670(22, 8);
  *(_DWORD *)(v148 + 16) = 1701210414;
  *(_BYTE *)(v148 + 20) = 113;
  *(_BYTE *)(v148 + 21) = 0;
  *(_QWORD *)v148 = 5;
  *(_DWORD *)(v148 + 8) = 0;
  *v24 = v148;
  ++*(_DWORD *)(a1 + 884);
  v149 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * (unsigned int)sub_C929D0(v1, v23));
  v25 = *v149;
  if ( !*v149 || v25 == -8 )
  {
    v150 = v149 + 1;
    do
    {
      do
        v25 = *v150++;
      while ( !v25 );
    }
    while ( v25 == -8 );
  }
LABEL_13:
  *(_DWORD *)(v25 + 8) = 83;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".ifge", 5u) + 8) = 84;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".ifgt", 5u) + 8) = 85;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".ifle", 5u) + 8) = 86;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".iflt", 5u) + 8) = 87;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".ifne", 5u) + 8) = 88;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".ifb", 4u) + 8) = 89;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".ifnb", 5u) + 8) = 90;
  v26 = sub_C92610();
  v27 = sub_C92740((__int64)v1, ".ifc", 4u, v26);
  v28 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * v27);
  v29 = *v28;
  if ( *v28 )
  {
    if ( v29 != -8 )
      goto LABEL_15;
    --*(_DWORD *)(a1 + 888);
  }
  v145 = sub_C7D670(21, 8);
  strcpy((char *)(v145 + 16), ".ifc");
  *(_QWORD *)v145 = 4;
  *(_DWORD *)(v145 + 8) = 0;
  *v28 = v145;
  ++*(_DWORD *)(a1 + 884);
  v146 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * (unsigned int)sub_C929D0(v1, v27));
  v29 = *v146;
  if ( !*v146 || v29 == -8 )
  {
    v147 = v146 + 1;
    do
    {
      do
        v29 = *v147++;
      while ( !v29 );
    }
    while ( v29 == -8 );
  }
LABEL_15:
  *(_DWORD *)(v29 + 8) = 91;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".ifeqs", 6u) + 8) = 92;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".ifnc", 5u) + 8) = 93;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".ifnes", 6u) + 8) = 94;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".ifdef", 6u) + 8) = 95;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".ifndef", 7u) + 8) = 96;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".ifnotdef", 9u) + 8) = 97;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".elseif", 7u) + 8) = 98;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".else", 5u) + 8) = 99;
  v30 = sub_C92610();
  v31 = sub_C92740((__int64)v1, ".end", 4u, v30);
  v32 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * v31);
  v33 = *v32;
  if ( *v32 )
  {
    if ( v33 != -8 )
      goto LABEL_17;
    --*(_DWORD *)(a1 + 888);
  }
  v142 = sub_C7D670(21, 8);
  strcpy((char *)(v142 + 16), ".end");
  *(_QWORD *)v142 = 4;
  *(_DWORD *)(v142 + 8) = 0;
  *v32 = v142;
  ++*(_DWORD *)(a1 + 884);
  v143 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * (unsigned int)sub_C929D0(v1, v31));
  v33 = *v143;
  if ( !*v143 || v33 == -8 )
  {
    v144 = v143 + 1;
    do
    {
      do
        v33 = *v144++;
      while ( !v33 );
    }
    while ( v33 == -8 );
  }
LABEL_17:
  *(_DWORD *)(v33 + 8) = 167;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".endif", 6u) + 8) = 100;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".skip", 5u) + 8) = 102;
  v34 = sub_C92610();
  v35 = sub_C92740((__int64)v1, ".space", 6u, v34);
  v36 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * v35);
  v37 = *v36;
  if ( *v36 )
  {
    if ( v37 != -8 )
      goto LABEL_19;
    --*(_DWORD *)(a1 + 888);
  }
  v139 = sub_C7D670(23, 8);
  strcpy((char *)(v139 + 16), ".space");
  *(_QWORD *)v139 = 6;
  *(_DWORD *)(v139 + 8) = 0;
  *v36 = v139;
  ++*(_DWORD *)(a1 + 884);
  v140 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * (unsigned int)sub_C929D0(v1, v35));
  v37 = *v140;
  if ( !*v140 || v37 == -8 )
  {
    v141 = v140 + 1;
    do
    {
      do
        v37 = *v141++;
      while ( v37 == -8 );
    }
    while ( !v37 );
  }
LABEL_19:
  *(_DWORD *)(v37 + 8) = 101;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".file", 5u) + 8) = 103;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".line", 5u) + 8) = 104;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".loc", 4u) + 8) = 105;
  v38 = sub_C92610();
  v39 = sub_C92740((__int64)v1, ".loc_label", 0xAu, v38);
  v40 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * v39);
  v41 = *v40;
  if ( *v40 )
  {
    if ( v41 != -8 )
      goto LABEL_21;
    --*(_DWORD *)(a1 + 888);
  }
  v136 = sub_C7D670(27, 8);
  strcpy((char *)(v136 + 16), ".loc_label");
  *(_QWORD *)v136 = 10;
  *(_DWORD *)(v136 + 8) = 0;
  *v40 = v136;
  ++*(_DWORD *)(a1 + 884);
  v137 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * (unsigned int)sub_C929D0(v1, v39));
  v41 = *v137;
  if ( !*v137 || v41 == -8 )
  {
    v138 = v137 + 1;
    do
    {
      do
        v41 = *v138++;
      while ( v41 == -8 );
    }
    while ( !v41 );
  }
LABEL_21:
  *(_DWORD *)(v41 + 8) = 106;
  v42 = sub_C92610();
  v43 = sub_C92740((__int64)v1, ".stabs", 6u, v42);
  v44 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * v43);
  v45 = *v44;
  if ( *v44 )
  {
    if ( v45 != -8 )
      goto LABEL_23;
    --*(_DWORD *)(a1 + 888);
  }
  v133 = sub_C7D670(23, 8);
  strcpy((char *)(v133 + 16), ".stabs");
  *(_QWORD *)v133 = 6;
  *(_DWORD *)(v133 + 8) = 0;
  *v44 = v133;
  ++*(_DWORD *)(a1 + 884);
  v134 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * (unsigned int)sub_C929D0(v1, v43));
  v45 = *v134;
  if ( !*v134 || v45 == -8 )
  {
    v135 = v134 + 1;
    do
    {
      do
        v45 = *v135++;
      while ( v45 == -8 );
    }
    while ( !v45 );
  }
LABEL_23:
  *(_DWORD *)(v45 + 8) = 107;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".cv_file", 8u) + 8) = 108;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".cv_func_id", 0xBu) + 8) = 109;
  v46 = sub_C92610();
  v47 = sub_C92740((__int64)v1, ".cv_loc", 7u, v46);
  v48 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * v47);
  v49 = *v48;
  if ( *v48 )
  {
    if ( v49 != -8 )
      goto LABEL_25;
    --*(_DWORD *)(a1 + 888);
  }
  v130 = sub_C7D670(24, 8);
  *(_DWORD *)(v130 + 16) = 1601594158;
  *(_WORD *)(v130 + 20) = 28524;
  *(_BYTE *)(v130 + 22) = 99;
  *(_BYTE *)(v130 + 23) = 0;
  *(_QWORD *)v130 = 7;
  *(_DWORD *)(v130 + 8) = 0;
  *v48 = v130;
  ++*(_DWORD *)(a1 + 884);
  v131 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * (unsigned int)sub_C929D0(v1, v47));
  v49 = *v131;
  if ( !*v131 || v49 == -8 )
  {
    v132 = v131 + 1;
    do
    {
      do
        v49 = *v132++;
      while ( v49 == -8 );
    }
    while ( !v49 );
  }
LABEL_25:
  *(_DWORD *)(v49 + 8) = 111;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".cv_linetable", 0xDu) + 8) = 112;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".cv_inline_linetable", 0x14u) + 8) = 113;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".cv_inline_site_id", 0x12u) + 8) = 110;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".cv_def_range", 0xDu) + 8) = 114;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".cv_string", 0xAu) + 8) = 116;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".cv_stringtable", 0xFu) + 8) = 115;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".cv_filechecksums", 0x11u) + 8) = 117;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".cv_filechecksumoffset", 0x16u) + 8) = 118;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".cv_fpo_data", 0xCu) + 8) = 119;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".sleb128", 8u) + 8) = 154;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".uleb128", 8u) + 8) = 155;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".cfi_sections", 0xDu) + 8) = 120;
  v50 = sub_C92610();
  v51 = sub_C92740((__int64)v1, ".cfi_startproc", 0xEu, v50);
  v52 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * v51);
  v53 = *v52;
  if ( *v52 )
  {
    if ( v53 != -8 )
      goto LABEL_27;
    --*(_DWORD *)(a1 + 888);
  }
  v127 = sub_C7D670(31, 8);
  strcpy((char *)(v127 + 16), ".cfi_startproc");
  *(_QWORD *)v127 = 14;
  *(_DWORD *)(v127 + 8) = 0;
  *v52 = v127;
  ++*(_DWORD *)(a1 + 884);
  v128 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * (unsigned int)sub_C929D0(v1, v51));
  v53 = *v128;
  if ( !*v128 || v53 == -8 )
  {
    v129 = v128 + 1;
    do
    {
      do
        v53 = *v129++;
      while ( v53 == -8 );
    }
    while ( !v53 );
  }
LABEL_27:
  *(_DWORD *)(v53 + 8) = 121;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".cfi_endproc", 0xCu) + 8) = 122;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".cfi_def_cfa", 0xCu) + 8) = 123;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".cfi_def_cfa_offset", 0x13u) + 8) = 124;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".cfi_adjust_cfa_offset", 0x16u) + 8) = 125;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".cfi_def_cfa_register", 0x15u) + 8) = 126;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".cfi_llvm_def_aspace_cfa", 0x18u) + 8) = 127;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".cfi_offset", 0xBu) + 8) = 128;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".cfi_rel_offset", 0xFu) + 8) = 129;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".cfi_personality", 0x10u) + 8) = 130;
  v54 = sub_C92610();
  v55 = sub_C92740((__int64)v1, ".cfi_lsda", 9u, v54);
  v56 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * v55);
  v57 = *v56;
  if ( *v56 )
  {
    if ( v57 != -8 )
      goto LABEL_29;
    --*(_DWORD *)(a1 + 888);
  }
  v124 = sub_C7D670(26, 8);
  strcpy((char *)(v124 + 16), ".cfi_lsda");
  *(_QWORD *)v124 = 9;
  *(_DWORD *)(v124 + 8) = 0;
  *v56 = v124;
  ++*(_DWORD *)(a1 + 884);
  v125 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * (unsigned int)sub_C929D0(v1, v55));
  v57 = *v125;
  if ( !*v125 || v57 == -8 )
  {
    v126 = v125 + 1;
    do
    {
      do
        v57 = *v126++;
      while ( v57 == -8 );
    }
    while ( !v57 );
  }
LABEL_29:
  *(_DWORD *)(v57 + 8) = 131;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".cfi_remember_state", 0x13u) + 8) = 132;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".cfi_restore_state", 0x12u) + 8) = 133;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".cfi_same_value", 0xFu) + 8) = 134;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".cfi_restore", 0xCu) + 8) = 135;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".cfi_escape", 0xBu) + 8) = 136;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".cfi_return_column", 0x12u) + 8) = 137;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".cfi_signal_frame", 0x11u) + 8) = 138;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".cfi_undefined", 0xEu) + 8) = 139;
  v58 = sub_C92610();
  v59 = sub_C92740((__int64)v1, ".cfi_register", 0xDu, v58);
  v60 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * v59);
  v61 = *v60;
  if ( *v60 )
  {
    if ( v61 != -8 )
      goto LABEL_31;
    --*(_DWORD *)(a1 + 888);
  }
  v121 = sub_C7D670(30, 8);
  strcpy((char *)(v121 + 16), ".cfi_register");
  *(_QWORD *)v121 = 13;
  *(_DWORD *)(v121 + 8) = 0;
  *v60 = v121;
  ++*(_DWORD *)(a1 + 884);
  v122 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * (unsigned int)sub_C929D0(v1, v59));
  v61 = *v122;
  if ( !*v122 || v61 == -8 )
  {
    v123 = v122 + 1;
    do
    {
      do
        v61 = *v123++;
      while ( v61 == -8 );
    }
    while ( !v61 );
  }
LABEL_31:
  *(_DWORD *)(v61 + 8) = 140;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".cfi_window_save", 0x10u) + 8) = 141;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".cfi_label", 0xAu) + 8) = 142;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".cfi_b_key_frame", 0x10u) + 8) = 143;
  v62 = sub_C92610();
  v63 = sub_C92740((__int64)v1, ".cfi_mte_tagged_frame", 0x15u, v62);
  v64 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * v63);
  v65 = *v64;
  if ( *v64 )
  {
    if ( v65 != -8 )
      goto LABEL_33;
    --*(_DWORD *)(a1 + 888);
  }
  v117 = sub_C7D670(38, 8);
  si128 = _mm_load_si128((const __m128i *)&xmmword_3F85320);
  *(_DWORD *)(v117 + 32) = 1835102822;
  *(_BYTE *)(v117 + 36) = 101;
  *(_BYTE *)(v117 + 37) = 0;
  *(_QWORD *)v117 = 21;
  *(_DWORD *)(v117 + 8) = 0;
  *(__m128i *)(v117 + 16) = si128;
  *v64 = v117;
  ++*(_DWORD *)(a1 + 884);
  v119 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * (unsigned int)sub_C929D0(v1, v63));
  v65 = *v119;
  if ( !*v119 || v65 == -8 )
  {
    v120 = v119 + 1;
    do
    {
      do
        v65 = *v120++;
      while ( v65 == -8 );
    }
    while ( !v65 );
  }
LABEL_33:
  *(_DWORD *)(v65 + 8) = 165;
  v66 = sub_C92610();
  v67 = sub_C92740((__int64)v1, ".cfi_val_offset", 0xFu, v66);
  v68 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * v67);
  v69 = *v68;
  if ( *v68 )
  {
    if ( v69 != -8 )
      goto LABEL_35;
    --*(_DWORD *)(a1 + 888);
  }
  v114 = sub_C7D670(32, 8);
  strcpy((char *)(v114 + 16), ".cfi_val_offset");
  *(_QWORD *)v114 = 15;
  *(_DWORD *)(v114 + 8) = 0;
  *v68 = v114;
  ++*(_DWORD *)(a1 + 884);
  v115 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * (unsigned int)sub_C929D0(v1, v67));
  v69 = *v115;
  if ( !*v115 || v69 == -8 )
  {
    v116 = v115 + 1;
    do
    {
      do
        v69 = *v116++;
      while ( v69 == -8 );
    }
    while ( !v69 );
  }
LABEL_35:
  *(_DWORD *)(v69 + 8) = 144;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".macros_on", 0xAu) + 8) = 145;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".macros_off", 0xBu) + 8) = 146;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".macro", 6u) + 8) = 149;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".exitm", 6u) + 8) = 150;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".endm", 5u) + 8) = 151;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".endmacro", 9u) + 8) = 152;
  v70 = sub_C92610();
  v71 = sub_C92740((__int64)v1, ".purgem", 7u, v70);
  v72 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * v71);
  v73 = *v72;
  if ( *v72 )
  {
    if ( v73 != -8 )
      goto LABEL_37;
    --*(_DWORD *)(a1 + 888);
  }
  v111 = sub_C7D670(24, 8);
  *(_WORD *)(v111 + 20) = 25959;
  *(_DWORD *)(v111 + 16) = 1920299054;
  *(_BYTE *)(v111 + 22) = 109;
  *(_BYTE *)(v111 + 23) = 0;
  *(_QWORD *)v111 = 7;
  *(_DWORD *)(v111 + 8) = 0;
  *v72 = v111;
  ++*(_DWORD *)(a1 + 884);
  v112 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * (unsigned int)sub_C929D0(v1, v71));
  v73 = *v112;
  if ( !*v112 || v73 == -8 )
  {
    v113 = v112 + 1;
    do
    {
      do
        v73 = *v113++;
      while ( v73 == -8 );
    }
    while ( !v73 );
  }
LABEL_37:
  *(_DWORD *)(v73 + 8) = 153;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".err", 4u) + 8) = 156;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".error", 6u) + 8) = 157;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".warning", 8u) + 8) = 158;
  v74 = sub_C92610();
  v75 = sub_C92740((__int64)v1, ".altmacro", 9u, v74);
  v76 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * v75);
  v77 = *v76;
  if ( *v76 )
  {
    if ( v77 != -8 )
      goto LABEL_39;
    --*(_DWORD *)(a1 + 888);
  }
  v109 = sub_C7D670(26, 8);
  strcpy((char *)(v109 + 16), ".altmacro");
  *(_QWORD *)v109 = 9;
  *(_DWORD *)(v109 + 8) = 0;
  *v76 = v109;
  ++*(_DWORD *)(a1 + 884);
  v110 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * (unsigned int)sub_C929D0(v1, v75));
  v77 = *v110;
  if ( !*v110 || v77 == -8 )
  {
    do
    {
      do
      {
        v77 = v110[1];
        ++v110;
      }
      while ( v77 == -8 );
    }
    while ( !v77 );
  }
LABEL_39:
  *(_DWORD *)(v77 + 8) = 147;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".noaltmacro", 0xBu) + 8) = 148;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".reloc", 6u) + 8) = 9;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".dc", 3u) + 8) = 18;
  v78 = sub_C92610();
  v79 = sub_C92740((__int64)v1, ".dc.a", 5u, v78);
  v80 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * v79);
  v81 = *v80;
  if ( *v80 )
  {
    if ( v81 != -8 )
      goto LABEL_41;
    --*(_DWORD *)(a1 + 888);
  }
  v107 = sub_C7D670(22, 8);
  *(_DWORD *)(v107 + 16) = 778265646;
  *(_BYTE *)(v107 + 20) = 97;
  *(_BYTE *)(v107 + 21) = 0;
  *(_QWORD *)v107 = 5;
  *(_DWORD *)(v107 + 8) = 0;
  *v80 = v107;
  ++*(_DWORD *)(a1 + 884);
  v108 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * (unsigned int)sub_C929D0(v1, v79));
  v81 = *v108;
  if ( !*v108 || v81 == -8 )
  {
    do
    {
      do
      {
        v81 = v108[1];
        ++v108;
      }
      while ( v81 == -8 );
    }
    while ( !v81 );
  }
LABEL_41:
  *(_DWORD *)(v81 + 8) = 19;
  v82 = sub_C92610();
  v83 = sub_C92740((__int64)v1, ".dc.b", 5u, v82);
  v84 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * v83);
  v85 = *v84;
  if ( *v84 )
  {
    if ( v85 != -8 )
      goto LABEL_43;
    --*(_DWORD *)(a1 + 888);
  }
  v105 = sub_C7D670(22, 8);
  *(_DWORD *)(v105 + 16) = 778265646;
  *(_BYTE *)(v105 + 20) = 98;
  *(_BYTE *)(v105 + 21) = 0;
  *(_QWORD *)v105 = 5;
  *(_DWORD *)(v105 + 8) = 0;
  *v84 = v105;
  ++*(_DWORD *)(a1 + 884);
  v106 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * (unsigned int)sub_C929D0(v1, v83));
  v85 = *v106;
  if ( !*v106 || v85 == -8 )
  {
    do
    {
      do
      {
        v85 = v106[1];
        ++v106;
      }
      while ( v85 == -8 );
    }
    while ( !v85 );
  }
LABEL_43:
  *(_DWORD *)(v85 + 8) = 20;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".dc.d", 5u) + 8) = 21;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".dc.l", 5u) + 8) = 22;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".dc.s", 5u) + 8) = 23;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".dc.w", 5u) + 8) = 24;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".dc.x", 5u) + 8) = 25;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".dcb", 4u) + 8) = 26;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".dcb.b", 6u) + 8) = 27;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".dcb.d", 6u) + 8) = 28;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".dcb.l", 6u) + 8) = 29;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".dcb.s", 6u) + 8) = 30;
  v86 = sub_C92610();
  v87 = sub_C92740((__int64)v1, ".dcb.w", 6u, v86);
  v88 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * v87);
  v89 = *v88;
  if ( *v88 )
  {
    if ( v89 != -8 )
      goto LABEL_45;
    --*(_DWORD *)(a1 + 888);
  }
  v103 = sub_C7D670(23, 8);
  strcpy((char *)(v103 + 16), ".dcb.w");
  *(_QWORD *)v103 = 6;
  *(_DWORD *)(v103 + 8) = 0;
  *v88 = v103;
  ++*(_DWORD *)(a1 + 884);
  v104 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * (unsigned int)sub_C929D0(v1, v87));
  v89 = *v104;
  if ( *v104 )
    goto LABEL_63;
  do
  {
    do
    {
      v89 = v104[1];
      ++v104;
    }
    while ( !v89 );
LABEL_63:
    ;
  }
  while ( v89 == -8 );
LABEL_45:
  *(_DWORD *)(v89 + 8) = 31;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".dcb.x", 6u) + 8) = 32;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".ds", 3u) + 8) = 33;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".ds.b", 5u) + 8) = 34;
  v90 = sub_C92610();
  v91 = sub_C92740((__int64)v1, ".ds.d", 5u, v90);
  v92 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * v91);
  v93 = *v92;
  if ( *v92 )
  {
    if ( v93 != -8 )
      goto LABEL_47;
    --*(_DWORD *)(a1 + 888);
  }
  v101 = sub_C7D670(22, 8);
  *(_DWORD *)(v101 + 16) = 779314222;
  *(_BYTE *)(v101 + 20) = 100;
  *(_BYTE *)(v101 + 21) = 0;
  *(_QWORD *)v101 = 5;
  *(_DWORD *)(v101 + 8) = 0;
  *v92 = v101;
  ++*(_DWORD *)(a1 + 884);
  v102 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * (unsigned int)sub_C929D0(v1, v91));
  v93 = *v102;
  if ( *v102 )
    goto LABEL_58;
  do
  {
    do
    {
      v93 = v102[1];
      ++v102;
    }
    while ( !v93 );
LABEL_58:
    ;
  }
  while ( v93 == -8 );
LABEL_47:
  *(_DWORD *)(v93 + 8) = 35;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".ds.l", 5u) + 8) = 36;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".ds.p", 5u) + 8) = 37;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".ds.s", 5u) + 8) = 38;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".ds.w", 5u) + 8) = 39;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".ds.x", 5u) + 8) = 40;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".print", 6u) + 8) = 159;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".addrsig", 8u) + 8) = 160;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".addrsig_sym", 0xCu) + 8) = 161;
  v94 = sub_C92610();
  v95 = sub_C92740((__int64)v1, ".pseudoprobe", 0xCu, v94);
  v96 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * v95);
  v97 = *v96;
  if ( *v96 )
  {
    if ( v97 != -8 )
      goto LABEL_49;
    --*(_DWORD *)(a1 + 888);
  }
  v99 = sub_C7D670(29, 8);
  strcpy((char *)(v99 + 16), ".pseudoprobe");
  *(_QWORD *)v99 = 12;
  *(_DWORD *)(v99 + 8) = 0;
  *v96 = v99;
  ++*(_DWORD *)(a1 + 884);
  v100 = (__int64 *)(*(_QWORD *)(a1 + 872) + 8LL * (unsigned int)sub_C929D0(v1, v95));
  v97 = *v100;
  if ( *v100 )
    goto LABEL_53;
  do
  {
    do
    {
      v97 = v100[1];
      ++v100;
    }
    while ( !v97 );
LABEL_53:
    ;
  }
  while ( v97 == -8 );
LABEL_49:
  *(_DWORD *)(v97 + 8) = 162;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".lto_discard", 0xCu) + 8) = 163;
  *(_DWORD *)(*sub_EA2DC0((__int64)v1, ".lto_set_conditional", 0x14u) + 8) = 164;
  result = *sub_EA2DC0((__int64)v1, ".memtag", 7u);
  *(_DWORD *)(result + 8) = 166;
  return result;
}
