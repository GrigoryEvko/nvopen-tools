// Function: sub_196C0A0
// Address: 0x196c0a0
//
__int64 __fastcall sub_196C0A0(__int64 *a1, double a2, double a3, double a4)
{
  __int64 v4; // r15
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r14
  __int64 v8; // r12
  __int64 v9; // r15
  int v10; // r13d
  _QWORD *v11; // rax
  __int64 *v12; // rdx
  __int64 v14; // r13
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  __int64 *v17; // r12
  __int64 v18; // rax
  unsigned int v19; // r14d
  __int64 v20; // r12
  __int64 v21; // rbx
  __int64 *v22; // r8
  __int64 v23; // rax
  int v24; // eax
  __int64 v25; // rax
  __int64 v26; // r13
  _QWORD *v27; // rax
  __int64 v28; // r13
  _QWORD *v29; // rax
  __int64 v30; // r13
  unsigned int v31; // esi
  char v32; // di
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rcx
  __int64 v36; // rax
  __int64 v37; // r12
  __int64 v38; // r12
  __int64 v39; // rdi
  unsigned __int64 v40; // rax
  __int64 v41; // rdi
  int v42; // edx
  __int64 v43; // rax
  __int64 *v44; // rax
  __int64 *v45; // rax
  __int64 v46; // rax
  __int64 v47; // rcx
  __int64 v48; // rax
  __int64 v49; // rdx
  unsigned __int64 v50; // r14
  __int64 *v51; // rax
  unsigned __int8 *v52; // rsi
  __int64 v53; // rax
  unsigned __int8 *v54; // rsi
  unsigned __int8 *v55; // rsi
  bool v56; // zf
  __int64 v57; // rdi
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // r12
  unsigned __int8 *v64; // rsi
  __int64 v65; // rdx
  __int64 v66; // rax
  __int64 *v67; // rax
  __int64 v68; // r12
  __int64 v69; // rax
  char v70; // di
  unsigned int v71; // ecx
  __int64 v72; // rdx
  __int64 v73; // rax
  __int64 v74; // rsi
  __int64 v75; // rcx
  __int64 v76; // r12
  unsigned __int64 v77; // rax
  __int64 v78; // r9
  __int64 v79; // rax
  __int64 v80; // rax
  __int64 v81; // r9
  __int64 v82; // r11
  __int64 v83; // r12
  unsigned __int8 *v84; // rsi
  unsigned __int8 *v85; // rsi
  __int64 v86; // rax
  __int64 v87; // rax
  __int64 v88; // rdx
  __int64 v89; // rcx
  __int64 v90; // r8
  __int64 v91; // r9
  __int64 v92; // r11
  __int64 v93; // r14
  int v94; // eax
  __int64 v95; // rax
  int v96; // edx
  __int64 v97; // rdx
  __int64 **v98; // rax
  __int64 *v99; // rcx
  unsigned __int64 v100; // rsi
  __int64 v101; // rcx
  __int64 v102; // rax
  __int64 v103; // rcx
  __int64 v104; // rsi
  int v105; // eax
  __int64 v106; // rax
  int v107; // edx
  __int64 v108; // rdx
  __int64 *v109; // rax
  __int64 v110; // rcx
  unsigned __int64 v111; // rsi
  __int64 v112; // rcx
  __int64 v113; // rax
  __int64 v114; // rcx
  __int64 v115; // rax
  __int16 v116; // dx
  __int64 v117; // rdx
  unsigned __int64 v118; // rcx
  __int64 v119; // rdx
  __int64 v120; // rax
  __int64 v121; // rcx
  unsigned __int64 v122; // rsi
  __int64 v123; // rcx
  __int64 *v124; // rax
  __int64 v125; // rax
  __int64 v126; // rsi
  unsigned __int8 *v127; // rsi
  __int64 v128; // rdx
  __int64 *v129; // rsi
  int v130; // edi
  __int64 v131; // rsi
  __int64 v132; // rax
  __int64 v133; // rsi
  __int64 v134; // rdx
  unsigned __int8 *v135; // rsi
  __int64 v136; // rax
  __int64 v137; // [rsp+8h] [rbp-138h]
  __int64 v138; // [rsp+8h] [rbp-138h]
  __int64 v139; // [rsp+8h] [rbp-138h]
  __int64 v140; // [rsp+8h] [rbp-138h]
  __int64 v141; // [rsp+8h] [rbp-138h]
  unsigned __int64 v142; // [rsp+10h] [rbp-130h]
  __int64 v143; // [rsp+18h] [rbp-128h]
  char v144; // [rsp+20h] [rbp-120h]
  __int64 v145; // [rsp+20h] [rbp-120h]
  __int64 v146; // [rsp+28h] [rbp-118h]
  __int64 v147; // [rsp+30h] [rbp-110h]
  __int64 *v148; // [rsp+38h] [rbp-108h]
  __int64 *v149; // [rsp+38h] [rbp-108h]
  __int64 v150; // [rsp+38h] [rbp-108h]
  __int64 v151; // [rsp+40h] [rbp-100h]
  int v152; // [rsp+40h] [rbp-100h]
  __int64 v153; // [rsp+40h] [rbp-100h]
  __int64 v154; // [rsp+40h] [rbp-100h]
  __int64 v155; // [rsp+40h] [rbp-100h]
  __int64 v156; // [rsp+40h] [rbp-100h]
  int v157; // [rsp+4Ch] [rbp-F4h]
  char v158; // [rsp+4Ch] [rbp-F4h]
  __int64 v159; // [rsp+50h] [rbp-F0h]
  __int64 *v160; // [rsp+50h] [rbp-F0h]
  __int64 v161; // [rsp+50h] [rbp-F0h]
  unsigned __int8 *v163; // [rsp+68h] [rbp-D8h] BYREF
  __int64 *v164[2]; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v165[2]; // [rsp+80h] [rbp-C0h] BYREF
  __int16 v166; // [rsp+90h] [rbp-B0h]
  unsigned __int8 *v167[2]; // [rsp+A0h] [rbp-A0h] BYREF
  __int16 v168; // [rsp+B0h] [rbp-90h]
  unsigned __int8 *v169; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v170; // [rsp+C8h] [rbp-78h]
  __int64 *v171; // [rsp+D0h] [rbp-70h]
  __int64 *v172; // [rsp+D8h] [rbp-68h]
  __int64 v173; // [rsp+E0h] [rbp-60h]
  int v174; // [rsp+E8h] [rbp-58h]
  __int64 v175; // [rsp+F0h] [rbp-50h]
  __int64 v176; // [rsp+F8h] [rbp-48h]

  v4 = *a1;
  v5 = sub_1969100(**(_QWORD **)(*a1 + 32));
  if ( v5 == v6 )
    goto LABEL_9;
  v7 = v6;
  v8 = v5;
  v9 = v4 + 56;
  v10 = 0;
  do
  {
    v11 = sub_1648700(v8);
    v10 -= !sub_1377F70(v9, v11[5]) - 1;
    do
      v8 = *(_QWORD *)(v8 + 8);
    while ( v8 && (unsigned __int8)(*((_BYTE *)sub_1648700(v8) + 16) - 25) > 9u );
  }
  while ( v7 != v8 );
  if ( v10 != 1 )
    goto LABEL_9;
  v12 = *(__int64 **)(*a1 + 32);
  if ( (unsigned int)((__int64)(*(_QWORD *)(*a1 + 40) - (_QWORD)v12) >> 3) != 1 )
    goto LABEL_9;
  v14 = *v12;
  v15 = sub_157EBA0(*v12);
  if ( *(_BYTE *)(v15 + 16) != 26 )
    goto LABEL_9;
  if ( (*(_DWORD *)(v15 + 20) & 0xFFFFFFF) != 3 )
    goto LABEL_9;
  v16 = sub_1969460(v15, v14);
  v151 = v16;
  if ( !v16 || (unsigned __int8)(*(_BYTE *)(v16 + 16) - 48) > 1u )
    goto LABEL_9;
  v17 = (*(_BYTE *)(v16 + 23) & 0x40) != 0
      ? *(__int64 **)(v16 - 8)
      : (__int64 *)(v16 - 24LL * (*(_DWORD *)(v16 + 20) & 0xFFFFFFF));
  v18 = v17[3];
  if ( *(_BYTE *)(v18 + 16) != 13 )
    goto LABEL_9;
  v19 = *(_DWORD *)(v18 + 32);
  LOBYTE(v9) = v19 <= 0x40 ? *(_QWORD *)(v18 + 24) == 1 : v19 - 1 == (unsigned int)sub_16A57B0(v18 + 24);
  if ( !(_BYTE)v9 )
    goto LABEL_9;
  v20 = sub_19695C0(*v17, v151, v14);
  if ( !v20 )
    goto LABEL_9;
  v159 = sub_157ED20(v14) + 24;
  if ( v159 == v14 + 40 )
    goto LABEL_9;
  v21 = v159;
  while ( 1 )
  {
    if ( *(_BYTE *)(v21 - 8) != 35 )
      goto LABEL_155;
    v143 = v21 - 24;
    v22 = (*(_BYTE *)(v21 - 1) & 0x40) != 0
        ? *(__int64 **)(v21 - 32)
        : (__int64 *)(v143 - 24LL * (*(_DWORD *)(v21 - 4) & 0xFFFFFFF));
    v23 = v22[3];
    if ( *(_BYTE *)(v23 + 16) != 13 )
      goto LABEL_155;
    if ( *(_DWORD *)(v23 + 32) > 0x40u )
      break;
    if ( *(_QWORD *)(v23 + 24) == 1 )
      goto LABEL_30;
LABEL_155:
    v21 = *(_QWORD *)(v21 + 8);
    if ( v14 + 40 == v21 )
      goto LABEL_9;
    if ( !v21 )
      BUG();
  }
  v157 = *(_DWORD *)(v23 + 32);
  v160 = v22;
  v24 = sub_16A57B0(v23 + 24);
  v22 = v160;
  if ( v24 != v157 - 1 )
    goto LABEL_155;
LABEL_30:
  v25 = sub_19695C0(*v22, v143, v14);
  if ( !v25 )
    goto LABEL_155;
  v26 = *(_QWORD *)(v25 + 8);
  v147 = v25;
  if ( v26 )
  {
    while ( 1 )
    {
      v27 = sub_1648700(v26);
      if ( !sub_1377F70(*a1 + 56, v27[5]) )
        break;
      v26 = *(_QWORD *)(v26 + 8);
      if ( !v26 )
        goto LABEL_158;
    }
    v158 = v9;
  }
  else
  {
LABEL_158:
    v158 = 0;
  }
  v28 = *(_QWORD *)(v21 - 16);
  if ( v28 )
  {
    while ( 1 )
    {
      v29 = sub_1648700(v28);
      if ( !sub_1377F70(*a1 + 56, v29[5]) )
        break;
      v28 = *(_QWORD *)(v28 + 8);
      if ( !v28 )
        goto LABEL_41;
    }
    if ( v158 )
    {
LABEL_9:
      LODWORD(v9) = 0;
      return (unsigned int)v9;
    }
  }
LABEL_41:
  v30 = sub_13FC520(*a1);
  v31 = *(_DWORD *)(v20 + 20) & 0xFFFFFFF;
  if ( v31 )
  {
    v32 = *(_BYTE *)(v20 + 23) & 0x40;
    v33 = 24LL * *(unsigned int *)(v20 + 56) + 8;
    v34 = 0;
    do
    {
      v35 = v20 - 24LL * v31;
      if ( v32 )
        v35 = *(_QWORD *)(v20 - 8);
      if ( v30 == *(_QWORD *)(v35 + v33) )
      {
        v36 = 24 * v34;
        goto LABEL_48;
      }
      ++v34;
      v33 += 8;
    }
    while ( v31 != (_DWORD)v34 );
    v36 = 0x17FFFFFFE8LL;
  }
  else
  {
    v36 = 0x17FFFFFFE8LL;
    v32 = *(_BYTE *)(v20 + 23) & 0x40;
  }
LABEL_48:
  if ( v32 )
    v37 = *(_QWORD *)(v20 - 8);
  else
    v37 = v20 - 24LL * v31;
  v38 = *(_QWORD *)(v37 + v36);
  if ( *(_BYTE *)(v151 + 16) == 49 && !(unsigned __int8)sub_14C2730((__int64 *)v38, a1[7], 0, 0, 0, 0) )
    goto LABEL_9;
  if ( v158 )
  {
    v164[0] = (__int64 *)v38;
    v124 = (__int64 *)sub_16498A0(v38);
    v45 = (__int64 *)sub_159C540(v124);
    v144 = 0;
  }
  else
  {
    v39 = sub_157F0B0(v30);
    if ( !v39 )
      goto LABEL_9;
    v40 = sub_157EBA0(v39);
    v41 = v40;
    if ( *(_BYTE *)(v40 + 16) != 26 )
      goto LABEL_9;
    v42 = *(_DWORD *)(v40 + 20);
    v43 = 0;
    if ( (v42 & 0xFFFFFFF) == 3 )
      v43 = sub_1969460(v41, v30);
    if ( v38 != v43 )
      goto LABEL_9;
    v164[0] = (__int64 *)v38;
    v44 = (__int64 *)sub_16498A0(v38);
    v45 = (__int64 *)sub_159C4F0(v44);
    v144 = v9;
  }
  v164[1] = v45;
  v46 = **(_QWORD **)(*a1 + 32);
  v47 = v46 + 40;
  v48 = *(_QWORD *)(v46 + 48);
  if ( v47 == v48 )
    goto LABEL_62;
  v49 = 0;
  do
  {
    v48 = *(_QWORD *)(v48 + 8);
    ++v49;
  }
  while ( v47 != v48 );
  if ( v49 != 6 )
  {
LABEL_62:
    if ( (int)sub_14A2710((__int64 *)a1[6], 31, *(_QWORD *)v38, v164, 2u) > 1 )
      goto LABEL_9;
  }
  v50 = sub_157EBA0(v30);
  v51 = (__int64 *)sub_16498A0(v50);
  v175 = 0;
  v176 = 0;
  v52 = *(unsigned __int8 **)(v50 + 48);
  v172 = v51;
  v174 = 0;
  v53 = *(_QWORD *)(v50 + 40);
  v169 = 0;
  v170 = v53;
  v173 = 0;
  v171 = (__int64 *)(v50 + 24);
  v167[0] = v52;
  if ( v52 )
  {
    sub_1623A60((__int64)v167, (__int64)v52, 2);
    if ( v169 )
      sub_161E7C0((__int64)&v169, (__int64)v169);
    v169 = v167[0];
    if ( v167[0] )
      sub_1623210((__int64)v167, v167[0], (__int64)&v169);
  }
  v54 = *(unsigned __int8 **)(v151 + 48);
  v167[0] = v54;
  if ( v54 )
  {
    sub_1623A60((__int64)v167, (__int64)v54, 2);
    v55 = v169;
    if ( v169 )
      goto LABEL_70;
LABEL_71:
    v169 = v167[0];
    if ( v167[0] )
      sub_1623210((__int64)v167, v167[0], (__int64)&v169);
  }
  else
  {
    v55 = v169;
    if ( v169 )
    {
LABEL_70:
      sub_161E7C0((__int64)&v169, (__int64)v55);
      goto LABEL_71;
    }
  }
  if ( v158 )
  {
    v56 = *(_BYTE *)(v151 + 16) == 49;
    v166 = 257;
    v57 = *(_QWORD *)v38;
    if ( v56 )
    {
      v136 = sub_15A0680(v57, 1, 0);
      if ( *(_BYTE *)(v38 + 16) > 0x10u || *(_BYTE *)(v136 + 16) > 0x10u )
      {
        v128 = v136;
        v168 = 257;
        v129 = (__int64 *)v38;
        v130 = 25;
        goto LABEL_187;
      }
      v59 = sub_15A2DA0((__int64 *)v38, v136, 0, a2, a3, a4);
LABEL_78:
      v38 = v59;
    }
    else
    {
      v58 = sub_15A0680(v57, 1, 0);
      if ( *(_BYTE *)(v38 + 16) <= 0x10u && *(_BYTE *)(v58 + 16) <= 0x10u )
      {
        v59 = sub_15A2D80((__int64 *)v38, v58, 0, a2, a3, a4);
        goto LABEL_78;
      }
      v128 = v58;
      v168 = 257;
      v129 = (__int64 *)v38;
      v130 = 24;
LABEL_187:
      v38 = sub_15FB440(v130, v129, v128, (__int64)v167, 0);
      if ( v170 )
      {
        v149 = v171;
        sub_157E9D0(v170 + 40, v38);
        v131 = *v149;
        v132 = *(_QWORD *)(v38 + 24) & 7LL;
        *(_QWORD *)(v38 + 32) = v149;
        v131 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v38 + 24) = v131 | v132;
        *(_QWORD *)(v131 + 8) = v38 + 24;
        *v149 = *v149 & 7 | (v38 + 24);
      }
      sub_164B780(v38, v165);
      if ( v169 )
      {
        v163 = v169;
        sub_1623A60((__int64)&v163, (__int64)v169, 2);
        v133 = *(_QWORD *)(v38 + 48);
        v134 = v38 + 48;
        if ( v133 )
        {
          sub_161E7C0(v38 + 48, v133);
          v134 = v38 + 48;
        }
        v135 = v163;
        *(_QWORD *)(v38 + 48) = v163;
        if ( v135 )
          sub_1623210((__int64)&v163, v135, v134);
      }
    }
  }
  v165[0] = v38;
  if ( v144 )
    v60 = sub_159C4F0(v172);
  else
    v60 = sub_159C540(v172);
  v165[1] = v60;
  v163 = *(unsigned __int8 **)v38;
  v61 = sub_15E26F0(*(__int64 **)(*(_QWORD *)(v170 + 56) + 40LL), 31, (__int64 *)&v163, 1);
  v168 = 257;
  v62 = sub_1285290((__int64 *)&v169, *(_QWORD *)(*(_QWORD *)v61 + 24LL), v61, (int)v165, 2, (__int64)v167, 0);
  v63 = v62;
  v64 = *(unsigned __int8 **)(v151 + 48);
  v167[0] = v64;
  if ( !v64 )
  {
    v65 = v62 + 48;
    if ( (unsigned __int8 **)(v62 + 48) == v167 )
      goto LABEL_85;
    v126 = *(_QWORD *)(v62 + 48);
    if ( !v126 )
      goto LABEL_85;
LABEL_180:
    v156 = v65;
    sub_161E7C0(v65, v126);
    v65 = v156;
    goto LABEL_181;
  }
  sub_1623A60((__int64)v167, (__int64)v64, 2);
  v65 = v63 + 48;
  if ( (unsigned __int8 **)(v63 + 48) == v167 )
  {
    if ( v167[0] )
      sub_161E7C0((__int64)v167, (__int64)v167[0]);
    goto LABEL_85;
  }
  v126 = *(_QWORD *)(v63 + 48);
  if ( v126 )
    goto LABEL_180;
LABEL_181:
  v127 = v167[0];
  *(unsigned __int8 **)(v63 + 48) = v167[0];
  if ( v127 )
    sub_1623210((__int64)v167, v127, v65);
LABEL_85:
  v168 = 257;
  v66 = sub_15A0680(*(_QWORD *)v63, *(_DWORD *)(*(_QWORD *)v63 + 8LL) >> 8, 0);
  v67 = (__int64 *)sub_156E1C0((__int64 *)&v169, v66, v63, (__int64)v167, 0, 0);
  v168 = 257;
  v68 = (__int64)v67;
  if ( v158 )
  {
    v125 = sub_15A0680(*v67, 1, 0);
    v148 = (__int64 *)sub_12899C0((__int64 *)&v169, v68, v125, (__int64)v167, 0, 0);
    v168 = 257;
  }
  else
  {
    v148 = v67;
  }
  v161 = sub_1904B50((__int64 *)&v169, v68, *(_QWORD *)(v21 - 24), (__int64 *)v167);
  v69 = 0x17FFFFFFE8LL;
  v70 = *(_BYTE *)(v147 + 23) & 0x40;
  v71 = *(_DWORD *)(v147 + 20) & 0xFFFFFFF;
  if ( v71 )
  {
    v72 = 24LL * *(unsigned int *)(v147 + 56) + 8;
    v73 = 0;
    do
    {
      v74 = v147 - 24LL * v71;
      if ( v70 )
        v74 = *(_QWORD *)(v147 - 8);
      if ( v30 == *(_QWORD *)(v74 + v72) )
      {
        v69 = 24 * v73;
        goto LABEL_94;
      }
      ++v73;
      v72 += 8;
    }
    while ( v71 != (_DWORD)v73 );
    v69 = 0x17FFFFFFE8LL;
  }
LABEL_94:
  if ( v70 )
    v75 = *(_QWORD *)(v147 - 8);
  else
    v75 = v147 - 24LL * v71;
  v76 = *(_QWORD *)(v75 + v69);
  if ( *(_BYTE *)(v76 + 16) != 13 )
    goto LABEL_99;
  if ( *(_DWORD *)(v76 + 32) <= 0x40u )
  {
    if ( !*(_QWORD *)(v76 + 24) )
      goto LABEL_100;
LABEL_99:
    v168 = 257;
    v161 = sub_12899C0((__int64 *)&v169, v161, v76, (__int64)v167, 0, 0);
    goto LABEL_100;
  }
  v152 = *(_DWORD *)(v76 + 32);
  if ( v152 != (unsigned int)sub_16A57B0(v76 + 24) )
    goto LABEL_99;
LABEL_100:
  v146 = **(_QWORD **)(*a1 + 32);
  v77 = sub_157EBA0(v146);
  v78 = *(_QWORD *)(v146 + 48);
  v142 = v77;
  v137 = *(_QWORD *)(v77 - 72);
  v79 = *v148;
  v168 = 259;
  v145 = v79;
  if ( v78 )
    v78 -= 24;
  v167[0] = "tcphi";
  v153 = v78;
  v80 = sub_1648B60(64);
  v81 = v153;
  v82 = v137;
  v83 = v80;
  if ( v80 )
  {
    v154 = v80;
    sub_15F1EA0(v80, v145, 53, 0, 0, v81);
    *(_DWORD *)(v83 + 56) = 2;
    sub_164B780(v83, (__int64 *)v167);
    sub_1648880(v83, *(_DWORD *)(v83 + 56), 1);
    v82 = v137;
  }
  else
  {
    v154 = 0;
  }
  v84 = *(unsigned __int8 **)(v82 + 48);
  v170 = *(_QWORD *)(v82 + 40);
  v171 = (__int64 *)(v82 + 24);
  v167[0] = v84;
  if ( v84 )
  {
    v138 = v82;
    sub_1623A60((__int64)v167, (__int64)v84, 2);
    v85 = v169;
    v82 = v138;
    if ( v169 )
      goto LABEL_106;
LABEL_107:
    v169 = v167[0];
    if ( v167[0] )
    {
      v140 = v82;
      sub_1623210((__int64)v167, v167[0], (__int64)&v169);
      v82 = v140;
    }
  }
  else
  {
    v85 = v169;
    if ( v169 )
    {
LABEL_106:
      v139 = v82;
      sub_161E7C0((__int64)&v169, (__int64)v85);
      v82 = v139;
      goto LABEL_107;
    }
  }
  v141 = v82;
  v167[0] = "tcdec";
  v168 = 259;
  v86 = sub_15A0680(v145, 1, 0);
  v87 = sub_156E1C0((__int64 *)&v169, v83, v86, (__int64)v167, 0, 1u);
  v92 = v141;
  v93 = v87;
  v94 = *(_DWORD *)(v83 + 20) & 0xFFFFFFF;
  if ( v94 == *(_DWORD *)(v83 + 56) )
  {
    sub_15F55D0(v83, v83, v88, v89, v90, v91);
    v92 = v141;
    v94 = *(_DWORD *)(v83 + 20) & 0xFFFFFFF;
  }
  v95 = (v94 + 1) & 0xFFFFFFF;
  v96 = v95 | *(_DWORD *)(v83 + 20) & 0xF0000000;
  *(_DWORD *)(v83 + 20) = v96;
  if ( (v96 & 0x40000000) != 0 )
    v97 = *(_QWORD *)(v83 - 8);
  else
    v97 = v154 - 24 * v95;
  v98 = (__int64 **)(v97 + 24LL * (unsigned int)(v95 - 1));
  if ( *v98 )
  {
    v99 = v98[1];
    v100 = (unsigned __int64)v98[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v100 = v99;
    if ( v99 )
      v99[2] = v100 | v99[2] & 3;
  }
  *v98 = v148;
  v101 = v148[1];
  v98[1] = (__int64 *)v101;
  if ( v101 )
    *(_QWORD *)(v101 + 16) = (unsigned __int64)(v98 + 1) | *(_QWORD *)(v101 + 16) & 3LL;
  v98[2] = (__int64 *)((unsigned __int64)(v148 + 1) | (unsigned __int64)v98[2] & 3);
  v148[1] = (__int64)v98;
  v102 = *(_DWORD *)(v83 + 20) & 0xFFFFFFF;
  if ( (*(_BYTE *)(v83 + 23) & 0x40) != 0 )
    v103 = *(_QWORD *)(v83 - 8);
  else
    v103 = v154 - 24 * v102;
  v104 = *(unsigned int *)(v83 + 56);
  *(_QWORD *)(v103 + 8LL * (unsigned int)(v102 - 1) + 24 * v104 + 8) = v30;
  v105 = *(_DWORD *)(v83 + 20) & 0xFFFFFFF;
  if ( v105 == *(_DWORD *)(v83 + 56) )
  {
    v150 = v92;
    sub_15F55D0(v83, v104, 3 * v104, v103, v90, v91);
    v92 = v150;
    v105 = *(_DWORD *)(v83 + 20) & 0xFFFFFFF;
  }
  v106 = (v105 + 1) & 0xFFFFFFF;
  v107 = v106 | *(_DWORD *)(v83 + 20) & 0xF0000000;
  *(_DWORD *)(v83 + 20) = v107;
  if ( (v107 & 0x40000000) != 0 )
    v108 = *(_QWORD *)(v83 - 8);
  else
    v108 = v154 - 24 * v106;
  v109 = (__int64 *)(v108 + 24LL * (unsigned int)(v106 - 1));
  if ( *v109 )
  {
    v110 = v109[1];
    v111 = v109[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v111 = v110;
    if ( v110 )
      *(_QWORD *)(v110 + 16) = v111 | *(_QWORD *)(v110 + 16) & 3LL;
  }
  *v109 = v93;
  if ( v93 )
  {
    v112 = *(_QWORD *)(v93 + 8);
    v109[1] = v112;
    if ( v112 )
      *(_QWORD *)(v112 + 16) = (unsigned __int64)(v109 + 1) | *(_QWORD *)(v112 + 16) & 3LL;
    v109[2] = (v93 + 8) | v109[2] & 3;
    *(_QWORD *)(v93 + 8) = v109;
  }
  v113 = *(_DWORD *)(v83 + 20) & 0xFFFFFFF;
  if ( (*(_BYTE *)(v83 + 23) & 0x40) != 0 )
    v114 = *(_QWORD *)(v83 - 8);
  else
    v114 = v154 - 24 * v113;
  *(_QWORD *)(v114 + 8LL * (unsigned int)(v113 - 1) + 24LL * *(unsigned int *)(v83 + 56) + 8) = v146;
  v115 = *(_QWORD *)(v142 - 24);
  if ( v146 != v115 || (v116 = 33, !v115) )
    v116 = 32;
  v56 = *(_QWORD *)(v92 - 48) == 0;
  *(_WORD *)(v92 + 18) = v116 | *(_WORD *)(v92 + 18) & 0x8000;
  if ( !v56 )
  {
    v117 = *(_QWORD *)(v92 - 40);
    v118 = *(_QWORD *)(v92 - 32) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v118 = v117;
    if ( v117 )
      *(_QWORD *)(v117 + 16) = v118 | *(_QWORD *)(v117 + 16) & 3LL;
  }
  *(_QWORD *)(v92 - 48) = v93;
  if ( v93 )
  {
    v119 = *(_QWORD *)(v93 + 8);
    *(_QWORD *)(v92 - 40) = v119;
    if ( v119 )
      *(_QWORD *)(v119 + 16) = (v92 - 40) | *(_QWORD *)(v119 + 16) & 3LL;
    *(_QWORD *)(v92 - 32) = (v93 + 8) | *(_QWORD *)(v92 - 32) & 3LL;
    *(_QWORD *)(v93 + 8) = v92 - 48;
  }
  v155 = v92;
  v120 = sub_15A0680(v145, 0, 0);
  if ( *(_QWORD *)(v155 - 24) )
  {
    v121 = *(_QWORD *)(v155 - 16);
    v122 = *(_QWORD *)(v155 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v122 = v121;
    if ( v121 )
      *(_QWORD *)(v121 + 16) = v122 | *(_QWORD *)(v121 + 16) & 3LL;
  }
  *(_QWORD *)(v155 - 24) = v120;
  if ( v120 )
  {
    v123 = *(_QWORD *)(v120 + 8);
    *(_QWORD *)(v155 - 16) = v123;
    if ( v123 )
      *(_QWORD *)(v123 + 16) = (v155 - 16) | *(_QWORD *)(v123 + 16) & 3LL;
    *(_QWORD *)(v155 - 24 + 16) = (v120 + 8) | *(_QWORD *)(v155 - 8) & 3LL;
    *(_QWORD *)(v120 + 8) = v155 - 24;
  }
  if ( v158 )
    sub_1648F20(v147, v161, v146);
  else
    sub_1648F20(v143, v161, v146);
  sub_1465150(a1[4], *a1);
  if ( v169 )
    sub_161E7C0((__int64)&v169, (__int64)v169);
  return (unsigned int)v9;
}
