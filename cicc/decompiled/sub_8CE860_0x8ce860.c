// Function: sub_8CE860
// Address: 0x8ce860
//
_BOOL8 __fastcall sub_8CE860(__int64 a1)
{
  __int64 v2; // r15
  _BOOL4 v3; // r13d
  __int64 *v4; // rax
  __int64 v5; // r9
  __int64 v6; // r10
  __int64 *v7; // rax
  __int64 v8; // rsi
  __int64 v10; // rax
  int v11; // eax
  __int64 v12; // r9
  __int64 i; // r10
  unsigned __int8 v14; // al
  char v15; // al
  unsigned __int8 v16; // al
  __int64 v17; // rdi
  __int64 v18; // rsi
  int v19; // eax
  int v20; // eax
  int v21; // eax
  __int64 v22; // rbx
  __int64 v23; // r14
  int v24; // eax
  char v25; // al
  int v26; // eax
  bool v27; // zf
  _QWORD *v28; // rbx
  _QWORD *v29; // r14
  unsigned __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // rax
  __int64 v33; // r9
  __int64 v34; // r10
  __int64 *v35; // rcx
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 *v39; // r8
  __int64 v40; // r9
  __int64 v41; // r10
  __int64 v42; // rdx
  __int64 j; // rcx
  __int64 v44; // rax
  __int64 v45; // r11
  __int64 v46; // r11
  __int64 v47; // rsi
  __int64 v48; // rdx
  _QWORD *v49; // rcx
  __int64 v50; // rcx
  unsigned __int64 m; // rdx
  __int64 v52; // rcx
  __int64 n; // r11
  unsigned __int8 v54; // al
  __int64 v55; // rdi
  __int64 v56; // rsi
  __int64 v57; // rcx
  __int64 v58; // r14
  __int64 ii; // rbx
  int v60; // eax
  __int64 v61; // rdx
  __int64 v62; // rdi
  __int64 v63; // rsi
  _QWORD *v64; // rsi
  _QWORD *v65; // rdi
  int v66; // edx
  __int64 v67; // r10
  unsigned __int64 v68; // rsi
  __int64 v69; // r9
  unsigned __int64 jj; // r11
  _QWORD **v71; // r11
  __int64 v72; // rdx
  __int64 *v73; // rax
  __int64 v74; // rax
  __int64 *v75; // rcx
  __int64 v76; // r10
  unsigned __int64 v77; // rbx
  unsigned __int64 kk; // r14
  __int64 v79; // rdi
  __int64 v80; // rsi
  __int64 v81; // rax
  __int64 *v82; // rdx
  __int64 v83; // rdx
  __int64 *v84; // rcx
  __int64 *v85; // rcx
  __int64 *v86; // rdx
  _QWORD *v87; // [rsp+0h] [rbp-70h]
  __int64 *v88; // [rsp+10h] [rbp-60h]
  __int64 v89; // [rsp+18h] [rbp-58h]
  unsigned __int64 v90; // [rsp+20h] [rbp-50h]
  __int64 v91; // [rsp+20h] [rbp-50h]
  __int64 v92; // [rsp+20h] [rbp-50h]
  __int64 v93; // [rsp+20h] [rbp-50h]
  __int64 v94; // [rsp+20h] [rbp-50h]
  __int64 v95; // [rsp+20h] [rbp-50h]
  __int64 v96; // [rsp+20h] [rbp-50h]
  __int64 v97; // [rsp+20h] [rbp-50h]
  __int64 v98; // [rsp+20h] [rbp-50h]
  __int64 v99; // [rsp+28h] [rbp-48h]
  __int64 v100; // [rsp+28h] [rbp-48h]
  __int64 v101; // [rsp+28h] [rbp-48h]
  __int64 v102; // [rsp+28h] [rbp-48h]
  __int64 v103; // [rsp+28h] [rbp-48h]
  __int64 v104; // [rsp+28h] [rbp-48h]
  __int64 v105; // [rsp+28h] [rbp-48h]
  __int64 v106; // [rsp+28h] [rbp-48h]
  __int64 v107; // [rsp+28h] [rbp-48h]
  __int64 v108; // [rsp+28h] [rbp-48h]
  __int64 v109; // [rsp+28h] [rbp-48h]
  __int64 v110; // [rsp+28h] [rbp-48h]
  __int64 v111; // [rsp+30h] [rbp-40h]
  __int64 v112; // [rsp+30h] [rbp-40h]
  __int64 v113; // [rsp+30h] [rbp-40h]
  __int64 v114; // [rsp+30h] [rbp-40h]
  __int64 v115; // [rsp+30h] [rbp-40h]
  __int64 v116; // [rsp+30h] [rbp-40h]
  __int64 v117; // [rsp+30h] [rbp-40h]
  __int64 v118; // [rsp+30h] [rbp-40h]
  __int64 *v119; // [rsp+30h] [rbp-40h]
  __int64 v120; // [rsp+30h] [rbp-40h]
  unsigned __int64 v121; // [rsp+30h] [rbp-40h]
  __int64 v122; // [rsp+30h] [rbp-40h]
  __int64 *v123; // [rsp+30h] [rbp-40h]
  __int64 v124; // [rsp+30h] [rbp-40h]
  __int64 v125; // [rsp+30h] [rbp-40h]
  __int64 v126; // [rsp+30h] [rbp-40h]
  __int64 v127; // [rsp+38h] [rbp-38h]
  __int64 v128; // [rsp+38h] [rbp-38h]
  __int64 v129; // [rsp+38h] [rbp-38h]
  __int64 v130; // [rsp+38h] [rbp-38h]
  __int64 v131; // [rsp+38h] [rbp-38h]
  __int64 v132; // [rsp+38h] [rbp-38h]
  __int64 v133; // [rsp+38h] [rbp-38h]
  __int64 *v134; // [rsp+38h] [rbp-38h]
  __int64 v135; // [rsp+38h] [rbp-38h]
  __int64 v136; // [rsp+38h] [rbp-38h]
  __int64 v137; // [rsp+38h] [rbp-38h]
  unsigned __int64 k; // [rsp+38h] [rbp-38h]
  __int64 v139; // [rsp+38h] [rbp-38h]
  __int64 v140; // [rsp+38h] [rbp-38h]
  __int64 *v141; // [rsp+38h] [rbp-38h]
  bool v142; // [rsp+38h] [rbp-38h]

  v2 = a1;
  v3 = sub_8C7610(a1);
  v4 = *(__int64 **)(a1 + 32);
  if ( v4 )
    v2 = *v4;
  v5 = *(_QWORD *)(a1 + 168);
  v6 = *(_QWORD *)(v2 + 168);
  if ( (*(_BYTE *)(a1 + 177) & 0x10) != 0 )
  {
    v10 = qword_4F60250;
    if ( qword_4F60250 )
    {
      while ( a1 != *(_QWORD *)(v10 + 16) )
      {
        v10 = *(_QWORD *)v10;
        if ( !v10 )
          goto LABEL_4;
      }
      *(_BYTE *)(v10 + 8) = 0;
      *(_QWORD *)(v10 + 16) = 0;
    }
  }
LABEL_4:
  if ( !v3 )
    return 0;
  if ( (unsigned __int8)(*(_BYTE *)(v2 + 140) - 9) > 2u )
    goto LABEL_6;
  v111 = v6;
  v127 = v5;
  v11 = sub_8D2490(a1);
  v12 = v127;
  i = v111;
  if ( !v11 )
    goto LABEL_17;
  v19 = sub_8D23B0(a1);
  v12 = v127;
  i = v111;
  if ( v19 || (v20 = sub_8D2490(v2), v12 = v127, i = v111, !v20) || (v21 = sub_8D23B0(v2), v12 = v127, i = v111, v21) )
  {
LABEL_17:
    v14 = *(_BYTE *)(a1 + 177);
    goto LABEL_18;
  }
  v22 = *(_QWORD *)(a1 + 160);
  if ( v22 )
  {
    while ( (*(_BYTE *)(v22 + 144) & 0x50) == 0x40 )
    {
      v22 = *(_QWORD *)(v22 + 112);
      if ( !v22 )
        goto LABEL_67;
    }
    v23 = *(_QWORD *)(v2 + 160);
    if ( !v23 )
      goto LABEL_6;
LABEL_37:
    while ( (*(_BYTE *)(v23 + 144) & 0x50) == 0x40 )
    {
      v23 = *(_QWORD *)(v23 + 112);
      if ( !v23 )
        goto LABEL_53;
    }
LABEL_38:
    if ( !v22 )
      goto LABEL_6;
    v112 = i;
    v128 = v12;
    if ( !(unsigned int)sub_8C78D0(v22) )
      return 0;
    v22 = *(_QWORD *)(v22 + 112);
    v12 = v128;
    for ( i = v112; v22; v22 = *(_QWORD *)(v22 + 112) )
    {
      if ( (*(_BYTE *)(v22 + 144) & 0x50) != 0x40 )
        break;
    }
    while ( 1 )
    {
      v23 = *(_QWORD *)(v23 + 112);
      if ( !v23 )
        break;
      if ( (*(_BYTE *)(v23 + 144) & 0x50) != 0x40 )
        goto LABEL_38;
    }
LABEL_53:
    if ( v22 )
      goto LABEL_6;
  }
  else
  {
LABEL_67:
    v23 = *(_QWORD *)(v2 + 160);
    if ( v23 )
    {
      v22 = 0;
      goto LABEL_37;
    }
  }
  if ( dword_4F077C4 != 2 )
    goto LABEL_55;
  v28 = *(_QWORD **)(i + 152);
  v29 = *(_QWORD **)(v12 + 152);
  v30 = v28[34];
  v31 = v29[34];
  if ( v30 && v31 )
  {
    do
    {
      v90 = v30;
      v99 = i;
      v115 = v12;
      v131 = v31;
      if ( !(unsigned int)sub_8CE3E0(v31) )
        return 0;
      v12 = v115;
      i = v99;
      v31 = *(_QWORD *)(v131 + 112);
      v30 = *(_QWORD *)(v90 + 112);
    }
    while ( v31 && v30 );
  }
  if ( __PAIR128__(v31, v30) != 0 )
    goto LABEL_6;
  v132 = sub_8C6310(v29[13]);
  v32 = sub_8C6310(v28[13]);
  v35 = (__int64 *)v132;
  v36 = v32;
  if ( v32 && v132 )
  {
    do
    {
      v91 = v36;
      v100 = v34;
      v116 = v33;
      v134 = v35;
      if ( !(unsigned int)sub_8CD5A0(v35) )
        return 0;
      v133 = sub_8C6310(v134[14]);
      v37 = sub_8C6310(*(_QWORD *)(v91 + 112));
      v35 = (__int64 *)v133;
      v33 = v116;
      v34 = v100;
      v36 = v37;
    }
    while ( v133 && v37 );
  }
  if ( __PAIR128__((unsigned __int64)v35, v36) != 0 )
    goto LABEL_6;
  v135 = sub_8C6270(v29[18]);
  v38 = sub_8C6270(v28[18]);
  v42 = v135;
  for ( j = v38; v42 && j; j = v44 )
  {
    v92 = v41;
    v101 = v40;
    v117 = j;
    v137 = v42;
    if ( !(unsigned int)sub_8CDA30(v42) )
      return 0;
    v136 = sub_8C6270(*(_QWORD *)(v137 + 112));
    v44 = sub_8C6270(*(_QWORD *)(v117 + 112));
    v42 = v136;
    v40 = v101;
    v41 = v92;
  }
  if ( __PAIR128__(j, v42) != 0 )
    goto LABEL_6;
  v45 = v29[14];
  for ( k = v28[14]; v45 && k; k = *(_QWORD *)(k + 112) )
  {
    v93 = v41;
    v102 = v40;
    v118 = v45;
    if ( !sub_8C7A50(v45) )
      return 0;
    v46 = v118;
    v40 = v102;
    v41 = v93;
    if ( *(char *)(a1 + 177) < 0
      && (*(_BYTE *)(*(_QWORD *)v118 + 81LL) & 2) != 0
      && (*(_BYTE *)(*(_QWORD *)k + 81LL) & 2) != 0 )
    {
      v39 = *(__int64 **)(*(_QWORD *)v118 + 88LL);
      v47 = *(_QWORD *)(*(_QWORD *)k + 88LL);
      v48 = v39[26];
      v49 = *(_QWORD **)(v47 + 208);
      while ( v48 && v49 )
      {
        v87 = v49;
        v88 = v39;
        v89 = v41;
        v94 = v40;
        v103 = v46;
        v119 = (__int64 *)v48;
        sub_8CD5A0(*(__int64 **)(v48 + 16));
        v46 = v103;
        v40 = v94;
        v48 = *v119;
        v49 = (_QWORD *)*v87;
        v41 = v89;
        v39 = v88;
      }
      if ( __PAIR128__((unsigned __int64)v49, v48) != 0 )
      {
        v95 = v41;
        v104 = v40;
        v120 = v46;
        sub_8C6700(v39, (unsigned int *)(v47 + 64), 0x704u, 0x703u);
        v41 = v95;
        v40 = v104;
        v46 = v120;
      }
    }
    v45 = *(_QWORD *)(v46 + 112);
  }
  if ( v45 | k )
    goto LABEL_6;
  v50 = v29[12];
  for ( m = v28[12]; v50 && m; m = *(_QWORD *)(v121 + 120) )
  {
    v96 = v41;
    v105 = v40;
    v121 = m;
    v139 = v50;
    if ( !(unsigned int)sub_8C77C0(v50) )
      return 0;
    v40 = v105;
    v41 = v96;
    v50 = *(_QWORD *)(v139 + 120);
  }
  if ( __PAIR128__(v50, m) != 0 )
    goto LABEL_6;
  v52 = **(_QWORD **)(a1 + 168);
  for ( n = **(_QWORD **)(v2 + 168); ; n = *v123 )
  {
    v142 = n == 0 || v52 == 0;
    if ( v142 )
      break;
    v54 = *(_BYTE *)(v52 + 96);
    if ( (v54 & 1) != 0 )
    {
      v55 = *(_QWORD *)(v52 + 40);
      v56 = *(_QWORD *)(n + 40);
      if ( v55 != v56 )
      {
        v97 = v41;
        v106 = v40;
        v122 = n;
        v140 = v52;
        if ( !(unsigned int)sub_8D97D0(v55, v56, 0, v52, v39) )
          goto LABEL_6;
        v52 = v140;
        n = v122;
        v40 = v106;
        v41 = v97;
        v54 = *(_BYTE *)(v140 + 96);
      }
      if ( ((*(_BYTE *)(n + 96) ^ v54) & 2) != 0
        || *(_BYTE *)(*(_QWORD *)(v52 + 112) + 25LL) != *(_BYTE *)(*(_QWORD *)(n + 112) + 25LL) )
      {
LABEL_6:
        v7 = *(__int64 **)(a1 + 32);
        v8 = a1;
        if ( v7 )
          v8 = *v7;
        sub_8C6700((__int64 *)a1, (unsigned int *)(v8 + 64), 0x42Au, 0x425u);
        return 0;
      }
    }
    v98 = v41;
    v107 = v40;
    v123 = (__int64 *)n;
    v141 = (__int64 *)v52;
    sub_8CBB20(0x25u, v52, (_QWORD *)n);
    v40 = v107;
    v41 = v98;
    v52 = *v141;
  }
  v57 = n | v52;
  if ( v57 )
    goto LABEL_6;
  v58 = v29[22];
  for ( ii = v28[22]; v58 && ii; ii = *(_QWORD *)ii )
  {
    if ( ((*(_BYTE *)(ii + 40) ^ *(_BYTE *)(v58 + 40)) & 1) != 0 )
      goto LABEL_6;
    if ( *(_BYTE *)(v58 + 42) != *(_BYTE *)(ii + 42) )
      goto LABEL_6;
    v60 = *(unsigned __int8 *)(v58 + 41);
    if ( ((*(_BYTE *)(ii + 41) ^ (unsigned __int8)v60) & 6) != 0 )
      goto LABEL_6;
    v61 = *(unsigned __int8 *)(v58 + 16);
    if ( (_BYTE)v61 != *(_BYTE *)(ii + 16) )
      goto LABEL_6;
    v62 = *(_QWORD *)(v58 + 48);
    v63 = *(_QWORD *)(ii + 48);
    if ( (*(_BYTE *)(v62 + 177) & 0x20) == 0 )
    {
      v81 = *(_QWORD *)(v58 + 24);
      if ( (_BYTE)v61 == 37 )
      {
        v81 = *(_QWORD *)(v81 + 40);
        v86 = *(__int64 **)(v81 + 32);
        if ( v86 )
          v81 = *v86;
        v83 = *(_QWORD *)(*(_QWORD *)(ii + 24) + 40LL);
        v84 = *(__int64 **)(v83 + 32);
        if ( v84 )
LABEL_182:
          v83 = *v84;
      }
      else
      {
        v82 = *(__int64 **)(v81 + 32);
        if ( v82 )
          v81 = *v82;
        v83 = *(_QWORD *)(ii + 24);
        v84 = *(__int64 **)(v83 + 32);
        if ( v84 )
          goto LABEL_182;
      }
      v85 = *(__int64 **)(v62 + 32);
      if ( v85 )
        v62 = *v85;
      v57 = *(_QWORD *)(v63 + 32);
      if ( v57 )
        v63 = *(_QWORD *)v57;
      v60 = v81 == v83 && v62 == v63;
      goto LABEL_144;
    }
    if ( v62 != v63 )
    {
      v108 = v41;
      v124 = v40;
      v60 = sub_8D97D0(v62, v63, 0, v57, v39);
      if ( !v60 )
        goto LABEL_6;
      v61 = *(unsigned __int8 *)(v58 + 16);
      v40 = v124;
      v41 = v108;
    }
    v64 = *(_QWORD **)(ii + 24);
    v65 = *(_QWORD **)(v58 + 24);
    if ( (_BYTE)v61 == 37 )
    {
      v65 = (_QWORD *)v65[5];
      v64 = (_QWORD *)v64[5];
      LOBYTE(v60) = v142;
      if ( v65 == v64 )
        goto LABEL_143;
      goto LABEL_142;
    }
    LOBYTE(v57) = (_BYTE)v61 == 6;
    LOBYTE(v60) = (_BYTE)v61 == 11;
    v57 = v60 | (unsigned int)v57;
    LOBYTE(v60) = v57 | ((_BYTE)v61 == 7 || (_BYTE)v61 == 59);
    if ( (_BYTE)v60 )
    {
      if ( (_BYTE)v61 != 6 )
      {
        if ( (_BYTE)v57 )
        {
          v65 = (_QWORD *)v65[19];
          v64 = (_QWORD *)v64[19];
        }
        else if ( (_BYTE)v61 == 7 )
        {
          v65 = (_QWORD *)v65[15];
          v64 = (_QWORD *)v64[15];
        }
        else
        {
          v66 = *((unsigned __int8 *)v65 + 120);
          if ( (_BYTE)v66 != *((_BYTE *)v64 + 120) )
            goto LABEL_6;
          v57 = (unsigned int)(v66 - 6);
          if ( (unsigned __int8)(v66 - 6) <= 1u || (_BYTE)v66 == 1 )
          {
            v65 = (_QWORD *)v65[24];
            v64 = (_QWORD *)v64[24];
          }
          else
          {
            v57 = (unsigned int)(v66 - 2);
            LOBYTE(v57) = (v66 - 2) & 0xFD;
            if ( (_BYTE)v57 )
            {
              if ( (((_BYTE)v66 - 3) & 0xFD) != 0 )
                sub_721090();
              v65 = *(_QWORD **)(v65[24] + 120LL);
              v64 = *(_QWORD **)(v64[24] + 120LL);
            }
            else
            {
              v65 = *(_QWORD **)(v65[24] + 152LL);
              v64 = *(_QWORD **)(v64[24] + 152LL);
            }
          }
        }
      }
      if ( v65 == v64 )
        goto LABEL_143;
LABEL_142:
      v109 = v41;
      v125 = v40;
      v60 = sub_8D97D0(v65, v64, 0, v57, v39);
      v41 = v109;
      v40 = v125;
      LOBYTE(v60) = v60 != 0;
LABEL_143:
      v60 = (unsigned __int8)v60;
      goto LABEL_144;
    }
    v110 = v41;
    v126 = v40;
    v60 = sub_73A2C0((__int64)v65, (__int64)v64, v61, v57, (_UNKNOWN *__ptr32 *)v39);
    v40 = v126;
    v41 = v110;
LABEL_144:
    if ( !v60 )
      goto LABEL_6;
    v58 = *(_QWORD *)v58;
  }
  if ( ii | v58 )
    goto LABEL_6;
  v68 = (unsigned __int64)sub_8C62C0(*(_QWORD **)(v40 + 136));
  for ( jj = (unsigned __int64)sub_8C62C0(*(_QWORD **)(v67 + 136)); v68 && jj; jj = (unsigned __int64)sub_8C62C0(*v71) )
  {
    v72 = *(_QWORD *)(v68 + 8);
    v73 = *(__int64 **)(v72 + 32);
    if ( v73 )
      v72 = *v73;
    v74 = *(_QWORD *)(jj + 8);
    v75 = *(__int64 **)(v74 + 32);
    if ( v75 )
      v74 = *v75;
    if ( v72 != v74 )
      goto LABEL_6;
    v68 = (unsigned __int64)sub_8C62C0(*(_QWORD **)v68);
  }
  if ( jj | v68 )
    goto LABEL_6;
  v77 = (unsigned __int64)sub_8C63A0(*(_QWORD **)(v69 + 144));
  for ( kk = (unsigned __int64)sub_8C63A0(*(_QWORD **)(v76 + 144));
        v77 && kk;
        kk = (unsigned __int64)sub_8C63A0(*(_QWORD **)kk) )
  {
    v79 = *(_QWORD *)(v77 + 8);
    v80 = *(_QWORD *)(kk + 8);
    if ( v79 != v80 && (v80 == 0 || v79 == 0 || !*qword_4D03FD0 || !(unsigned int)sub_8C7EB0(v79, v80, 6u)) )
      goto LABEL_6;
    v77 = (unsigned __int64)sub_8C63A0(*(_QWORD **)v77);
  }
  if ( kk | v77 )
    goto LABEL_6;
LABEL_55:
  v25 = *(_BYTE *)(v2 + 176) ^ *(_BYTE *)(a1 + 176);
  if ( (v25 & 0x7E) != 0 )
    goto LABEL_6;
  if ( v25 < 0 )
    goto LABEL_6;
  v14 = *(_BYTE *)(a1 + 177);
  if ( ((v14 ^ *(_BYTE *)(v2 + 177)) & 0xD) != 0
    || ((*(_BYTE *)(v2 + 179) ^ *(_BYTE *)(a1 + 179)) & 8) != 0
    || *(_DWORD *)(a1 + 184) != *(_DWORD *)(v2 + 184)
    || ((*(_BYTE *)(v2 + 179) ^ *(_BYTE *)(a1 + 179)) & 1) != 0
    || v12 && (*(_QWORD *)(v12 + 72) != *(_QWORD *)(i + 72) || *(_BYTE *)(v12 + 113) != *(_BYTE *)(i + 113)) )
  {
    goto LABEL_6;
  }
LABEL_18:
  v15 = *(_BYTE *)(v2 + 177) ^ v14;
  if ( (v15 & 0x30) != 0 || v15 < 0 )
    goto LABEL_6;
  v16 = *(_BYTE *)(v2 + 178);
  if ( ((v16 ^ *(_BYTE *)(a1 + 178)) & 1) != 0 )
  {
    if ( (*(_BYTE *)(a1 + 178) & 1) != 0
      || (v114 = i,
          v130 = v12,
          v26 = sub_8D2490(a1),
          v12 = v130,
          i = v114,
          v27 = v26 == 0,
          v16 = *(_BYTE *)(v2 + 178),
          v27) )
    {
      if ( (v16 & 1) != 0 )
        goto LABEL_21;
      v113 = i;
      v129 = v12;
      v24 = sub_8D2490(v2);
      v12 = v129;
      i = v113;
      if ( !v24 )
        goto LABEL_21;
      v16 = *(_BYTE *)(a1 + 178);
    }
    if ( (v16 & 1) != 0 )
      goto LABEL_6;
  }
LABEL_21:
  if ( v12 )
  {
    v17 = *(_QWORD *)(v12 + 120);
    v18 = *(_QWORD *)(i + 120);
    if ( v17 != v18 && (v17 == 0 || *qword_4D03FD0 == 0 || !v18 || !(unsigned int)sub_8C7EB0(v17, v18, 8u)) )
      goto LABEL_6;
  }
  return v3;
}
