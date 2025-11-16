// Function: sub_1242650
// Address: 0x1242650
//
__int64 __fastcall sub_1242650(__int64 a1, __int64 a2, unsigned __int64 a3, unsigned int a4)
{
  __int64 v4; // r12
  unsigned __int64 v5; // rsi
  const char *v6; // r13
  const char *v7; // r12
  __int64 *v8; // rbx
  __int64 *v9; // r15
  __int64 v10; // rdi
  __int64 v11; // rbx
  __int64 v12; // r15
  __int64 v13; // rdi
  __int64 v14; // rbx
  __int64 v15; // r12
  __int64 v16; // rdi
  __int64 v17; // rdi
  __int64 v18; // r13
  __int64 v19; // r12
  __int64 v20; // rbx
  __int64 v21; // r15
  __int64 v22; // rdi
  __int64 v23; // rdi
  __int64 v24; // rdi
  __int64 v25; // rdi
  __int64 v26; // rbx
  __int64 v27; // r12
  __int64 v28; // rdi
  __int64 v29; // rbx
  __int64 v30; // r12
  __int64 v31; // rdi
  __int64 v33; // rax
  unsigned int v34; // ebx
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // r15
  const char *v46; // rbx
  __int64 *v47; // rdx
  __int64 *v48; // rdi
  const char *v49; // r13
  __int64 v50; // r14
  __int64 *v51; // rbx
  __int64 *v52; // r12
  __int64 v53; // rdi
  __int64 v54; // rdx
  __int64 v55; // rdi
  const char *v56; // r13
  __int64 v57; // r14
  __int64 v58; // rbx
  __int64 v59; // r12
  __int64 v60; // rdi
  __int64 v61; // rbx
  __int64 v62; // rdi
  __int64 v63; // rdi
  __int64 v64; // r12
  __int64 v65; // rdi
  __int64 v66; // rbx
  __int64 v67; // r13
  __int64 v68; // rdi
  __int64 v69; // rdi
  __int64 v70; // rdi
  __int64 v71; // rdi
  __int64 v72; // rbx
  __int64 v73; // r13
  __int64 v74; // r12
  __int64 v75; // rdi
  __int64 v76; // rbx
  __int64 v77; // r13
  __int64 v78; // r12
  __int64 v79; // rdi
  int v80; // r15d
  unsigned __int8 v81; // dl
  unsigned int v82; // eax
  __int64 v83; // [rsp+10h] [rbp-2A0h]
  int v84; // [rsp+18h] [rbp-298h]
  const char *i; // [rsp+18h] [rbp-298h]
  __int64 j; // [rsp+18h] [rbp-298h]
  __int64 v87; // [rsp+18h] [rbp-298h]
  int v88; // [rsp+20h] [rbp-290h]
  unsigned __int64 v92; // [rsp+48h] [rbp-268h]
  unsigned __int8 v93; // [rsp+48h] [rbp-268h]
  unsigned int v94; // [rsp+54h] [rbp-25Ch] BYREF
  int v95; // [rsp+58h] [rbp-258h] BYREF
  int v96; // [rsp+5Ch] [rbp-254h] BYREF
  __m128i v97; // [rsp+60h] [rbp-250h] BYREF
  _QWORD v98[2]; // [rsp+70h] [rbp-240h] BYREF
  _QWORD v99[2]; // [rsp+80h] [rbp-230h] BYREF
  __int64 v100; // [rsp+90h] [rbp-220h] BYREF
  __int64 v101; // [rsp+98h] [rbp-218h]
  __int64 v102; // [rsp+A0h] [rbp-210h]
  __int64 v103; // [rsp+B0h] [rbp-200h] BYREF
  __int64 v104; // [rsp+B8h] [rbp-1F8h]
  __int64 v105; // [rsp+C0h] [rbp-1F0h]
  const char *v106; // [rsp+D0h] [rbp-1E0h] BYREF
  const char *v107; // [rsp+D8h] [rbp-1D8h]
  __int64 v108; // [rsp+E0h] [rbp-1D0h]
  _QWORD v109[2]; // [rsp+F0h] [rbp-1C0h] BYREF
  __int64 v110; // [rsp+100h] [rbp-1B0h]
  __int64 v111[2]; // [rsp+110h] [rbp-1A0h] BYREF
  __int64 v112; // [rsp+120h] [rbp-190h]
  __int64 v113[2]; // [rsp+130h] [rbp-180h] BYREF
  __int64 v114; // [rsp+140h] [rbp-170h]
  __int64 v115; // [rsp+150h] [rbp-160h] BYREF
  __int64 v116; // [rsp+158h] [rbp-158h]
  __int64 v117; // [rsp+160h] [rbp-150h]
  __int64 v118; // [rsp+170h] [rbp-140h] BYREF
  __int64 v119; // [rsp+178h] [rbp-138h]
  __int64 v120; // [rsp+180h] [rbp-130h]
  __int64 v121; // [rsp+190h] [rbp-120h] BYREF
  __int64 v122; // [rsp+198h] [rbp-118h]
  __int64 v123; // [rsp+1A0h] [rbp-110h]
  __int64 v124; // [rsp+1B0h] [rbp-100h] BYREF
  __int64 v125; // [rsp+1B8h] [rbp-F8h]
  __int64 v126; // [rsp+1C0h] [rbp-F0h]
  const char *v127; // [rsp+1D0h] [rbp-E0h] BYREF
  const char *v128; // [rsp+1D8h] [rbp-D8h]
  __int64 v129; // [rsp+1E0h] [rbp-D0h]
  char v130; // [rsp+1F0h] [rbp-C0h]
  char v131; // [rsp+1F1h] [rbp-BFh]
  __int64 v132; // [rsp+200h] [rbp-B0h] BYREF
  __int64 v133; // [rsp+208h] [rbp-A8h]
  __int64 v134; // [rsp+210h] [rbp-A0h]
  __int64 v135; // [rsp+218h] [rbp-98h]
  __int64 v136; // [rsp+220h] [rbp-90h]
  __int64 v137; // [rsp+228h] [rbp-88h]
  __int64 v138; // [rsp+230h] [rbp-80h]
  __int64 v139; // [rsp+238h] [rbp-78h]
  __int64 v140; // [rsp+240h] [rbp-70h]
  __int64 v141; // [rsp+248h] [rbp-68h]
  __int64 v142; // [rsp+250h] [rbp-60h]
  __int64 v143; // [rsp+258h] [rbp-58h]
  __int64 v144; // [rsp+260h] [rbp-50h]
  __int64 v145; // [rsp+268h] [rbp-48h]
  __int64 v146; // [rsp+270h] [rbp-40h]

  v4 = a1;
  v92 = *(_QWORD *)(a1 + 232);
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  LOWORD(v94) = 0;
  v5 = 16;
  v97 = 0u;
  v98[0] = v99;
  v98[1] = 0;
  v132 = 0;
  v133 = 0;
  v134 = 0;
  v135 = 0;
  v136 = 0;
  v137 = 0;
  v138 = 0;
  v139 = 0;
  v140 = 0;
  v141 = 0;
  v142 = 0;
  v143 = 0;
  v144 = 0;
  v145 = 0;
  v146 = 0;
  v100 = 0;
  v101 = 0;
  v102 = 0;
  v99[0] = &v100;
  v99[1] = 0;
  v103 = 0;
  v104 = 0;
  v105 = 0;
  v106 = 0;
  v107 = 0;
  v108 = 0;
  v96 = 0;
  if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here") )
    goto LABEL_2;
  v5 = 12;
  if ( (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here") )
    goto LABEL_2;
  v5 = (unsigned __int64)&v97;
  if ( (unsigned __int8)sub_1212200(a1, &v97) )
    goto LABEL_2;
  v5 = 4;
  if ( (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' here") )
    goto LABEL_2;
  v5 = (unsigned __int64)&v94;
  if ( (unsigned __int8)sub_1211B70(a1, &v94) )
    goto LABEL_2;
  v5 = 4;
  if ( (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' here") )
    goto LABEL_2;
  v5 = 427;
  if ( (unsigned __int8)sub_120AFE0(a1, 427, "expected 'insts' here") )
    goto LABEL_2;
  v5 = 16;
  if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here") )
    goto LABEL_2;
  v5 = (unsigned __int64)&v95;
  if ( (unsigned __int8)sub_120BD00(a1, &v95) )
    goto LABEL_2;
  while ( *(_DWORD *)(a1 + 240) == 4 )
  {
    v81 = sub_1205540(a1);
    if ( !v81 )
      break;
    v82 = *(_DWORD *)(a1 + 240);
    if ( v82 <= 0x1C4 )
    {
      if ( v82 > 0x1AB )
      {
        switch ( v82 )
        {
          case 0x1ACu:
            v5 = (unsigned __int64)&v96;
            if ( (unsigned __int8)sub_1211000(a1, &v96) )
              goto LABEL_2;
            continue;
          case 0x1B7u:
            v5 = (unsigned __int64)v98;
            if ( (unsigned __int8)sub_123B0B0(a1, (__int64)v98) )
              goto LABEL_2;
            continue;
          case 0x1B9u:
            v5 = (unsigned __int64)&v100;
            if ( (unsigned __int8)sub_1239E50(a1, &v100) )
              goto LABEL_2;
            continue;
          case 0x1C3u:
            v5 = (unsigned __int64)v99;
            if ( (unsigned __int8)sub_123A2F0(a1, (__int64)v99) )
              goto LABEL_2;
            continue;
          case 0x1C4u:
            v5 = (unsigned __int64)&v132;
            if ( !(unsigned __int8)sub_123DC50(a1, &v132) )
              continue;
            goto LABEL_2;
          default:
            break;
        }
      }
LABEL_183:
      v5 = *(_QWORD *)(a1 + 232);
      v93 = v81;
      v131 = 1;
      v127 = "expected optional function summary field";
      v130 = 3;
      sub_11FD800(a1 + 176, v5, (__int64)&v127, 1);
      goto LABEL_3;
    }
    if ( v82 == 492 )
    {
      v5 = (unsigned __int64)&v103;
      if ( (unsigned __int8)sub_12425E0(a1, &v103) )
        goto LABEL_2;
    }
    else
    {
      if ( v82 != 495 )
        goto LABEL_183;
      v5 = (unsigned __int64)&v106;
      if ( (unsigned __int8)sub_1241A90(a1, (__int64)&v106) )
        goto LABEL_2;
    }
  }
  v5 = 13;
  if ( (unsigned __int8)sub_120AFE0(a1, 13, "expected ')' here") )
  {
LABEL_2:
    v93 = 1;
  }
  else
  {
    v33 = v132;
    v132 = 0;
    v34 = v94;
    v109[0] = v33;
    v84 = v95;
    v109[1] = v133;
    v88 = v96;
    v110 = v134;
    v134 = 0;
    v111[0] = v135;
    v133 = 0;
    v111[1] = v136;
    v136 = 0;
    v112 = v137;
    v137 = 0;
    v113[0] = v138;
    v135 = 0;
    v113[1] = v139;
    v139 = 0;
    v114 = v140;
    v140 = 0;
    v138 = 0;
    v35 = v141;
    v141 = 0;
    v115 = v35;
    v36 = v142;
    v142 = 0;
    v116 = v36;
    v37 = v143;
    v143 = 0;
    v117 = v37;
    v38 = v144;
    v144 = 0;
    v118 = v38;
    v39 = v145;
    v145 = 0;
    v119 = v39;
    v40 = v146;
    v146 = 0;
    v120 = v40;
    v41 = v100;
    v100 = 0;
    v121 = v41;
    v42 = v101;
    v101 = 0;
    v122 = v42;
    v43 = v102;
    v102 = 0;
    v123 = v43;
    v124 = v103;
    v125 = v104;
    v126 = v105;
    v105 = 0;
    v127 = v106;
    v104 = 0;
    v128 = v107;
    v103 = 0;
    v129 = v108;
    v108 = 0;
    v107 = 0;
    v106 = 0;
    v44 = sub_22077B0(112);
    v45 = v44;
    if ( v44 )
    {
      v5 = v34;
      sub_9C6E00(
        v44,
        v34,
        v84,
        v88,
        (__int64)v99,
        (__int64)v98,
        v109,
        v111,
        v113,
        &v115,
        &v118,
        &v121,
        &v124,
        (__int64)&v127);
    }
    v46 = v127;
    for ( i = v128; i != v46; v46 += 112 )
    {
      v47 = (__int64 *)*((_QWORD *)v46 + 12);
      v48 = (__int64 *)*((_QWORD *)v46 + 11);
      if ( v47 != v48 )
      {
        v49 = v46;
        v50 = v4;
        v51 = (__int64 *)*((_QWORD *)v46 + 11);
        v52 = v47;
        do
        {
          v53 = *v51;
          if ( *v51 )
          {
            v5 = v51[2] - v53;
            j_j___libc_free_0(v53, v5);
          }
          v51 += 3;
        }
        while ( v52 != v51 );
        v48 = (__int64 *)*((_QWORD *)v49 + 11);
        v46 = v49;
        v4 = v50;
      }
      if ( v48 )
      {
        v5 = *((_QWORD *)v46 + 13) - (_QWORD)v48;
        j_j___libc_free_0(v48, v5);
      }
      v54 = *((_QWORD *)v46 + 9);
      v55 = *((_QWORD *)v46 + 8);
      if ( v54 != v55 )
      {
        v56 = v46;
        v57 = v4;
        v58 = *((_QWORD *)v46 + 8);
        v59 = v54;
        do
        {
          v60 = *(_QWORD *)(v58 + 8);
          if ( v60 != v58 + 24 )
            _libc_free(v60, v5);
          v58 += 72;
        }
        while ( v59 != v58 );
        v55 = *((_QWORD *)v56 + 8);
        v46 = v56;
        v4 = v57;
      }
      if ( v55 )
      {
        v5 = *((_QWORD *)v46 + 10) - v55;
        j_j___libc_free_0(v55, v5);
      }
      if ( *(const char **)v46 != v46 + 24 )
        _libc_free(*(_QWORD *)v46, v5);
    }
    if ( v127 )
    {
      v5 = v129 - (_QWORD)v127;
      j_j___libc_free_0(v127, v129 - (_QWORD)v127);
    }
    v61 = v124;
    for ( j = v125; j != v61; v61 += 136 )
    {
      v62 = *(_QWORD *)(v61 + 72);
      if ( v62 != v61 + 88 )
        _libc_free(v62, v5);
      v63 = *(_QWORD *)(v61 + 8);
      if ( v63 != v61 + 24 )
        _libc_free(v63, v5);
    }
    if ( v124 )
      j_j___libc_free_0(v124, v126 - v124);
    v87 = v122;
    if ( v122 != v121 )
    {
      v83 = v4;
      v64 = v121;
      do
      {
        v65 = *(_QWORD *)(v64 + 40);
        v66 = *(_QWORD *)(v64 + 48);
        v67 = v65;
        if ( v66 != v65 )
        {
          do
          {
            if ( *(_DWORD *)(v67 + 40) > 0x40u )
            {
              v68 = *(_QWORD *)(v67 + 32);
              if ( v68 )
                j_j___libc_free_0_0(v68);
            }
            if ( *(_DWORD *)(v67 + 24) > 0x40u )
            {
              v69 = *(_QWORD *)(v67 + 16);
              if ( v69 )
                j_j___libc_free_0_0(v69);
            }
            v67 += 48;
          }
          while ( v66 != v67 );
          v65 = *(_QWORD *)(v64 + 40);
        }
        if ( v65 )
          j_j___libc_free_0(v65, *(_QWORD *)(v64 + 56) - v65);
        if ( *(_DWORD *)(v64 + 32) > 0x40u )
        {
          v70 = *(_QWORD *)(v64 + 24);
          if ( v70 )
            j_j___libc_free_0_0(v70);
        }
        if ( *(_DWORD *)(v64 + 16) > 0x40u )
        {
          v71 = *(_QWORD *)(v64 + 8);
          if ( v71 )
            j_j___libc_free_0_0(v71);
        }
        v64 += 64;
      }
      while ( v87 != v64 );
      v4 = v83;
    }
    if ( v121 )
      j_j___libc_free_0(v121, v123 - v121);
    v72 = v118;
    if ( v119 != v118 )
    {
      v73 = v4;
      v74 = v119;
      do
      {
        v75 = *(_QWORD *)(v72 + 16);
        if ( v75 )
          j_j___libc_free_0(v75, *(_QWORD *)(v72 + 32) - v75);
        v72 += 40;
      }
      while ( v74 != v72 );
      v4 = v73;
    }
    if ( v118 )
      j_j___libc_free_0(v118, v120 - v118);
    v76 = v115;
    if ( v116 != v115 )
    {
      v77 = v4;
      v78 = v116;
      do
      {
        v79 = *(_QWORD *)(v76 + 16);
        if ( v79 )
          j_j___libc_free_0(v79, *(_QWORD *)(v76 + 32) - v79);
        v76 += 40;
      }
      while ( v78 != v76 );
      v4 = v77;
    }
    if ( v115 )
      j_j___libc_free_0(v115, v117 - v115);
    if ( v113[0] )
      j_j___libc_free_0(v113[0], v114 - v113[0]);
    if ( v111[0] )
      j_j___libc_free_0(v111[0], v112 - v111[0]);
    if ( v109[0] )
      j_j___libc_free_0(v109[0], v110 - v109[0]);
    v124 = v45;
    *(__m128i *)(v45 + 24) = v97;
    v80 = v94 & 0xF;
    v5 = (unsigned __int64)&v127;
    sub_2241BD0(&v127, a2);
    v93 = sub_123DE00(v4, &v127, a3, v80, a4, &v124, v92);
    sub_2240A30(&v127);
    sub_9C9050(&v124);
  }
LABEL_3:
  v6 = v107;
  v7 = v106;
  if ( v107 != v106 )
  {
    do
    {
      v8 = (__int64 *)*((_QWORD *)v7 + 12);
      v9 = (__int64 *)*((_QWORD *)v7 + 11);
      if ( v8 != v9 )
      {
        do
        {
          v10 = *v9;
          if ( *v9 )
          {
            v5 = v9[2] - v10;
            j_j___libc_free_0(v10, v5);
          }
          v9 += 3;
        }
        while ( v8 != v9 );
        v9 = (__int64 *)*((_QWORD *)v7 + 11);
      }
      if ( v9 )
      {
        v5 = *((_QWORD *)v7 + 13) - (_QWORD)v9;
        j_j___libc_free_0(v9, v5);
      }
      v11 = *((_QWORD *)v7 + 9);
      v12 = *((_QWORD *)v7 + 8);
      if ( v11 != v12 )
      {
        do
        {
          v13 = *(_QWORD *)(v12 + 8);
          if ( v13 != v12 + 24 )
            _libc_free(v13, v5);
          v12 += 72;
        }
        while ( v11 != v12 );
        v12 = *((_QWORD *)v7 + 8);
      }
      if ( v12 )
      {
        v5 = *((_QWORD *)v7 + 10) - v12;
        j_j___libc_free_0(v12, v5);
      }
      if ( *(const char **)v7 != v7 + 24 )
        _libc_free(*(_QWORD *)v7, v5);
      v7 += 112;
    }
    while ( v6 != v7 );
    v7 = v106;
  }
  if ( v7 )
  {
    v5 = v108 - (_QWORD)v7;
    j_j___libc_free_0(v7, v108 - (_QWORD)v7);
  }
  v14 = v104;
  v15 = v103;
  if ( v104 != v103 )
  {
    do
    {
      v16 = *(_QWORD *)(v15 + 72);
      if ( v16 != v15 + 88 )
        _libc_free(v16, v5);
      v17 = *(_QWORD *)(v15 + 8);
      if ( v17 != v15 + 24 )
        _libc_free(v17, v5);
      v15 += 136;
    }
    while ( v14 != v15 );
    v15 = v103;
  }
  if ( v15 )
  {
    v5 = v105 - v15;
    j_j___libc_free_0(v15, v105 - v15);
  }
  if ( (__int64 *)v99[0] != &v100 )
    _libc_free(v99[0], v5);
  v18 = v101;
  v19 = v100;
  if ( v101 != v100 )
  {
    do
    {
      v20 = *(_QWORD *)(v19 + 48);
      v21 = *(_QWORD *)(v19 + 40);
      if ( v20 != v21 )
      {
        do
        {
          if ( *(_DWORD *)(v21 + 40) > 0x40u )
          {
            v22 = *(_QWORD *)(v21 + 32);
            if ( v22 )
              j_j___libc_free_0_0(v22);
          }
          if ( *(_DWORD *)(v21 + 24) > 0x40u )
          {
            v23 = *(_QWORD *)(v21 + 16);
            if ( v23 )
              j_j___libc_free_0_0(v23);
          }
          v21 += 48;
        }
        while ( v20 != v21 );
        v21 = *(_QWORD *)(v19 + 40);
      }
      if ( v21 )
      {
        v5 = *(_QWORD *)(v19 + 56) - v21;
        j_j___libc_free_0(v21, v5);
      }
      if ( *(_DWORD *)(v19 + 32) > 0x40u )
      {
        v24 = *(_QWORD *)(v19 + 24);
        if ( v24 )
          j_j___libc_free_0_0(v24);
      }
      if ( *(_DWORD *)(v19 + 16) > 0x40u )
      {
        v25 = *(_QWORD *)(v19 + 8);
        if ( v25 )
          j_j___libc_free_0_0(v25);
      }
      v19 += 64;
    }
    while ( v18 != v19 );
    v19 = v100;
  }
  if ( v19 )
  {
    v5 = v102 - v19;
    j_j___libc_free_0(v19, v102 - v19);
  }
  v26 = v145;
  v27 = v144;
  if ( v145 != v144 )
  {
    do
    {
      v28 = *(_QWORD *)(v27 + 16);
      if ( v28 )
      {
        v5 = *(_QWORD *)(v27 + 32) - v28;
        j_j___libc_free_0(v28, v5);
      }
      v27 += 40;
    }
    while ( v26 != v27 );
    v27 = v144;
  }
  if ( v27 )
  {
    v5 = v146 - v27;
    j_j___libc_free_0(v27, v146 - v27);
  }
  v29 = v142;
  v30 = v141;
  if ( v142 != v141 )
  {
    do
    {
      v31 = *(_QWORD *)(v30 + 16);
      if ( v31 )
      {
        v5 = *(_QWORD *)(v30 + 32) - v31;
        j_j___libc_free_0(v31, v5);
      }
      v30 += 40;
    }
    while ( v29 != v30 );
    v30 = v141;
  }
  if ( v30 )
  {
    v5 = v143 - v30;
    j_j___libc_free_0(v30, v143 - v30);
  }
  if ( v138 )
  {
    v5 = v140 - v138;
    j_j___libc_free_0(v138, v140 - v138);
  }
  if ( v135 )
  {
    v5 = v137 - v135;
    j_j___libc_free_0(v135, v137 - v135);
  }
  if ( v132 )
  {
    v5 = v134 - v132;
    j_j___libc_free_0(v132, v134 - v132);
  }
  if ( (_QWORD *)v98[0] != v99 )
    _libc_free(v98[0], v5);
  return v93;
}
