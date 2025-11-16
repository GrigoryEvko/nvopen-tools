// Function: sub_1958390
// Address: 0x1958390
//
__int64 __fastcall sub_1958390(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 *v5; // rax
  __int64 v6; // rdx
  __int64 *v7; // r15
  __int64 *v8; // rdx
  char v9; // dl
  __int64 v10; // r12
  __int64 v11; // r14
  unsigned __int64 v12; // r14
  char v13; // al
  unsigned int v14; // r13d
  bool v15; // al
  unsigned int v16; // r13d
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 *v19; // rdx
  __int64 v20; // rbx
  __int64 *v21; // rsi
  __int64 *v22; // rcx
  unsigned int v23; // r13d
  char v24; // r8
  __int64 v25; // rax
  __int64 v26; // r10
  unsigned int v27; // edx
  __int64 v28; // rsi
  __int64 v29; // rdi
  __int64 v30; // rsi
  __int64 v31; // rcx
  unsigned __int64 v32; // rcx
  __int64 v33; // rsi
  __int64 v34; // rdx
  unsigned __int64 v35; // r14
  __int64 v36; // rbx
  __int64 *v37; // r8
  __int64 *v38; // rbx
  __int64 v39; // r13
  __int64 *v40; // r14
  unsigned __int64 v41; // r13
  __int64 v43; // r15
  unsigned __int64 v44; // rdi
  unsigned __int64 v45; // r12
  int v46; // eax
  int v47; // ebx
  unsigned int v48; // r13d
  int v49; // r8d
  int v50; // r9d
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // r13
  int v54; // r12d
  int v55; // eax
  unsigned __int64 v56; // rcx
  __int64 v57; // rdx
  unsigned __int64 v58; // rcx
  __int64 v59; // rdx
  unsigned __int64 v60; // rcx
  __int64 v61; // rdx
  __int64 *v62; // r14
  __int64 v63; // rax
  __int64 *v64; // r12
  __int64 v65; // rdx
  __int64 *v66; // r13
  __int64 v67; // rbx
  __int64 *i; // r13
  __int64 v69; // rsi
  __int64 v70; // rdx
  __int64 *v71; // rax
  __int64 v72; // rcx
  __int64 v73; // rax
  __int64 *v74; // rdi
  unsigned __int64 v75; // r12
  unsigned int v76; // ecx
  __int64 v77; // rax
  __int64 v78; // r10
  int v79; // edx
  __int64 v80; // rdx
  int v81; // edx
  unsigned int v82; // ecx
  __int64 v83; // rdi
  __int64 v84; // rdx
  unsigned __int64 v85; // r12
  unsigned int v86; // eax
  __int64 v87; // rdx
  __int64 *v88; // r12
  __int64 v89; // rcx
  __int64 *v90; // rax
  unsigned int v91; // r13d
  __int64 *v92; // rax
  __int64 *v93; // rbx
  __int64 v94; // rax
  __int64 v95; // rax
  unsigned __int64 v96; // rax
  int v97; // eax
  unsigned __int64 v98; // rdi
  unsigned __int64 v99; // rbx
  unsigned int v100; // r12d
  unsigned __int8 v101; // r13
  __int64 v102; // r15
  _QWORD *v103; // r12
  _QWORD *v104; // rdi
  int v105; // esi
  __int64 v106; // r11
  unsigned int v107; // ecx
  __int64 v108; // rdi
  unsigned int v109; // r13d
  unsigned __int64 v110; // r12
  __int64 v111; // rcx
  const __m128i *v112; // rax
  __int64 v113; // rdx
  const __m128i *v114; // rsi
  __int64 v115; // rdx
  const __m128i *v116; // rdx
  signed __int64 v117; // rdx
  int v118; // esi
  __int64 v119; // [rsp+8h] [rbp-388h]
  __int64 v121; // [rsp+10h] [rbp-380h]
  unsigned __int8 v123; // [rsp+30h] [rbp-360h]
  int v124; // [rsp+30h] [rbp-360h]
  __int64 *v125; // [rsp+30h] [rbp-360h]
  int v126; // [rsp+38h] [rbp-358h]
  __int64 v127; // [rsp+38h] [rbp-358h]
  __int64 v129; // [rsp+48h] [rbp-348h]
  __int64 *v130; // [rsp+50h] [rbp-340h]
  int v131; // [rsp+50h] [rbp-340h]
  __int64 v132; // [rsp+58h] [rbp-338h]
  __m128i v133; // [rsp+60h] [rbp-330h] BYREF
  __int64 v134; // [rsp+70h] [rbp-320h]
  unsigned int v135; // [rsp+78h] [rbp-318h]
  __int64 *v136; // [rsp+80h] [rbp-310h]
  __int64 v137; // [rsp+88h] [rbp-308h]
  _BYTE v138[128]; // [rsp+90h] [rbp-300h] BYREF
  const __m128i *v139; // [rsp+110h] [rbp-280h] BYREF
  __int64 v140; // [rsp+118h] [rbp-278h]
  _QWORD v141[16]; // [rsp+120h] [rbp-270h] BYREF
  __int64 v142; // [rsp+1A0h] [rbp-1F0h] BYREF
  __int64 *v143; // [rsp+1A8h] [rbp-1E8h]
  __int64 *v144; // [rsp+1B0h] [rbp-1E0h]
  __int64 v145; // [rsp+1B8h] [rbp-1D8h]
  int v146; // [rsp+1C0h] [rbp-1D0h]
  _BYTE v147[136]; // [rsp+1C8h] [rbp-1C8h] BYREF
  __int64 *v148; // [rsp+250h] [rbp-140h] BYREF
  __int64 v149; // [rsp+258h] [rbp-138h]
  _BYTE v150[304]; // [rsp+260h] [rbp-130h] BYREF

  v136 = (__int64 *)v138;
  v137 = 0x800000000LL;
  v123 = sub_1954CE0(a1, a2, a3);
  if ( !v123 )
    goto LABEL_69;
  v5 = (__int64 *)v147;
  v148 = (__int64 *)v150;
  v6 = 2LL * (unsigned int)v137;
  v149 = 0x1000000000LL;
  v142 = 0;
  v143 = (__int64 *)v147;
  v144 = (__int64 *)v147;
  v145 = 16;
  v146 = 0;
  v130 = &v136[v6];
  if ( v136 == &v136[v6] )
  {
    v123 = 0;
    goto LABEL_67;
  }
  v126 = 0;
  v7 = v136;
  v8 = (__int64 *)v147;
  v129 = 0;
  v132 = 0;
  while ( 1 )
  {
    v20 = v7[1];
    if ( v5 != v8 )
      goto LABEL_4;
    v21 = &v5[HIDWORD(v145)];
    if ( v5 == v21 )
    {
LABEL_74:
      if ( HIDWORD(v145) >= (unsigned int)v145 )
      {
LABEL_4:
        sub_16CCBA0((__int64)&v142, v7[1]);
        if ( !v9 )
          goto LABEL_20;
      }
      else
      {
        ++HIDWORD(v145);
        *v21 = v20;
        ++v142;
      }
LABEL_5:
      v10 = *v7;
      v11 = 0;
      if ( *(_BYTE *)(*v7 + 16) != 9 )
      {
        v12 = sub_157EBA0(a3);
        v13 = *(_BYTE *)(v12 + 16);
        if ( v13 == 26 )
        {
          v14 = *(_DWORD *)(v10 + 32);
          if ( v14 <= 0x40 )
            v15 = *(_QWORD *)(v10 + 24) == 0;
          else
            v15 = v14 == (unsigned int)sub_16A57B0(v10 + 24);
          v11 = *(_QWORD *)(v12 - 24LL * v15 - 24);
          goto LABEL_10;
        }
        if ( v13 == 27 )
        {
          v23 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
          v24 = *(_BYTE *)(v12 + 23) & 0x40;
          v119 = (v23 >> 1) - 1;
          v25 = v119 >> 2;
          if ( v119 >> 2 )
          {
            v26 = 4 * v25;
            v25 = 0;
            v27 = 8;
            while ( 1 )
            {
              v29 = v25 + 1;
              v32 = v12 - 24LL * v23;
              if ( v24 )
                v32 = *(_QWORD *)(v12 - 8);
              v33 = *(_QWORD *)(v32 + 24LL * (v27 - 6));
              if ( v33 )
              {
                if ( v10 == v33 )
                  goto LABEL_46;
              }
              v28 = *(_QWORD *)(v32 + 24LL * (v27 - 4));
              if ( v28 && v10 == v28 )
                goto LABEL_47;
              v29 = v25 + 3;
              v30 = *(_QWORD *)(v32 + 24LL * (v27 - 2));
              if ( v30 && v10 == v30 )
              {
                LODWORD(v29) = v25 + 2;
                if ( v119 != v25 + 2 )
                  goto LABEL_48;
                goto LABEL_91;
              }
              v25 += 4;
              v31 = *(_QWORD *)(v32 + 24LL * v27);
              if ( v31 && v10 == v31 )
                goto LABEL_47;
              v27 += 8;
              if ( v25 == v26 )
              {
                v52 = v119 - v25;
                goto LABEL_88;
              }
            }
          }
          v52 = ((*(_DWORD *)(v12 + 20) & 0xFFFFFFFu) >> 1) - 1;
LABEL_88:
          switch ( v52 )
          {
            case 2LL:
              v29 = v25;
              break;
            case 3LL:
              v29 = v25 + 1;
              if ( v24 )
                v56 = *(_QWORD *)(v12 - 8);
              else
                v56 = v12 - 24LL * v23;
              v57 = *(_QWORD *)(v56 + 24LL * (unsigned int)(2 * (v25 + 1)));
              if ( v57 && v10 == v57 )
              {
LABEL_46:
                v29 = v25;
                goto LABEL_47;
              }
              break;
            case 1LL:
LABEL_111:
              if ( v24 )
                v60 = *(_QWORD *)(v12 - 8);
              else
                v60 = v12 - 24LL * v23;
              v61 = *(_QWORD *)(v60 + 24LL * (unsigned int)(2 * v25 + 2));
              if ( !v61 || v10 != v61 )
                goto LABEL_91;
              goto LABEL_46;
            default:
              goto LABEL_91;
          }
          v25 = v29 + 1;
          if ( v24 )
            v58 = *(_QWORD *)(v12 - 8);
          else
            v58 = v12 - 24LL * v23;
          v59 = *(_QWORD *)(v58 + 24LL * (unsigned int)(2 * (v29 + 1)));
          if ( v59 && v10 == v59 )
          {
LABEL_47:
            if ( v119 != v29 )
            {
LABEL_48:
              if ( (_DWORD)v29 != -2 )
              {
                v34 = 24LL * (unsigned int)(2 * v29 + 3);
LABEL_50:
                if ( v24 )
                  v35 = *(_QWORD *)(v12 - 8);
                else
                  v35 = v12 - 24LL * v23;
                v11 = *(_QWORD *)(v35 + v34);
                goto LABEL_10;
              }
            }
LABEL_91:
            v34 = 24;
            goto LABEL_50;
          }
          goto LABEL_111;
        }
        v11 = *(_QWORD *)(v10 - 24);
      }
LABEL_10:
      v16 = v149;
      if ( (_DWORD)v149 )
      {
        v17 = v132;
        v18 = -1;
        if ( v11 != v132 )
          v17 = -1;
        v132 = v17;
        if ( v10 == v129 )
          v18 = v129;
        v129 = v18;
      }
      else
      {
        v129 = v10;
        v132 = v11;
      }
      ++v126;
      if ( *(_BYTE *)(sub_157EBA0(v20) + 16) != 28 )
      {
        if ( v16 >= HIDWORD(v149) )
        {
          sub_16CD150((__int64)&v148, v150, 0, 16, v3, v4);
          v16 = v149;
        }
        v19 = &v148[2 * v16];
        *v19 = v20;
        v19[1] = v11;
        LODWORD(v149) = v149 + 1;
      }
      goto LABEL_20;
    }
    v22 = 0;
    while ( v20 != *v5 )
    {
      if ( *v5 == -2 )
        v22 = v5;
      if ( v21 == ++v5 )
      {
        if ( !v22 )
          goto LABEL_74;
        *v22 = v20;
        --v146;
        ++v142;
        goto LABEL_5;
      }
    }
LABEL_20:
    v7 += 2;
    if ( v130 == v7 )
      break;
    v8 = v144;
    v5 = v143;
  }
  v36 = (unsigned int)v149;
  if ( !(_DWORD)v149 )
  {
    v123 = 0;
    v41 = (unsigned __int64)v148;
    goto LABEL_65;
  }
  if ( (unsigned __int64)(v132 - 1) <= 0xFFFFFFFFFFFFFFFDLL )
  {
    v53 = *(_QWORD *)(a3 + 8);
    if ( v53 )
    {
      while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v53) + 16) - 25) > 9u )
      {
        v53 = *(_QWORD *)(v53 + 8);
        if ( !v53 )
          goto LABEL_102;
      }
      v54 = 0;
      while ( 1 )
      {
        v53 = *(_QWORD *)(v53 + 8);
        if ( !v53 )
          break;
        while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v53) + 16) - 25) <= 9u )
        {
          v53 = *(_QWORD *)(v53 + 8);
          ++v54;
          if ( !v53 )
            goto LABEL_98;
        }
      }
LABEL_98:
      v55 = v54 + 1;
    }
    else
    {
LABEL_102:
      v55 = 0;
    }
    if ( v55 != v126 )
    {
      v37 = v148;
      v139 = (const __m128i *)v141;
      v38 = &v148[2 * v36];
      v140 = 0x1000000000LL;
      goto LABEL_57;
    }
    v139 = 0;
    v140 = 0;
    v141[0] = 0;
    v96 = sub_157EBA0(a3);
    v97 = sub_15F4D60(v96);
    sub_1953AE0(&v139, (unsigned int)(v97 - 1));
    v98 = sub_157EBA0(a3);
    if ( v98 )
    {
      v131 = sub_15F4D60(v98);
      v99 = sub_157EBA0(a3);
      if ( v131 )
      {
        v100 = 0;
        v101 = 0;
        do
        {
          v102 = sub_15F4DF0(v99, v100);
          if ( ((v101 ^ 1) & (v102 == v132)) != 0 )
          {
            v101 = (v101 ^ 1) & (v102 == v132);
          }
          else
          {
            sub_157F2D0(v102, a3, 1);
            v133.m128i_i64[1] = v102 | 4;
            v133.m128i_i64[0] = a3;
            sub_19541D0((__int64)&v139, &v133);
          }
          ++v100;
        }
        while ( v100 != v131 );
      }
    }
    v103 = (_QWORD *)sub_157EBA0(a3);
    v104 = sub_1648A60(56, 1u);
    if ( v104 )
      sub_15F8320((__int64)v104, v132, (__int64)v103);
    sub_15F20C0(v103);
    sub_15CD9D0(*(_QWORD *)(a1 + 24), v139->m128i_i64, (v140 - (__int64)v139) >> 4);
    if ( *(_BYTE *)(a2 + 16) > 0x17u )
    {
      if ( *(_QWORD *)(a2 + 8) || (unsigned __int8)sub_15F3040(a2) || sub_15F3330(a2) )
      {
        if ( (unsigned __int64)(v129 - 1) <= 0xFFFFFFFFFFFFFFFDLL && a3 == *(_QWORD *)(a2 + 40) )
          sub_1952070((_QWORD *)a2, v129);
      }
      else
      {
        sub_15F20C0((_QWORD *)a2);
      }
    }
    if ( v139 )
      j_j___libc_free_0(v139, v141[0] - (_QWORD)v139);
LABEL_64:
    v41 = (unsigned __int64)v148;
    goto LABEL_65;
  }
  if ( v132 != -1 )
    goto LABEL_56;
  v62 = v148;
  v63 = 16LL * (unsigned int)v149;
  v64 = &v148[(unsigned __int64)v63 / 8];
  v65 = v63 >> 4;
  if ( !(v63 >> 6) )
  {
LABEL_196:
    switch ( v65 )
    {
      case 2LL:
        v67 = a1 + 56;
        break;
      case 3LL:
        v67 = a1 + 56;
        if ( sub_1377F70(a1 + 56, v62[1]) )
          goto LABEL_126;
        v62 += 2;
        break;
      case 1LL:
        v67 = a1 + 56;
LABEL_232:
        if ( sub_1377F70(v67, v62[1]) )
          goto LABEL_126;
        goto LABEL_199;
      default:
LABEL_199:
        v62 = v64;
        goto LABEL_131;
    }
    if ( sub_1377F70(v67, v62[1]) )
      goto LABEL_126;
    v62 += 2;
    goto LABEL_232;
  }
  v66 = &v148[8 * (v63 >> 6)];
  v67 = a1 + 56;
  while ( !sub_1377F70(v67, v62[1]) )
  {
    if ( sub_1377F70(v67, v62[3]) )
    {
      v62 += 2;
      break;
    }
    if ( sub_1377F70(v67, v62[5]) )
    {
      v62 += 4;
      break;
    }
    if ( sub_1377F70(v67, v62[7]) )
    {
      v62 += 6;
      break;
    }
    v62 += 8;
    if ( v66 == v62 )
    {
      v65 = ((char *)v64 - (char *)v62) >> 4;
      goto LABEL_196;
    }
  }
LABEL_126:
  if ( v64 != v62 )
  {
    for ( i = v62 + 2; v64 != i; i += 2 )
    {
      if ( !sub_1377F70(v67, i[1]) )
      {
        v62 += 2;
        *(v62 - 2) = *i;
        *(v62 - 1) = i[1];
      }
    }
  }
LABEL_131:
  v41 = (unsigned __int64)v148;
  v69 = (char *)&v148[2 * (unsigned int)v149] - (char *)v64;
  v70 = v69 >> 4;
  if ( v69 > 0 )
  {
    v71 = v62;
    do
    {
      v72 = *v64;
      v71 += 2;
      v64 += 2;
      *(v71 - 2) = v72;
      *(v71 - 1) = *(v64 - 1);
      --v70;
    }
    while ( v70 );
    v41 = (unsigned __int64)v148;
    v62 = (__int64 *)((char *)v62 + v69);
  }
  v123 = 0;
  v73 = (__int64)((__int64)v62 - v41) >> 4;
  LODWORD(v149) = v73;
  if ( (_DWORD)v73 )
  {
    v133 = 0u;
    v74 = 0;
    v134 = 0;
    v75 = v41 + 16LL * (unsigned int)v73;
    v135 = 0;
    while ( 2 )
    {
      v80 = *(_QWORD *)(v41 + 8);
      if ( !v80 )
        goto LABEL_140;
      if ( !v135 )
      {
        ++v133.m128i_i64[0];
        goto LABEL_144;
      }
      v76 = (v135 - 1) & (((unsigned int)v80 >> 9) ^ ((unsigned int)v80 >> 4));
      v77 = (__int64)&v74[2 * v76];
      v78 = *(_QWORD *)v77;
      if ( v80 == *(_QWORD *)v77 )
      {
LABEL_138:
        v79 = *(_DWORD *)(v77 + 8) + 1;
        goto LABEL_139;
      }
      LODWORD(v4) = 1;
      v3 = 0;
      while ( v78 != -8 )
      {
        if ( !v3 && v78 == -16 )
          v3 = v77;
        v76 = (v135 - 1) & (v4 + v76);
        v77 = (__int64)&v74[2 * v76];
        v78 = *(_QWORD *)v77;
        if ( v80 == *(_QWORD *)v77 )
          goto LABEL_138;
        LODWORD(v4) = v4 + 1;
      }
      if ( v3 )
        v77 = v3;
      ++v133.m128i_i64[0];
      v81 = v134 + 1;
      if ( 4 * ((int)v134 + 1) >= 3 * v135 )
      {
LABEL_144:
        sub_13FEAC0((__int64)&v133, 2 * v135);
        if ( !v135 )
          goto LABEL_272;
        v4 = *(_QWORD *)(v41 + 8);
        LODWORD(v3) = v135 - 1;
        v81 = v134 + 1;
        v82 = (v135 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
        v77 = v133.m128i_i64[1] + 16LL * v82;
        v83 = *(_QWORD *)v77;
        if ( v4 != *(_QWORD *)v77 )
        {
          v118 = 1;
          v106 = 0;
          while ( v83 != -8 )
          {
            if ( v83 == -16 && !v106 )
              v106 = v77;
            v82 = v3 & (v118 + v82);
            v77 = v133.m128i_i64[1] + 16LL * v82;
            v83 = *(_QWORD *)v77;
            if ( v4 == *(_QWORD *)v77 )
              goto LABEL_146;
            ++v118;
          }
          goto LABEL_214;
        }
      }
      else if ( v135 - HIDWORD(v134) - v81 <= v135 >> 3 )
      {
        sub_13FEAC0((__int64)&v133, v135);
        if ( !v135 )
        {
LABEL_272:
          LODWORD(v134) = v134 + 1;
          BUG();
        }
        v4 = *(_QWORD *)(v41 + 8);
        LODWORD(v3) = v135 - 1;
        v105 = 1;
        v106 = 0;
        v81 = v134 + 1;
        v107 = (v135 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
        v77 = v133.m128i_i64[1] + 16LL * v107;
        v108 = *(_QWORD *)v77;
        if ( v4 != *(_QWORD *)v77 )
        {
          while ( v108 != -8 )
          {
            if ( !v106 && v108 == -16 )
              v106 = v77;
            v107 = v3 & (v105 + v107);
            v77 = v133.m128i_i64[1] + 16LL * v107;
            v108 = *(_QWORD *)v77;
            if ( v4 == *(_QWORD *)v77 )
              goto LABEL_146;
            ++v105;
          }
LABEL_214:
          if ( v106 )
            v77 = v106;
        }
      }
LABEL_146:
      LODWORD(v134) = v81;
      if ( *(_QWORD *)v77 != -8 )
        --HIDWORD(v134);
      v84 = *(_QWORD *)(v41 + 8);
      *(_DWORD *)(v77 + 8) = 0;
      *(_QWORD *)v77 = v84;
      v79 = 1;
LABEL_139:
      *(_DWORD *)(v77 + 8) = v79;
      v74 = (__int64 *)v133.m128i_i64[1];
LABEL_140:
      v41 += 16LL;
      if ( v75 == v41 )
      {
        if ( (_DWORD)v134 )
        {
          v87 = v135;
          v132 = *v74;
          v88 = &v74[2 * v135];
          if ( v74 == v88 )
          {
            v90 = &v74[2 * v135];
          }
          else
          {
            while ( 1 )
            {
              v89 = *v74;
              v90 = v74;
              v74 += 2;
              if ( v89 != -16 && v89 != -8 )
                break;
              if ( v74 == v88 )
              {
                v132 = v90[2];
                v90 = v88;
                goto LABEL_157;
              }
            }
            v132 = v89;
          }
LABEL_157:
          v91 = *((_DWORD *)v90 + 2);
          v92 = v90 + 2;
          v139 = (const __m128i *)v141;
          v140 = 0x400000000LL;
          if ( v92 != v88 )
          {
            while ( 1 )
            {
              v93 = v92;
              if ( *v92 != -8 && *v92 != -16 )
                break;
              v92 += 2;
              if ( v88 == v92 )
                goto LABEL_161;
            }
            if ( v92 != v88 )
            {
              do
              {
                if ( *((_DWORD *)v93 + 2) >= v91 )
                {
                  if ( *((_DWORD *)v93 + 2) == v91 )
                  {
                    v95 = (unsigned int)v140;
                    if ( (unsigned int)v140 >= HIDWORD(v140) )
                    {
                      sub_16CD150((__int64)&v139, v141, 0, 8, v3, v4);
                      v95 = (unsigned int)v140;
                    }
                    v139->m128i_i64[v95] = *v93;
                    v87 = v135;
                    LODWORD(v140) = v140 + 1;
                  }
                  else
                  {
                    LODWORD(v140) = 0;
                    v91 = *((_DWORD *)v93 + 2);
                    v132 = *v93;
                  }
                }
                for ( v93 += 2; v88 != v93; v93 += 2 )
                {
                  if ( *v93 != -16 && *v93 != -8 )
                    break;
                }
              }
              while ( v93 != (__int64 *)(v133.m128i_i64[1] + 16 * v87) );
              v94 = (unsigned int)v140;
              if ( !(_DWORD)v140 )
              {
LABEL_174:
                if ( v139 != (const __m128i *)v141 )
                  _libc_free((unsigned __int64)v139);
                goto LABEL_161;
              }
              if ( (unsigned int)v140 >= HIDWORD(v140) )
              {
                sub_16CD150((__int64)&v139, v141, 0, 8, v3, v4);
                v94 = (unsigned int)v140;
              }
              v109 = 0;
              v139->m128i_i64[v94] = v132;
              LODWORD(v140) = v140 + 1;
              v110 = sub_157EBA0(a3);
              while ( 2 )
              {
                v111 = sub_15F4DF0(v110, v109);
                v112 = v139;
                v113 = 8LL * (unsigned int)v140;
                v114 = (const __m128i *)((char *)v139 + v113);
                v115 = v113 >> 5;
                if ( v115 )
                {
                  v116 = &v139[2 * v115];
                  while ( v111 != v112->m128i_i64[0] )
                  {
                    if ( v111 == v112->m128i_i64[1] )
                    {
                      v112 = (const __m128i *)((char *)v112 + 8);
                      goto LABEL_229;
                    }
                    if ( v111 == v112[1].m128i_i64[0] )
                    {
                      ++v112;
                      goto LABEL_229;
                    }
                    if ( v111 == v112[1].m128i_i64[1] )
                    {
                      v112 = (const __m128i *)((char *)v112 + 24);
                      goto LABEL_229;
                    }
                    v112 += 2;
                    if ( v116 == v112 )
                      goto LABEL_239;
                  }
                  break;
                }
LABEL_239:
                v117 = (char *)v114 - (char *)v112;
                if ( (char *)v114 - (char *)v112 == 16 )
                {
LABEL_257:
                  if ( v111 != v112->m128i_i64[0] )
                  {
                    v112 = (const __m128i *)((char *)v112 + 8);
LABEL_242:
                    if ( v111 != v112->m128i_i64[0] )
                    {
LABEL_243:
                      ++v109;
                      continue;
                    }
                  }
                }
                else
                {
                  if ( v117 != 24 )
                  {
                    if ( v117 == 8 )
                      goto LABEL_242;
                    goto LABEL_243;
                  }
                  if ( v111 != v112->m128i_i64[0] )
                  {
                    v112 = (const __m128i *)((char *)v112 + 8);
                    goto LABEL_257;
                  }
                }
                break;
              }
LABEL_229:
              if ( v114 != v112 )
              {
                v132 = sub_15F4DF0(v110, v109);
                goto LABEL_174;
              }
              goto LABEL_243;
            }
          }
LABEL_161:
          v74 = (__int64 *)v133.m128i_i64[1];
        }
        else
        {
          v132 = 0;
        }
        j___libc_free_0(v74);
        v36 = (unsigned int)v149;
LABEL_56:
        v37 = v148;
        v139 = (const __m128i *)v141;
        v38 = &v148[2 * v36];
        v140 = 0x1000000000LL;
        if ( v148 != v38 )
        {
LABEL_57:
          v39 = v132;
          v40 = v37;
          do
          {
            if ( v40[1] == v39 )
            {
              v43 = *v40;
              v127 = *v40;
              v44 = sub_157EBA0(*v40);
              if ( v44 )
              {
                v124 = sub_15F4D60(v44);
                v45 = sub_157EBA0(v43);
                v46 = v124;
                if ( v124 )
                {
                  v125 = v38;
                  v47 = v46;
                  v121 = v39;
                  v48 = 0;
                  do
                  {
                    while ( a3 != sub_15F4DF0(v45, v48) )
                    {
                      if ( ++v48 == v47 )
                        goto LABEL_85;
                    }
                    v51 = (unsigned int)v140;
                    if ( (unsigned int)v140 >= HIDWORD(v140) )
                    {
                      sub_16CD150((__int64)&v139, v141, 0, 8, v49, v50);
                      v51 = (unsigned int)v140;
                    }
                    ++v48;
                    v139->m128i_i64[v51] = v127;
                    LODWORD(v140) = v140 + 1;
                  }
                  while ( v48 != v47 );
LABEL_85:
                  v38 = v125;
                  v39 = v121;
                }
              }
            }
            v40 += 2;
          }
          while ( v38 != v40 );
        }
        if ( !v132 )
        {
          v85 = sub_157EBA0(a3);
          v86 = sub_1952280(a3);
          v132 = sub_15F4DF0(v85, v86);
        }
        v123 = sub_1958300(a1, a3, (__int64)&v139, v132);
        if ( v139 != (const __m128i *)v141 )
          _libc_free((unsigned __int64)v139);
        goto LABEL_64;
      }
      continue;
    }
  }
LABEL_65:
  if ( (_BYTE *)v41 != v150 )
    _libc_free(v41);
LABEL_67:
  if ( v144 != v143 )
    _libc_free((unsigned __int64)v144);
LABEL_69:
  if ( v136 != (__int64 *)v138 )
    _libc_free((unsigned __int64)v136);
  return v123;
}
