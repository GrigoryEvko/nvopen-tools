// Function: sub_F990C0
// Address: 0xf990c0
//
__int64 __fastcall sub_F990C0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 *a5, char a6)
{
  __int64 v8; // r13
  _BYTE *v9; // rax
  __int64 v10; // r12
  unsigned int v11; // r14d
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 *v16; // r15
  __int64 v17; // r9
  __int64 v18; // r8
  unsigned __int64 v19; // rax
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rcx
  __int64 v25; // r9
  bool v26; // r15
  __int64 v27; // r10
  __int64 v28; // rdx
  __int64 v29; // rax
  unsigned int v30; // eax
  __int64 v31; // rax
  int v32; // edx
  __int64 v33; // r15
  __int64 v34; // rbx
  __int64 v35; // rax
  __int64 v36; // rbx
  unsigned __int8 **v37; // rax
  unsigned __int8 *v38; // rdx
  unsigned __int8 v39; // cl
  __int64 v40; // rax
  _BYTE *v41; // rdi
  unsigned __int8 *v42; // rdx
  _BYTE *v43; // rdi
  unsigned __int8 *v44; // r15
  char v45; // al
  __int64 v46; // rdx
  char v47; // bl
  unsigned __int64 *v48; // rdx
  __int64 v49; // rax
  unsigned __int64 *v50; // r15
  __int64 v51; // r12
  char *v52; // rax
  char *v53; // rdx
  __int64 v54; // rax
  __int64 *v55; // rbx
  __int64 v56; // rax
  __int64 v57; // rcx
  unsigned __int64 *v58; // rax
  unsigned __int64 *v59; // rcx
  __int64 *v60; // r14
  __int64 v61; // rdx
  __int64 v62; // rax
  __int64 v63; // r14
  unsigned __int8 *v64; // r14
  __int64 v65; // rax
  __int64 v66; // r15
  unsigned __int8 *v67; // r15
  __int64 v68; // rcx
  __int64 v69; // r8
  __int64 v70; // r9
  int v71; // r15d
  unsigned int v72; // r13d
  __int64 v73; // r14
  __int64 v74; // r12
  unsigned int v75; // esi
  __int64 v76; // rax
  __int64 v77; // rcx
  __int64 v78; // r8
  __int64 v79; // r9
  char v80; // al
  int v81; // eax
  int v82; // eax
  bool v83; // zf
  char v84; // al
  signed __int64 v85; // rcx
  __int64 v86; // r15
  _BYTE *v87; // rax
  int v88; // eax
  char v89; // r15
  __int64 v90; // r14
  char v91; // r12
  unsigned int i; // ebx
  char *v93; // rax
  char v94; // al
  char v95; // al
  __int64 v96; // [rsp+0h] [rbp-200h]
  __int64 v97; // [rsp+20h] [rbp-1E0h]
  int v98; // [rsp+28h] [rbp-1D8h]
  unsigned __int8 v99; // [rsp+2Eh] [rbp-1D2h]
  unsigned __int8 v100; // [rsp+2Fh] [rbp-1D1h]
  __int64 v101; // [rsp+30h] [rbp-1D0h]
  __int64 v102; // [rsp+30h] [rbp-1D0h]
  unsigned int v103; // [rsp+38h] [rbp-1C8h]
  __int64 v104; // [rsp+38h] [rbp-1C8h]
  __int64 v106; // [rsp+48h] [rbp-1B8h]
  __int64 v108; // [rsp+50h] [rbp-1B0h]
  int v109; // [rsp+50h] [rbp-1B0h]
  unsigned __int8 *v111; // [rsp+58h] [rbp-1A8h]
  unsigned __int64 *v112; // [rsp+58h] [rbp-1A8h]
  __int64 v113; // [rsp+58h] [rbp-1A8h]
  unsigned __int8 *v114; // [rsp+58h] [rbp-1A8h]
  unsigned __int8 *v115; // [rsp+58h] [rbp-1A8h]
  unsigned __int8 *v116; // [rsp+58h] [rbp-1A8h]
  __int64 v117; // [rsp+60h] [rbp-1A0h] BYREF
  __int64 v118; // [rsp+68h] [rbp-198h] BYREF
  __int64 v119; // [rsp+70h] [rbp-190h] BYREF
  int v120; // [rsp+78h] [rbp-188h]
  unsigned __int64 *v121; // [rsp+80h] [rbp-180h] BYREF
  __int64 v122; // [rsp+88h] [rbp-178h]
  _BYTE v123[16]; // [rsp+90h] [rbp-170h] BYREF
  __int64 v124; // [rsp+A0h] [rbp-160h] BYREF
  unsigned int v125; // [rsp+A8h] [rbp-158h]
  int v126; // [rsp+B8h] [rbp-148h]
  __int64 v127; // [rsp+C0h] [rbp-140h] BYREF
  char *v128; // [rsp+C8h] [rbp-138h]
  __int64 v129; // [rsp+D0h] [rbp-130h]
  int v130; // [rsp+D8h] [rbp-128h]
  char v131; // [rsp+DCh] [rbp-124h]
  char v132; // [rsp+E0h] [rbp-120h] BYREF
  __int64 *v133; // [rsp+100h] [rbp-100h] BYREF
  __int64 v134; // [rsp+108h] [rbp-F8h]
  _BYTE v135[16]; // [rsp+110h] [rbp-F0h] BYREF
  __int16 v136; // [rsp+120h] [rbp-E0h]
  __int64 *v137; // [rsp+140h] [rbp-C0h] BYREF
  __int64 v138; // [rsp+148h] [rbp-B8h]
  _QWORD v139[4]; // [rsp+150h] [rbp-B0h] BYREF
  __int64 v140; // [rsp+170h] [rbp-90h]
  __int64 v141; // [rsp+178h] [rbp-88h]
  __int16 v142; // [rsp+180h] [rbp-80h]
  __int64 v143; // [rsp+188h] [rbp-78h]
  void **v144; // [rsp+190h] [rbp-70h]
  void **v145; // [rsp+198h] [rbp-68h]
  __int64 v146; // [rsp+1A0h] [rbp-60h]
  int v147; // [rsp+1A8h] [rbp-58h]
  __int16 v148; // [rsp+1ACh] [rbp-54h]
  char v149; // [rsp+1AEh] [rbp-52h]
  __int64 v150; // [rsp+1B0h] [rbp-50h]
  __int64 v151; // [rsp+1B8h] [rbp-48h]
  void *v152; // [rsp+1C0h] [rbp-40h] BYREF
  void *v153; // [rsp+1C8h] [rbp-38h] BYREF

  v8 = *(_QWORD *)(a1 + 40);
  v9 = sub_F35A80(v8, &v117, &v118);
  if ( !v9 )
    return 0;
  v10 = (__int64)v9;
  v106 = *((_QWORD *)v9 - 12);
  if ( *(_BYTE *)v106 == 17 )
    return 0;
  v100 = sub_DF9950((__int64)a2);
  if ( !v100 )
    return 0;
  v13 = *(_QWORD *)(a1 - 8);
  v96 = *(_QWORD *)(v10 + 40);
  v14 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  v121 = (unsigned __int64 *)v123;
  v122 = 0x200000000LL;
  v15 = 32LL * *(unsigned int *)(a1 + 72);
  v16 = (__int64 *)(v13 + v15);
  v17 = v13 + v15 + 8 * v14;
  if ( v17 != v13 + v15 )
  {
    do
    {
      while ( 1 )
      {
        v18 = *v16;
        v19 = *(_QWORD *)(*v16 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v19 == *v16 + 48 )
          goto LABEL_173;
        if ( !v19 )
          BUG();
        if ( (unsigned int)*(unsigned __int8 *)(v19 - 24) - 30 > 0xA )
LABEL_173:
          BUG();
        if ( (*(_DWORD *)(v19 - 20) & 0x7FFFFFF) == 1 )
          break;
        if ( (__int64 *)v17 == ++v16 )
          goto LABEL_16;
      }
      v20 = (unsigned int)v122;
      v21 = (unsigned int)v122 + 1LL;
      if ( v21 > HIDWORD(v122) )
      {
        v97 = v17;
        v102 = *v16;
        sub_C8D5F0((__int64)&v121, v123, v21, 8u, v18, v17);
        v20 = (unsigned int)v122;
        v17 = v97;
        v18 = v102;
      }
      ++v16;
      v121[v20] = v18;
      LODWORD(v122) = v122 + 1;
    }
    while ( (__int64 *)v17 != v16 );
  }
LABEL_16:
  if ( (*(_BYTE *)(v10 + 7) & 0x20) == 0 )
  {
    v26 = 0;
    goto LABEL_25;
  }
  v22 = 15;
  v23 = sub_B91C10(v10, 15);
  v26 = v23 != 0;
  if ( !v23 )
  {
LABEL_25:
    v22 = (unsigned __int64)&v133;
    if ( (unsigned __int8)sub_BC8C50(v10, &v133, &v137) )
    {
      v22 = (unsigned __int64)v133 + (_QWORD)v137;
      if ( (__int64 *)((char *)v133 + (_QWORD)v137) )
      {
        v103 = sub_F02DD0((unsigned __int64)v133, v22);
        v24 = (unsigned int)sub_DF95A0(a2);
        v30 = 0x80000000 - v103;
        if ( (_DWORD)v122 == 1 )
        {
          if ( *(_QWORD *)(v10 - 32) == v8 )
            v30 = v103;
          if ( (unsigned int)v24 <= v30 )
            goto LABEL_31;
        }
        else
        {
          if ( v30 < v103 )
            v30 = v103;
          if ( v30 >= (unsigned int)v24 )
            goto LABEL_31;
        }
      }
    }
  }
  if ( *(_BYTE *)v106 == 84 && v8 == *(_QWORD *)(v106 + 40) )
  {
LABEL_31:
    v11 = 0;
    goto LABEL_32;
  }
  v27 = *(_QWORD *)(v8 + 56);
  v28 = 4;
  v29 = v27;
  if ( !v27 )
LABEL_177:
    BUG();
  while ( *(_BYTE *)(v29 - 24) == 84 )
  {
    v28 = (unsigned int)(v28 - 1);
    if ( !(_DWORD)v28 )
      goto LABEL_31;
    v29 = *(_QWORD *)(v29 + 8);
    if ( !v29 )
      goto LABEL_177;
  }
  v127 = 0;
  v128 = &v132;
  v129 = 4;
  v130 = 0;
  v131 = 1;
  v119 = 0;
  v120 = 0;
  v104 = (unsigned int)qword_4F8D268;
  if ( a6 && v26 )
  {
    v31 = sub_DF96E0((__int64)a2);
    v24 = v32 == 1;
    v28 = v31 + v104;
    v98 = v24;
    if ( __OFADD__(v31, v104) )
    {
      v27 = *(_QWORD *)(v8 + 56);
      if ( v31 <= 0 )
        v104 = 0x8000000000000000LL;
      else
        v104 = 0x7FFFFFFFFFFFFFFFLL;
    }
    else
    {
      v104 += v31;
      v27 = *(_QWORD *)(v8 + 56);
    }
  }
  else
  {
    v98 = 0;
  }
  v101 = (__int64)a2;
  v33 = v27;
  v99 = 0;
  while ( 1 )
  {
    if ( !v33 )
      BUG();
    if ( *(_BYTE *)(v33 - 24) != 84 )
      break;
    v34 = *(_QWORD *)(v33 + 8);
    v142 = 257;
    v137 = a5;
    v138 = 0;
    memset(v139, 0, 24);
    v139[3] = v33 - 24;
    v140 = 0;
    v141 = 0;
    v35 = sub_1020E10(v33 - 24, &v137, v28, v24, 257, v25);
    if ( v35 )
    {
      v22 = v35;
      sub_BD84D0(v33 - 24, v35);
      sub_B43D60((_QWORD *)(v33 - 24));
      v99 = v100;
    }
    else
    {
      v22 = v8;
      if ( !(unsigned __int8)sub_F947D0(
                               **(_QWORD **)(v33 - 32),
                               v8,
                               v10,
                               (__int64)&v127,
                               (__int64)&v119,
                               v104,
                               v98,
                               v101,
                               a4,
                               0)
        || (v22 = v8,
            !(unsigned __int8)sub_F947D0(
                                *(_QWORD *)(*(_QWORD *)(v33 - 32) + 32LL),
                                v8,
                                v10,
                                (__int64)&v127,
                                (__int64)&v119,
                                v104,
                                v98,
                                v101,
                                a4,
                                0)) )
      {
        v11 = v99;
        goto LABEL_50;
      }
    }
    v33 = v34;
  }
  v36 = *(_QWORD *)(v8 + 56);
  v11 = v99;
  if ( !v36 )
    BUG();
  if ( *(_BYTE *)(v36 - 24) != 84 )
    goto LABEL_115;
  v22 = 1;
  if ( !sub_BCAC40(*(_QWORD *)(v36 - 16), 1) )
    goto LABEL_73;
  v37 = *(unsigned __int8 ***)(v36 - 32);
  v38 = *v37;
  v39 = **v37;
  if ( v39 <= 0x1Cu )
    goto LABEL_63;
  v22 = (unsigned int)v39 - 42;
  if ( (unsigned int)v22 <= 0x11 )
  {
LABEL_66:
    v44 = v37[4];
    v137 = 0;
    if ( *v38 == 59 )
    {
      v22 = (unsigned __int64)v38;
      v114 = v38;
      v80 = sub_F90BD0(&v137, (__int64)v38);
      v38 = v114;
      if ( v80 )
      {
        v38 = v44;
        v44 = v114;
      }
    }
    v137 = 0;
    v139[0] = 0;
    v133 = 0;
    if ( *v44 != 59 )
      goto LABEL_50;
    v22 = (unsigned __int64)v44;
    v111 = v38;
    v45 = sub_F90BD0(&v133, (__int64)v44);
    v46 = (__int64)v111;
    v47 = v45;
    if ( !v45 )
      goto LABEL_50;
    if ( *v111 != 59
      || (v22 = (unsigned __int64)v111, v95 = sub_F90BD0(&v137, (__int64)v111), v46 = (__int64)v111, !v95) )
    {
      if ( *(_BYTE *)v46 != 17 )
      {
        v86 = *(_QWORD *)(v46 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v86 + 8) - 17 > 1 || *(_BYTE *)v46 > 0x15u )
          goto LABEL_50;
        v22 = 0;
        v116 = (unsigned __int8 *)v46;
        v87 = sub_AD7630(v46, 0, v46);
        v46 = (__int64)v116;
        if ( !v87 || *v87 != 17 )
        {
          if ( *(_BYTE *)(v86 + 8) != 17 )
            goto LABEL_50;
          v88 = *(_DWORD *)(v86 + 32);
          v89 = 0;
          v90 = v10;
          v91 = v47;
          v109 = v88;
          for ( i = 0; v109 != i; ++i )
          {
            v22 = i;
            v93 = (char *)sub_AD69F0(v116, i);
            if ( !v93 )
              goto LABEL_164;
            v94 = *v93;
            if ( v94 != 13 )
            {
              if ( v94 != 17 )
              {
LABEL_164:
                v11 = v99;
                goto LABEL_50;
              }
              v89 = v91;
            }
          }
          v10 = v90;
          v46 = (__int64)v116;
          v11 = v99;
          if ( !v89 )
            goto LABEL_50;
        }
      }
      if ( v139[0] )
        *(_QWORD *)v139[0] = v46;
    }
    goto LABEL_73;
  }
  if ( v39 != 86 )
    goto LABEL_63;
  if ( (v38[7] & 0x40) != 0 )
  {
    v40 = *((_QWORD *)v38 - 1);
    v41 = *(_BYTE **)(v40 + 32);
    if ( *v41 > 0x15u || *v41 == 5 )
      goto LABEL_60;
  }
  else
  {
    v82 = *((_DWORD *)v38 + 1);
    v22 = (unsigned __int64)&v38[-32 * (v82 & 0x7FFFFFF)];
    v41 = *(_BYTE **)(v22 + 32);
    if ( *v41 > 0x15u || *v41 == 5 )
      goto LABEL_126;
  }
  v115 = v38;
  v84 = sub_AD6CA0((__int64)v41);
  v38 = v115;
  if ( !v84 )
  {
LABEL_64:
    v37 = *(unsigned __int8 ***)(v36 - 32);
LABEL_65:
    v38 = *v37;
    goto LABEL_66;
  }
  if ( (v115[7] & 0x40) == 0 )
  {
    v82 = *((_DWORD *)v115 + 1);
LABEL_126:
    v42 = &v38[-32 * (v82 & 0x7FFFFFF)];
    goto LABEL_61;
  }
  v40 = *((_QWORD *)v115 - 1);
LABEL_60:
  v42 = (unsigned __int8 *)v40;
LABEL_61:
  v43 = (_BYTE *)*((_QWORD *)v42 + 8);
  if ( *v43 > 0x15u || *v43 == 5 )
  {
    v37 = *(unsigned __int8 ***)(v36 - 32);
  }
  else
  {
    v83 = (unsigned __int8)sub_AD6CA0((__int64)v43) == 0;
    v37 = *(unsigned __int8 ***)(v36 - 32);
    if ( v83 )
      goto LABEL_65;
  }
LABEL_63:
  if ( (unsigned __int8)sub_F91130(v37[4]) || (unsigned __int8)sub_F91130((unsigned __int8 *)v106) )
    goto LABEL_64;
LABEL_73:
  v48 = v121;
  v49 = (unsigned int)v122;
  v112 = &v121[(unsigned int)v122];
  if ( v112 == v121 )
    goto LABEL_89;
  v108 = v10;
  v50 = v121;
  do
  {
    v51 = *(_QWORD *)(*v50 + 56);
    while ( 1 )
    {
      if ( !v51 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v51 - 24) - 30 <= 0xA )
        break;
      if ( v131 )
      {
        v52 = v128;
        v53 = &v128[8 * HIDWORD(v129)];
        if ( v128 != v53 )
        {
          while ( v51 - 24 != *(_QWORD *)v52 )
          {
            v52 += 8;
            if ( v53 == v52 )
              goto LABEL_85;
          }
          goto LABEL_83;
        }
LABEL_85:
        if ( !sub_B46AA0(v51 - 24) )
          goto LABEL_50;
        v51 = *(_QWORD *)(v51 + 8);
      }
      else
      {
        v22 = v51 - 24;
        if ( !sub_C8CA60((__int64)&v127, v51 - 24) )
          goto LABEL_85;
LABEL_83:
        v51 = *(_QWORD *)(v51 + 8);
      }
    }
    ++v50;
  }
  while ( v112 != v50 );
  v10 = v108;
  v49 = (unsigned int)v122;
  v48 = v121;
LABEL_89:
  v54 = v49;
  v55 = (__int64 *)&v48[v54];
  v56 = (v54 * 8) >> 5;
  if ( !v56 )
  {
    v58 = v48;
LABEL_138:
    v85 = (char *)v55 - (char *)v58;
    if ( (char *)v55 - (char *)v58 != 16 )
    {
      if ( v85 != 24 )
      {
        if ( v85 != 8 )
          goto LABEL_97;
        goto LABEL_141;
      }
      if ( (*(_WORD *)(*v58 + 2) & 0x7FFF) != 0 )
        goto LABEL_96;
      ++v58;
    }
    if ( (*(_WORD *)(*v58 + 2) & 0x7FFF) != 0 )
      goto LABEL_96;
    ++v58;
LABEL_141:
    if ( (*(_WORD *)(*v58 + 2) & 0x7FFF) != 0 )
      goto LABEL_96;
    goto LABEL_97;
  }
  v57 = 4 * v56;
  v58 = v48;
  v59 = &v48[v57];
  while ( 1 )
  {
    v22 = *v58;
    if ( (*(_WORD *)(*v58 + 2) & 0x7FFF) != 0 )
      break;
    v22 = v58[1];
    if ( (*(_WORD *)(v22 + 2) & 0x7FFF) != 0 )
    {
      ++v58;
      break;
    }
    v22 = v58[2];
    if ( (*(_WORD *)(v22 + 2) & 0x7FFF) != 0 )
    {
      v58 += 2;
      break;
    }
    v22 = v58[3];
    if ( (*(_WORD *)(v22 + 2) & 0x7FFF) != 0 )
    {
      v58 += 3;
      break;
    }
    v58 += 4;
    if ( v59 == v58 )
      goto LABEL_138;
  }
LABEL_96:
  if ( v55 == (__int64 *)v58 )
  {
LABEL_97:
    v60 = (__int64 *)v48;
    if ( v48 != (unsigned __int64 *)v55 )
    {
      do
      {
        v61 = *v60++;
        sub_F57C50(v96, v10, v61);
      }
      while ( v55 != v60 );
    }
    v62 = sub_BD5C60(v10);
    v149 = 7;
    v143 = v62;
    v144 = &v152;
    v145 = &v153;
    v137 = v139;
    v152 = &unk_49DA1B0;
    v148 = 512;
    v142 = 0;
    v153 = &unk_49DA0B0;
    v138 = 0x200000000LL;
    v146 = 0;
    v147 = 0;
    v150 = 0;
    v151 = 0;
    v140 = 0;
    v141 = 0;
    sub_D5F1F0((__int64)&v137, v10);
    v63 = *(_QWORD *)(v8 + 56);
    if ( !v63 )
LABEL_172:
      BUG();
    while ( *(_BYTE *)(v63 - 24) == 84 )
    {
      v64 = (unsigned __int8 *)(v63 - 24);
      v113 = sub_F0A930((__int64)v64, v117);
      v65 = sub_F0A930((__int64)v64, v118);
      v136 = 257;
      v66 = v65;
      if ( (unsigned __int8)sub_920620((__int64)v64) )
      {
        v81 = sub_B45210((__int64)v64);
        BYTE4(v124) = 1;
        LODWORD(v124) = v81;
      }
      else
      {
        BYTE4(v124) = 0;
      }
      v67 = (unsigned __int8 *)sub_B36280((unsigned int **)&v137, v106, v113, v66, v124, (__int64)&v133, v10);
      sub_BD84D0((__int64)v64, (__int64)v67);
      sub_BD6B90(v67, v64);
      sub_B43D60(v64);
      v63 = *(_QWORD *)(v8 + 56);
      if ( !v63 )
        goto LABEL_172;
    }
    v22 = v8;
    sub_F902B0((__int64 *)&v137, v8);
    v133 = (__int64 *)v135;
    v134 = 0x300000000LL;
    if ( a3 )
    {
      sub_F35FA0((__int64)&v133, v96, v8 & 0xFFFFFFFFFFFFFFFBLL, v68, v69, v70);
      sub_F34070((__int64)&v124, v96);
      v71 = v126;
      v72 = v125;
      if ( v125 != v126 )
      {
        v73 = v10;
        v74 = v124;
        do
        {
          v75 = v72++;
          v76 = sub_B46EC0(v74, v75);
          sub_F35FA0((__int64)&v133, v96, v76 | 4, v77, v78, v79);
        }
        while ( v71 != v72 );
        v10 = v73;
      }
      sub_B43D60((_QWORD *)v10);
      v22 = (unsigned __int64)v133;
      sub_FFB3D0(a3, v133, (unsigned int)v134);
    }
    else
    {
      sub_B43D60((_QWORD *)v10);
    }
    if ( v133 != (__int64 *)v135 )
      _libc_free(v133, v22);
    nullsub_61();
    v152 = &unk_49DA1B0;
    nullsub_63();
    if ( v137 != v139 )
      _libc_free(v137, v22);
LABEL_115:
    v11 = v100;
  }
LABEL_50:
  if ( !v131 )
    _libc_free(v128, v22);
LABEL_32:
  if ( v121 != (unsigned __int64 *)v123 )
    _libc_free(v121, v22);
  return v11;
}
