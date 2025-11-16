// Function: sub_1714690
// Address: 0x1714690
//
__int64 __fastcall sub_1714690(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 *v4; // rcx
  int v5; // r8d
  int v6; // r9d
  int v7; // r15d
  __int16 v8; // ax
  __int64 v9; // r14
  __int64 v10; // rsi
  int v11; // ebx
  __int64 v12; // r12
  __int64 v13; // r13
  bool v14; // zf
  unsigned __int64 i; // rbx
  char v16; // dl
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rsi
  __int64 *v21; // rax
  __int64 *v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rsi
  unsigned int v25; // r12d
  char v26; // r9
  unsigned int v27; // edi
  unsigned int v28; // eax
  __int64 v29; // rdx
  unsigned int v30; // ebx
  unsigned int v31; // eax
  __int64 *v32; // rcx
  __int64 v33; // r8
  __int64 v34; // rdx
  unsigned int v35; // edx
  unsigned __int64 v36; // rax
  __int64 v37; // r14
  __int64 v38; // rsi
  __int64 v39; // r15
  unsigned int v40; // r12d
  __int64 v41; // rcx
  char v42; // dl
  __int64 v43; // r13
  unsigned int v44; // r14d
  __int64 v45; // rbx
  int v46; // eax
  _BYTE *v47; // rdx
  __int64 v48; // r13
  __int64 *v49; // rax
  __int64 *v50; // rsi
  bool v51; // al
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rbx
  __int64 v55; // rax
  char v56; // bl
  __int64 *v57; // rax
  __int128 v58; // rdi
  __int64 v59; // rcx
  __int64 v60; // rax
  __int64 v61; // rax
  bool v62; // al
  char v63; // bl
  _BYTE *v64; // rdx
  int v65; // eax
  void *v66; // rdi
  char *v67; // rsi
  char *v68; // rax
  int v69; // edx
  __int64 v70; // rax
  __int64 v71; // rdx
  __int64 *v72; // r14
  __int64 v73; // rax
  __int64 *v74; // r9
  __int64 v75; // rcx
  __int64 v76; // r15
  char *v77; // rax
  char *v78; // r13
  __int64 v79; // r12
  __int64 v81; // rax
  __int64 v82; // rbx
  __int64 v83; // rdx
  __int64 v84; // rcx
  __int64 v85; // r8
  __int64 v86; // r9
  __int64 v87; // r13
  __int64 v88; // rsi
  __int16 v89; // ax
  __int16 v90; // dx
  __int16 v91; // cx
  __int64 v92; // [rsp+18h] [rbp-2F8h]
  unsigned __int8 v93; // [rsp+26h] [rbp-2EAh]
  char v94; // [rsp+27h] [rbp-2E9h]
  int dest; // [rsp+38h] [rbp-2D8h]
  char *desta; // [rsp+38h] [rbp-2D8h]
  int j; // [rsp+40h] [rbp-2D0h]
  __int64 v99; // [rsp+48h] [rbp-2C8h]
  _BYTE *src; // [rsp+50h] [rbp-2C0h]
  __int64 *srca; // [rsp+50h] [rbp-2C0h]
  __int64 v102; // [rsp+58h] [rbp-2B8h]
  unsigned int v103; // [rsp+58h] [rbp-2B8h]
  __int64 v104; // [rsp+58h] [rbp-2B8h]
  _BYTE *v105; // [rsp+60h] [rbp-2B0h] BYREF
  __int64 v106; // [rsp+68h] [rbp-2A8h]
  _BYTE v107[128]; // [rsp+70h] [rbp-2A0h] BYREF
  _BYTE *v108; // [rsp+F0h] [rbp-220h] BYREF
  __int64 v109; // [rsp+F8h] [rbp-218h]
  _BYTE v110[128]; // [rsp+100h] [rbp-210h] BYREF
  __int64 v111; // [rsp+180h] [rbp-190h] BYREF
  __int64 *v112; // [rsp+188h] [rbp-188h]
  __int64 *v113; // [rsp+190h] [rbp-180h]
  __int64 v114; // [rsp+198h] [rbp-178h]
  int v115; // [rsp+1A0h] [rbp-170h]
  _BYTE v116[136]; // [rsp+1A8h] [rbp-168h] BYREF
  __int64 v117; // [rsp+230h] [rbp-E0h] BYREF
  __int64 *v118; // [rsp+238h] [rbp-D8h]
  __int64 *v119; // [rsp+240h] [rbp-D0h]
  __int64 v120; // [rsp+248h] [rbp-C8h]
  int v121; // [rsp+250h] [rbp-C0h]
  _BYTE v122[184]; // [rsp+258h] [rbp-B8h] BYREF

  v2 = sub_15E38F0(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 56LL));
  v111 = 0;
  v114 = 16;
  v7 = sub_14DD7D0(v2);
  v105 = v107;
  v106 = 0x1000000000LL;
  v8 = *(_WORD *)(a2 + 18);
  v115 = 0;
  v93 = v8 & 1;
  v112 = (__int64 *)v116;
  v113 = (__int64 *)v116;
  dest = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( !dest )
  {
    v79 = 0;
    goto LABEL_131;
  }
  v94 = 0;
  v9 = 0;
  while ( 2 )
  {
    v10 = a2;
    v11 = v9 + 1;
    if ( (*(_BYTE *)(a2 + 23) & 0x40) == 0 )
    {
      v10 = a2;
      v3 = v9 - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      v12 = *(_QWORD *)(a2 + 24 * v3);
      v99 = *(_QWORD *)v12;
      if ( *(_BYTE *)(*(_QWORD *)v12 + 8LL) == 14 )
        goto LABEL_5;
LABEL_67:
      v48 = sub_1649C60(v12);
      v49 = v112;
      if ( v113 != v112 )
        goto LABEL_68;
      v50 = &v112[HIDWORD(v114)];
      if ( v112 != v50 )
      {
        v4 = 0;
        while ( 1 )
        {
          v3 = *v49;
          if ( v48 == *v49 )
            goto LABEL_69;
          if ( v3 == -2 )
            v4 = v49;
          if ( v50 == ++v49 )
          {
            if ( !v4 )
              break;
            *v4 = v48;
            --v115;
            ++v111;
            goto LABEL_97;
          }
        }
      }
      if ( HIDWORD(v114) < (unsigned int)v114 )
      {
        ++HIDWORD(v114);
        *v50 = v48;
        v61 = (unsigned int)v106;
        ++v111;
        if ( (unsigned int)v106 >= HIDWORD(v106) )
        {
LABEL_139:
          v50 = (__int64 *)v107;
          sub_16CD150((__int64)&v105, v107, 0, 8, v5, v6);
          v61 = (unsigned int)v106;
        }
      }
      else
      {
LABEL_68:
        v50 = (__int64 *)v48;
        sub_16CCBA0((__int64)&v111, v48);
        if ( !(_BYTE)v3 )
        {
LABEL_69:
          v94 = 1;
          goto LABEL_70;
        }
LABEL_97:
        v61 = (unsigned int)v106;
        if ( (unsigned int)v106 >= HIDWORD(v106) )
          goto LABEL_139;
      }
      v3 = (__int64)v105;
      *(_QWORD *)&v105[8 * v61] = v12;
      LODWORD(v106) = v106 + 1;
LABEL_70:
      if ( v7 == 11 )
        goto LABEL_34;
      if ( v7 < 4 )
        goto LABEL_34;
      v51 = sub_1593BB0(v48, (__int64)v50, v3, (__int64)v4);
      if ( !v51 )
        goto LABEL_34;
      v25 = v106;
      if ( v11 == dest )
        goto LABEL_147;
      if ( (unsigned int)v106 > 1 )
      {
        v94 = v51;
        v93 = 0;
        goto LABEL_37;
      }
      v93 = 0;
      goto LABEL_149;
    }
    v12 = *(_QWORD *)(*(_QWORD *)(a2 - 8) + 24 * v9);
    v99 = *(_QWORD *)v12;
    if ( *(_BYTE *)(*(_QWORD *)v12 + 8LL) != 14 )
      goto LABEL_67;
LABEL_5:
    v13 = *(_QWORD *)(v99 + 32);
    if ( !(_DWORD)v13 )
    {
      v81 = (unsigned int)v106;
      if ( (unsigned int)v106 >= HIDWORD(v106) )
      {
        sub_16CD150((__int64)&v105, v107, 0, 8, v5, v6);
        v81 = (unsigned int)v106;
      }
      *(_QWORD *)&v105[8 * v81] = v12;
      v25 = v106 + 1;
      LODWORD(v106) = v106 + 1;
      if ( v11 == dest )
        goto LABEL_147;
      v93 = 0;
      if ( v25 > 1 )
      {
        v94 = 1;
        goto LABEL_37;
      }
      goto LABEL_149;
    }
    v14 = *(_BYTE *)(v12 + 16) == 10;
    v108 = v110;
    v109 = 0x1000000000LL;
    if ( v14 )
    {
      v52 = sub_15A06D0(*(__int64 ***)(v99 + 24), v10, v3, (__int64)v4);
      v54 = v52;
      if ( v7 == 11 || v7 < 4 )
      {
        v55 = (unsigned int)v109;
        if ( (unsigned int)v109 < HIDWORD(v109) )
          goto LABEL_79;
      }
      else
      {
        if ( sub_1593BB0(v52, v10, v53, (__int64)v4) )
        {
LABEL_31:
          if ( v108 != v110 )
            _libc_free((unsigned __int64)v108);
          v94 = 1;
          goto LABEL_34;
        }
        v55 = (unsigned int)v109;
        if ( (unsigned int)v109 < HIDWORD(v109) )
        {
LABEL_79:
          *(_QWORD *)&v108[8 * v55] = v54;
          v56 = 0;
          v24 = (unsigned int)(v109 + 1);
          LODWORD(v109) = v109 + 1;
          if ( (_DWORD)v13 != 1 )
            goto LABEL_80;
          goto LABEL_81;
        }
      }
      sub_16CD150((__int64)&v108, v110, 0, 8, v5, v6);
      v55 = (unsigned int)v109;
      goto LABEL_79;
    }
    v117 = 0;
    v118 = (__int64 *)v122;
    v119 = (__int64 *)v122;
    v120 = 16;
    v121 = 0;
    v102 = (unsigned int)v13;
    if ( (unsigned int)v13 > 0x10 )
      sub_16CD150((__int64)&v108, v110, (unsigned int)v13, 8, v5, v6);
    for ( i = 0; i != v102; ++i )
    {
      while ( 1 )
      {
        v17 = *(_QWORD *)(v12 + 24 * (i - (*(_DWORD *)(v12 + 20) & 0xFFFFFFF)));
        v18 = sub_1649C60(v17);
        v20 = v18;
        if ( v7 != 11 && v7 >= 4 )
        {
          v20 = v18;
          if ( sub_1593BB0(v18, v18, v19, (__int64)v4) )
          {
            if ( v119 != v118 )
              _libc_free((unsigned __int64)v119);
            goto LABEL_31;
          }
        }
        v21 = v118;
        if ( v119 != v118 )
          goto LABEL_10;
        v22 = &v118[HIDWORD(v120)];
        v5 = HIDWORD(v120);
        if ( v118 != v22 )
        {
          v4 = 0;
          do
          {
            if ( v20 == *v21 )
              goto LABEL_11;
            if ( *v21 == -2 )
              v4 = v21;
            ++v21;
          }
          while ( v22 != v21 );
          if ( v4 )
            break;
        }
        if ( HIDWORD(v120) < (unsigned int)v120 )
        {
          v5 = ++HIDWORD(v120);
          *v22 = v20;
          v23 = (unsigned int)v109;
          ++v117;
          if ( (unsigned int)v109 < HIDWORD(v109) )
            goto LABEL_24;
          goto LABEL_88;
        }
LABEL_10:
        sub_16CCBA0((__int64)&v117, v20);
        if ( v16 )
          goto LABEL_23;
LABEL_11:
        if ( ++i == v102 )
          goto LABEL_25;
      }
      *v4 = v20;
      --v121;
      ++v117;
LABEL_23:
      v23 = (unsigned int)v109;
      if ( (unsigned int)v109 < HIDWORD(v109) )
        goto LABEL_24;
LABEL_88:
      sub_16CD150((__int64)&v108, v110, 0, 8, v5, v6);
      v23 = (unsigned int)v109;
LABEL_24:
      *(_QWORD *)&v108[8 * v23] = v17;
      LODWORD(v109) = v109 + 1;
    }
LABEL_25:
    v24 = (unsigned int)v109;
    if ( (unsigned int)v109 >= i )
    {
      if ( v119 == v118 )
      {
        v56 = 0;
        goto LABEL_81;
      }
      _libc_free((unsigned __int64)v119);
      v56 = 0;
      v60 = (unsigned int)v106;
      if ( (unsigned int)v106 < HIDWORD(v106) )
        goto LABEL_82;
LABEL_101:
      sub_16CD150((__int64)&v105, v107, 0, 8, v5, v6);
      v60 = (unsigned int)v106;
      goto LABEL_82;
    }
    if ( v119 != v118 )
    {
      _libc_free((unsigned __int64)v119);
      v24 = (unsigned int)v109;
    }
LABEL_80:
    v56 = 1;
    v57 = sub_1645D80(*(__int64 **)(v99 + 24), v24);
    *((_QWORD *)&v58 + 1) = v108;
    *(_QWORD *)&v58 = v57;
    v94 = 1;
    v12 = sub_159DFD0(v58, (unsigned int)v109, v59);
LABEL_81:
    v60 = (unsigned int)v106;
    if ( (unsigned int)v106 >= HIDWORD(v106) )
      goto LABEL_101;
LABEL_82:
    v3 = (__int64)v105;
    *(_QWORD *)&v105[8 * v60] = v12;
    v25 = v106 + 1;
    LODWORD(v106) = v106 + 1;
    if ( !v56 || (_DWORD)v109 )
    {
      if ( v108 != v110 )
        _libc_free((unsigned __int64)v108);
LABEL_34:
      if ( dest == (_DWORD)++v9 )
      {
        v25 = v106;
        goto LABEL_36;
      }
      continue;
    }
    break;
  }
  if ( v108 != v110 )
  {
    _libc_free((unsigned __int64)v108);
    v25 = v106;
  }
LABEL_147:
  v93 = 0;
LABEL_36:
  if ( v25 > 1 )
  {
LABEL_37:
    v26 = v94;
    v27 = 0;
LABEL_38:
    while ( 2 )
    {
      if ( v27 == v25 )
      {
        v30 = v27;
        v31 = v27;
      }
      else
      {
        v28 = v27;
        do
        {
          v29 = v28;
          v30 = v28++;
          if ( *(_BYTE *)(**(_QWORD **)&v105[8 * v29] + 8LL) != 14 )
          {
            v31 = v27;
            goto LABEL_43;
          }
        }
        while ( v25 != v28 );
        v30 = v25;
        v31 = v27;
      }
LABEL_43:
      v32 = (__int64 *)&v105[8 * v27 + 8];
      do
      {
        v34 = v31++;
        if ( v31 >= v30 )
        {
          v27 = v30 + 1;
          if ( v30 + 2 < v25 )
            goto LABEL_38;
          goto LABEL_47;
        }
        v33 = *v32++;
      }
      while ( *(_QWORD *)(*(_QWORD *)v33 + 32LL) >= *(_QWORD *)(**(_QWORD **)&v105[8 * v34] + 32LL) );
      v70 = 8LL * v30;
      v71 = 8LL * v27;
      v72 = (__int64 *)&v105[v70];
      v73 = v70 - v71;
      v74 = (__int64 *)&v105[v71];
      v75 = v73 >> 3;
      if ( v73 <= 0 )
      {
LABEL_127:
        v76 = 0;
        sub_1713E20(v74, v72, (__int64 (__fastcall *)(__int64, __int64))sub_1704280);
        v78 = 0;
      }
      else
      {
        while ( 1 )
        {
          v76 = 8 * v75;
          srca = v74;
          v104 = v75;
          v77 = (char *)sub_2207800(8 * v75, &unk_435FF63);
          v74 = srca;
          v78 = v77;
          if ( v77 )
            break;
          v75 = v104 >> 1;
          if ( !(v104 >> 1) )
            goto LABEL_127;
        }
        sub_1714590(srca, v72, v77, (void *)v104, (__int64 (__fastcall *)(__int64, __int64))sub_1704280);
      }
      j_j___libc_free_0(v78, v76);
      v27 = v30 + 1;
      v26 = 1;
      if ( v30 + 2 < v25 )
        continue;
      break;
    }
LABEL_47:
    v25 = v106;
    v94 = v26;
    v35 = v106;
    if ( (unsigned int)v106 > 1 )
    {
      for ( j = 0; ; ++j )
      {
        v36 = (unsigned __int64)v105;
        v37 = *(_QWORD *)&v105[8 * j];
        if ( *(_BYTE *)(*(_QWORD *)v37 + 8LL) == 14 )
        {
          v92 = *(_QWORD *)(*(_QWORD *)v37 + 32LL);
          v103 = v35 - 1;
          if ( v35 - 1 != j )
            break;
        }
LABEL_112:
        v25 = v35;
        if ( v35 <= j + 2 )
          goto LABEL_128;
      }
      src = *(_BYTE **)&v105[8 * j];
      while ( 2 )
      {
        v38 = v36 + 8LL * v103;
        v39 = *(_QWORD *)v38;
        desta = (char *)v38;
        if ( *(_BYTE *)(**(_QWORD **)v38 + 8LL) == 14 )
        {
          if ( !(_DWORD)v92 )
          {
            v66 = (void *)(v36 + 8LL * v103);
            v67 = (char *)(v38 + 8);
            v68 = (char *)(v36 + 8LL * (unsigned int)v106);
            v69 = v106;
            if ( v68 == v67 )
              goto LABEL_116;
            goto LABEL_115;
          }
          v40 = *(_QWORD *)(*(_QWORD *)v39 + 32LL);
          if ( (unsigned int)v92 <= v40 )
          {
            v41 = (__int64)src;
            v42 = src[16];
            if ( *(_BYTE *)(v39 + 16) != 10 )
            {
              v43 = 0;
              if ( v42 == 10 )
              {
                while ( 1 )
                {
                  v62 = sub_1593BB0(
                          *(_QWORD *)(v39
                                    + 24 * ((unsigned int)v43 - (unsigned __int64)(*(_DWORD *)(v39 + 20) & 0xFFFFFFF))),
                          v38,
                          *(_DWORD *)(v39 + 20) & 0xFFFFFFF,
                          v41);
                  if ( v62 )
                    break;
                  LODWORD(v43) = v43 + 1;
                  if ( v40 == (_DWORD)v43 )
                    goto LABEL_64;
                }
                v63 = v62;
                v64 = &v105[8 * (unsigned int)v106];
                v65 = v106;
                if ( v64 != (_BYTE *)(v38 + 8) )
                {
                  memmove((void *)v38, (const void *)(v38 + 8), (size_t)&v64[-v38 - 8]);
                  v65 = v106;
                }
                v94 = v63;
                LODWORD(v106) = v65 - 1;
              }
              else
              {
                do
                {
                  v44 = 0;
                  v45 = sub_1649C60(*(_QWORD *)&src[24 * (v43 - (*((_DWORD *)src + 5) & 0xFFFFFFF))]);
                  while ( v45 != sub_1649C60(*(_QWORD *)(v39
                                                       + 24
                                                       * (v44 - (unsigned __int64)(*(_DWORD *)(v39 + 20) & 0xFFFFFFF)))) )
                  {
                    if ( v40 == ++v44 )
                      goto LABEL_64;
                  }
                  ++v43;
                }
                while ( v43 != (unsigned int)v92 );
                v46 = v106;
                v47 = &v105[8 * (unsigned int)v106];
                if ( v47 != (_BYTE *)(v38 + 8) )
                {
                  memmove((void *)v38, (const void *)(v38 + 8), (size_t)&v47[-v38 - 8]);
                  v46 = v106;
                }
                v94 = 1;
                LODWORD(v106) = v46 - 1;
              }
              goto LABEL_64;
            }
            if ( v42 == 10 )
            {
              v66 = (void *)(v36 + 8LL * v103);
              v68 = (char *)(v36 + 8LL * (unsigned int)v106);
              v67 = (char *)(v38 + 8);
              v69 = v106;
              if ( v68 == desta + 8 )
              {
LABEL_116:
                v94 = 1;
                LODWORD(v106) = v69 - 1;
                goto LABEL_64;
              }
LABEL_115:
              memmove(v66, v67, v68 - v67);
              v69 = v106;
              goto LABEL_116;
            }
          }
        }
LABEL_64:
        if ( --v103 == j )
        {
          v35 = v106;
          goto LABEL_112;
        }
        v36 = (unsigned __int64)v105;
        continue;
      }
    }
  }
LABEL_128:
  if ( v94 )
  {
LABEL_149:
    v82 = 0;
    LOWORD(v119) = 257;
    v79 = sub_15F5910(*(_QWORD *)a2, v25, (__int64)&v117, 0);
    v87 = 8LL * (unsigned int)v106;
    if ( !(_DWORD)v106 )
      goto LABEL_154;
    do
    {
      v88 = *(_QWORD *)&v105[v82];
      v82 += 8;
      sub_15F5A60(v79, v88, v83, v84, v85, v86);
    }
    while ( v82 != v87 );
    if ( (_DWORD)v106 )
    {
      v89 = *(_WORD *)(v79 + 18);
      v90 = v93;
      v91 = v89 & 0x7FFE;
    }
    else
    {
LABEL_154:
      v89 = *(_WORD *)(v79 + 18);
      v90 = 1;
      v91 = v89 & 0x7FFE;
    }
    *(_WORD *)(v79 + 18) = v91 | v90 | v89 & 0x8000;
    goto LABEL_131;
  }
  v79 = 0;
  if ( v93 != (*(_WORD *)(a2 + 18) & 1) )
  {
    v79 = a2;
    *(_WORD *)(a2 + 18) = *(_WORD *)(a2 + 18) & 0xFFFE | v93;
  }
LABEL_131:
  if ( v113 != v112 )
    _libc_free((unsigned __int64)v113);
  if ( v105 != v107 )
    _libc_free((unsigned __int64)v105);
  return v79;
}
