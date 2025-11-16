// Function: sub_9FDC80
// Address: 0x9fdc80
//
__int64 *__fastcall sub_9FDC80(__int64 *a1, __int64 a2, _BYTE *a3)
{
  __int64 *v3; // r15
  __int64 v4; // r14
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r13
  unsigned int v9; // eax
  __int64 *v10; // rbx
  __int64 v11; // rsi
  __int64 v12; // rcx
  unsigned __int64 v13; // rax
  __int64 *v14; // rdi
  __int64 v15; // rsi
  __int64 v16; // rdx
  unsigned __int8 v17; // al
  __int64 v18; // rsi
  char v19; // al
  _QWORD *v20; // r10
  _QWORD *v21; // r12
  _QWORD *v22; // rbx
  __int64 v23; // r12
  __int64 v24; // rcx
  const char *v25; // rax
  char v26; // al
  const char *v27; // rax
  int v28; // r9d
  __int64 v29; // rax
  _QWORD *v30; // rax
  _QWORD *v31; // rbx
  _QWORD *k; // r12
  _QWORD *v33; // r15
  __int64 v34; // rax
  __int64 v35; // r14
  unsigned __int8 v36; // al
  _BYTE *v37; // rdi
  __int64 v38; // rcx
  __int64 v39; // rax
  __int64 v40; // rsi
  __int64 v41; // rdx
  __int64 v42; // rax
  unsigned int v43; // r14d
  int v44; // edx
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rbx
  int v48; // ebx
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 v51; // rdx
  __int64 v52; // rdx
  __int64 v53; // rax
  __int64 v54; // rsi
  __int64 v55; // rdx
  __int64 v56; // rax
  _QWORD *v57; // rax
  __int64 v58; // rax
  __int64 v59; // rdx
  char v60; // al
  unsigned int v61; // r13d
  int v62; // eax
  int v63; // edx
  __int64 v64; // rax
  _QWORD *v65; // r8
  _QWORD *v66; // r9
  _QWORD *j; // r13
  _QWORD *v68; // r14
  __int64 v69; // rdx
  __int64 v70; // rax
  __int64 v71; // r13
  __int64 v72; // r12
  __int64 v73; // r15
  __int64 i; // r14
  __int64 v75; // rdi
  __int64 v76; // rax
  _QWORD *v77; // rax
  __int64 *v78; // rdx
  __int64 v79; // rax
  __int64 *v80; // rcx
  __int64 v81; // rax
  __int64 *v82; // rbx
  __int64 *v83; // r13
  __int64 v84; // r12
  __int64 v85; // rax
  _BYTE *v86; // rdi
  __int64 v87; // rax
  unsigned int v88; // r12d
  __int64 v89; // rbx
  __int64 v90; // rax
  __int64 v91; // r13
  __int64 v92; // rbx
  __int64 v93; // rdi
  _QWORD *v94; // [rsp+0h] [rbp-2E0h]
  __int64 v95; // [rsp+8h] [rbp-2D8h]
  _QWORD *v96; // [rsp+8h] [rbp-2D8h]
  __int64 v97; // [rsp+10h] [rbp-2D0h]
  __int64 v98; // [rsp+10h] [rbp-2D0h]
  __int64 *v99; // [rsp+18h] [rbp-2C8h]
  __int64 *v100; // [rsp+18h] [rbp-2C8h]
  _QWORD *v101; // [rsp+20h] [rbp-2C0h]
  __int64 v102; // [rsp+20h] [rbp-2C0h]
  __int64 v103; // [rsp+28h] [rbp-2B8h]
  __int64 v104; // [rsp+30h] [rbp-2B0h]
  __int64 v105; // [rsp+30h] [rbp-2B0h]
  __int64 v106; // [rsp+38h] [rbp-2A8h]
  __int64 v107; // [rsp+38h] [rbp-2A8h]
  unsigned __int8 v108; // [rsp+48h] [rbp-298h]
  __int64 v109; // [rsp+48h] [rbp-298h]
  __int64 v110; // [rsp+48h] [rbp-298h]
  __int64 v111; // [rsp+48h] [rbp-298h]
  unsigned __int64 v112; // [rsp+58h] [rbp-288h] BYREF
  __int64 v113; // [rsp+60h] [rbp-280h] BYREF
  char v114; // [rsp+68h] [rbp-278h]
  __int64 v115[4]; // [rsp+70h] [rbp-270h] BYREF
  char v116; // [rsp+90h] [rbp-250h]
  char v117; // [rsp+91h] [rbp-24Fh]
  __int64 v118[2]; // [rsp+A0h] [rbp-240h] BYREF
  _BYTE v119[16]; // [rsp+B0h] [rbp-230h] BYREF
  __int64 v120; // [rsp+C0h] [rbp-220h]

  v3 = a1;
  if ( *a3 || (v4 = (__int64)a3, (a3[35] & 8) == 0) )
  {
    *a1 = 1;
    return v3;
  }
  v6 = *(unsigned int *)(a2 + 1664);
  v7 = *(_QWORD *)(a2 + 1648);
  v8 = a2;
  if ( (_DWORD)v6 )
  {
    v9 = (v6 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
    v10 = (__int64 *)(v7 + 16LL * v9);
    v11 = *v10;
    if ( v4 == *v10 )
      goto LABEL_7;
    v28 = 1;
    while ( v11 != -4096 )
    {
      v9 = (v6 - 1) & (v28 + v9);
      v10 = (__int64 *)(v7 + 16LL * v9);
      v11 = *v10;
      if ( v4 == *v10 )
        goto LABEL_7;
      ++v28;
    }
  }
  v10 = (__int64 *)(v7 + 16LL * (unsigned int)v6);
LABEL_7:
  if ( !v10[1] )
  {
    v23 = v8 + 32;
    do
    {
      sub_9CDFE0(v118, v23, *(_QWORD *)(v8 + 448), v7);
      v13 = v118[0] & 0xFFFFFFFFFFFFFFFELL;
      if ( (v118[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
        goto LABEL_28;
      if ( *(_DWORD *)(v8 + 64) || *(_QWORD *)(v8 + 40) > *(_QWORD *)(v8 + 48) )
      {
        if ( *(_BYTE *)(v8 + 1632) )
        {
          v118[0] = (__int64)v119;
          v118[1] = 0x4000000000LL;
          sub_9CEA50((__int64)&v113, v23, 0, v24);
          v6 = v114 & 1;
          v26 = (2 * v6) | v114 & 0xFD;
          v114 = v26;
          if ( (_BYTE)v6 )
          {
            v114 = v26 & 0xFD;
            v29 = v113;
            v113 = 0;
            v112 = v29 | 1;
          }
          else
          {
            if ( (_DWORD)v113 != 2 )
            {
              v117 = 1;
              v27 = "Expect SubBlock";
              goto LABEL_42;
            }
            if ( HIDWORD(v113) == 12 )
            {
              sub_9DDE80(v115, v8);
              if ( (v115[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
              {
                v112 = v115[0] & 0xFFFFFFFFFFFFFFFELL | 1;
              }
              else
              {
                v64 = *(_QWORD *)(v8 + 48);
                v6 = *(unsigned int *)(v8 + 64);
                v112 = 1;
                *(_QWORD *)(v8 + 448) = 8 * v64 - v6;
              }
            }
            else
            {
              v117 = 1;
              v27 = "Expect function block";
LABEL_42:
              v115[0] = (__int64)v27;
              v116 = 3;
              sub_9C81F0((__int64 *)&v112, v8 + 8, (__int64)v115);
            }
            if ( (v114 & 2) != 0 )
              sub_9CEF10(&v113);
            if ( (v114 & 1) == 0 )
            {
LABEL_45:
              v13 = v112 & 0xFFFFFFFFFFFFFFFELL;
              goto LABEL_35;
            }
          }
          if ( v113 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v113 + 8LL))(v113);
          goto LABEL_45;
        }
        BYTE1(v120) = 1;
        v25 = "Trying to materialize functions before seeing function blocks";
      }
      else
      {
        BYTE1(v120) = 1;
        v25 = "Could not find function in stream";
      }
      v118[0] = (__int64)v25;
      LOBYTE(v120) = 3;
      sub_9C81F0((__int64 *)&v112, v8 + 8, (__int64)v118);
      v13 = v112 & 0xFFFFFFFFFFFFFFFELL;
LABEL_35:
      if ( v13 )
        goto LABEL_28;
    }
    while ( !v10[1] );
  }
  sub_9CE050((unsigned __int64 *)v118, v8, v6, v7);
  v13 = v118[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (v118[0] & 0xFFFFFFFFFFFFFFFELL) != 0
    || (sub_9CDFE0(v118, v8 + 32, v10[1], v12),
        v13 = v118[0] & 0xFFFFFFFFFFFFFFFELL,
        (v118[0] & 0xFFFFFFFFFFFFFFFELL) != 0)
    || (*(_BYTE *)(v4 + 128) = 1,
        v14 = v118,
        v15 = v8,
        sub_9F2A40(v118, v8, v4),
        v13 = v118[0] & 0xFFFFFFFFFFFFFFFELL,
        (v118[0] & 0xFFFFFFFFFFFFFFFELL) != 0) )
  {
LABEL_28:
    *v3 = v13 | 1;
    return v3;
  }
  *(_WORD *)(v4 + 34) &= ~0x800u;
  if ( *(_BYTE *)(v8 + 1834) )
  {
    v108 = *(_BYTE *)(v8 + 1835);
    if ( v108 )
    {
      v118[0] = (__int64)"Mixed debug intrinsics and debug records in bitcode module!";
      LOWORD(v120) = 259;
      sub_9C81F0(v3, v8 + 8, (__int64)v118);
      return v3;
    }
    if ( LODWORD(qword_4F80E68[8]) != 1 )
    {
      v17 = *(_BYTE *)(*(_QWORD *)(v4 + 40) + 872LL);
LABEL_15:
      sub_B2BA20(v4, v17);
      goto LABEL_16;
    }
    goto LABEL_178;
  }
  if ( LODWORD(qword_4F80E68[8]) == 1 )
  {
    v108 = *(_BYTE *)(v8 + 1835);
    if ( !v108 )
    {
      v88 = *(unsigned __int8 *)(*(_QWORD *)(v4 + 40) + 872LL);
      goto LABEL_192;
    }
LABEL_178:
    LOBYTE(qword_4F80F48[8]) = v108;
    if ( !qword_4F80F48[14]
      || (v14 = &qword_4F80F48[12],
          v15 = v8 + 1835,
          ((void (__fastcall *)(_QWORD *, __int64))qword_4F80F48[15])(&qword_4F80F48[12], v8 + 1835),
          v16 = *(unsigned __int8 *)(v8 + 1835),
          unk_4F80E08 = v16,
          unk_4F81788 = v16,
          !unk_4F817B8) )
    {
      sub_4263D6(v14, v15, v16);
    }
    unk_4F817C0(&unk_4F817A8, v8 + 1835);
    v87 = *(_QWORD *)(v4 + 40);
    v88 = v108;
    v107 = v87;
    if ( *(_BYTE *)(v87 + 872) != v108 )
    {
      v89 = *(_QWORD *)(v87 + 32);
      v90 = v87 + 24;
      if ( v89 != v90 )
      {
        v105 = v8;
        v91 = v89;
        v92 = v90;
        do
        {
          v93 = v91 - 56;
          if ( !v91 )
            v93 = 0;
          sub_B2BA20(v93, v108);
          v91 = *(_QWORD *)(v91 + 8);
        }
        while ( v92 != v91 );
        v8 = v105;
      }
      *(_BYTE *)(v107 + 872) = v108;
      goto LABEL_16;
    }
LABEL_192:
    sub_B2BA20(v4, v88);
    goto LABEL_16;
  }
  v17 = *(_BYTE *)(*(_QWORD *)(v4 + 40) + 872LL);
  if ( v17 )
    goto LABEL_15;
  v17 = *(_BYTE *)(v8 + 1835);
  if ( !v17 )
    goto LABEL_15;
  sub_B2B9F0(v4, 0);
LABEL_16:
  if ( *(_BYTE *)(v8 + 1836) )
    sub_AEAD90(v4);
  if ( *(_DWORD *)(v8 + 1616) )
  {
    v78 = *(__int64 **)(v8 + 1608);
    v79 = 2LL * *(unsigned int *)(v8 + 1624);
    v80 = &v78[v79];
    if ( v78 != &v78[v79] )
    {
      while ( 1 )
      {
        v81 = *v78;
        v82 = v78;
        if ( *v78 != -4096 && v81 != -8192 )
          break;
        v78 += 2;
        if ( v80 == v78 )
          goto LABEL_19;
      }
      if ( v80 != v78 )
      {
        v111 = v8;
        v83 = v80;
        do
        {
          v84 = *(_QWORD *)(v81 + 16);
          while ( v84 )
          {
            v85 = v84;
            v84 = *(_QWORD *)(v84 + 8);
            v86 = *(_BYTE **)(v85 + 24);
            if ( *v86 == 85 )
              sub_A939D0(v86, v82[1]);
          }
          v82 += 2;
          if ( v82 == v83 )
            break;
          while ( 1 )
          {
            v81 = *v82;
            if ( *v82 != -8192 && v81 != -4096 )
              break;
            v82 += 2;
            if ( v83 == v82 )
              goto LABEL_172;
          }
        }
        while ( v82 != v83 );
LABEL_172:
        v8 = v111;
      }
    }
  }
LABEL_19:
  v106 = v8 + 808;
  v18 = sub_A03C30(v8 + 808, v4);
  if ( v18 )
    sub_B994C0(v4, v18);
  v19 = sub_A03CD0(v106, v18);
  v20 = *(_QWORD **)(v4 + 80);
  v103 = v4 + 72;
  if ( v19 )
    goto LABEL_22;
  v65 = *(_QWORD **)(v4 + 80);
  if ( (_QWORD *)v103 == v20 )
  {
    v66 = 0;
  }
  else
  {
    if ( !v20 )
      goto LABEL_202;
    v66 = (_QWORD *)v20[4];
    if ( v66 == v20 + 3 )
    {
      do
      {
        v65 = (_QWORD *)v65[1];
        if ( (_QWORD *)v103 == v65 )
          goto LABEL_22;
        if ( !v65 )
          goto LABEL_202;
        v66 = (_QWORD *)v65[4];
      }
      while ( v66 == v65 + 3 );
    }
  }
  if ( (_QWORD *)v103 != v65 )
  {
    v100 = v3;
    v102 = v8 + 1840;
    v98 = v8;
    j = v66;
    v104 = v4;
    v68 = v65;
    do
    {
      if ( !j )
LABEL_205:
        BUG();
      if ( (*((_BYTE *)j - 17) & 0x20) != 0 )
      {
        v69 = sub_B91C10(j - 3, 1);
        if ( v69 )
        {
          if ( !(unsigned __int8)sub_BF6420(v102, j - 3, v69) )
          {
            sub_A03CC0(v106, 1);
            v70 = *(_QWORD *)(v104 + 40);
            v110 = v70 + 24;
            if ( *(_QWORD *)(v70 + 32) != v70 + 24 )
            {
              v96 = j;
              v71 = *(_QWORD *)(v70 + 32);
              v94 = v68;
              do
              {
                if ( !v71 )
                  BUG();
                if ( (*(_BYTE *)(v71 - 21) & 8) == 0 )
                {
                  v72 = *(_QWORD *)(v71 + 24);
                  v73 = v71 + 16;
                  if ( v71 + 16 != v72 )
                  {
                    while ( v72 )
                    {
                      i = *(_QWORD *)(v72 + 32);
                      if ( i == v72 + 24 )
                      {
                        v72 = *(_QWORD *)(v72 + 8);
                        if ( v73 != v72 )
                          continue;
                      }
                      goto LABEL_133;
                    }
LABEL_202:
                    BUG();
                  }
                  i = 0;
LABEL_133:
                  while ( v73 != v72 )
                  {
                    v75 = i - 24;
                    if ( !i )
                      v75 = 0;
                    sub_B99FD0(v75, 1, 0);
                    for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v72 + 32) )
                    {
                      v76 = v72 - 24;
                      if ( !v72 )
                        v76 = 0;
                      if ( i != v76 + 48 )
                        break;
                      v72 = *(_QWORD *)(v72 + 8);
                      if ( v73 == v72 )
                        goto LABEL_133;
                      if ( !v72 )
                        goto LABEL_202;
                    }
                  }
                }
                v71 = *(_QWORD *)(v71 + 8);
              }
              while ( v110 != v71 );
              j = v96;
              v68 = v94;
            }
          }
        }
      }
      for ( j = (_QWORD *)j[1]; ; j = (_QWORD *)v68[4] )
      {
        v77 = v68 - 3;
        if ( !v68 )
          v77 = 0;
        if ( j != v77 + 6 )
          break;
        v68 = (_QWORD *)v68[1];
        if ( (_QWORD *)v103 == v68 )
          goto LABEL_154;
        if ( !v68 )
          goto LABEL_202;
      }
    }
    while ( v68 != (_QWORD *)v103 );
LABEL_154:
    v4 = v104;
    v3 = v100;
    v8 = v98;
    v20 = *(_QWORD **)(v104 + 80);
  }
LABEL_22:
  v21 = v20;
  if ( (_QWORD *)v103 != v20 )
  {
    while ( v21 )
    {
      v22 = (_QWORD *)v21[4];
      if ( v22 != v21 + 3 )
        goto LABEL_57;
      v21 = (_QWORD *)v21[1];
      if ( (_QWORD *)v103 == v21 )
        goto LABEL_26;
    }
    goto LABEL_202;
  }
  v22 = 0;
LABEL_57:
  if ( (_QWORD *)v103 == v21 )
    goto LABEL_26;
  v99 = v3;
  v30 = v22;
  v31 = v21;
  v97 = v8;
  k = v30;
  v95 = v4;
  do
  {
    if ( !k )
      goto LABEL_205;
    v33 = k - 3;
    if ( (*((_BYTE *)k - 17) & 0x20) == 0 )
      goto LABEL_65;
    v34 = sub_B91C10(k - 3, 2);
    v35 = v34;
    if ( !v34 )
      goto LABEL_65;
    v36 = *(_BYTE *)(v34 - 16);
    if ( (v36 & 2) != 0 )
    {
      v37 = **(_BYTE ***)(v35 - 32);
      if ( v37 )
        goto LABEL_64;
    }
    else
    {
      v37 = *(_BYTE **)(v35 - 8LL * ((v36 >> 2) & 0xF) - 16);
      if ( v37 )
      {
LABEL_64:
        if ( !*v37 )
        {
          v58 = sub_B91420(v37, 2);
          if ( v59 != 14
            || *(_QWORD *)v58 != 0x775F68636E617262LL
            || *(_DWORD *)(v58 + 8) != 1751607653
            || *(_WORD *)(v58 + 12) != 29556 )
          {
            goto LABEL_84;
          }
          v60 = *((_BYTE *)k - 24);
          switch ( v60 )
          {
            case 31:
              v61 = ((*((_DWORD *)k - 5) & 0x7FFFFFF) == 3) + 1;
              break;
            case 32:
              v61 = (*((_DWORD *)k - 5) & 0x7FFFFFFu) >> 1;
              break;
            case 85:
              v61 = 1;
              break;
            case 33:
              v61 = (*((_DWORD *)k - 5) & 0x7FFFFFF) - 1;
              break;
            default:
              v61 = 2;
              if ( v60 != 86 )
                goto LABEL_84;
              break;
          }
          v62 = sub_BC8810(v35);
          if ( (*(_BYTE *)(v35 - 16) & 2) != 0 )
            v63 = *(_DWORD *)(v35 - 24);
          else
            v63 = (*(_WORD *)(v35 - 16) >> 6) & 0xF;
          if ( v62 + v61 != v63 )
            sub_B99FD0(k - 3, 2, 0);
        }
      }
    }
LABEL_65:
    if ( (unsigned __int8)(*((_BYTE *)k - 24) - 34) > 0x33u )
      goto LABEL_84;
    v38 = 0x8000000000041LL;
    if ( !_bittest64(&v38, (unsigned int)*((unsigned __int8 *)k - 24) - 34) )
      goto LABEL_84;
    v118[0] = k[6];
    v39 = sub_A74610(v118);
    v40 = **(_QWORD **)(k[7] + 16LL);
    sub_A751C0(v118, v40, v39, 3);
    v42 = sub_BD5C60(k - 3, v40, v41);
    v43 = 0;
    k[6] = sub_A7A440(k + 6, v42, 0, v118);
    sub_9C9F50((__int64)v119, v120);
    v101 = v31;
    while ( 1 )
    {
      v44 = *((unsigned __int8 *)k - 24);
      if ( v44 == 40 )
      {
        v109 = 32LL * (unsigned int)sub_B491D0(v33);
      }
      else
      {
        v109 = 0;
        if ( v44 != 85 )
        {
          if ( v44 != 34 )
            goto LABEL_206;
          v109 = 64;
        }
      }
      if ( *((char *)k - 17) >= 0 )
        goto LABEL_81;
      v45 = sub_BD2BC0(v33);
      v47 = v45 + v46;
      if ( *((char *)k - 17) >= 0 )
      {
        if ( (unsigned int)(v47 >> 4) )
LABEL_206:
          BUG();
LABEL_81:
        v51 = 0;
        goto LABEL_78;
      }
      if ( !(unsigned int)((v47 - sub_BD2BC0(v33)) >> 4) )
        goto LABEL_81;
      if ( *((char *)k - 17) >= 0 )
        goto LABEL_206;
      v48 = *(_DWORD *)(sub_BD2BC0(v33) + 8);
      if ( *((char *)k - 17) >= 0 )
        BUG();
      v49 = sub_BD2BC0(v33);
      v51 = 32LL * (unsigned int)(*(_DWORD *)(v49 + v50 - 4) - v48);
LABEL_78:
      if ( v43 >= (unsigned int)((32LL * (*((_DWORD *)k - 5) & 0x7FFFFFF) - 32 - v109 - v51) >> 5) )
        break;
      v118[0] = k[6];
      v52 = sub_A744E0(v118, v43);
      v53 = v43++;
      v54 = *(_QWORD *)(v33[4 * (v53 - (*((_DWORD *)k - 5) & 0x7FFFFFF))] + 8LL);
      sub_A751C0(v118, v54, v52, 3);
      v56 = sub_BD5C60(v33, v54, v55);
      k[6] = sub_A7A440(k + 6, v56, v43, v118);
      sub_9C9F50((__int64)v119, v120);
    }
    v31 = v101;
LABEL_84:
    for ( k = (_QWORD *)k[1]; ; k = (_QWORD *)v31[4] )
    {
      v57 = v31 - 3;
      if ( !v31 )
        v57 = 0;
      if ( k != v57 + 6 )
        break;
      v31 = (_QWORD *)v31[1];
      if ( (_QWORD *)v103 == v31 )
        goto LABEL_92;
      if ( !v31 )
        goto LABEL_202;
    }
  }
  while ( v31 != (_QWORD *)v103 );
LABEL_92:
  v3 = v99;
  v8 = v97;
  v4 = v95;
LABEL_26:
  sub_A86E70(v4);
  sub_9FF010(v3, v8);
  return v3;
}
