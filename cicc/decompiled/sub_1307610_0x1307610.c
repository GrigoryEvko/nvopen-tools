// Function: sub_1307610
// Address: 0x1307610
//
void *__fastcall sub_1307610(unsigned __int64 a1, int a2)
{
  __int64 v2; // r15
  int v4; // ebx
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // r12
  unsigned __int64 v8; // r14
  unsigned int v9; // r9d
  unsigned int v10; // r8d
  unsigned __int64 v11; // rax
  __int64 v12; // rcx
  __int64 *v13; // rdx
  __int64 v14; // r11
  __int64 v15; // rsi
  __int64 *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r10
  void **v19; // rax
  void *v20; // r8
  void **v21; // rdi
  __int64 v22; // rdx
  void *v23; // r8
  unsigned __int64 v24; // r15
  __int64 v25; // r10
  __int64 v26; // rsi
  __int64 v27; // r11
  __int64 v28; // r14
  void **v29; // rax
  void **v30; // rdi
  __int64 v31; // rdx
  __int64 v33; // rcx
  unsigned __int64 v34; // rax
  char v35; // cl
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // r9
  void **v39; // rax
  void **v40; // rdi
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // r10
  __int64 v44; // r13
  __int64 v45; // rsi
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  unsigned __int64 v49; // rcx
  char v50; // al
  char v51; // cl
  __int64 v52; // rax
  __int64 v53; // r10
  __int64 v54; // r9
  void **v55; // rax
  void **v56; // rdx
  __int64 v57; // rax
  __int64 v58; // r10
  __int64 v59; // rcx
  __int64 v60; // rdx
  __int64 v61; // rsi
  __int64 v62; // rax
  void *v63; // rax
  __int64 v64; // rbx
  __int64 v65; // rax
  void *v66; // rax
  __int64 v67; // rax
  __int64 v68; // rax
  unsigned __int64 v69; // rcx
  void *v70; // rax
  __int64 v71; // rdi
  __int64 v72; // rax
  __int64 v73; // rcx
  __int64 v74; // rax
  __int64 v75; // rax
  __int64 v76; // [rsp+0h] [rbp-A0h]
  __int64 v77; // [rsp+8h] [rbp-98h]
  __int64 v78; // [rsp+8h] [rbp-98h]
  __int64 v79; // [rsp+10h] [rbp-90h]
  unsigned int v80; // [rsp+10h] [rbp-90h]
  __int64 v81; // [rsp+18h] [rbp-88h]
  unsigned int v82; // [rsp+18h] [rbp-88h]
  __int64 v83; // [rsp+18h] [rbp-88h]
  __int64 v84; // [rsp+18h] [rbp-88h]
  __int64 v85; // [rsp+20h] [rbp-80h]
  __int64 v86; // [rsp+20h] [rbp-80h]
  __int64 v87; // [rsp+20h] [rbp-80h]
  __int64 v88; // [rsp+20h] [rbp-80h]
  __int64 *v89; // [rsp+20h] [rbp-80h]
  __int64 v90; // [rsp+20h] [rbp-80h]
  __int64 v91; // [rsp+20h] [rbp-80h]
  __int64 v92; // [rsp+28h] [rbp-78h]
  __int64 v93; // [rsp+28h] [rbp-78h]
  __int64 v94; // [rsp+28h] [rbp-78h]
  __int64 *v95; // [rsp+28h] [rbp-78h]
  __int64 v96; // [rsp+28h] [rbp-78h]
  void *v97; // [rsp+28h] [rbp-78h]
  __int64 v98; // [rsp+28h] [rbp-78h]
  unsigned int v99; // [rsp+28h] [rbp-78h]
  unsigned int v100; // [rsp+28h] [rbp-78h]
  unsigned int v101; // [rsp+28h] [rbp-78h]
  __int64 v102; // [rsp+28h] [rbp-78h]
  size_t v103; // [rsp+30h] [rbp-70h]
  unsigned int v104; // [rsp+30h] [rbp-70h]
  unsigned int v105; // [rsp+30h] [rbp-70h]
  unsigned int v106; // [rsp+30h] [rbp-70h]
  unsigned int v107; // [rsp+30h] [rbp-70h]
  unsigned int v108; // [rsp+30h] [rbp-70h]
  unsigned int v109; // [rsp+30h] [rbp-70h]
  unsigned int v110; // [rsp+38h] [rbp-68h]
  void *v111; // [rsp+38h] [rbp-68h]
  unsigned int v112; // [rsp+38h] [rbp-68h]
  void *v113; // [rsp+38h] [rbp-68h]
  __int64 v114; // [rsp+38h] [rbp-68h]
  unsigned int v115; // [rsp+38h] [rbp-68h]
  unsigned int v116; // [rsp+38h] [rbp-68h]
  unsigned int v117; // [rsp+38h] [rbp-68h]
  __int64 v118; // [rsp+38h] [rbp-68h]
  __int64 v119; // [rsp+38h] [rbp-68h]
  unsigned int v120; // [rsp+38h] [rbp-68h]
  unsigned int v121; // [rsp+38h] [rbp-68h]
  void *v122; // [rsp+38h] [rbp-68h]
  __int64 v123; // [rsp+38h] [rbp-68h]
  unsigned int v124; // [rsp+38h] [rbp-68h]
  unsigned __int64 v125; // [rsp+40h] [rbp-60h] BYREF
  __int64 v126; // [rsp+48h] [rbp-58h]
  __int64 v127; // [rsp+50h] [rbp-50h]
  __int64 v128; // [rsp+58h] [rbp-48h]
  __int64 v129; // [rsp+60h] [rbp-40h]

  v2 = a2;
  if ( !a2 )
  {
    LOBYTE(v4) = __readfsbyte(0xFFFFF8C8);
    if ( (_BYTE)v4 )
    {
      v5 = __readfsqword(0) - 2664;
      v6 = sub_1313D30(v5, 0);
      LOBYTE(v4) = *(_BYTE *)(v6 + 816);
      v7 = v6;
      if ( (_BYTE)v4 )
      {
        v8 = 0;
        v9 = -2;
        LOBYTE(v4) = 0;
        v10 = -1;
LABEL_5:
        if ( dword_4C6F034[0] )
        {
          v107 = v10;
          v117 = v9;
          v50 = sub_13022D0(v5, 0);
          v9 = v117;
          v10 = v107;
          if ( v50 )
          {
            *__errno_location() = 12;
            return 0;
          }
        }
        if ( (v8 & (v8 - 1)) != 0 )
          goto LABEL_119;
        if ( unk_4F96994 )
          LOBYTE(v4) = 1;
        if ( !v8 )
        {
          if ( a1 > 0x1000 )
          {
            if ( a1 > 0x7000000000000000LL )
              goto LABEL_119;
            v51 = 7;
            _BitScanReverse64((unsigned __int64 *)&v52, 2 * a1 - 1);
            if ( (unsigned int)v52 >= 7 )
              v51 = v52;
            if ( (unsigned int)v52 < 6 )
              LODWORD(v52) = 6;
            v12 = ((unsigned int)(((a1 - 1) & (-1LL << (v51 - 3))) >> (v51 - 3)) & 3) + 4 * (_DWORD)v52 - 23;
          }
          else
          {
            v12 = byte_5060800[(a1 + 7) >> 3];
          }
          if ( (unsigned int)v12 > 0xE7 )
          {
LABEL_119:
            v8 = 0;
LABEL_120:
            v20 = 0;
LABEL_40:
            v111 = v20;
            v127 = 0;
            v125 = a1;
            v126 = v2;
            sub_1346E80(7, v20, v8, &v125);
            return v111;
          }
          v103 = qword_505FA40[(unsigned int)v12];
LABEL_17:
          if ( *(char *)(v7 + 1) > 0 )
          {
            v16 = qword_50579C0;
            v10 = 0;
            v14 = 0;
            goto LABEL_25;
          }
          if ( v9 == -2 )
          {
            v14 = v7 + 856;
            if ( *(_BYTE *)v7 )
            {
LABEL_23:
              v15 = 0;
              if ( v10 == -1 )
                goto LABEL_26;
              v16 = &qword_50579C0[v10];
LABEL_25:
              v15 = *v16;
              if ( !*v16 )
              {
                v100 = v12;
                v90 = v14;
                v121 = v10;
                v68 = sub_1300B80(v7, v10, (__int64)&off_49E8000);
                v12 = v100;
                v14 = v90;
                v15 = v68;
                if ( !v68 && unk_505F9B8 <= v121 )
                  goto LABEL_119;
              }
LABEL_26:
              v110 = (unsigned __int8)v4;
              if ( !v8 )
              {
                if ( v14 )
                {
                  if ( a1 <= 0x3800 )
                  {
                    v92 = (unsigned int)v12;
                    v17 = 24LL * (unsigned int)v12;
                    v18 = v14 + v17;
                    v19 = *(void ***)(v14 + v17 + 8);
                    v20 = *v19;
                    v21 = v19 + 1;
                    if ( (_WORD)v19 == *(_WORD *)(v14 + v17 + 24) )
                    {
                      if ( (_WORD)v19 == *(_WORD *)(v18 + 28) )
                      {
                        v77 = 24LL * (unsigned int)v12;
                        v79 = v14;
                        v82 = v12;
                        v86 = v14 + v17;
                        v57 = sub_1302E60(v7, v15);
                        v58 = v86;
                        v59 = v82;
                        if ( !v57 )
                          goto LABEL_120;
                        if ( !*(_WORD *)(unk_5060A20 + 2 * v92) )
                        {
                          v20 = (void *)sub_1317CF0(v7, v57, a1, v82, (unsigned __int8)v4);
                          goto LABEL_34;
                        }
                        v60 = v79 + v77 + 8;
                        v61 = v79;
                        v87 = v79;
                        v78 = v58;
                        v76 = v57;
                        v80 = v82;
                        v83 = v60;
                        sub_1310140(v7, v61, v60, v59, 1);
                        v62 = sub_13100A0(v7, v76, v87, v83, v80);
                        v18 = v78;
                        v20 = (void *)v62;
                        if ( !(_BYTE)v125 )
                          goto LABEL_120;
                      }
                      else
                      {
                        *(_QWORD *)(v18 + 8) = v21;
                        *(_WORD *)(v18 + 24) = (_WORD)v21;
                      }
                    }
                    else
                    {
                      *(_QWORD *)(v18 + 8) = v21;
                    }
                    if ( (_BYTE)v4 )
                    {
                      v118 = v18;
                      v63 = memset(v20, 0, qword_505FA40[v92]);
                      v18 = v118;
                      v20 = v63;
                    }
                    ++*(_QWORD *)(v18 + 16);
LABEL_34:
                    if ( v20 )
                      goto LABEL_35;
                    goto LABEL_119;
                  }
                  if ( a1 <= unk_5060A10 )
                  {
                    v53 = 24LL * (unsigned int)v12;
                    v54 = v14 + v53;
                    v55 = *(void ***)(v14 + v53 + 8);
                    v20 = *v55;
                    v56 = v55 + 1;
                    if ( (_WORD)v55 == *(_WORD *)(v14 + v53 + 24) )
                    {
                      v71 = v14 + 24LL * (unsigned int)v12;
                      if ( (_WORD)v55 == *(_WORD *)(v71 + 28) )
                      {
                        v84 = 24LL * (unsigned int)v12;
                        v91 = v14;
                        v101 = v12;
                        v72 = sub_1302E60(v7, v15);
                        v73 = v101;
                        if ( !v72 )
                          goto LABEL_119;
                        v102 = v72;
                        sub_1310140(v7, v91, v91 + v84 + 8, v73, 0);
                        _BitScanReverse64((unsigned __int64 *)&v74, 2 * a1 - 1);
                        if ( (unsigned __int64)(int)v74 < 7 )
                          LOBYTE(v74) = 7;
                        v20 = (void *)sub_1309DC0(
                                        v7,
                                        v102,
                                        -(1LL << ((unsigned __int8)v74 - 3))
                                      & ((1LL << ((unsigned __int8)v74 - 3)) + a1 - 1),
                                        (unsigned __int8)v4);
                        if ( !v20 )
                          goto LABEL_119;
LABEL_35:
                        LOBYTE(v125) = 1;
                        v126 = v7 + 824;
                        v127 = v7 + 8;
                        v128 = v7 + 16;
                        v129 = v7 + 832;
                        v22 = *(_QWORD *)(v7 + 824);
                        *(_QWORD *)(v7 + 824) = v22 + v103;
                        if ( *(_QWORD *)(v7 + 16) - v22 <= v103 )
                        {
                          v97 = v20;
                          sub_13133F0(v7, &v125);
                          v20 = v97;
                        }
                        v8 = (unsigned __int64)v20;
                        if ( !(_BYTE)v4 && unk_4F969A2 )
                        {
                          v122 = v20;
                          off_4C6F0B8(v20, v103);
                          v20 = v122;
                        }
                        goto LABEL_40;
                      }
                      *(_QWORD *)(v54 + 8) = v56;
                      *(_WORD *)(v71 + 24) = (_WORD)v56;
                    }
                    else
                    {
                      *(_QWORD *)(v54 + 8) = v56;
                    }
                    if ( (_BYTE)v4 )
                    {
                      v123 = v14 + v53;
                      v70 = memset(v20, 0, qword_505FA40[(unsigned int)v12]);
                      v54 = v123;
                      v20 = v70;
                    }
                    ++*(_QWORD *)(v54 + 16);
                    goto LABEL_34;
                  }
                }
LABEL_138:
                v20 = (void *)sub_1317CF0(v7, v15, a1, v12, v110);
                goto LABEL_34;
              }
              goto LABEL_145;
            }
          }
          else if ( v9 != -1 )
          {
            v13 = (__int64 *)(*(_QWORD *)&dword_5060A08 + 8LL * v9);
            v14 = *v13;
            if ( *v13 )
            {
              if ( v14 == 1 )
              {
                v89 = (__int64 *)(*(_QWORD *)&dword_5060A08 + 8LL * v9);
                v99 = v12;
                v120 = v10;
                v67 = sub_1311F90(v7);
                v12 = v99;
                v10 = v120;
                v14 = v67;
                *v89 = v67;
              }
              goto LABEL_23;
            }
LABEL_127:
            sub_130ACF0((unsigned int)"<jemalloc>: invalid tcache id (%u).\n", v9, (_DWORD)v13, v12, v10, v9);
            abort();
          }
          if ( v10 == -1 )
          {
            v110 = (unsigned __int8)v4;
            if ( !v8 )
            {
              v15 = 0;
              goto LABEL_138;
            }
            v14 = 0;
            v15 = 0;
LABEL_145:
            v20 = (void *)sub_1318040(v7, v15, v103, v8, v110, v14);
            goto LABEL_34;
          }
          v14 = 0;
          v16 = &qword_50579C0[v10];
          goto LABEL_25;
        }
        if ( a1 > 0x3800 || v8 > 0x1000 )
        {
          if ( v8 > 0x7000000000000000LL )
            goto LABEL_119;
          if ( a1 <= 0x4000 )
            goto LABEL_15;
          if ( a1 > 0x7000000000000000LL )
            goto LABEL_119;
          _BitScanReverse64((unsigned __int64 *)&v41, 2 * a1 - 1);
          if ( (unsigned __int64)(int)v41 < 7 )
            LOBYTE(v41) = 7;
          v103 = -(1LL << ((unsigned __int8)v41 - 3)) & ((1LL << ((unsigned __int8)v41 - 3)) + a1 - 1);
          if ( a1 > v103 || __CFADD__(v103, ((v8 + 4095) & 0xFFFFFFFFFFFFF000LL) + unk_50607C0 - 4096) )
            goto LABEL_119;
        }
        else
        {
          v11 = -(__int64)v8 & (v8 + a1 - 1);
          if ( v11 > 0x1000 )
          {
            _BitScanReverse64(&v69, 2 * v11 - 1);
            v103 = -(1LL << ((unsigned __int8)v69 - 3)) & (v11 + (1LL << ((unsigned __int8)v69 - 3)) - 1);
          }
          else
          {
            v103 = qword_505FA40[byte_5060800[(v11 + 7) >> 3]];
          }
          if ( v103 > 0x3FFF )
          {
LABEL_15:
            if ( unk_50607C0 + ((v8 + 4095) & 0xFFFFFFFFFFFFF000LL) + 12288 <= 0x3FFF )
              goto LABEL_119;
            v103 = 0x4000;
            v12 = 0;
            goto LABEL_17;
          }
        }
        if ( v103 - 1 > 0x6FFFFFFFFFFFFFFFLL )
          goto LABEL_119;
        v12 = 0;
        goto LABEL_17;
      }
      v9 = -2;
      v10 = -1;
    }
    else
    {
      v10 = -1;
      v9 = -2;
      v7 = __readfsqword(0) - 2664;
    }
    goto LABEL_42;
  }
  v9 = -2;
  v8 = (1LL << a2) & 0xFFFFFFFFFFFFFFFELL;
  v4 = ((unsigned int)a2 >> 6) & 1;
  if ( (a2 & 0xFFF00) != 0 )
  {
    v9 = -1;
    if ( (a2 & 0xFFF00) != 0x100 )
      v9 = ((a2 >> 8) & 0xFFF) - 2;
  }
  v10 = ((unsigned int)a2 >> 20) - 1;
  if ( (a2 & 0xFFF00000) == 0 )
    v10 = -1;
  if ( __readfsbyte(0xFFFFF8C8) )
  {
    v109 = v10;
    v124 = v9;
    v5 = __readfsqword(0) - 2664;
    v75 = sub_1313D30(v5, 0);
    v9 = v124;
    v10 = v109;
    v7 = v75;
    if ( *(_BYTE *)(v75 + 816) )
      goto LABEL_5;
  }
  else
  {
    v7 = __readfsqword(0) - 2664;
  }
  if ( (v8 & (v8 - 1)) != 0 )
    return 0;
  if ( !v8 )
  {
LABEL_42:
    if ( a1 > 0x1000 )
    {
      if ( a1 > 0x7000000000000000LL )
        return 0;
      v35 = 7;
      _BitScanReverse64((unsigned __int64 *)&v36, 2 * a1 - 1);
      if ( (unsigned int)v36 >= 7 )
        v35 = v36;
      if ( (unsigned int)v36 < 6 )
        LODWORD(v36) = 6;
      v12 = ((unsigned int)(((a1 - 1) & (-1LL << (v35 - 3))) >> (v35 - 3)) & 3) + 4 * (_DWORD)v36 - 23;
    }
    else
    {
      v12 = byte_5060800[(a1 + 7) >> 3];
    }
    if ( (unsigned int)v12 <= 0xE7 )
    {
      v8 = 0;
      v24 = qword_505FA40[(unsigned int)v12];
      goto LABEL_46;
    }
    return 0;
  }
  if ( a1 > 0x3800 || v8 > 0x1000 )
  {
    if ( v8 > 0x7000000000000000LL )
      return 0;
    if ( a1 <= 0x4000 )
      goto LABEL_85;
    if ( a1 > 0x7000000000000000LL )
      return 0;
    _BitScanReverse64((unsigned __int64 *)&v33, 2 * a1 - 1);
    if ( (unsigned __int64)(int)v33 < 7 )
      LOBYTE(v33) = 7;
    v24 = -(1LL << ((unsigned __int8)v33 - 3)) & ((1LL << ((unsigned __int8)v33 - 3)) + a1 - 1);
    if ( a1 > v24 || __CFADD__(v24, ((v8 + 4095) & 0xFFFFFFFFFFFFF000LL) + unk_50607C0 - 4096) )
      return 0;
LABEL_87:
    if ( v24 - 1 > 0x6FFFFFFFFFFFFFFFLL )
      return 0;
    v12 = 0;
    goto LABEL_46;
  }
  v34 = -(__int64)v8 & (v8 + a1 - 1);
  if ( v34 > 0x1000 )
  {
    _BitScanReverse64(&v49, 2 * v34 - 1);
    v24 = -(1LL << ((unsigned __int8)v49 - 3)) & (v34 + (1LL << ((unsigned __int8)v49 - 3)) - 1);
  }
  else
  {
    v24 = qword_505FA40[byte_5060800[(v34 + 7) >> 3]];
  }
  if ( v24 <= 0x3FFF )
    goto LABEL_87;
LABEL_85:
  if ( ((v8 + 4095) & 0xFFFFFFFFFFFFF000LL) + unk_50607C0 + 12288 <= 0x3FFF )
    return 0;
  v12 = 0;
  v24 = 0x4000;
LABEL_46:
  v25 = v7 + 856;
  if ( v9 == -2 )
    goto LABEL_51;
  if ( v9 != -1 )
  {
    v13 = (__int64 *)(*(_QWORD *)&dword_5060A08 + 8LL * v9);
    v25 = *v13;
    if ( !*v13 )
      goto LABEL_127;
    if ( v25 == 1 )
    {
      v95 = (__int64 *)(*(_QWORD *)&dword_5060A08 + 8LL * v9);
      v105 = v10;
      v115 = v12;
      v47 = sub_1311F90(v7);
      v10 = v105;
      v12 = v115;
      v25 = v47;
      *v95 = v47;
    }
LABEL_51:
    v26 = 0;
    if ( v10 == -1 )
      goto LABEL_53;
    goto LABEL_52;
  }
  if ( v10 == -1 )
  {
    v112 = (unsigned __int8)v4;
    if ( !v8 )
    {
      v26 = 0;
      goto LABEL_92;
    }
    v25 = 0;
    v26 = 0;
LABEL_99:
    v23 = (void *)sub_1318040(v7, v26, v24, v8, v112, v25);
    goto LABEL_61;
  }
  v25 = 0;
LABEL_52:
  v26 = qword_50579C0[v10];
  if ( !v26 )
  {
    v106 = v12;
    v96 = v25;
    v116 = v10;
    v48 = sub_1300B80(v7, v10, (__int64)&off_49E8000);
    v12 = v106;
    v25 = v96;
    v26 = v48;
    if ( !v48 && v116 >= unk_505F9B8 )
      return 0;
  }
LABEL_53:
  v112 = (unsigned __int8)v4;
  if ( v8 )
    goto LABEL_99;
  if ( !v25 )
    goto LABEL_92;
  if ( a1 > 0x3800 )
  {
    if ( a1 <= unk_5060A10 )
    {
      v37 = 24LL * (unsigned int)v12;
      v38 = v25 + v37;
      v39 = *(void ***)(v25 + v37 + 8);
      v23 = *v39;
      v40 = v39 + 1;
      if ( (_WORD)v39 == *(_WORD *)(v25 + v37 + 24) )
      {
        if ( (_WORD)v39 == *(_WORD *)(v38 + 28) )
        {
          v88 = 24LL * (unsigned int)v12;
          v98 = v25;
          v108 = v12;
          v64 = sub_1302E60(v7, v26);
          if ( !v64 )
            return 0;
          sub_1310140(v7, v98, v98 + v88 + 8, v108, 0);
          _BitScanReverse64((unsigned __int64 *)&v65, 2 * a1 - 1);
          if ( (unsigned __int64)(int)v65 < 7 )
            LOBYTE(v65) = 7;
          v23 = (void *)sub_1309DC0(
                          v7,
                          v64,
                          -(1LL << ((unsigned __int8)v65 - 3)) & ((1LL << ((unsigned __int8)v65 - 3)) + a1 - 1),
                          v112);
          if ( !v23 )
            return 0;
          goto LABEL_62;
        }
        *(_QWORD *)(v38 + 8) = v40;
        *(_WORD *)(v38 + 24) = (_WORD)v40;
      }
      else
      {
        *(_QWORD *)(v38 + 8) = v40;
      }
      if ( (_BYTE)v4 )
      {
        v119 = v25 + v37;
        v66 = memset(v23, 0, qword_505FA40[(unsigned int)v12]);
        v38 = v119;
        v23 = v66;
      }
      ++*(_QWORD *)(v38 + 16);
      goto LABEL_61;
    }
LABEL_92:
    v23 = (void *)sub_1317CF0(v7, v26, a1, v12, v112);
    goto LABEL_61;
  }
  v27 = (unsigned int)v12;
  v28 = v25 + 24LL * (unsigned int)v12;
  v29 = *(void ***)(v28 + 8);
  v23 = *v29;
  v30 = v29 + 1;
  if ( (_WORD)v29 == *(_WORD *)(v28 + 24) )
  {
    if ( (_WORD)v29 == *(_WORD *)(v28 + 28) )
    {
      v81 = (unsigned int)v12;
      v85 = 24LL * (unsigned int)v12;
      v93 = v25;
      v104 = v12;
      v42 = sub_1302E60(v7, v26);
      v43 = v93;
      if ( !v42 )
        return 0;
      if ( !*(_WORD *)(unk_5060A20 + 2 * v81) )
      {
        v23 = (void *)sub_1317CF0(v7, v42, a1, v104, (unsigned __int8)v4);
        goto LABEL_61;
      }
      v44 = v93 + v85 + 8;
      v45 = v93;
      v94 = v42;
      v114 = v43;
      sub_1310140(v7, v45, v44, v104, 1);
      v46 = sub_13100A0(v7, v94, v114, v44, v104);
      v27 = v81;
      v23 = (void *)v46;
      if ( !(_BYTE)v125 )
        return 0;
    }
    else
    {
      *(_QWORD *)(v28 + 8) = v30;
      *(_WORD *)(v28 + 24) = (_WORD)v30;
    }
  }
  else
  {
    *(_QWORD *)(v28 + 8) = v30;
  }
  if ( (_BYTE)v4 )
    v23 = memset(v23, 0, qword_505FA40[v27]);
  ++*(_QWORD *)(v28 + 16);
LABEL_61:
  if ( !v23 )
    return 0;
LABEL_62:
  LOBYTE(v125) = 1;
  v126 = v7 + 824;
  v127 = v7 + 8;
  v128 = v7 + 16;
  v129 = v7 + 832;
  v31 = *(_QWORD *)(v7 + 824);
  *(_QWORD *)(v7 + 824) = v31 + v24;
  if ( *(_QWORD *)(v7 + 16) - v31 <= v24 )
  {
    v113 = v23;
    sub_13133F0(v7, &v125);
    return v113;
  }
  return v23;
}
