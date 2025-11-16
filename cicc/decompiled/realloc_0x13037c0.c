// Function: realloc
// Address: 0x13037c0
//
// Alternative name is '__libc_realloc'
char *__fastcall realloc(unsigned __int64 src, unsigned __int64 a2, int a3, int a4, int a5, int a6)
{
  unsigned __int64 v7; // r12
  signed int v8; // edx
  __int64 v10; // r14
  unsigned int v11; // r13d
  __int64 v12; // rdx
  __int64 v13; // r10
  unsigned __int64 v14; // r15
  __int64 v15; // rbx
  __int64 *v16; // rax
  __int64 v17; // r8
  _QWORD *v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // r12
  __int64 v21; // r14
  unsigned __int64 v22; // r15
  unsigned __int64 v23; // rbx
  __int64 v24; // rdx
  __int64 v25; // rcx
  _QWORD *v26; // rax
  __int64 v27; // r9
  __int64 v28; // rcx
  __int64 v29; // rax
  size_t v30; // r10
  __int64 v31; // rbx
  __int64 v32; // rcx
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // r12
  __int64 v36; // rdi
  unsigned __int8 v37; // bl
  __int64 v38; // rcx
  __int64 v39; // r10
  size_t v40; // r15
  unsigned int v41; // r9d
  __int64 v42; // r11
  void **v43; // rax
  void *v44; // r8
  void **v45; // rsi
  __int64 v46; // rdx
  void *v47; // rdx
  char v48; // cl
  __int64 v49; // rbx
  int v50; // edx
  unsigned __int64 v51; // rax
  __int64 v52; // rax
  __int64 *v53; // rcx
  _QWORD *v54; // rsi
  __int64 v55; // rax
  __int64 v56; // rsi
  int *v57; // rax
  __int64 v58; // rax
  unsigned __int16 v59; // cx
  __int64 v60; // rdx
  __int64 v61; // r15
  __int64 v62; // rax
  __int64 v63; // r11
  void **v64; // rax
  void **v65; // rsi
  __int64 v66; // rdx
  void *v67; // rax
  __int64 v68; // rax
  __int64 v69; // rsi
  _QWORD *v70; // rax
  __int64 v71; // rax
  __int64 v72; // rcx
  __int64 v73; // rax
  __int64 i; // rax
  int v75; // esi
  __int64 v76; // rdi
  __int64 v77; // rsi
  void *v78; // rax
  __int64 v79; // rax
  __int64 v80; // rax
  __int64 v81; // rcx
  __int64 v82; // rax
  __int64 v83; // rdx
  __int64 j; // rax
  int v85; // edi
  __int64 v86; // rcx
  __int64 v87; // rdx
  __int64 v88; // [rsp+8h] [rbp-B8h]
  __int64 v89; // [rsp+10h] [rbp-B0h]
  __int64 v90; // [rsp+18h] [rbp-A8h]
  unsigned int v91; // [rsp+18h] [rbp-A8h]
  char v92; // [rsp+20h] [rbp-A0h]
  unsigned int v93; // [rsp+20h] [rbp-A0h]
  __int64 v94; // [rsp+20h] [rbp-A0h]
  __int64 v95; // [rsp+20h] [rbp-A0h]
  __int64 v96; // [rsp+28h] [rbp-98h]
  __int64 v97; // [rsp+28h] [rbp-98h]
  unsigned int v98; // [rsp+28h] [rbp-98h]
  __int64 v99; // [rsp+28h] [rbp-98h]
  __int64 v100; // [rsp+30h] [rbp-90h]
  __int64 v101; // [rsp+30h] [rbp-90h]
  __int64 v102; // [rsp+30h] [rbp-90h]
  __int64 v103; // [rsp+30h] [rbp-90h]
  void *v104; // [rsp+30h] [rbp-90h]
  __int64 v105; // [rsp+30h] [rbp-90h]
  void *v106; // [rsp+30h] [rbp-90h]
  size_t v107; // [rsp+38h] [rbp-88h]
  __int64 v108; // [rsp+38h] [rbp-88h]
  __int64 v109; // [rsp+38h] [rbp-88h]
  void *v110; // [rsp+38h] [rbp-88h]
  __int64 v111; // [rsp+38h] [rbp-88h]
  unsigned int v112; // [rsp+38h] [rbp-88h]
  size_t v113; // [rsp+38h] [rbp-88h]
  size_t v114; // [rsp+38h] [rbp-88h]
  size_t v115; // [rsp+38h] [rbp-88h]
  __int64 v116; // [rsp+38h] [rbp-88h]
  __int64 v117; // [rsp+38h] [rbp-88h]
  size_t v118; // [rsp+38h] [rbp-88h]
  void *v119; // [rsp+38h] [rbp-88h]
  size_t v120; // [rsp+38h] [rbp-88h]
  __int64 v121; // [rsp+38h] [rbp-88h]
  size_t v122; // [rsp+38h] [rbp-88h]
  unsigned __int64 v123; // [rsp+40h] [rbp-80h] BYREF
  __int128 v124; // [rsp+48h] [rbp-78h]
  unsigned __int64 v125; // [rsp+60h] [rbp-60h] BYREF
  unsigned __int64 v126; // [rsp+68h] [rbp-58h]
  __int64 v127; // [rsp+70h] [rbp-50h]
  __int64 v128; // [rsp+78h] [rbp-48h]
  __int64 v129; // [rsp+80h] [rbp-40h]

  v7 = a2;
  if ( !a2 )
  {
    if ( !src )
      goto LABEL_7;
    _InterlockedAdd64(&qword_4F96998, 1u);
    if ( !unk_4C6F0D8 )
    {
      v8 = 256;
      a2 = 1;
      return sub_13010C0(src, a2, v8, 1, a5, a6);
    }
    if ( unk_4C6F0D8 != 1 )
    {
      sub_130D560((unsigned int)"Called realloc(non-null-ptr, 0) with zero_realloc:abort set\n", 0, a3, a4, a5, a6);
      return 0;
    }
    v20 = __readfsqword(0) - 2664;
    if ( __readfsbyte(0xFFFFF8C8) )
      v20 = sub_1313D30(v20, 0);
    if ( *(_BYTE *)(v20 + 1) )
    {
      v21 = 0;
    }
    else
    {
      v21 = v20 + 856;
      if ( !*(_BYTE *)v20 )
        v21 = 0;
    }
    v123 = src;
    v124 = 0;
    v22 = src & 0xFFFFFFFFC0000000LL;
    sub_1346FC0(3, src, &v123);
    v23 = v20 + ((src >> 26) & 0xF0);
    v24 = v20 + 432;
    v25 = *(_QWORD *)(v23 + 432);
    if ( (src & 0xFFFFFFFFC0000000LL) == v25 )
    {
      v26 = (_QWORD *)(*(_QWORD *)(v23 + 440) + ((src >> 9) & 0x1FFFF8));
    }
    else if ( v22 == *(_QWORD *)(v20 + 688) )
    {
      *(_QWORD *)(v20 + 688) = v25;
      v58 = *(_QWORD *)(v20 + 696);
      *(_QWORD *)(v20 + 696) = *(_QWORD *)(v23 + 440);
LABEL_85:
      *(_QWORD *)(v23 + 440) = v58;
      *(_QWORD *)(v23 + 432) = v22;
      v26 = (_QWORD *)(((src >> 9) & 0x1FFFF8) + v58);
    }
    else
    {
      for ( i = 1; i != 8; ++i )
      {
        v75 = i;
        if ( v22 == *(_QWORD *)(v20 + 16 * i + 688) )
        {
          v76 = v20 + 16 * i;
          v58 = *(_QWORD *)(v76 + 696);
          v77 = v20 + 16LL * (unsigned int)(v75 - 1);
          *(_QWORD *)(v76 + 688) = *(_QWORD *)(v77 + 688);
          *(_QWORD *)(v76 + 696) = *(_QWORD *)(v77 + 696);
          *(_QWORD *)(v77 + 688) = v25;
          *(_QWORD *)(v77 + 696) = *(_QWORD *)(v23 + 440);
          goto LABEL_85;
        }
      }
      v26 = (_QWORD *)sub_130D370(v20, &unk_5060AE0, v24, src, 1, 0);
      v24 = v20 + 432;
    }
    v27 = HIWORD(*v26);
    v28 = *v26 & 1LL;
    v29 = (unsigned int)v27;
    v30 = qword_505FA40[(unsigned int)v27];
    if ( unk_4F969A1 )
    {
      v92 = v28;
      v90 = (unsigned int)v27;
      v96 = v27;
      v100 = v24;
      v107 = qword_505FA40[(unsigned int)v27];
      off_4C6F0B0((void *)src, v30);
      v29 = v90;
      LOBYTE(v28) = v92;
      v27 = v96;
      v24 = v100;
      v30 = v107;
    }
    if ( !v21 )
    {
      v113 = v30;
      sub_12FCB00(v20, src);
      v30 = v113;
      goto LABEL_33;
    }
    if ( (_BYTE)v28 )
    {
      v31 = v21 + 24 * v27;
      v32 = *(_QWORD *)(v31 + 8);
      if ( *(_WORD *)(v21 + 24 * v29 + 26) != (_WORD)v32 )
      {
        *(_QWORD *)(v31 + 8) = v32 - 8;
        *(_QWORD *)(v32 - 8) = src;
        goto LABEL_33;
      }
      v59 = *(_WORD *)(unk_5060A20 + 2 * v27);
      if ( !v59 )
      {
        v120 = v30;
        sub_1315B20(v20, src);
        v30 = v120;
LABEL_33:
        LOBYTE(v125) = 0;
        v126 = v20 + 840;
        v127 = v20 + 24;
        v128 = v20 + 32;
        v129 = v20 + 848;
        v33 = *(_QWORD *)(v20 + 840);
        *(_QWORD *)(v20 + 840) = v30 + v33;
        if ( v30 >= *(_QWORD *)(v20 + 32) - v33 )
          sub_13133F0(v20, &v125);
        return 0;
      }
      v103 = v29;
      v114 = v30;
      sub_13108D0(v20, v21, v21 + 24 * v27 + 8, (unsigned int)v27, (int)v59 >> unk_4C6F1EC);
      v60 = *(_QWORD *)(v31 + 8);
      v30 = v114;
      if ( *(_WORD *)(v21 + 24 * v103 + 26) == (_WORD)v60 )
        goto LABEL_33;
    }
    else
    {
      if ( unk_5060A18 <= (unsigned int)v27 )
      {
        v69 = *(_QWORD *)(v23 + 432);
        if ( v22 == v69 )
        {
          v70 = (_QWORD *)(*(_QWORD *)(v23 + 440) + ((src >> 9) & 0x1FFFF8));
        }
        else if ( v22 == *(_QWORD *)(v20 + 688) )
        {
          *(_QWORD *)(v20 + 688) = v69;
          v79 = *(_QWORD *)(v20 + 696);
          *(_QWORD *)(v20 + 696) = *(_QWORD *)(v23 + 440);
LABEL_130:
          *(_QWORD *)(v23 + 440) = v79;
          *(_QWORD *)(v23 + 432) = v22;
          v70 = (_QWORD *)(((src >> 9) & 0x1FFFF8) + v79);
        }
        else
        {
          for ( j = 1; j != 8; ++j )
          {
            v85 = j;
            if ( v22 == *(_QWORD *)(v20 + 16 * j + 688) )
            {
              v86 = v20 + 16 * j;
              v79 = *(_QWORD *)(v86 + 696);
              v87 = v20 + 16LL * (unsigned int)(v85 - 1);
              *(_QWORD *)(v86 + 688) = *(_QWORD *)(v87 + 688);
              *(_QWORD *)(v86 + 696) = *(_QWORD *)(v87 + 696);
              *(_QWORD *)(v87 + 688) = v69;
              *(_QWORD *)(v87 + 696) = *(_QWORD *)(v23 + 440);
              goto LABEL_130;
            }
          }
          v122 = v30;
          v70 = (_QWORD *)sub_130D370(v20, &unk_5060AE0, v24, src, 1, 0);
          v30 = v122;
        }
        v118 = v30;
        sub_130A160(v20, ((__int64)(*v70 << 16) >> 16) & 0xFFFFFFFFFFFFFF80LL);
        v30 = v118;
        goto LABEL_33;
      }
      v31 = v21 + 24 * v27;
      v61 = v21 + 24 * v29;
      v60 = *(_QWORD *)(v31 + 8);
      if ( *(_WORD *)(v61 + 26) == (_WORD)v60 )
      {
        v115 = v30;
        sub_1310E90(
          v20,
          v21,
          v21 + 24 * v27 + 8,
          (unsigned int)v27,
          (int)*(unsigned __int16 *)(unk_5060A20 + 2 * v27) >> unk_4C6F1E8);
        v62 = *(_QWORD *)(v31 + 8);
        v30 = v115;
        if ( *(_WORD *)(v61 + 26) != (_WORD)v62 )
        {
          *(_QWORD *)(v31 + 8) = v62 - 8;
          *(_QWORD *)(v62 - 8) = src;
        }
        goto LABEL_33;
      }
    }
    *(_QWORD *)(v31 + 8) = v60 - 8;
    *(_QWORD *)(v60 - 8) = src;
    goto LABEL_33;
  }
  if ( src )
  {
    v8 = 0;
    return sub_13010C0(src, a2, v8, 1, a5, a6);
  }
LABEL_7:
  v10 = __readfsqword(0) - 2664;
  if ( __readfsbyte(0xFFFFF8C8) )
  {
    v36 = v10;
    v10 = sub_1313D30(v10, 0);
    if ( *(_BYTE *)(v10 + 816) )
    {
      if ( dword_4C6F034[0] && (unsigned __int8)sub_13022D0(v36, 0) )
        goto LABEL_41;
      v37 = unk_4F96994;
      if ( a2 > 0x1000 )
        v38 = (unsigned int)sub_12FCA60(a2);
      else
        v38 = byte_5060800[(a2 + 7) >> 3];
      if ( (unsigned int)v38 > 0xE7 )
        goto LABEL_77;
      v39 = (unsigned int)v38;
      v40 = qword_505FA40[(unsigned int)v38];
      if ( *(char *)(v10 + 1) > 0 )
      {
        v56 = qword_50579C0[0];
        if ( !qword_50579C0[0] )
        {
          v112 = v38;
          v55 = sub_1300B80(v10, 0, (__int64)&off_49E8000);
          v38 = v112;
          v56 = v55;
          if ( !v55 && !unk_505F9B8 )
            goto LABEL_77;
        }
        v41 = v37;
      }
      else
      {
        v41 = v37;
        if ( *(_BYTE *)v10 )
        {
          v109 = v10 + 856;
          if ( a2 <= 0x3800 )
          {
            v42 = v10 + 24LL * (unsigned int)v38;
            v43 = *(void ***)(v42 + 864);
            v44 = *v43;
            v45 = v43 + 1;
            if ( (_WORD)v43 == *(_WORD *)(v42 + 880) )
            {
              if ( (_WORD)v43 == *(_WORD *)(v42 + 884) )
              {
                v88 = (unsigned int)v38;
                v89 = 24LL * (unsigned int)v38;
                v93 = v38;
                v71 = sub_1303050(v10, 0);
                v72 = v93;
                v105 = v71;
                if ( !v71 )
                  goto LABEL_77;
                if ( !*(_WORD *)(unk_5060A20 + 2 * v88) )
                {
                  v44 = (void *)sub_1317CF0(v10, v71, v7, v93, v37);
                  goto LABEL_55;
                }
                v91 = v93;
                v94 = v109 + v89 + 8;
                sub_1310140(v10, v109, v94, v72, 1);
                v73 = sub_13100A0(v10, v105, v109, v94, v91);
                v42 = v10 + v89;
                v39 = v88;
                v44 = (void *)v73;
                if ( !(_BYTE)v125 )
                  goto LABEL_77;
              }
              else
              {
                *(_QWORD *)(v42 + 864) = v45;
                *(_WORD *)(v42 + 880) = (_WORD)v45;
              }
            }
            else
            {
              *(_QWORD *)(v42 + 864) = v45;
            }
            if ( v37 )
            {
              v116 = v42;
              v67 = memset(v44, 0, qword_505FA40[v39]);
              v42 = v116;
              v44 = v67;
            }
            ++*(_QWORD *)(v42 + 872);
LABEL_55:
            if ( v44 )
            {
LABEL_56:
              LOBYTE(v125) = 1;
              v126 = v10 + 824;
              v127 = v10 + 8;
              v128 = v10 + 16;
              v129 = v10 + 832;
              v46 = *(_QWORD *)(v10 + 824);
              *(_QWORD *)(v10 + 824) = v40 + v46;
              if ( v40 >= *(_QWORD *)(v10 + 16) - v46 )
              {
                v104 = v44;
                sub_13133F0(v10, &v125);
                v44 = v104;
              }
              v47 = v44;
              if ( !v37 && unk_4F969A2 )
              {
                v106 = v44;
                v119 = v44;
                off_4C6F0B8(v44, v40);
                v44 = v119;
                v47 = v106;
              }
              goto LABEL_61;
            }
LABEL_77:
            v57 = __errno_location();
            v47 = 0;
            v44 = 0;
            *v57 = 12;
LABEL_61:
            v125 = src;
            v110 = v44;
            v127 = 0;
            v126 = v7;
            sub_1346E80(8, v44, v47, &v125);
            return (char *)v110;
          }
          if ( a2 <= unk_5060A10 )
          {
            v63 = v10 + 24LL * (unsigned int)v38;
            v64 = *(void ***)(v63 + 864);
            v44 = *v64;
            v65 = v64 + 1;
            if ( (_WORD)v64 == *(_WORD *)(v63 + 880) )
            {
              if ( (_WORD)v64 == *(_WORD *)(v63 + 884) )
              {
                v95 = 24LL * (unsigned int)v38;
                v98 = v38;
                v80 = sub_1303050(v10, 0);
                v81 = v98;
                if ( v80 )
                {
                  v99 = v80;
                  sub_1310140(v10, v109, v109 + v95 + 8, v81, 0);
                  if ( v7 > 0x7000000000000000LL )
                  {
                    v83 = 0;
                  }
                  else
                  {
                    _BitScanReverse64((unsigned __int64 *)&v82, 2 * v7 - 1);
                    if ( (unsigned __int64)(int)v82 < 7 )
                      LOBYTE(v82) = 7;
                    v83 = -(1LL << ((unsigned __int8)v82 - 3)) & ((1LL << ((unsigned __int8)v82 - 3)) + v7 - 1);
                  }
                  v44 = (void *)sub_1309DC0(v10, v99, v83, v37);
                  if ( v44 )
                    goto LABEL_56;
                }
                goto LABEL_77;
              }
              *(_QWORD *)(v63 + 864) = v65;
              *(_WORD *)(v63 + 880) = (_WORD)v65;
            }
            else
            {
              *(_QWORD *)(v63 + 864) = v65;
            }
            if ( v37 )
            {
              v121 = v10 + 24LL * (unsigned int)v38;
              v78 = memset(v44, 0, qword_505FA40[(unsigned int)v38]);
              v63 = v121;
              v44 = v78;
            }
            ++*(_QWORD *)(v63 + 872);
            goto LABEL_55;
          }
        }
        v56 = 0;
      }
      v44 = (void *)sub_1317CF0(v10, v56, v7, v38, v41);
      goto LABEL_55;
    }
  }
  if ( a2 > 0x1000 )
  {
    if ( a2 > 0x7000000000000000LL )
      goto LABEL_41;
    v48 = 7;
    _BitScanReverse64((unsigned __int64 *)&v49, 2 * a2 - 1);
    v50 = 6;
    if ( (unsigned int)v49 >= 7 )
      v48 = v49;
    v51 = (((a2 - 1) & (-1LL << (v48 - 3))) >> (v48 - 3)) & 3;
    if ( (unsigned int)v49 >= 6 )
      v50 = v49;
    v11 = v51 + 4 * v50 - 23;
    if ( (_DWORD)v51 + 4 * v50 == 255 )
      goto LABEL_41;
    v12 = v11;
    v13 = v10 + 856;
    v14 = qword_505FA40[v11];
    if ( a2 > 0x3800 )
    {
      if ( a2 > unk_5060A10 )
      {
        v17 = sub_1317CF0(v10, 0, a2, v11, 0);
      }
      else
      {
        v52 = v10 + 24LL * v11;
        v53 = *(__int64 **)(v52 + 864);
        v17 = *v53;
        v54 = v53 + 1;
        if ( (_WORD)v53 == *(_WORD *)(v52 + 880) )
        {
          v66 = v10 + 24LL * v11;
          if ( (_WORD)v53 == *(_WORD *)(v66 + 884) )
          {
            v68 = sub_1303050(v10, 0);
            if ( !v68 )
              goto LABEL_41;
            v117 = v68;
            sub_1310140(v10, v10 + 856, v10 + 856 + 24LL * v11 + 8, v11, 0);
            if ( (unsigned __int64)(int)v49 < 7 )
              LOBYTE(v49) = 7;
            v17 = sub_1309DC0(
                    v10,
                    v117,
                    -(1LL << ((unsigned __int8)v49 - 3)) & ((1LL << ((unsigned __int8)v49 - 3)) + v7 - 1),
                    0);
            if ( !v17 )
              goto LABEL_41;
LABEL_15:
            LOBYTE(v125) = 1;
            v126 = v10 + 824;
            v127 = v10 + 8;
            v128 = v10 + 16;
            v129 = v10 + 832;
            v19 = *(_QWORD *)(v10 + 824);
            *(_QWORD *)(v10 + 824) = v19 + v14;
            if ( *(_QWORD *)(v10 + 16) - v19 <= v14 )
            {
              v111 = v17;
              sub_13133F0(v10, &v125);
              return (char *)v111;
            }
            return (char *)v17;
          }
          *(_QWORD *)(v52 + 864) = v54;
          *(_WORD *)(v66 + 880) = (_WORD)v54;
        }
        else
        {
          *(_QWORD *)(v52 + 864) = v54;
        }
        ++*(_QWORD *)(v52 + 872);
      }
      goto LABEL_14;
    }
  }
  else
  {
    v11 = byte_5060800[(a2 + 7) >> 3];
    if ( v11 > 0xE7 )
      goto LABEL_41;
    v12 = byte_5060800[(a2 + 7) >> 3];
    v13 = v10 + 856;
    v14 = qword_505FA40[v12];
  }
  v15 = v10 + 24 * v12;
  v16 = *(__int64 **)(v15 + 864);
  v17 = *v16;
  v18 = v16 + 1;
  if ( (_WORD)v16 != *(_WORD *)(v15 + 880) )
  {
    *(_QWORD *)(v15 + 864) = v18;
LABEL_13:
    ++*(_QWORD *)(v15 + 872);
    goto LABEL_14;
  }
  if ( (_WORD)v16 != *(_WORD *)(v15 + 884) )
  {
    *(_QWORD *)(v15 + 864) = v18;
    *(_WORD *)(v15 + 880) = (_WORD)v18;
    goto LABEL_13;
  }
  v97 = v12;
  v101 = 24 * v12;
  v108 = v13;
  v34 = sub_1302E60(v10, 0);
  if ( !v34 )
    goto LABEL_41;
  if ( *(_WORD *)(unk_5060A20 + 2 * v97) )
  {
    v35 = v108 + v101 + 8;
    v102 = v34;
    sub_1310140(v10, v108, v35, v11, 1);
    v17 = sub_13100A0(v10, v102, v108, v35, v11);
    if ( !(_BYTE)v125 )
      goto LABEL_41;
    goto LABEL_13;
  }
  v17 = sub_1317CF0(v10, v34, v7, v11, 0);
LABEL_14:
  if ( v17 )
    goto LABEL_15;
LABEL_41:
  *__errno_location() = 12;
  return 0;
}
