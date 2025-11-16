// Function: sub_1134F50
// Address: 0x1134f50
//
unsigned __int8 *__fastcall sub_1134F50(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 *v4; // r14
  __int64 v5; // r12
  __int64 v6; // r15
  __int64 v7; // r13
  char v8; // dl
  unsigned __int16 v9; // bx
  char v10; // al
  char v11; // al
  __int64 *v12; // r13
  _BYTE *v13; // rsi
  _BYTE *v14; // r13
  __int64 v15; // rbx
  __int64 v16; // rax
  __int64 v17; // r13
  unsigned int **v18; // rdi
  __int64 v19; // rax
  unsigned __int8 *result; // rax
  __int64 *v21; // rdi
  __int64 v22; // rdx
  char *v23; // rax
  char *v24; // rsi
  __int64 v25; // rcx
  __int64 v26; // rdx
  char *v27; // rdx
  _BYTE *v28; // rax
  __int64 v29; // r10
  __int64 v30; // rcx
  __int64 v31; // rax
  unsigned int v32; // esi
  unsigned int v35; // ecx
  __int64 *v36; // r11
  __int64 *v37; // r12
  __int64 v38; // rsi
  __int64 v39; // r9
  __int64 v40; // rdx
  unsigned __int64 v41; // r8
  bool v42; // al
  unsigned int v43; // eax
  unsigned int v44; // eax
  __int64 **v45; // rax
  __int64 v46; // r10
  __int64 *v47; // rax
  unsigned __int16 v48; // r12
  __int64 v49; // rbx
  __int64 v50; // rax
  __int64 v51; // rsi
  __int64 v52; // rbx
  int v53; // eax
  __int64 v54; // rax
  __int64 v55; // r15
  __int64 v56; // r10
  __int64 v57; // rax
  __int64 v58; // rbx
  __int64 v59; // rbx
  __int64 v60; // r12
  __int64 v61; // rdx
  unsigned int v62; // esi
  __int64 v63; // rcx
  __int64 v64; // rbx
  __int64 v65; // rdx
  __int64 *v66; // rbx
  __int64 *v67; // rdx
  const char *v68; // rsi
  __int64 v69; // r8
  __int64 *v70; // rcx
  const char *v71; // r12
  __int64 *v72; // r13
  __int64 v73; // r14
  __int64 v74; // rbx
  int v75; // eax
  int v76; // eax
  unsigned int v77; // edi
  __int64 v78; // rax
  __int64 v79; // rdi
  __int64 v80; // rdi
  __int64 v81; // rdx
  int v82; // r8d
  int v83; // eax
  __int64 v84; // rdx
  unsigned __int64 v85; // rax
  _QWORD *v86; // rax
  __int64 *v87; // rax
  const char *v88; // rax
  __int64 v89; // rdx
  const char *v90; // rax
  __int64 v91; // rdx
  const char *v92; // rax
  unsigned __int64 v93; // rsi
  __int64 v94; // rax
  __int64 v95; // rax
  __int64 v96; // rax
  const char *v97; // rdx
  unsigned __int64 v98; // rax
  __int64 *v99; // [rsp+8h] [rbp-108h]
  __int64 *v100; // [rsp+8h] [rbp-108h]
  __int64 v101; // [rsp+10h] [rbp-100h]
  __int64 *v102; // [rsp+10h] [rbp-100h]
  __int64 v103; // [rsp+18h] [rbp-F8h]
  __int64 *v104; // [rsp+18h] [rbp-F8h]
  __int64 v105; // [rsp+18h] [rbp-F8h]
  __int64 v106; // [rsp+18h] [rbp-F8h]
  __int64 v107; // [rsp+18h] [rbp-F8h]
  __int64 v108; // [rsp+18h] [rbp-F8h]
  char v109; // [rsp+18h] [rbp-F8h]
  __int64 v110; // [rsp+20h] [rbp-F0h]
  __int64 *v111; // [rsp+20h] [rbp-F0h]
  __int64 *v112; // [rsp+20h] [rbp-F0h]
  __int64 *v113; // [rsp+20h] [rbp-F0h]
  __int64 *v114; // [rsp+20h] [rbp-F0h]
  __int64 *v115; // [rsp+20h] [rbp-F0h]
  __int64 v116; // [rsp+20h] [rbp-F0h]
  __int64 *v117; // [rsp+20h] [rbp-F0h]
  __int64 v118; // [rsp+28h] [rbp-E8h]
  unsigned __int8 *v119; // [rsp+28h] [rbp-E8h]
  __int64 v120; // [rsp+28h] [rbp-E8h]
  __int64 v121; // [rsp+28h] [rbp-E8h]
  __int64 v122; // [rsp+28h] [rbp-E8h]
  __int64 *v123; // [rsp+28h] [rbp-E8h]
  __int64 v124; // [rsp+28h] [rbp-E8h]
  __int64 v125; // [rsp+28h] [rbp-E8h]
  __int64 *v126; // [rsp+30h] [rbp-E0h]
  unsigned int v127; // [rsp+30h] [rbp-E0h]
  int v128; // [rsp+30h] [rbp-E0h]
  __int64 v129; // [rsp+30h] [rbp-E0h]
  int v130; // [rsp+30h] [rbp-E0h]
  __int64 v131; // [rsp+30h] [rbp-E0h]
  unsigned int **v132; // [rsp+30h] [rbp-E0h]
  unsigned __int64 v133; // [rsp+38h] [rbp-D8h]
  __int64 v134; // [rsp+38h] [rbp-D8h]
  unsigned __int8 *v135; // [rsp+38h] [rbp-D8h]
  const char *v136; // [rsp+38h] [rbp-D8h]
  __int64 *v137; // [rsp+38h] [rbp-D8h]
  __int64 v138[4]; // [rsp+40h] [rbp-D0h] BYREF
  __int16 v139; // [rsp+60h] [rbp-B0h]
  const char *v140[4]; // [rsp+70h] [rbp-A0h] BYREF
  __int16 v141; // [rsp+90h] [rbp-80h]
  const char *v142; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v143; // [rsp+A8h] [rbp-68h]
  _QWORD v144[2]; // [rsp+B0h] [rbp-60h] BYREF
  __int16 v145; // [rsp+C0h] [rbp-50h]

  v4 = (__int64 *)a1;
  v5 = a2;
  v6 = *(_QWORD *)(a2 - 32);
  v7 = *(_QWORD *)(a2 - 64);
  v8 = *(_BYTE *)v6;
  v9 = *(_WORD *)(a2 + 2) & 0x3F;
  v10 = *(_BYTE *)v6;
  if ( v9 != 34 || v8 != 17 )
  {
LABEL_2:
    if ( (unsigned __int8)v10 > 0x15u )
      return 0;
LABEL_3:
    v11 = *(_BYTE *)v7;
    if ( *(_BYTE *)v7 <= 0x1Cu )
      return sub_F16720(v4, (unsigned __int8 *)v5);
    if ( v11 != 84 )
    {
      if ( v11 == 86 )
      {
        if ( (*(_BYTE *)(v7 + 7) & 0x40) != 0 )
        {
          v12 = *(__int64 **)(v7 - 8);
          v118 = *v12;
          if ( *v12 )
            goto LABEL_8;
        }
        else
        {
          v12 = (__int64 *)(v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF));
          v118 = *v12;
          if ( *v12 )
          {
LABEL_8:
            v13 = (_BYTE *)v12[4];
            if ( *v13 <= 0x15u )
            {
              v14 = (_BYTE *)v12[8];
              if ( *v14 <= 0x15u )
              {
                v126 = a3;
                v133 = v9;
                v15 = sub_1016CC0(v9 & 0x3F, v13, (_BYTE *)v6, a3);
                v16 = sub_1016CC0(v133, v14, (_BYTE *)v6, v126);
                v17 = v16;
                if ( v15 )
                {
                  if ( v16 )
                  {
                    sub_D5F1F0(v4[4], v5);
                    v18 = (unsigned int **)v4[4];
                    v145 = 257;
                    v19 = sub_B36550(v18, v118, v15, v17, (__int64)&v142, 0);
                    return sub_F162A0((__int64)v4, v5, v19);
                  }
                }
              }
            }
          }
        }
      }
      return sub_F16720(v4, (unsigned __int8 *)v5);
    }
    v21 = (__int64 *)v7;
    v22 = 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF);
    v23 = (char *)(v7 - v22);
    if ( (*(_BYTE *)(v7 + 7) & 0x40) != 0 )
      v23 = *(char **)(v7 - 8);
    v24 = &v23[v22];
    v25 = v22 >> 5;
    v26 = v22 >> 7;
    if ( v26 )
    {
      v27 = &v23[128 * v26];
      while ( **(_BYTE **)v23 <= 0x15u )
      {
        if ( **((_BYTE **)v23 + 4) > 0x15u )
        {
          v23 += 32;
          break;
        }
        if ( **((_BYTE **)v23 + 8) > 0x15u )
        {
          v23 += 64;
          break;
        }
        if ( **((_BYTE **)v23 + 12) > 0x15u )
        {
          v23 += 96;
          break;
        }
        v23 += 128;
        if ( v27 == v23 )
        {
          v25 = (v24 - v23) >> 5;
          goto LABEL_46;
        }
      }
LABEL_22:
      if ( v24 != v23 )
        return sub_F16720(v4, (unsigned __int8 *)v5);
      goto LABEL_50;
    }
LABEL_46:
    if ( v25 != 2 )
    {
      if ( v25 != 3 )
      {
        if ( v25 != 1 )
          goto LABEL_50;
        goto LABEL_49;
      }
      if ( **(_BYTE **)v23 > 0x15u )
        goto LABEL_22;
      v23 += 32;
    }
    if ( **(_BYTE **)v23 > 0x15u )
      goto LABEL_22;
    v23 += 32;
LABEL_49:
    if ( **(_BYTE **)v23 > 0x15u )
      goto LABEL_22;
LABEL_50:
    v142 = (const char *)v144;
    v143 = 0x600000000LL;
    if ( (*(_BYTE *)(v7 + 7) & 0x40) != 0 )
    {
      v36 = *(__int64 **)(v7 - 8);
      v21 = &v36[4 * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF)];
    }
    else
    {
      v36 = (__int64 *)(v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF));
    }
    if ( v36 == v21 )
    {
LABEL_75:
      sub_D5F1F0(v4[4], v7);
      v51 = *(_QWORD *)(v5 + 8);
      v139 = 257;
      v52 = v4[4];
      v53 = *(_DWORD *)(v7 + 4) & 0x7FFFFFF;
      v141 = 257;
      v128 = v53;
      v54 = sub_BD2DA0(80);
      v55 = v54;
      if ( v54 )
      {
        v122 = v54;
        sub_B44260(v54, v51, 55, 0x8000000u, 0, 0);
        *(_DWORD *)(v55 + 72) = v128;
        sub_BD6B50((unsigned __int8 *)v55, v140);
        sub_BD2A10(v55, *(_DWORD *)(v55 + 72), 1);
        v56 = v122;
      }
      else
      {
        v56 = 0;
      }
      if ( (unsigned __int8)sub_920620(v56) )
      {
        v81 = *(_QWORD *)(v52 + 96);
        v82 = *(_DWORD *)(v52 + 104);
        if ( v81 )
        {
          v130 = *(_DWORD *)(v52 + 104);
          sub_B99FD0(v55, 3u, v81);
          v82 = v130;
        }
        sub_B45150(v55, v82);
      }
      (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v52 + 88) + 16LL))(
        *(_QWORD *)(v52 + 88),
        v55,
        v138,
        *(_QWORD *)(v52 + 56),
        *(_QWORD *)(v52 + 64));
      v57 = *(_QWORD *)v52;
      v58 = 16LL * *(unsigned int *)(v52 + 8);
      if ( v57 != v57 + v58 )
      {
        v129 = v5;
        v59 = v57 + v58;
        v60 = v57;
        do
        {
          v61 = *(_QWORD *)(v60 + 8);
          v62 = *(_DWORD *)v60;
          v60 += 16;
          sub_B99FD0(v55, v62, v61);
        }
        while ( v59 != v60 );
        v5 = v129;
      }
      v63 = *(_QWORD *)(v7 - 8);
      v64 = 32LL * *(unsigned int *)(v7 + 72);
      v65 = v64 + 8LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF);
      v66 = (__int64 *)(v63 + v64);
      v67 = (__int64 *)(v63 + v65);
      v68 = &v142[8 * (unsigned int)v143];
      if ( v67 != v66 && v68 != v142 )
      {
        v69 = v5;
        v70 = v4;
        v71 = v142;
        v72 = v66;
        do
        {
          v73 = *v72;
          v74 = *(_QWORD *)v71;
          v75 = *(_DWORD *)(v55 + 4) & 0x7FFFFFF;
          if ( v75 == *(_DWORD *)(v55 + 72) )
          {
            v106 = v69;
            v114 = v70;
            v123 = v67;
            sub_B48D90(v55);
            v69 = v106;
            v70 = v114;
            v67 = v123;
            v75 = *(_DWORD *)(v55 + 4) & 0x7FFFFFF;
          }
          v76 = (v75 + 1) & 0x7FFFFFF;
          v77 = v76 | *(_DWORD *)(v55 + 4) & 0xF8000000;
          v78 = *(_QWORD *)(v55 - 8) + 32LL * (unsigned int)(v76 - 1);
          *(_DWORD *)(v55 + 4) = v77;
          if ( *(_QWORD *)v78 )
          {
            v79 = *(_QWORD *)(v78 + 8);
            **(_QWORD **)(v78 + 16) = v79;
            if ( v79 )
              *(_QWORD *)(v79 + 16) = *(_QWORD *)(v78 + 16);
          }
          *(_QWORD *)v78 = v74;
          if ( v74 )
          {
            v80 = *(_QWORD *)(v74 + 16);
            *(_QWORD *)(v78 + 8) = v80;
            if ( v80 )
              *(_QWORD *)(v80 + 16) = v78 + 8;
            *(_QWORD *)(v78 + 16) = v74 + 16;
            *(_QWORD *)(v74 + 16) = v78;
          }
          ++v72;
          v71 += 8;
          *(_QWORD *)(*(_QWORD *)(v55 - 8)
                    + 32LL * *(unsigned int *)(v55 + 72)
                    + 8LL * ((*(_DWORD *)(v55 + 4) & 0x7FFFFFFu) - 1)) = v73;
        }
        while ( v67 != v72 && v68 != v71 );
        v4 = v70;
        v5 = v69;
      }
      v38 = v5;
      result = sub_F162A0((__int64)v4, v5, v55);
    }
    else
    {
      v103 = v5;
      v37 = v36;
      while ( 1 )
      {
        v38 = *v37;
        result = (unsigned __int8 *)sub_9719A0(v9, (_BYTE *)*v37, v6, v4[11], 0, 0);
        if ( !result )
          break;
        v40 = (unsigned int)v143;
        v41 = (unsigned int)v143 + 1LL;
        if ( v41 > HIDWORD(v143) )
        {
          v119 = result;
          sub_C8D5F0((__int64)&v142, v144, (unsigned int)v143 + 1LL, 8u, v41, v39);
          v40 = (unsigned int)v143;
          result = v119;
        }
        v37 += 4;
        *(_QWORD *)&v142[8 * v40] = result;
        LODWORD(v143) = v143 + 1;
        if ( v21 == v37 )
        {
          v5 = v103;
          goto LABEL_75;
        }
      }
    }
    if ( v142 != (const char *)v144 )
    {
      v135 = result;
      _libc_free(v142, v38);
      return v135;
    }
    return result;
  }
  if ( *(_BYTE *)v7 != 42 )
    goto LABEL_3;
  v28 = *(_BYTE **)(v7 - 64);
  if ( *v28 != 42 )
    goto LABEL_3;
  v134 = *((_QWORD *)v28 - 8);
  if ( !v134 )
    goto LABEL_3;
  v29 = *((_QWORD *)v28 - 4);
  if ( !v29 )
    goto LABEL_3;
  v30 = *(_QWORD *)(v7 - 32);
  if ( *(_BYTE *)v30 != 17 )
    goto LABEL_3;
  v31 = *(_QWORD *)(v7 + 16);
  if ( !v31 || *(_QWORD *)(v31 + 8) )
    goto LABEL_3;
  v32 = *(_DWORD *)(v30 + 32);
  if ( v32 > 0x40 )
  {
    v115 = a3;
    v124 = v29;
    v131 = v30 + 24;
    v83 = sub_C44630(v30 + 24);
    a3 = v115;
    if ( v83 != 1 )
      goto LABEL_3;
    LODWORD(_RAX) = sub_C44590(v131);
    v8 = 17;
    a3 = v115;
    v29 = v124;
  }
  else
  {
    _RAX = *(_QWORD *)(v30 + 24);
    if ( !_RAX || (_RAX & (_RAX - 1)) != 0 )
      goto LABEL_3;
    __asm { tzcnt   rax, rax }
    if ( (unsigned int)_RAX > v32 )
      LODWORD(_RAX) = *(_DWORD *)(v30 + 32);
  }
  if ( (_RAX & 0xFFFFFFF7) != 7 && (_DWORD)_RAX != 31 )
    goto LABEL_3;
  v35 = *(_DWORD *)(v6 + 32);
  v127 = _RAX + 1;
  if ( (_DWORD)_RAX + 1 == v35 )
  {
    v10 = v8;
    goto LABEL_2;
  }
  LODWORD(v143) = *(_DWORD *)(v6 + 32);
  if ( v35 <= 0x40 )
  {
    v142 = 0;
    if ( (_DWORD)_RAX == -1 )
    {
      if ( *(_QWORD *)(v6 + 24) )
      {
LABEL_130:
        v10 = *(_BYTE *)v6;
        goto LABEL_2;
      }
      goto LABEL_66;
    }
    if ( v127 > 0x40 )
      goto LABEL_62;
    v97 = 0;
    v98 = 0xFFFFFFFFFFFFFFFFLL >> (63 - (unsigned __int8)_RAX);
LABEL_128:
    v142 = (const char *)((unsigned __int64)v97 | v98);
    goto LABEL_63;
  }
  v109 = _RAX;
  v102 = a3;
  v116 = v29;
  sub_C43690((__int64)&v142, 0, 0);
  v29 = v116;
  a3 = v102;
  if ( !v127 )
    goto LABEL_63;
  if ( v127 <= 0x40 )
  {
    v97 = v142;
    v98 = 0xFFFFFFFFFFFFFFFFLL >> (63 - v109);
    if ( (unsigned int)v143 > 0x40 )
    {
      *(_QWORD *)v142 |= v98;
      goto LABEL_63;
    }
    goto LABEL_128;
  }
LABEL_62:
  v104 = a3;
  v110 = v29;
  sub_C43C90(&v142, 0, v127);
  a3 = v104;
  v29 = v110;
LABEL_63:
  if ( *(_DWORD *)(v6 + 32) > 0x40u )
  {
    v111 = a3;
    v120 = v29;
    v42 = sub_C43C50(v6 + 24, (const void **)&v142);
    a3 = v111;
    if ( v42 )
    {
      v29 = v120;
      goto LABEL_66;
    }
LABEL_120:
    if ( (unsigned int)v143 > 0x40 && v142 )
    {
      v137 = a3;
      j_j___libc_free_0_0(v142);
      a3 = v137;
      v10 = *(_BYTE *)v6;
      goto LABEL_2;
    }
    goto LABEL_130;
  }
  if ( *(const char **)(v6 + 24) != v142 )
    goto LABEL_120;
LABEL_66:
  if ( (unsigned int)v143 > 0x40 && v142 )
  {
    v117 = a3;
    v125 = v29;
    j_j___libc_free_0_0(v142);
    v29 = v125;
    a3 = v117;
  }
  v112 = a3;
  v121 = v29;
  v43 = sub_9AF930(v134, *(_QWORD *)(a1 + 88), 0, *(_QWORD *)(a1 + 64), v5, *(_QWORD *)(a1 + 80));
  a3 = v112;
  if ( v127 < v43 )
    goto LABEL_118;
  v44 = sub_9AF930(v121, *(_QWORD *)(a1 + 88), 0, *(_QWORD *)(a1 + 64), v5, *(_QWORD *)(a1 + 80));
  a3 = v112;
  if ( v127 < v44 )
    goto LABEL_118;
  v45 = (__int64 **)sub_986520(v7);
  v46 = v121;
  a3 = v112;
  v47 = *v45;
  v113 = v47;
  if ( v47[2] )
  {
    v105 = v5;
    v48 = v9;
    v49 = v47[2];
    do
    {
      v50 = *(_QWORD *)(v49 + 24);
      if ( v7 != v50 )
      {
        if ( *(_BYTE *)v50 != 67
          || (v99 = a3,
              v142 = (const char *)sub_BCAE30(*(_QWORD *)(v50 + 8)),
              v143 = v84,
              v85 = sub_CA1930(&v142),
              a3 = v99,
              v85 > v127) )
        {
          v9 = v48;
          v4 = (__int64 *)a1;
          v10 = *(_BYTE *)v6;
          v5 = v105;
          goto LABEL_2;
        }
      }
      v49 = *(_QWORD *)(v49 + 8);
    }
    while ( v49 );
    v9 = v48;
    v46 = v121;
    v5 = v105;
    v4 = (__int64 *)a1;
  }
  v100 = a3;
  v101 = v46;
  v86 = (_QWORD *)sub_BD5C60((__int64)v113);
  v138[0] = sub_BCCE00(v86, v127);
  v87 = (__int64 *)sub_B43CA0(v5);
  v107 = sub_B6E160(v87, 0x138u, (__int64)v138, 1);
  v132 = (unsigned int **)v4[4];
  sub_D5F1F0((__int64)v132, (__int64)v113);
  v88 = sub_BD5D20(v134);
  v143 = v89;
  v144[0] = ".trunc";
  v145 = 773;
  v142 = v88;
  v136 = (const char *)sub_A82DA0(v132, v134, v138[0], (__int64)&v142, 0, 0);
  v90 = sub_BD5D20(v101);
  v143 = v91;
  v144[0] = ".trunc";
  v145 = 773;
  v142 = v90;
  v92 = (const char *)sub_A82DA0(v132, v101, v138[0], (__int64)&v142, 0, 0);
  v142 = "sadd";
  v145 = 259;
  v140[0] = v136;
  v93 = 0;
  v140[1] = v92;
  if ( v107 )
    v93 = *(_QWORD *)(v107 + 24);
  v94 = sub_921880(v132, v93, v107, (int)v140, 2, (__int64)&v142, 0);
  v142 = "sadd.result";
  v145 = 259;
  LODWORD(v140[0]) = 0;
  v108 = v94;
  v95 = sub_94D3D0(v132, v94, (__int64)v140, 1, (__int64)&v142);
  v145 = 257;
  v96 = sub_A82F30(v132, v95, v113[1], (__int64)&v142, 0);
  sub_F162A0((__int64)v4, (__int64)v113, v96);
  sub_F207A0((__int64)v4, v113);
  v142 = "sadd.overflow";
  v145 = 259;
  LODWORD(v140[0]) = 1;
  result = (unsigned __int8 *)sub_9C6C50(v108, (__int64)v140, 1, (__int64)&v142, 0, 0);
  a3 = v100;
  if ( !result )
  {
LABEL_118:
    v10 = *(_BYTE *)v6;
    goto LABEL_2;
  }
  return result;
}
