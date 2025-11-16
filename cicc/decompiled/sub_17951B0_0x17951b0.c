// Function: sub_17951B0
// Address: 0x17951b0
//
unsigned __int8 *__fastcall sub_17951B0(_WORD *a1, __int64 a2, __int64 a3, __int64 a4, double a5, double a6, double a7)
{
  __int64 v8; // r12
  _WORD *v9; // rbx
  __int64 v10; // rcx
  unsigned __int8 v11; // al
  __int64 v12; // rdx
  __int64 v13; // r13
  __int64 *v14; // r9
  int v15; // r8d
  int v16; // r15d
  __int64 v17; // rsi
  unsigned __int8 v18; // al
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // rax
  int v22; // eax
  int v23; // eax
  __int64 *v24; // rax
  __int64 v25; // rdx
  unsigned __int64 v26; // rcx
  unsigned int v27; // edx
  bool v28; // al
  __int64 v30; // rdx
  __int64 v31; // rcx
  unsigned int v32; // eax
  int v33; // eax
  int v34; // eax
  __int64 *v35; // r13
  unsigned int v36; // eax
  char v37; // cl
  char v38; // dl
  unsigned int v39; // r9d
  unsigned int v40; // r10d
  unsigned int v41; // esi
  unsigned __int64 v42; // r9
  bool v43; // si
  __int64 v44; // rdi
  unsigned int v45; // eax
  bool v46; // cl
  int v47; // edi
  unsigned int v48; // r9d
  unsigned int v49; // esi
  __int64 v50; // rdx
  int v51; // eax
  int v52; // edx
  __int64 v53; // rcx
  __int64 v54; // rax
  __int64 *v55; // r12
  __int64 v56; // rax
  __int64 v57; // rdx
  __int64 *v58; // r12
  __int64 v59; // rax
  _BYTE *v60; // r9
  unsigned int v61; // edx
  int v62; // r12d
  unsigned int v63; // eax
  unsigned int v64; // eax
  __int64 v65; // rdx
  __int64 v66; // rdx
  __int64 *v67; // rax
  __int64 *v68; // rbx
  __int64 v69; // rax
  __int64 v70; // rax
  __int64 v71; // rbx
  __int64 v72; // rax
  unsigned __int8 *v73; // rax
  __int64 v74; // rax
  __int64 v75; // rdi
  __int64 v76; // rax
  __int64 v77; // rdi
  unsigned __int64 *v78; // rbx
  __int64 v79; // rax
  unsigned __int64 v80; // rcx
  __int64 *v81; // rsi
  __int64 *v82; // rdi
  __int64 v83; // rdx
  bool v84; // zf
  __int64 v85; // rsi
  __int64 v86; // rsi
  unsigned __int8 *v87; // rsi
  unsigned int v88; // r15d
  bool v89; // al
  __int64 v90; // rax
  _BYTE *v91; // r9
  __int64 v92; // rax
  unsigned __int64 *v93; // rbx
  __int64 v94; // rax
  unsigned __int64 v95; // rcx
  __int64 v96; // rsi
  __int64 v97; // rsi
  unsigned __int8 *v98; // rsi
  __int64 v99; // rax
  _BYTE *v100; // r9
  unsigned int v101; // r15d
  __int64 v102; // r13
  unsigned int j; // ebx
  __int64 v104; // rax
  char v105; // dl
  unsigned int v106; // edx
  __int64 v107; // rbx
  unsigned int v108; // r12d
  __int64 v109; // rax
  char v110; // dl
  bool v111; // al
  __int64 v112; // rbx
  unsigned int i; // r12d
  __int64 v114; // rax
  int v115; // eax
  bool v116; // al
  unsigned int v117; // [rsp+0h] [rbp-110h]
  int v118; // [rsp+4h] [rbp-10Ch]
  char v119; // [rsp+8h] [rbp-108h]
  unsigned int v120; // [rsp+8h] [rbp-108h]
  char v121; // [rsp+8h] [rbp-108h]
  unsigned int v122; // [rsp+Ch] [rbp-104h]
  char v123; // [rsp+13h] [rbp-FDh]
  char v124; // [rsp+14h] [rbp-FCh]
  unsigned int v125; // [rsp+14h] [rbp-FCh]
  __int64 v126; // [rsp+18h] [rbp-F8h]
  int v127; // [rsp+18h] [rbp-F8h]
  int v128; // [rsp+18h] [rbp-F8h]
  bool v129; // [rsp+20h] [rbp-F0h]
  unsigned __int8 v130; // [rsp+20h] [rbp-F0h]
  __int64 v131; // [rsp+20h] [rbp-F0h]
  unsigned int v132; // [rsp+20h] [rbp-F0h]
  unsigned int v133; // [rsp+28h] [rbp-E8h]
  unsigned int v134; // [rsp+28h] [rbp-E8h]
  __int64 v135; // [rsp+28h] [rbp-E8h]
  _BYTE *v136; // [rsp+30h] [rbp-E0h]
  unsigned int v137; // [rsp+30h] [rbp-E0h]
  int v138; // [rsp+30h] [rbp-E0h]
  _BYTE *v139; // [rsp+30h] [rbp-E0h]
  unsigned int v140; // [rsp+30h] [rbp-E0h]
  _BYTE *v141; // [rsp+30h] [rbp-E0h]
  int v142; // [rsp+30h] [rbp-E0h]
  int v143; // [rsp+30h] [rbp-E0h]
  int v144; // [rsp+30h] [rbp-E0h]
  int v145; // [rsp+30h] [rbp-E0h]
  int v146; // [rsp+30h] [rbp-E0h]
  int v147; // [rsp+30h] [rbp-E0h]
  unsigned int v149; // [rsp+38h] [rbp-D8h]
  __int64 v150; // [rsp+48h] [rbp-C8h] BYREF
  __int64 *v151; // [rsp+50h] [rbp-C0h] BYREF
  __int64 *v152; // [rsp+58h] [rbp-B8h] BYREF
  __int64 v153[2]; // [rsp+60h] [rbp-B0h] BYREF
  __int16 v154; // [rsp+70h] [rbp-A0h]
  __int64 v155[2]; // [rsp+80h] [rbp-90h] BYREF
  __int16 v156; // [rsp+90h] [rbp-80h]
  __int64 v157; // [rsp+A0h] [rbp-70h] BYREF
  unsigned int v158; // [rsp+A8h] [rbp-68h]
  __int16 v159; // [rsp+B0h] [rbp-60h]
  unsigned __int8 *v160; // [rsp+C0h] [rbp-50h] BYREF
  __int64 *v161; // [rsp+C8h] [rbp-48h]
  __int16 v162; // [rsp+D0h] [rbp-40h]

  v8 = a2;
  v9 = a1;
  v10 = *(_QWORD *)a2;
  v11 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
  v12 = v11;
  if ( v11 == 16 )
    v12 = *(unsigned __int8 *)(**(_QWORD **)(v10 + 16) + 8LL);
  if ( (_BYTE)v12 != 11 )
    return 0;
  LOBYTE(v12) = v11 == 16;
  if ( (v11 == 16) != (*(_BYTE *)(*(_QWORD *)a1 + 8LL) == 16) )
    return 0;
  v13 = *((_QWORD *)a1 - 6);
  v14 = (__int64 *)*((_QWORD *)a1 - 3);
  v15 = a1[9] & 0x7FFF;
  v16 = v15;
  if ( (unsigned int)(v15 - 32) <= 1 )
  {
    if ( *((_BYTE *)v14 + 16) > 0x10u )
      return 0;
    v139 = (_BYTE *)*((_QWORD *)a1 - 3);
    if ( sub_1593BB0((__int64)v139, a2, v12, v10) )
      goto LABEL_27;
    if ( v139[16] == 13 )
    {
      v88 = *((_DWORD *)v139 + 8);
      if ( v88 <= 0x40 )
        v89 = *((_QWORD *)v139 + 3) == 0;
      else
        v89 = v88 == (unsigned int)sub_16A57B0((__int64)(v139 + 24));
LABEL_97:
      if ( !v89 )
        return 0;
      goto LABEL_27;
    }
    if ( *(_BYTE *)(*(_QWORD *)v139 + 8LL) != 16 )
      return 0;
    v99 = sub_15A1020(v139, a2, v30, v31);
    v100 = v139;
    if ( v99 && *(_BYTE *)(v99 + 16) == 13 )
    {
      v101 = *(_DWORD *)(v99 + 32);
      if ( v101 <= 0x40 )
      {
        v89 = *(_QWORD *)(v99 + 24) == 0;
        goto LABEL_97;
      }
      if ( v101 != (unsigned int)sub_16A57B0(v99 + 24) )
        return 0;
    }
    else
    {
      v147 = *(_QWORD *)(*(_QWORD *)v139 + 32LL);
      if ( v147 )
      {
        v112 = (__int64)v100;
        for ( i = 0; i != v147; ++i )
        {
          v114 = sub_15A0A60(v112, i);
          if ( !v114 )
            return 0;
          v31 = *(unsigned __int8 *)(v114 + 16);
          if ( (_BYTE)v31 != 9 )
          {
            if ( (_BYTE)v31 != 13 )
              return 0;
            v31 = *(unsigned int *)(v114 + 32);
            if ( (unsigned int)v31 <= 0x40 )
            {
              v116 = *(_QWORD *)(v114 + 24) == 0;
            }
            else
            {
              v132 = *(_DWORD *)(v114 + 32);
              v115 = sub_16A57B0(v114 + 24);
              v31 = v132;
              v116 = v132 == v115;
            }
            if ( !v116 )
              return 0;
          }
        }
        v8 = a2;
        v9 = a1;
      }
    }
LABEL_27:
    v161 = &v157;
    if ( !(unsigned __int8)sub_1793FF0((__int64)&v160, v13, v30, v31) )
      return 0;
    v32 = *(_DWORD *)(v157 + 8);
    if ( v32 <= 0x40 )
    {
      v137 = -1;
      v26 = *(_QWORD *)v157;
      if ( *(_QWORD *)v157 )
      {
        _BitScanReverse64(&v26, v26);
        v26 ^= 0x3Fu;
        v137 = 63 - v26;
      }
    }
    else
    {
      v140 = v32 - 1;
      v33 = sub_16A57B0(v157);
      v26 = v140 - v33;
      v137 = v140 - v33;
    }
    v34 = (unsigned __int16)v9[9];
    v126 = v13;
    v124 = 0;
    BYTE1(v34) &= ~0x80u;
    v129 = v34 == 32;
    goto LABEL_31;
  }
  v17 = (v15 - 38) & 0xFFFFFFFD;
  if ( ((v15 - 38) & 0xFFFFFFFD) != 0 )
    return 0;
  v18 = *((_BYTE *)v14 + 16);
  if ( v15 == 38 )
  {
    if ( v18 == 13 )
    {
      v27 = *((_DWORD *)v14 + 8);
      if ( v27 <= 0x40 )
      {
        v28 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v27) == v14[3];
      }
      else
      {
        v138 = *((_DWORD *)v14 + 8);
        v28 = v138 == (unsigned int)sub_16A58F0((__int64)(v14 + 3));
      }
    }
    else
    {
      if ( *(_BYTE *)(*v14 + 8) != 16 || v18 > 0x10u )
        return 0;
      v141 = (_BYTE *)*((_QWORD *)a1 - 3);
      v59 = sub_15A1020(v141, v17, v12, *v14);
      v60 = v141;
      if ( !v59 || *(_BYTE *)(v59 + 16) != 13 )
      {
        v145 = *(_QWORD *)(*(_QWORD *)v141 + 32LL);
        if ( v145 )
        {
          v135 = v13;
          v102 = (__int64)v60;
          for ( j = 0; j != v145; ++j )
          {
            v104 = sub_15A0A60(v102, j);
            if ( !v104 )
              return 0;
            v105 = *(_BYTE *)(v104 + 16);
            if ( v105 != 9 )
            {
              if ( v105 != 13 )
                return 0;
              v106 = *(_DWORD *)(v104 + 32);
              if ( v106 <= 0x40 )
              {
                if ( *(_QWORD *)(v104 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v106) )
                  return 0;
              }
              else
              {
                v127 = *(_DWORD *)(v104 + 32);
                if ( v127 != (unsigned int)sub_16A58F0(v104 + 24) )
                  return 0;
              }
            }
          }
          v13 = v135;
          v9 = a1;
        }
        goto LABEL_10;
      }
      v61 = *(_DWORD *)(v59 + 32);
      if ( v61 <= 0x40 )
      {
        v28 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v61) == *(_QWORD *)(v59 + 24);
      }
      else
      {
        v142 = *(_DWORD *)(v59 + 32);
        v28 = v142 == (unsigned int)sub_16A58F0(v59 + 24);
      }
    }
LABEL_23:
    if ( !v28 )
      return 0;
    goto LABEL_10;
  }
  if ( v18 > 0x10u )
    return 0;
  v133 = (v15 - 38) & 0xFFFFFFFD;
  v136 = (_BYTE *)*((_QWORD *)a1 - 3);
  if ( sub_1593BB0((__int64)v136, v17, v12, v10) )
    goto LABEL_10;
  if ( v136[16] == 13 )
  {
    if ( *((_DWORD *)v136 + 8) <= 0x40u )
    {
      v28 = *((_QWORD *)v136 + 3) == 0;
    }
    else
    {
      v75 = (__int64)(v136 + 24);
      v143 = *((_DWORD *)v136 + 8);
      v28 = v143 == (unsigned int)sub_16A57B0(v75);
    }
    goto LABEL_23;
  }
  if ( *(_BYTE *)(*(_QWORD *)v136 + 8LL) != 16 )
    return 0;
  v90 = sub_15A1020(v136, v133, v19, v20);
  v91 = v136;
  if ( v90 && *(_BYTE *)(v90 + 16) == 13 )
  {
    if ( *(_DWORD *)(v90 + 32) <= 0x40u )
    {
      v28 = *(_QWORD *)(v90 + 24) == 0;
    }
    else
    {
      v144 = *(_DWORD *)(v90 + 32);
      v28 = v144 == (unsigned int)sub_16A57B0(v90 + 24);
    }
    goto LABEL_23;
  }
  v146 = *(_QWORD *)(*(_QWORD *)v136 + 32LL);
  if ( v146 )
  {
    v107 = (__int64)v91;
    v131 = v8;
    v108 = v133;
    do
    {
      v109 = sub_15A0A60(v107, v108);
      if ( !v109 )
        return 0;
      v110 = *(_BYTE *)(v109 + 16);
      if ( v110 != 9 )
      {
        if ( v110 != 13 )
          return 0;
        if ( *(_DWORD *)(v109 + 32) <= 0x40u )
        {
          v111 = *(_QWORD *)(v109 + 24) == 0;
        }
        else
        {
          v128 = *(_DWORD *)(v109 + 32);
          v111 = v128 == (unsigned int)sub_16A57B0(v109 + 24);
        }
        if ( !v111 )
          return 0;
      }
      ++v108;
    }
    while ( v146 != v108 );
    v9 = a1;
    v8 = v131;
  }
LABEL_10:
  v21 = *(_QWORD *)(v13 + 8);
  if ( !v21 || *(_QWORD *)(v21 + 8) )
    return 0;
  v22 = *(unsigned __int8 *)(v13 + 16);
  if ( (unsigned __int8)v22 > 0x17u )
  {
    v23 = v22 - 24;
  }
  else
  {
    if ( (_BYTE)v22 != 5 )
      return 0;
    v23 = *(unsigned __int16 *)(v13 + 18);
  }
  if ( v23 != 36 )
    return 0;
  v24 = (*(_BYTE *)(v13 + 23) & 0x40) != 0
      ? *(__int64 **)(v13 - 8)
      : (__int64 *)(v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF));
  v126 = *v24;
  if ( !*v24 )
    return 0;
  v129 = v16 == 38;
  v124 = 1;
  v137 = sub_16431D0(*(_QWORD *)v13) - 1;
LABEL_31:
  v160 = (unsigned __int8 *)v8;
  v35 = (__int64 *)v8;
  v161 = &v150;
  v36 = sub_1794FA0((__int64)&v160, a3, v25, v26);
  v37 = 0;
  v38 = v36;
  if ( (_BYTE)v36 )
    goto LABEL_32;
  v35 = (__int64 *)a3;
  v161 = &v150;
  v160 = (unsigned __int8 *)a3;
  v37 = sub_1794FA0((__int64)&v160, v8, v36, 0);
  if ( !v37 )
    return 0;
  v38 = 0;
LABEL_32:
  v39 = *(_DWORD *)(v150 + 8);
  v40 = v39 - 1;
  v122 = v39 - 1;
  if ( v39 > 0x40 )
  {
    v121 = v37;
    v123 = v38;
    v64 = sub_16A57B0(v150);
    v37 = v121;
    v38 = v123;
    v39 = v64;
    v134 = v122 - v64;
  }
  else
  {
    v41 = v39 - 64;
    v134 = -1;
    if ( *(_QWORD *)v150 )
    {
      _BitScanReverse64(&v42, *(_QWORD *)v150);
      v39 = v41 + (v42 ^ 0x3F);
      v134 = v40 - v39;
    }
  }
  v43 = v129;
  v44 = *v35;
  v119 = v38;
  v130 = v38 & !v129;
  if ( v130 )
  {
    v120 = v39;
    v62 = sub_16431D0(v44);
    v63 = sub_16431D0(*(_QWORD *)v126);
    v48 = v120;
    v47 = 1;
    v49 = v63;
    v46 = v63 != v62;
  }
  else
  {
    v117 = v39;
    v130 = v43 & v37;
    v118 = sub_16431D0(v44);
    v45 = sub_16431D0(*(_QWORD *)v126);
    v46 = v118 != v45;
    v47 = v130;
    v48 = v117;
    v49 = v45;
    if ( v119 )
      v8 = a3;
    a3 = v8;
  }
  v50 = *((_QWORD *)v9 + 1);
  v51 = v46 + v47 + (v137 != v134);
  if ( v50 )
    v52 = *(_QWORD *)(v50 + 8) == 0;
  else
    v52 = 0;
  v53 = *(_QWORD *)(a3 + 8);
  if ( v53 )
    v52 += *(_QWORD *)(v53 + 8) == 0;
  if ( v51 > v52 )
    return 0;
  if ( !v124 )
    goto LABEL_45;
  v158 = v49;
  v71 = 1LL << v137;
  if ( v49 > 0x40 )
  {
    v125 = v48;
    sub_16A4EF0((__int64)&v157, 0, 0);
    v48 = v125;
    if ( v158 > 0x40 )
    {
      *(_QWORD *)(v157 + 8LL * (v137 >> 6)) |= v71;
      goto LABEL_77;
    }
  }
  else
  {
    v157 = 0;
  }
  v157 |= v71;
LABEL_77:
  v149 = v48;
  v162 = 257;
  v72 = sub_15A1070(*(_QWORD *)v126, (__int64)&v157);
  v73 = sub_1729500(a4, (unsigned __int8 *)v126, v72, (__int64 *)&v160, a5, a6, a7);
  v48 = v149;
  v126 = (__int64)v73;
  if ( v158 > 0x40 && v157 )
  {
    j_j___libc_free_0_0(v157);
    v48 = v149;
  }
LABEL_45:
  if ( v137 < v134 )
  {
    v66 = *v35;
    v162 = 257;
    v67 = sub_1793A00(a4, v126, v66, (__int64 *)&v160);
    v154 = 257;
    v68 = v67;
    v69 = sub_15A0680(*v67, v134 - v137, 0);
    if ( *((_BYTE *)v68 + 16) <= 0x10u && *(_BYTE *)(v69 + 16) <= 0x10u )
    {
      v58 = (__int64 *)sub_15A2D50(v68, v69, 0, 0, a5, a6, a7);
      v70 = sub_14DBA30((__int64)v58, *(_QWORD *)(a4 + 96), 0);
      if ( v70 )
        v58 = (__int64 *)v70;
      goto LABEL_52;
    }
    v159 = 257;
    v76 = sub_15FB440(23, v68, v69, (__int64)&v157, 0);
    v77 = *(_QWORD *)(a4 + 8);
    v58 = (__int64 *)v76;
    if ( v77 )
    {
      v78 = *(unsigned __int64 **)(a4 + 16);
      sub_157E9D0(v77 + 40, v76);
      v79 = v58[3];
      v80 = *v78;
      v58[4] = (__int64)v78;
      v80 &= 0xFFFFFFFFFFFFFFF8LL;
      v58[3] = v80 | v79 & 7;
      *(_QWORD *)(v80 + 8) = v58 + 3;
      *v78 = *v78 & 7 | (unsigned __int64)(v58 + 3);
    }
    v81 = v153;
    v82 = v58;
    sub_164B780((__int64)v58, v153);
    v84 = *(_QWORD *)(a4 + 80) == 0;
    v151 = v58;
    if ( !v84 )
    {
      (*(void (__fastcall **)(__int64, __int64 **))(a4 + 88))(a4 + 64, &v151);
      v85 = *(_QWORD *)a4;
      if ( *(_QWORD *)a4 )
      {
        v160 = *(unsigned __int8 **)a4;
        sub_1623A60((__int64)&v160, v85, 2);
        v86 = v58[6];
        if ( v86 )
          sub_161E7C0((__int64)(v58 + 6), v86);
        v87 = v160;
        v58[6] = (__int64)v160;
        if ( v87 )
          sub_1623210((__int64)&v160, v87, (__int64)(v58 + 6));
      }
      goto LABEL_52;
    }
    goto LABEL_158;
  }
  if ( v137 <= v134 )
  {
    v65 = *v35;
    v162 = 257;
    v58 = sub_1793A00(a4, v126, v65, (__int64 *)&v160);
    goto LABEL_52;
  }
  v156 = 257;
  v54 = sub_15A0680(*(_QWORD *)v126, v137 - v122 + v48, 0);
  if ( *(_BYTE *)(v126 + 16) > 0x10u || *(_BYTE *)(v54 + 16) > 0x10u )
  {
    v162 = 257;
    v55 = (__int64 *)sub_15FB440(24, (__int64 *)v126, v54, (__int64)&v160, 0);
    v92 = *(_QWORD *)(a4 + 8);
    if ( v92 )
    {
      v93 = *(unsigned __int64 **)(a4 + 16);
      sub_157E9D0(v92 + 40, (__int64)v55);
      v94 = v55[3];
      v95 = *v93;
      v55[4] = (__int64)v93;
      v95 &= 0xFFFFFFFFFFFFFFF8LL;
      v55[3] = v95 | v94 & 7;
      *(_QWORD *)(v95 + 8) = v55 + 3;
      *v93 = *v93 & 7 | (unsigned __int64)(v55 + 3);
    }
    v81 = v155;
    v82 = v55;
    sub_164B780((__int64)v55, v155);
    v84 = *(_QWORD *)(a4 + 80) == 0;
    v152 = v55;
    if ( !v84 )
    {
      (*(void (__fastcall **)(__int64, __int64 **))(a4 + 88))(a4 + 64, &v152);
      v96 = *(_QWORD *)a4;
      if ( *(_QWORD *)a4 )
      {
        v157 = *(_QWORD *)a4;
        sub_1623A60((__int64)&v157, v96, 2);
        v97 = v55[6];
        if ( v97 )
          sub_161E7C0((__int64)(v55 + 6), v97);
        v98 = (unsigned __int8 *)v157;
        v55[6] = v157;
        if ( v98 )
          sub_1623210((__int64)&v157, v98, (__int64)(v55 + 6));
      }
      goto LABEL_51;
    }
LABEL_158:
    sub_4263D6(v82, v81, v83);
  }
  v55 = (__int64 *)sub_15A2D80((__int64 *)v126, v54, 0, a5, a6, a7);
  v56 = sub_14DBA30((__int64)v55, *(_QWORD *)(a4 + 96), 0);
  if ( v56 )
    v55 = (__int64 *)v56;
LABEL_51:
  v57 = *v35;
  v162 = 257;
  v58 = sub_1793A00(a4, (__int64)v55, v57, (__int64 *)&v160);
LABEL_52:
  if ( v130 )
  {
    v162 = 257;
    v74 = sub_15A1070(*v58, v150);
    v58 = (__int64 *)sub_172B670(a4, (__int64)v58, v74, (__int64 *)&v160, a5, a6, a7);
  }
  v162 = 257;
  return sub_172AC10(a4, (__int64)v58, (__int64)v35, (__int64 *)&v160, a5, a6, a7);
}
