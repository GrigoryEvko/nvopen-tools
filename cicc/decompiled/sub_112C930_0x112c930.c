// Function: sub_112C930
// Address: 0x112c930
//
void *__fastcall sub_112C930(const __m128i *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int16 v7; // r14
  bool v8; // zf
  void *v9; // rbx
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int16 v15; // ax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 *v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 *v21; // rsi
  __int64 v22; // r12
  unsigned __int8 v23; // al
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 *v27; // rax
  __int64 *v28; // r12
  unsigned int v29; // edx
  __int64 v30; // rax
  unsigned int v31; // ebx
  __int64 v32; // rsi
  __int64 v33; // r12
  __int64 v34; // r13
  _QWORD *v35; // rax
  unsigned int v36; // eax
  __int64 *v37; // rsi
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r8
  int v41; // eax
  __int64 v42; // rdx
  unsigned int v43; // r8d
  __int64 *v44; // rsi
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // r8
  unsigned int v48; // ebx
  unsigned int v49; // eax
  __int64 v50; // r9
  bool v51; // al
  __int64 *v52; // rsi
  unsigned int v53; // eax
  __int64 v54; // r12
  _QWORD *v55; // rax
  __int64 *v56; // rdi
  unsigned __int64 v57; // rcx
  __int64 *v59; // rdi
  unsigned int v60; // ebx
  __int64 v62; // rcx
  char v64; // bl
  unsigned __int64 v65; // rax
  unsigned int v66; // eax
  __int64 v67; // r12
  _QWORD *v68; // rax
  __int64 v69; // r14
  unsigned int v70; // ebx
  __int64 v71; // r15
  _BYTE *v72; // rbx
  unsigned int **v73; // r14
  const char *v74; // rax
  __int64 **v75; // rdx
  __int64 v76; // rax
  __int16 v77; // r12
  __int64 v78; // r14
  __int16 v79; // r12
  _QWORD *v80; // rax
  __int64 v81; // rdx
  __int64 v82; // rcx
  unsigned int v83; // r8d
  __int64 *v84; // rbx
  __int64 v85; // rax
  __int64 v86; // r13
  _QWORD *v87; // rax
  int v88; // edx
  unsigned __int64 v89; // rdx
  unsigned int v90; // eax
  __int64 v91; // rdx
  __int64 v92; // rcx
  __int64 v93; // r8
  __int64 v94; // r13
  _QWORD *v95; // rax
  __int64 v96; // rsi
  _BYTE *v97; // r9
  __int64 v98; // r9
  unsigned __int8 *v99; // r11
  __int64 v100; // rdx
  __int64 v101; // rcx
  __int64 v102; // rcx
  unsigned __int8 v103; // al
  char v104; // al
  unsigned int v105; // edx
  char v106; // r8
  unsigned int v107; // eax
  unsigned int **v108; // r14
  const char *v109; // rax
  unsigned int **v110; // r15
  __int64 **v111; // rdx
  const char *v112; // rax
  __int64 v113; // rdx
  _BYTE *v114; // rax
  __int64 v115; // rax
  unsigned int **v116; // r14
  _BYTE *v117; // r15
  const char *v118; // rax
  __int64 **v119; // rdx
  __int64 v120; // rax
  __int16 v121; // r12
  __int64 v122; // r14
  __int64 v123; // r13
  __int16 v124; // r12
  _QWORD *v125; // rax
  char v126; // al
  unsigned int v127; // [rsp+Ch] [rbp-F4h]
  unsigned int v128; // [rsp+Ch] [rbp-F4h]
  __int64 v129; // [rsp+10h] [rbp-F0h]
  __int64 *v130; // [rsp+10h] [rbp-F0h]
  __int64 v131; // [rsp+10h] [rbp-F0h]
  __int64 v132; // [rsp+10h] [rbp-F0h]
  unsigned int v133; // [rsp+18h] [rbp-E8h]
  unsigned int v134; // [rsp+18h] [rbp-E8h]
  unsigned __int8 *v135; // [rsp+20h] [rbp-E0h]
  unsigned __int8 *v136; // [rsp+20h] [rbp-E0h]
  unsigned __int8 *v137; // [rsp+20h] [rbp-E0h]
  __int64 v138; // [rsp+28h] [rbp-D8h]
  __int64 v139; // [rsp+28h] [rbp-D8h]
  bool v140; // [rsp+28h] [rbp-D8h]
  unsigned int v141; // [rsp+28h] [rbp-D8h]
  __int64 v142; // [rsp+28h] [rbp-D8h]
  _BYTE *v143; // [rsp+28h] [rbp-D8h]
  __int64 *v145; // [rsp+40h] [rbp-C0h] BYREF
  __int64 *v146; // [rsp+48h] [rbp-B8h] BYREF
  _BYTE *v147; // [rsp+50h] [rbp-B0h] BYREF
  unsigned int v148; // [rsp+58h] [rbp-A8h]
  __int64 v149; // [rsp+60h] [rbp-A0h] BYREF
  unsigned int v150; // [rsp+68h] [rbp-98h]
  __int64 v151; // [rsp+70h] [rbp-90h] BYREF
  __int64 v152; // [rsp+78h] [rbp-88h]
  __int64 v153[2]; // [rsp+80h] [rbp-80h] BYREF
  __int16 v154; // [rsp+90h] [rbp-70h]
  __int64 *v155; // [rsp+A0h] [rbp-60h] BYREF
  __int64 **v156; // [rsp+A8h] [rbp-58h] BYREF
  char v157; // [rsp+B0h] [rbp-50h]
  __int16 v158; // [rsp+C0h] [rbp-40h]

  v7 = *(_WORD *)(a2 + 2) & 0x3F;
  if ( v7 == 33 && (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a2 + 8) + 8LL) - 17 <= 1 )
  {
    v31 = *(_DWORD *)(a4 + 8);
    if ( v31 <= 0x40 )
    {
      if ( *(_QWORD *)a4 )
        goto LABEL_2;
    }
    else if ( v31 != (unsigned int)sub_C444A0(a4) )
    {
      goto LABEL_2;
    }
    v32 = *(_QWORD *)(a3 - 32);
    v155 = 0;
    if ( (unsigned __int8)sub_993A50(&v155, v32) )
    {
      v33 = *(_QWORD *)(a3 - 64);
      v158 = 257;
      v34 = *(_QWORD *)(a2 + 8);
      v35 = sub_BD2C40(72, unk_3F10A14);
      v9 = v35;
      if ( v35 )
        sub_B51510((__int64)v35, v33, v34, (__int64)&v155, 0, 0);
      return v9;
    }
  }
LABEL_2:
  v8 = *(_BYTE *)a3 == 57;
  v157 = 0;
  v155 = (__int64 *)&v146;
  v156 = &v145;
  if ( !v8 )
    return 0;
  if ( !*(_QWORD *)(a3 - 64) )
    return 0;
  _RSI = *(_QWORD *)(a3 - 32);
  v146 = *(__int64 **)(a3 - 64);
  if ( !(unsigned __int8)sub_991580((__int64)&v156, _RSI) )
    return 0;
  v15 = *(_WORD *)(a2 + 2) & 0x3F;
  if ( v15 != 38 )
    goto LABEL_8;
  v44 = v145;
  sub_9865C0((__int64)&v151, (__int64)v145);
  sub_987160((__int64)&v151, (__int64)v44, v45, v46, v47);
  v48 = v152;
  _RSI = (__int64)&v155;
  LODWORD(v152) = 0;
  v155 = (__int64 *)v151;
  v139 = v151;
  LODWORD(v156) = v48;
  v49 = sub_C49970(a4, (unsigned __int64 *)&v155);
  v50 = v139;
  v14 = v49;
  v51 = 0;
  if ( (int)v14 <= 0 )
  {
    v56 = v145;
    v12 = *((unsigned int *)v145 + 2);
    v13 = (unsigned int)(v12 - 1);
    v14 = 1LL << ((unsigned __int8)v12 - 1);
    _RSI = *v145;
    if ( (unsigned int)v12 > 0x40 )
    {
      v13 = (unsigned int)v13 >> 6;
      v14 &= *(_QWORD *)(_RSI + 8 * v13);
      if ( !v14 )
        goto LABEL_48;
      v127 = *((_DWORD *)v145 + 2);
      v129 = v139;
      v141 = sub_C44500((__int64)v145);
      LODWORD(_RAX) = sub_C44590((__int64)v56);
      v12 = v127;
      v50 = v129;
      v13 = v141;
    }
    else
    {
      if ( (v14 & _RSI) == 0 )
        goto LABEL_48;
      if ( (_DWORD)v12 )
      {
        v13 = 64;
        if ( _RSI << (64 - (unsigned __int8)v12) != -1 )
        {
          _BitScanReverse64(&v57, ~(_RSI << (64 - (unsigned __int8)v12)));
          v13 = (unsigned int)v57 ^ 0x3F;
        }
      }
      else
      {
        v13 = 0;
      }
      __asm { tzcnt   rax, rsi }
      if ( (unsigned int)_RAX > (unsigned int)v12 )
        LODWORD(_RAX) = *((_DWORD *)v145 + 2);
    }
    v51 = (_DWORD)v13 + (_DWORD)_RAX == (_DWORD)v12;
  }
LABEL_48:
  if ( v48 > 0x40 )
  {
    if ( v50 )
    {
      v140 = v51;
      j_j___libc_free_0_0(v50);
      v51 = v140;
      if ( (unsigned int)v152 > 0x40 )
      {
        if ( v151 )
        {
          j_j___libc_free_0_0(v151);
          v51 = v140;
        }
      }
    }
  }
  if ( v51 )
  {
    v52 = v145;
    v150 = *((_DWORD *)v145 + 2);
    if ( v150 > 0x40 )
      sub_C43780((__int64)&v149, (const void **)v145);
    else
      v149 = *v145;
    sub_987160((__int64)&v149, (__int64)v52, v12, v13, v14);
    v53 = v150;
    v150 = 0;
    LODWORD(v152) = v53;
    v151 = v149;
    v54 = sub_AD8D80(v146[1], (__int64)&v151);
    v158 = 257;
    v55 = sub_BD2C40(72, unk_3F10FD0);
    v9 = v55;
    if ( v55 )
      sub_1113300((__int64)v55, 38, (__int64)v146, v54, (__int64)&v155);
    if ( (unsigned int)v152 > 0x40 && v151 )
      j_j___libc_free_0_0(v151);
    if ( v150 > 0x40 && v149 )
      j_j___libc_free_0_0(v149);
    return v9;
  }
  v15 = *(_WORD *)(a2 + 2) & 0x3F;
LABEL_8:
  if ( v15 == 40 && !(unsigned __int8)sub_986B30((__int64 *)a4, _RSI, v12, v13, v14) )
  {
    sub_9865C0((__int64)&v147, a4);
    sub_C46F20((__int64)&v147, 1u);
    v36 = v148;
    v37 = v145;
    v148 = 0;
    v150 = v36;
    v149 = (__int64)v147;
    sub_9865C0((__int64)&v151, (__int64)v145);
    sub_987160((__int64)&v151, (__int64)v37, v38, v39, v40);
    v41 = v152;
    _RSI = (__int64)&v155;
    LODWORD(v152) = 0;
    LODWORD(v156) = v41;
    v155 = (__int64 *)v151;
    if ( (int)sub_C49970((__int64)&v149, (unsigned __int64 *)&v155) > 0 )
    {
LABEL_44:
      sub_969240((__int64 *)&v155);
      sub_969240(&v151);
      sub_969240(&v149);
      sub_969240((__int64 *)&v147);
      goto LABEL_9;
    }
    v59 = v145;
    v60 = *((_DWORD *)v145 + 2);
    _RAX = *v145;
    _RSI = 1LL << ((unsigned __int8)v60 - 1);
    if ( v60 > 0x40 )
    {
      _RSI &= *(_QWORD *)(_RAX + 8LL * ((v60 - 1) >> 6));
      if ( !_RSI )
        goto LABEL_44;
      v130 = v145;
      v128 = sub_C44500((__int64)v145);
      LODWORD(_RAX) = sub_C44590((__int64)v130);
      v62 = v128;
      v59 = v130;
    }
    else
    {
      if ( (_RSI & _RAX) == 0 )
        goto LABEL_44;
      if ( v60 )
      {
        v62 = 64;
        _RSI = ~(_RAX << (64 - (unsigned __int8)v60));
        if ( _RAX << (64 - (unsigned __int8)v60) != -1 )
        {
          _BitScanReverse64((unsigned __int64 *)&_RSI, _RSI);
          v62 = (unsigned int)_RSI ^ 0x3F;
        }
      }
      else
      {
        v62 = 0;
      }
      __asm { tzcnt   rax, rax }
      if ( (unsigned int)_RAX > v60 )
        LODWORD(_RAX) = *((_DWORD *)v145 + 2);
    }
    if ( v60 != (_DWORD)v62 + (_DWORD)_RAX )
      goto LABEL_44;
    v64 = sub_986B30(v59, _RSI, v42, v62, v43);
    sub_969240((__int64 *)&v155);
    sub_969240(&v151);
    sub_969240(&v149);
    sub_969240((__int64 *)&v147);
    if ( !v64 )
    {
      sub_9865C0((__int64)&v149, (__int64)v145);
      if ( v150 > 0x40 )
      {
        sub_C43D10((__int64)&v149);
      }
      else
      {
        v65 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v150) & ~v149;
        if ( !v150 )
          v65 = 0;
        v149 = v65;
      }
      sub_C46250((__int64)&v149);
      v66 = v150;
      v150 = 0;
      LODWORD(v152) = v66;
      v151 = v149;
      v67 = sub_AD8D80(v146[1], (__int64)&v151);
      v158 = 257;
      v68 = sub_BD2C40(72, unk_3F10FD0);
      v9 = v68;
      if ( v68 )
        sub_1113300((__int64)v68, 40, (__int64)v146, v67, (__int64)&v155);
      sub_969240(&v151);
      sub_969240(&v149);
      return v9;
    }
  }
LABEL_9:
  v16 = *(_QWORD *)(a3 + 16);
  if ( !v16 || *(_QWORD *)(v16 + 8) )
    return 0;
  if ( !(unsigned __int8)sub_F0C3D0((__int64)a1) && (*(_WORD *)(a2 + 2) & 0x3Fu) - 32 <= 1 && sub_9867B0(a4) )
  {
    v84 = v145;
    if ( (unsigned __int8)sub_986B30(v145, _RSI, v81, v82, v83) )
    {
      v85 = sub_AD6530(v146[1], _RSI);
      v158 = 257;
      v86 = v85;
      v87 = sub_BD2C40(72, unk_3F10FD0);
      v9 = v87;
      if ( v87 )
        sub_1113300((__int64)v87, (v7 == 33) + 39, (__int64)v146, v86, (__int64)&v155);
      return v9;
    }
    sub_9865C0((__int64)&v147, (__int64)v84);
    sub_10BE760((__int64)&v151, a1, *(_QWORD *)(a3 - 64), 0, a3);
    v88 = v152;
    if ( (unsigned int)v152 > 0x40 )
    {
      v88 = sub_C44500((__int64)&v151);
    }
    else if ( (_DWORD)v152 )
    {
      _BitScanReverse64(&v89, ~(v151 << (64 - (unsigned __int8)v152)));
      v88 = v89 ^ 0x3F;
      if ( v151 << (64 - (unsigned __int8)v152) == -1 )
        v88 = 64;
    }
    sub_109DDE0((__int64)&v149, *((_DWORD *)v145 + 2), v88);
    if ( v150 > 0x40 )
      sub_C43BD0(&v149, v145);
    else
      v149 |= *v145;
    v90 = v150;
    v150 = 0;
    LODWORD(v156) = v90;
    v155 = (__int64 *)v149;
    sub_1110A30((__int64 *)&v147, (__int64 *)&v155);
    sub_969240((__int64 *)&v155);
    sub_969240(&v149);
    if ( (unsigned __int8)sub_109DE70((__int64)&v147) )
    {
      sub_9865C0((__int64)&v149, (__int64)&v147);
      sub_AADAA0((__int64)&v155, (__int64)&v149, v91, v92, v93);
      v94 = sub_AD8D80(*(_QWORD *)(a3 + 8), (__int64)&v155);
      sub_969240((__int64 *)&v155);
      sub_969240(&v149);
      v158 = 257;
      v95 = sub_BD2C40(72, unk_3F10FD0);
      v9 = v95;
      if ( v95 )
        sub_1113300((__int64)v95, (v7 != 33) + 35, (__int64)v146, v94, (__int64)&v155);
      sub_969240(v153);
      sub_969240(&v151);
      sub_969240((__int64 *)&v147);
      return v9;
    }
    sub_969240(v153);
    sub_969240(&v151);
    sub_969240((__int64 *)&v147);
  }
  v17 = *(_QWORD *)(a3 - 64);
  v18 = v145;
  v19 = *(_QWORD *)(v17 + 16);
  if ( v19 )
  {
    if ( !*(_QWORD *)(v19 + 8) && *(_BYTE *)v17 == 67 )
    {
      v142 = *(_QWORD *)(v17 - 32);
      if ( v142 )
      {
        if ( (*(_WORD *)(a2 + 2) & 0x3Fu) - 32 <= 1
          || !sub_986C60((__int64 *)a4, *(_DWORD *)(a4 + 8) - 1) && !sub_986C60(v18, *((_DWORD *)v18 + 2) - 1) )
        {
          v18 = v145;
          if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a2 + 8) + 8LL) - 17 > 1 )
          {
            v69 = *(_QWORD *)(v142 + 8);
            v70 = sub_BCB060(v69);
            sub_C449B0((__int64)&v155, (const void **)a4, v70);
            v71 = sub_AD8D80(v69, (__int64)&v155);
            sub_969240((__int64 *)&v155);
            sub_C449B0((__int64)&v155, (const void **)v145, v70);
            v72 = (_BYTE *)sub_AD8D80(v69, (__int64)&v155);
            sub_969240((__int64 *)&v155);
            v73 = (unsigned int **)a1[2].m128i_i64[0];
            v74 = sub_BD5D20(a3);
            v158 = 261;
            v156 = v75;
            v155 = (__int64 *)v74;
            v76 = sub_A82350(v73, (_BYTE *)v142, v72, (__int64)&v155);
            v77 = *(_WORD *)(a2 + 2);
            v78 = v76;
            v158 = 257;
            v79 = v77 & 0x3F;
            v80 = sub_BD2C40(72, unk_3F10FD0);
            v9 = v80;
            if ( v80 )
              sub_1113300((__int64)v80, v79, v78, v71, (__int64)&v155);
            return v9;
          }
        }
      }
    }
  }
  v9 = sub_1121F70((__int64)a1, a2, a3, a4, (__int64)v18);
  if ( v9 )
    return v9;
  if ( sub_B532B0(*(_WORD *)(a2 + 2) & 0x3F) )
    goto LABEL_18;
  if ( !sub_9867B0(a4) )
    goto LABEL_18;
  v20 = *(_QWORD *)(*(_QWORD *)(a3 - 64) + 16LL);
  if ( !v20 )
    goto LABEL_18;
  if ( *(_QWORD *)(v20 + 8) )
    goto LABEL_18;
  v96 = *(_QWORD *)(a3 - 32);
  v155 = 0;
  if ( !(unsigned __int8)sub_993A50(&v155, v96) )
    goto LABEL_18;
  v97 = *(_BYTE **)(a3 - 64);
  v143 = *(_BYTE **)(a3 - 32);
  v155 = &v149;
  v156 = (__int64 **)&v147;
  if ( *v97 != 58 )
    goto LABEL_18;
  if ( !(unsigned __int8)sub_11108F0(&v155, (__int64)v97) )
    goto LABEL_18;
  if ( *(_BYTE *)v149 != 55 )
    goto LABEL_18;
  if ( v147 != *(_BYTE **)(v149 - 64) )
    goto LABEL_18;
  v99 = *(unsigned __int8 **)(v149 - 32);
  if ( !v99 )
    goto LABEL_18;
  v100 = *(_QWORD *)(a3 + 16);
  if ( v100 )
    LODWORD(v100) = *(_QWORD *)(v100 + 8) == 0;
  v101 = *(_QWORD *)(v98 + 16);
  if ( v101 && !*(_QWORD *)(v101 + 8) )
    LODWORD(v100) = v100 + 1;
  v102 = *(_QWORD *)(v149 + 16);
  v103 = *v99;
  if ( !v102 || *(_QWORD *)(v102 + 8) )
  {
    if ( v103 > 0x15u )
      goto LABEL_18;
    v131 = v98;
    v133 = v100;
    v135 = *(unsigned __int8 **)(v149 - 32);
    v104 = sub_1110A70(v135);
    v99 = v135;
    v105 = v133;
    v106 = v104;
    v98 = v131;
    v107 = 1;
    if ( !v106 )
      goto LABEL_18;
  }
  else
  {
    v105 = v100 + 1;
    if ( v103 <= 0x15u )
    {
      v132 = v98;
      v134 = v105;
      v137 = *(unsigned __int8 **)(v149 - 32);
      v126 = sub_1110A70(v137);
      v99 = v137;
      v105 = v134;
      v98 = v132;
      if ( v126 )
      {
LABEL_139:
        v136 = v99;
        v108 = (unsigned int **)a1[2].m128i_i64[0];
        v109 = sub_BD5D20(v98);
        v110 = (unsigned int **)a1[2].m128i_i64[0];
        v158 = 261;
        v155 = (__int64 *)v109;
        v156 = v111;
        v112 = sub_BD5D20(v149);
        v152 = v113;
        v154 = 261;
        v151 = (__int64)v112;
        v114 = (_BYTE *)sub_920A70(v110, v143, v136, (__int64)&v151, 1u, 0);
        v115 = sub_A82480(v108, v114, v143, (__int64)&v155);
        v116 = (unsigned int **)a1[2].m128i_i64[0];
        v117 = (_BYTE *)v115;
        v118 = sub_BD5D20(a3);
        v156 = v119;
        v158 = 261;
        v155 = (__int64 *)v118;
        v120 = sub_A82350(v116, v147, v117, (__int64)&v155);
        v121 = *(_WORD *)(a2 + 2);
        v158 = 257;
        v122 = v120;
        v123 = *(_QWORD *)(a2 - 32);
        v124 = v121 & 0x3F;
        v125 = sub_BD2C40(72, unk_3F10FD0);
        v9 = v125;
        if ( v125 )
          sub_1113300((__int64)v125, v124, v122, v123, (__int64)&v155);
        return v9;
      }
    }
    v107 = 3;
  }
  if ( v107 <= v105 )
    goto LABEL_139;
LABEL_18:
  if ( !(unsigned __int8)sub_B2D610(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 72LL), 30)
    && (*(_WORD *)(a2 + 2) & 0x3Fu) - 32 <= 1 )
  {
    v21 = v146;
    v155 = &v149;
    if ( (unsigned __int8)sub_111E110(&v155, (__int64)v146) )
    {
      v22 = *(_QWORD *)(v149 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v22 + 8) - 17 <= 1 )
        v22 = **(_QWORD **)(v22 + 16);
      v23 = *(_BYTE *)(v22 + 8);
      if ( v23 <= 3u || v23 == 5 )
      {
        if ( sub_9867B0(a4) || (v21 = v145, sub_AAD8B0(a4, v145)) )
        {
          v138 = sub_BCAC60(v22, (__int64)v21, v24, v25, v26);
          v27 = (__int64 *)sub_C33340();
          v28 = v27;
          if ( (__int64 *)v138 == v27 )
            sub_C3C500(&v155, (__int64)v27);
          else
            sub_C373C0(&v155, v138);
          if ( v155 == v28 )
            sub_C3CF20((__int64)&v155, 0);
          else
            sub_C36EF0((_DWORD **)&v155, 0);
          if ( v155 == v28 )
            sub_C3E660((__int64)&v151, (__int64)&v155);
          else
            sub_C3A850((__int64)&v151, (__int64 *)&v155);
          sub_91D830(&v155);
          if ( sub_AAD8B0((__int64)v145, &v151) )
          {
            v29 = !sub_9867B0(a4) ? 519 : 240;
            if ( v7 == 33 )
              v29 ^= 0x3FFu;
            v30 = sub_B37A80(a1[2].m128i_i64[0], v149, v29);
            v9 = sub_F162A0((__int64)a1, a2, v30);
            sub_969240(&v151);
          }
          else
          {
            sub_969240(&v151);
          }
          return v9;
        }
        return 0;
      }
    }
  }
  return v9;
}
