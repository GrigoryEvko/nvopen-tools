// Function: sub_1E87320
// Address: 0x1e87320
//
signed __int64 __fastcall sub_1E87320(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int16 *v7; // r13
  __int16 *v8; // rax
  __int16 v9; // ax
  __int64 v10; // rbx
  __int64 v11; // r14
  __int16 v12; // ax
  __int64 v13; // rdx
  int v14; // edi
  __int16 v15; // ax
  __int64 v16; // rax
  __int16 v17; // ax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  int v21; // eax
  __int64 v22; // rsi
  int v23; // edx
  int v24; // edi
  unsigned int v25; // eax
  __int64 v26; // rcx
  __int64 v27; // r14
  unsigned int v28; // r10d
  int v29; // r9d
  __int64 v30; // rcx
  unsigned __int64 v31; // rax
  unsigned __int64 v32; // rdx
  _QWORD *v33; // rax
  _QWORD *i; // rcx
  __int64 v35; // rdx
  int v36; // eax
  unsigned __int64 v37; // r11
  bool v38; // cl
  char v39; // si
  __int64 v40; // rax
  __int64 *v41; // r9
  __int64 v42; // r11
  unsigned int v43; // r14d
  __int64 v44; // rdx
  __int64 v45; // rdi
  __int64 (*v46)(); // rax
  signed __int64 result; // rax
  __int64 v48; // kr00_8
  __int64 v49; // rcx
  __int64 v50; // rax
  unsigned __int64 v51; // rax
  __int64 v52; // rax
  unsigned int v53; // r14d
  unsigned int v54; // ebx
  __int64 v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rax
  __int64 v58; // rax
  _BYTE *v59; // rdx
  void *v60; // rax
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rax
  __int64 v64; // r14
  __int64 v65; // r13
  __int64 v66; // rax
  int v67; // esi
  unsigned __int64 v68; // rbx
  _QWORD *v69; // rax
  __int64 v70; // r13
  __int64 v71; // rbx
  __int64 v72; // rax
  __int64 v73; // r9
  __int64 v74; // r10
  __int64 v75; // rsi
  int v76; // eax
  __int64 v77; // rdi
  bool v78; // dl
  unsigned __int8 v79; // r8
  _DWORD *v80; // rbx
  __int64 v81; // r14
  __int64 v82; // rsi
  void *v83; // rax
  __int64 v84; // r13
  __int64 v85; // rdx
  __int64 v86; // rcx
  __int64 v87; // r13
  __int64 v88; // rdx
  __int64 v89; // rcx
  unsigned int v90; // r13d
  unsigned int v91; // r14d
  void *v92; // rax
  __int64 v93; // rax
  __int64 v94; // rax
  __int64 v95; // rax
  __int64 v96; // rax
  __int64 v97; // rdi
  int v98; // eax
  __int64 v99; // r8
  bool v100; // dl
  unsigned __int8 v101; // r11
  int v102; // eax
  __int64 v103; // r8
  bool v104; // dl
  unsigned __int8 v105; // r11
  int v106; // eax
  __int64 v107; // r11
  bool v108; // dl
  unsigned __int8 v109; // r8
  __int64 v110; // rax
  _BYTE *v111; // rax
  __int64 v112; // rax
  _BYTE *v113; // rax
  __int64 v114; // rax
  _BYTE *v115; // rax
  __int64 v116; // rax
  _BYTE *v117; // rax
  unsigned int v118; // ebx
  __int16 v119; // dx
  int v120; // [rsp+0h] [rbp-A0h]
  unsigned __int64 v121; // [rsp+8h] [rbp-98h]
  __int64 v122; // [rsp+8h] [rbp-98h]
  unsigned int v123; // [rsp+8h] [rbp-98h]
  unsigned __int64 v124; // [rsp+18h] [rbp-88h] BYREF
  unsigned __int64 v125; // [rsp+20h] [rbp-80h] BYREF
  unsigned __int64 v126; // [rsp+28h] [rbp-78h] BYREF
  const char *v127[2]; // [rsp+30h] [rbp-70h] BYREF
  unsigned __int64 v128; // [rsp+40h] [rbp-60h] BYREF
  __int64 v129; // [rsp+48h] [rbp-58h]
  _BYTE v130[80]; // [rsp+50h] [rbp-50h] BYREF

  v7 = *(unsigned __int16 **)(a2 + 16);
  v8 = (__int16 *)v7;
  if ( *(_DWORD *)(a2 + 40) < (unsigned int)v7[1] )
  {
    sub_1E86C30(a1, "Too few operands", a2);
    v60 = sub_16E8CB0();
    v61 = sub_16E7A90((__int64)v60, v7[1]);
    v62 = sub_1263B40(v61, " operands expected, but ");
    v63 = sub_16E7A90(v62, *(unsigned int *)(a2 + 40));
    sub_1263B40(v63, " given.\n");
    v8 = *(__int16 **)(a2 + 16);
  }
  v9 = *v8;
  if ( (v9 == 45 || !v9) && (**(_BYTE **)(*(_QWORD *)(a1 + 16) + 352LL) & 2) != 0 )
  {
    sub_1E86C30(a1, "Found PHI instruction with NoPHIs property set", a2);
    if ( **(_WORD **)(a2 + 16) != 1 )
      goto LABEL_7;
  }
  else if ( v9 != 1 )
  {
    goto LABEL_7;
  }
  if ( *(_DWORD *)(a2 + 40) <= 1u )
  {
    sub_1E86C30(a1, "Too few operands on inline asm", a2);
    goto LABEL_7;
  }
  v52 = *(_QWORD *)(a2 + 32);
  if ( *(_BYTE *)v52 != 9 )
  {
    sub_1E86C30(a1, "Asm string must be an external symbol", a2);
    v52 = *(_QWORD *)(a2 + 32);
  }
  if ( *(_BYTE *)(v52 + 40) != 1 )
  {
    sub_1E86C30(a1, "Asm flags must be an immediate", a2);
    v52 = *(_QWORD *)(a2 + 32);
  }
  if ( *(_QWORD *)(v52 + 64) > 0x3Fu )
    sub_1E86D40(a1, "Unknown asm flags", v52 + 40, 1u, 0);
  v53 = *(_DWORD *)(a2 + 40);
  if ( v53 <= 2 )
  {
    v54 = 2;
LABEL_189:
    if ( v53 >= v54 )
      goto LABEL_7;
  }
  else
  {
    v54 = 2;
    while ( 1 )
    {
      v55 = *(_QWORD *)(a2 + 32) + 40LL * v54;
      if ( *(_BYTE *)v55 != 1 )
        break;
      v54 += (((unsigned int)*(_QWORD *)(v55 + 24) >> 3) & 0x1FFF) + 1;
      if ( v53 <= v54 )
        goto LABEL_189;
    }
    if ( v53 >= v54 )
      goto LABEL_90;
  }
  sub_1E86C30(a1, "Missing operands in last group", a2);
  v53 = *(_DWORD *)(a2 + 40);
LABEL_90:
  if ( v53 <= v54 )
    goto LABEL_7;
  v56 = *(_QWORD *)(a2 + 32);
  v57 = v54;
  if ( *(_BYTE *)(v56 + 40LL * v54) == 14 )
  {
    if ( ++v54 >= v53 )
      goto LABEL_7;
    v57 = v54;
  }
  v58 = 40 * v57;
  while ( 1 )
  {
    v59 = (_BYTE *)(v58 + v56);
    if ( *v59 || (v59[3] & 0x20) == 0 )
    {
      v122 = v58;
      sub_1E86D40(a1, "Expected implicit register after groups", (__int64)v59, v54, 0);
      v58 = v122;
    }
    ++v54;
    v58 += 40;
    if ( v54 == v53 )
      break;
    v56 = *(_QWORD *)(a2 + 32);
  }
LABEL_7:
  v10 = *(_QWORD *)(a2 + 56);
  v11 = v10 + 8LL * *(unsigned __int8 *)(a2 + 49);
  if ( v10 != v11 )
  {
    while ( 1 )
    {
      v12 = *(_WORD *)(*(_QWORD *)v10 + 32LL);
      if ( (v12 & 1) != 0 )
      {
        v13 = *(_QWORD *)(a2 + 16);
        if ( *(_WORD *)v13 == 1 )
        {
          LOBYTE(v14) = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 64LL);
          if ( (v14 & 8) != 0 )
          {
            if ( (v12 & 2) == 0 )
              goto LABEL_10;
            goto LABEL_22;
          }
        }
        v15 = *(_WORD *)(a2 + 46);
        if ( (v15 & 4) != 0 || (v15 & 8) == 0 )
          v16 = (*(_QWORD *)(v13 + 8) >> 16) & 1LL;
        else
          LOBYTE(v16) = sub_1E15D00(a2, 0x10000u, 1);
        if ( !(_BYTE)v16 )
          sub_1E86C30(a1, "Missing mayLoad flag", a2);
        if ( (*(_WORD *)(*(_QWORD *)v10 + 32LL) & 2) == 0 )
          goto LABEL_10;
      }
      else if ( (v12 & 2) == 0 )
      {
        goto LABEL_10;
      }
      v13 = *(_QWORD *)(a2 + 16);
      if ( *(_WORD *)v13 == 1 )
      {
        v14 = *(_DWORD *)(*(_QWORD *)(a2 + 32) + 64LL);
LABEL_22:
        if ( (v14 & 0x10) == 0 )
          goto LABEL_23;
LABEL_10:
        v10 += 8;
        if ( v10 == v11 )
          break;
      }
      else
      {
LABEL_23:
        v17 = *(_WORD *)(a2 + 46);
        if ( (v17 & 4) != 0 || (v17 & 8) == 0 )
          v18 = (*(_QWORD *)(v13 + 8) >> 17) & 1LL;
        else
          LOBYTE(v18) = sub_1E15D00(a2, 0x20000u, 1);
        if ( (_BYTE)v18 )
          goto LABEL_10;
        v10 += 8;
        sub_1E86C30(a1, "Missing mayStore flag", a2);
        if ( v10 == v11 )
          break;
      }
    }
  }
  v19 = *(_QWORD *)(a1 + 568);
  if ( !v19 )
    goto LABEL_34;
  v20 = *(_QWORD *)(v19 + 272);
  v21 = *(_DWORD *)(v20 + 384);
  if ( v21 )
  {
    v22 = *(_QWORD *)(v20 + 368);
    v23 = v21 - 1;
    v24 = 1;
    v25 = (v21 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v26 = *(_QWORD *)(v22 + 16LL * v25);
    if ( a2 == v26 )
    {
LABEL_31:
      if ( (unsigned __int16)(**(_WORD **)(a2 + 16) - 12) <= 1u )
      {
        sub_1E86C30(a1, "Debug instruction has a slot index", a2);
      }
      else if ( (*(_BYTE *)(a2 + 46) & 4) != 0 )
      {
        sub_1E86C30(a1, "Instruction inside bundle has a slot index", a2);
      }
      goto LABEL_34;
    }
    while ( v26 != -8 )
    {
      LODWORD(a5) = v24 + 1;
      v25 = v23 & (v24 + v25);
      v26 = *(_QWORD *)(v22 + 16LL * v25);
      if ( a2 == v26 )
        goto LABEL_31;
      ++v24;
    }
  }
  if ( (unsigned __int16)(**(_WORD **)(a2 + 16) - 12) > 1u && (*(_BYTE *)(a2 + 46) & 4) == 0 )
    sub_1E86C30(a1, "Missing slot index", a2);
LABEL_34:
  if ( (unsigned int)*v7 - 34 > 0x5B )
    goto LABEL_63;
  if ( *(_BYTE *)(a1 + 61) )
    sub_1E86C30(a1, "Unexpected generic instruction in a Selected function", a2);
  v27 = 0;
  v128 = (unsigned __int64)v130;
  v129 = 0x400000000LL;
  if ( v7[1] )
  {
    do
    {
      v28 = v27;
      v29 = *(unsigned __int8 *)(*((_QWORD *)v7 + 5) + 8 * v27 + 3);
      if ( (unsigned __int8)(v29 - 6) <= 5u )
      {
        v30 = (unsigned int)v129;
        v31 = v29 - 5;
        if ( v31 < (unsigned int)v129 )
          v31 = (unsigned int)v129;
        v32 = v31;
        if ( (unsigned int)v129 < v31 )
        {
          if ( HIDWORD(v129) < v31 )
          {
            v120 = *(unsigned __int8 *)(*((_QWORD *)v7 + 5) + 8 * v27 + 3);
            v121 = v31;
            sub_16CD150((__int64)&v128, v130, v31, 8, a5, v29);
            v30 = (unsigned int)v129;
            v29 = v120;
            v28 = v27;
            v32 = v121;
          }
          v33 = (_QWORD *)(v128 + 8 * v30);
          for ( i = (_QWORD *)(v128 + 8 * v32); i != v33; ++v33 )
          {
            if ( v33 )
              *v33 = 0;
          }
          LODWORD(v129) = v32;
        }
        v35 = *(_QWORD *)(a2 + 32) + 40 * v27;
        v36 = *(_DWORD *)(v35 + 8);
        if ( v36 >= 0
          || (v49 = *(_QWORD *)(a1 + 48), v50 = v36 & 0x7FFFFFFF, (unsigned int)v50 >= *(_DWORD *)(v49 + 336)) )
        {
          v37 = 0;
          v38 = 0;
          v39 = 0;
        }
        else
        {
          v51 = *(_QWORD *)(*(_QWORD *)(v49 + 328) + 8 * v50);
          v39 = v51 & 1;
          v37 = v51 >> 2;
          v38 = (v51 & 2) != 0;
        }
        v40 = 4 * v37;
        a5 = (4 * v37) | v39 & 1 | (2LL * v38);
        if ( (4 * v37) | (unsigned __int8)(2 * v38) & 0xFC )
        {
          v41 = (__int64 *)(v128 + 8LL * (v29 - 6));
          v42 = *v41;
          if ( (*v41 & 0xFFFFFFFFFFFFFFFCLL) != 0 )
          {
            if ( (((unsigned __int8)a5 ^ (unsigned __int8)v42) & 3) != 0 || ((v42 ^ a5) & 0xFFFFFFFFFFFFFFFCLL) != 0 )
              sub_1E86D40(a1, "Type mismatch in generic instruction", v35, v28, a5);
          }
          else
          {
            *(_BYTE *)v41 = *(_BYTE *)v41 & 0xFC | v39 | (2 * v38);
            *v41 = *v41 & 3 | v40;
          }
        }
        else
        {
          sub_1E86D40(a1, "Generic instruction is missing a virtual register type", v35, v28, 0);
        }
      }
      ++v27;
    }
    while ( v7[1] > (unsigned int)v27 );
    if ( !*(_DWORD *)(a2 + 40) )
      goto LABEL_61;
  }
  else if ( !*(_DWORD *)(a2 + 40) )
  {
    goto LABEL_63;
  }
  v43 = 0;
  do
  {
    v44 = *(_QWORD *)(a2 + 32) + 40LL * v43;
    if ( !*(_BYTE *)v44 && *(int *)(v44 + 8) > 0 )
      sub_1E86D40(a1, "Generic instruction cannot have physical register", v44, v43, 0);
    ++v43;
  }
  while ( v43 < *(_DWORD *)(a2 + 40) );
LABEL_61:
  if ( (_BYTE *)v128 != v130 )
    _libc_free(v128);
LABEL_63:
  v45 = *(_QWORD *)(a1 + 32);
  v127[0] = 0;
  v127[1] = 0;
  v46 = *(__int64 (**)())(*(_QWORD *)v45 + 896LL);
  if ( v46 != sub_1E84D10
    && !((unsigned __int8 (__fastcall *)(__int64, unsigned __int64, const char **))v46)(v45, a2, v127) )
  {
    sub_1E86C30(a1, v127[0], a2);
  }
  result = **(unsigned __int16 **)(a2 + 16);
  if ( (unsigned __int16)result <= 0x53u )
  {
    if ( (unsigned __int16)result > 0xEu )
    {
      LODWORD(result) = result - 15;
      v48 = (unsigned int)result;
      result = (unsigned __int16)result;
      switch ( (__int16)result )
      {
        case 0:
          if ( *(_DWORD *)(a1 + 56) )
            return result;
          v80 = *(_DWORD **)(a2 + 32);
          v81 = *(_QWORD *)(a1 + 48);
          v126 = sub_1E85F00(v81, v80[2]);
          v123 = v80[12];
          result = sub_1E85F00(v81, v123);
          v128 = result;
          if ( (result & 0xFFFFFFFFFFFFFFFCLL) == 0 )
            goto LABEL_147;
          v82 = v123;
          if ( (v126 & 0xFFFFFFFFFFFFFFFCLL) == 0
            || (((unsigned __int8)v126 ^ (unsigned __int8)result) & 3) == 0
            && ((v126 ^ result) & 0xFFFFFFFFFFFFFFFCLL) == 0 )
          {
            goto LABEL_143;
          }
          sub_1E86C30(a1, "Copy Instruction is illegal with mismatching types", a2);
          v83 = sub_16E8CB0();
          v84 = sub_1263B40((__int64)v83, "Def = ");
          sub_39462E0(&v126, v84, v85, v86);
          v87 = sub_1263B40(v84, ", Src = ");
          sub_39462E0(&v128, v87, v88, v89);
          result = sub_1263B40(v87, "\n");
          if ( (v128 & 0xFFFFFFFFFFFFFFFCLL) != 0 )
            goto LABEL_142;
LABEL_147:
          if ( (v126 & 0xFFFFFFFFFFFFFFFCLL) != 0 )
          {
LABEL_142:
            v81 = *(_QWORD *)(a1 + 48);
            v82 = (unsigned int)v80[12];
LABEL_143:
            v90 = sub_1F4B530(*(_QWORD *)(a1 + 40), v82, v81);
            result = sub_1F4B530(*(_QWORD *)(a1 + 40), (unsigned int)v80[2], *(_QWORD *)(a1 + 48));
            v91 = result;
            if ( v90 != (_DWORD)result && (*v80 & 0xFFF00) == 0 && (v80[10] & 0xFFF00) == 0 )
            {
              sub_1E86C30(a1, "Copy Instruction is illegal with mismatching sizes", a2);
              v92 = sub_16E8CB0();
              v93 = sub_1263B40((__int64)v92, "Def Size = ");
              v94 = sub_16E7A90(v93, v91);
              v95 = sub_1263B40(v94, ", Src Size = ");
              v96 = sub_16E7A90(v95, v90);
              return sub_1263B40(v96, "\n");
            }
          }
          return result;
        case 8:
          result = *(_QWORD *)(a2 + 32);
          if ( *(_BYTE *)result != 1 || *(_BYTE *)(result + 40) != 1 || *(_BYTE *)(result + 80) != 1 )
            return sub_1E86C30(a1, "meta operands to STATEPOINT not constant!", a2);
          return result;
        case 30:
          v70 = *(_QWORD *)(a2 + 32);
          v71 = *(_QWORD *)(a1 + 48);
          v126 = sub_1E85F00(v71, *(_DWORD *)(v70 + 8));
          if ( (v126 & 0xFFFFFFFFFFFFFFFCLL) == 0 )
            return sub_1E86C30(a1, "Generic Instruction G_PHI has operands with incompatible/missing types", a2);
          v72 = *(unsigned int *)(a2 + 40);
          v128 = a1;
          v73 = v70 + 40;
          v129 = (__int64)&v126;
          v72 *= 40;
          v74 = v70 + v72;
          result = 0xCCCCCCCCCCCCCCCDLL * ((v72 - 40) >> 3);
          if ( result >> 2 <= 0 )
            goto LABEL_173;
          v75 = v70 + 160 * (result >> 2) + 40;
          break;
        case 40:
        case 43:
          if ( *(_BYTE *)(a2 + 49) != 1 )
            return sub_1E86C30(a1, "Generic instruction accessing memory must have one mem operand", a2);
          return result;
        case 61:
        case 62:
        case 67:
        case 68:
          goto LABEL_106;
        default:
          return v48;
      }
      while ( 1 )
      {
        if ( !*(_BYTE *)v73 )
        {
          v76 = *(_DWORD *)(v73 + 8);
          if ( v76 >= 0 || (v110 = v76 & 0x7FFFFFFF, (unsigned int)v110 >= *(_DWORD *)(v71 + 336)) )
          {
            v77 = 0;
            v78 = 0;
            v79 = 0;
          }
          else
          {
            v111 = (_BYTE *)(*(_QWORD *)(v71 + 328) + 8 * v110);
            v79 = *v111 & 1;
            v78 = (*v111 & 2) != 0;
            v77 = *(_QWORD *)v111 >> 2;
          }
          result = (4 * v77) | v79 | (2LL * v78);
          if ( !((4 * v77) | (unsigned __int16)(v79 | (unsigned __int16)(2 * v78)) & 0xFFFC) )
            break;
          if ( (((unsigned __int8)v126 ^ (unsigned __int8)result) & 3) != 0 )
            break;
          result ^= v126;
          if ( (result & 0xFFFFFFFFFFFFFFFCLL) != 0 )
            break;
        }
        v97 = v73 + 40;
        if ( !*(_BYTE *)(v73 + 40) )
        {
          v98 = *(_DWORD *)(v73 + 48);
          if ( v98 >= 0 || (v112 = v98 & 0x7FFFFFFF, (unsigned int)v112 >= *(_DWORD *)(v71 + 336)) )
          {
            v99 = 0;
            v100 = 0;
            v101 = 0;
          }
          else
          {
            v113 = (_BYTE *)(*(_QWORD *)(v71 + 328) + 8 * v112);
            v101 = *v113 & 1;
            v100 = (*v113 & 2) != 0;
            v99 = *(_QWORD *)v113 >> 2;
          }
          result = (4 * v99) | v101 | (2LL * v100);
          if ( !((4 * v99) | (unsigned __int16)(v101 | (unsigned __int16)(2 * v100)) & 0xFFFC) )
            goto LABEL_178;
          if ( (((unsigned __int8)v126 ^ (unsigned __int8)result) & 3) != 0 )
            goto LABEL_178;
          result ^= v126;
          if ( (result & 0xFFFFFFFFFFFFFFFCLL) != 0 )
            goto LABEL_178;
        }
        v97 = v73 + 80;
        if ( !*(_BYTE *)(v73 + 80) )
        {
          v102 = *(_DWORD *)(v73 + 88);
          if ( v102 >= 0 || (v114 = v102 & 0x7FFFFFFF, (unsigned int)v114 >= *(_DWORD *)(v71 + 336)) )
          {
            v103 = 0;
            v104 = 0;
            v105 = 0;
          }
          else
          {
            v115 = (_BYTE *)(*(_QWORD *)(v71 + 328) + 8 * v114);
            v105 = *v115 & 1;
            v104 = (*v115 & 2) != 0;
            v103 = *(_QWORD *)v115 >> 2;
          }
          result = (4 * v103) | v105 | (2LL * v104);
          if ( !((4 * v103) | (unsigned __int16)(v105 | (unsigned __int16)(2 * v104)) & 0xFFFC) )
            goto LABEL_178;
          if ( (((unsigned __int8)v126 ^ (unsigned __int8)result) & 3) != 0 )
            goto LABEL_178;
          result ^= v126;
          if ( (result & 0xFFFFFFFFFFFFFFFCLL) != 0 )
            goto LABEL_178;
        }
        v97 = v73 + 120;
        if ( !*(_BYTE *)(v73 + 120) )
        {
          v106 = *(_DWORD *)(v73 + 128);
          if ( v106 >= 0 || (v116 = v106 & 0x7FFFFFFF, (unsigned int)v116 >= *(_DWORD *)(v71 + 336)) )
          {
            v107 = 0;
            v108 = 0;
            v109 = 0;
          }
          else
          {
            v117 = (_BYTE *)(*(_QWORD *)(v71 + 328) + 8 * v116);
            v109 = *v117 & 1;
            v108 = (*v117 & 2) != 0;
            v107 = *(_QWORD *)v117 >> 2;
          }
          result = (4 * v107) | v109 | (2LL * v108);
          if ( !((4 * v107) | (unsigned __int16)(v109 | (unsigned __int16)(2 * v108)) & 0xFFFC)
            || (((unsigned __int8)v126 ^ (unsigned __int8)result) & 3) != 0
            || (result ^= v126, (result & 0xFFFFFFFFFFFFFFFCLL) != 0) )
          {
LABEL_178:
            v73 = v97;
            break;
          }
        }
        v73 += 160;
        if ( v75 == v73 )
        {
          result = 0xCCCCCCCCCCCCCCCDLL * ((v74 - v73) >> 3);
LABEL_173:
          if ( result != 2 )
          {
            if ( result != 3 )
            {
              if ( result != 1 )
                return result;
LABEL_176:
              result = sub_1E85000((_QWORD **)&v128, v73);
              if ( (_BYTE)result )
                return result;
              break;
            }
            result = sub_1E85000((_QWORD **)&v128, v73);
            if ( !(_BYTE)result )
              break;
            v73 += 40;
          }
          result = sub_1E85000((_QWORD **)&v128, v73);
          if ( !(_BYTE)result )
            break;
          v73 += 40;
          goto LABEL_176;
        }
      }
      if ( v74 != v73 )
        return sub_1E86C30(a1, "Generic Instruction G_PHI has operands with incompatible/missing types", a2);
    }
    return result;
  }
  result = (unsigned int)(result - 110);
  if ( (unsigned __int16)result > 1u )
    return result;
LABEL_106:
  result = v7[1];
  if ( (unsigned int)result <= *(_DWORD *)(a2 + 40) )
  {
    v64 = *(_QWORD *)(a2 + 32);
    v65 = *(_QWORD *)(a1 + 48);
    v66 = sub_1E85F00(v65, *(_DWORD *)(v64 + 8));
    v67 = *(_DWORD *)(v64 + 48);
    v124 = v66;
    v68 = v66;
    result = sub_1E85F00(v65, v67);
    v125 = result;
    if ( (v68 & 0xFFFFFFFFFFFFFFFCLL) != 0 && (result & 0xFFFFFFFFFFFFFFFCLL) != 0 )
    {
      if ( (v124 & 2) != 0 )
      {
        v126 = sub_1E85E90(&v124);
        v69 = (_QWORD *)v125;
        if ( (v125 & 0xFFFFFFFFFFFFFFFCLL) == 0 )
        {
LABEL_113:
          v128 = (unsigned __int64)v69;
          if ( (v126 & 0xFFFFFFFFFFFFFFFCLL) != 0 && (v126 & 3) == 1
            || (v128 & 0xFFFFFFFFFFFFFFFCLL) != 0 && (v128 & 3) == 1 )
          {
            sub_1E86C30(a1, "Generic extend/truncate can not operate on pointers", a2);
          }
          if ( (v124 & 0xFFFFFFFFFFFFFFFCLL) != 0 )
          {
            if ( (v124 & 2) != 0 )
            {
              if ( (v125 & 0xFFFFFFFFFFFFFFFCLL) == 0 || (v125 & 2) == 0 )
                return sub_1E86C30(a1, "Generic extend/truncate must be all-vector or all-scalar", a2);
              if ( (unsigned __int16)(v125 >> 2) != (unsigned __int16)(v124 >> 2) )
                sub_1E86C30(a1, "Generic vector extend/truncate must preserve number of lanes", a2);
            }
            else if ( (v125 & 0xFFFFFFFFFFFFFFFCLL) != 0 && (v125 & 2) != 0 )
            {
              return sub_1E86C30(a1, "Generic extend/truncate must be all-vector or all-scalar", a2);
            }
          }
          else if ( (v125 & 0xFFFFFFFFFFFFFFFCLL) != 0 && (v125 & 2) != 0 )
          {
            return sub_1E86C30(a1, "Generic extend/truncate must be all-vector or all-scalar", a2);
          }
          v118 = sub_1E85E20(&v126);
          result = sub_1E85E20(&v128);
          v119 = **(_WORD **)(a2 + 16);
          if ( v119 == 77 || v119 == 111 )
          {
            if ( v118 >= (unsigned int)result )
              return sub_1E86C30(a1, "Generic truncate has destination type no smaller than source", a2);
          }
          else if ( v118 <= (unsigned int)result )
          {
            return sub_1E86C30(a1, "Generic extend has destination type no larger than source", a2);
          }
          return result;
        }
      }
      else
      {
        v126 = v68;
      }
      if ( (v125 & 2) != 0 )
        v69 = (_QWORD *)sub_1E85E90(&v125);
      else
        v69 = (_QWORD *)v125;
      goto LABEL_113;
    }
  }
  return result;
}
