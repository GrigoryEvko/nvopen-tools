// Function: sub_2EF7830
// Address: 0x2ef7830
//
void __fastcall sub_2EF7830(__int64 *a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v4; // rax
  int v5; // eax
  __int64 v6; // rdx
  unsigned int v7; // r13d
  unsigned int v8; // ebx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r15
  _BYTE *v13; // rdx
  int v14; // eax
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 (*v17)(); // rax
  __int64 v18; // r8
  __int64 v19; // rsi
  __int64 v20; // rax
  int *v21; // rdx
  int v22; // eax
  int *v23; // rbx
  __int64 v24; // rcx
  int *v25; // r13
  __int64 v26; // r15
  __int16 v27; // ax
  int v28; // edi
  int v29; // eax
  __int64 v30; // rax
  int v31; // eax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rdi
  int v36; // eax
  int v37; // ecx
  unsigned int v38; // eax
  __int64 v39; // rdx
  __int16 v40; // ax
  __int64 v41; // rdi
  __int64 (*v42)(); // rax
  unsigned __int16 v43; // ax
  int v44; // r13d
  unsigned int i; // ebx
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // r14
  __int64 v49; // r15
  __int64 v50; // rsi
  __int64 v51; // rax
  __m128i *v52; // rdx
  __int64 v53; // rdi
  __m128i si128; // xmm0
  __int64 v55; // rax
  _QWORD *v56; // rdx
  int v57; // r9d
  __int16 v58; // ax
  unsigned int v59; // ebx
  __int64 v60; // r15
  bool v61; // al
  __int64 v62; // r10
  _DWORD *v63; // r15
  __int64 v64; // rdx
  int v65; // r13d
  int v66; // ebx
  unsigned __int64 v67; // r8
  bool v68; // di
  bool v69; // al
  unsigned __int8 v70; // r9
  unsigned __int64 v71; // r8
  bool v72; // di
  bool v73; // al
  unsigned __int8 v74; // r9
  __int64 v75; // rax
  unsigned __int64 v76; // r8
  __int64 v77; // r12
  __int64 v78; // r12
  __int64 v79; // rax
  __int64 v80; // rdx
  __int64 v81; // rdi
  __int64 v82; // rdx
  unsigned __int64 v83; // r14
  __int64 v84; // rax
  __int64 *v85; // rdx
  unsigned int v86; // r10d
  __int64 v87; // r8
  char v88; // r9
  __int64 v89; // rax
  unsigned __int64 v90; // r8
  __int64 v91; // r12
  __int64 v92; // rax
  __int64 v93; // r12
  __int64 v94; // rax
  __int64 v95; // rdi
  unsigned int v96; // ebx
  char v97; // al
  __int64 v98; // rax
  int v99; // eax
  __int64 v100; // rax
  __int64 v101; // rdx
  _DWORD *v102; // rdx
  __int64 *v103; // rdx
  __int64 v104; // rcx
  unsigned int v105; // ebx
  unsigned int v106; // eax
  unsigned int v107; // eax
  unsigned int v108; // eax
  unsigned int v109; // r14d
  unsigned int v110; // r13d
  unsigned int v111; // eax
  __int64 v112; // r15
  __int64 v113; // rax
  __int64 v114; // rax
  unsigned __int64 v115; // r8
  char v116; // al
  __int64 v117; // rax
  unsigned __int64 v118; // r8
  char v119; // al
  __int64 v120; // rax
  __int64 v121; // rax
  __int64 v122; // rax
  __int64 v123; // rax
  unsigned int v124; // [rsp+Ch] [rbp-94h]
  __int64 v125; // [rsp+10h] [rbp-90h]
  unsigned __int64 v126; // [rsp+10h] [rbp-90h]
  __int64 v127; // [rsp+10h] [rbp-90h]
  __int64 v128; // [rsp+10h] [rbp-90h]
  __int64 v129; // [rsp+18h] [rbp-88h]
  char v130; // [rsp+18h] [rbp-88h]
  char v131; // [rsp+18h] [rbp-88h]
  char v132; // [rsp+18h] [rbp-88h]
  __int64 v133; // [rsp+20h] [rbp-80h]
  char v134; // [rsp+20h] [rbp-80h]
  __int64 v135; // [rsp+28h] [rbp-78h] BYREF
  __int64 v136; // [rsp+30h] [rbp-70h] BYREF
  __int64 v137; // [rsp+38h] [rbp-68h] BYREF
  char *v138[2]; // [rsp+40h] [rbp-60h] BYREF
  __int64 v139; // [rsp+50h] [rbp-50h] BYREF
  __int64 v140; // [rsp+58h] [rbp-48h]
  __int64 *v141; // [rsp+60h] [rbp-40h] BYREF
  __int64 *v142; // [rsp+68h] [rbp-38h]

  v2 = a2;
  v4 = *(_QWORD *)(a2 + 16);
  v135 = a2;
  v133 = v4;
  if ( *(unsigned __int16 *)(v4 + 2) > (*(_DWORD *)(a2 + 40) & 0xFFFFFFu) )
  {
    sub_2EF06E0((__int64)a1, "Too few operands", a2);
    v51 = sub_CB59D0(a1[2], *(unsigned __int16 *)(v133 + 2));
    v52 = *(__m128i **)(v51 + 32);
    v53 = v51;
    if ( *(_QWORD *)(v51 + 24) - (_QWORD)v52 <= 0x17u )
    {
      v53 = sub_CB6200(v51, " operands expected, but ", 0x18u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_4453DC0);
      v52[1].m128i_i64[0] = 0x20747562202C6465LL;
      *v52 = si128;
      *(_QWORD *)(v51 + 32) += 24LL;
    }
    v55 = sub_CB59D0(v53, *(_DWORD *)(v135 + 40) & 0xFFFFFF);
    v56 = *(_QWORD **)(v55 + 32);
    if ( *(_QWORD *)(v55 + 24) - (_QWORD)v56 <= 7u )
    {
      sub_CB6200(v55, " given.\n", 8u);
      v2 = v135;
    }
    else
    {
      *v56 = 0xA2E6E6576696720LL;
      v2 = v135;
      *(_QWORD *)(v55 + 32) += 8LL;
    }
  }
  if ( (*(_BYTE *)(v2 + 46) & 2) != 0 && (*(_BYTE *)(v133 + 28) & 0x10) == 0 )
  {
    sub_2EF06E0((__int64)a1, "NoConvergent flag expected only on convergent instructions.", v2);
    v2 = v135;
  }
  v5 = *(unsigned __int16 *)(v2 + 68);
  if ( v5 != 68 && *(_WORD *)(v2 + 68) )
  {
    if ( a1[11] )
      goto LABEL_10;
    a1[11] = v2;
  }
  else if ( (*(_BYTE *)(a1[4] + 344) & 2) != 0 )
  {
    sub_2EF06E0((__int64)a1, "Found PHI instruction with NoPHIs property set", v2);
    v2 = v135;
    if ( a1[11] )
      goto LABEL_9;
  }
  else if ( a1[11] )
  {
LABEL_9:
    sub_2EF06E0((__int64)a1, "Found PHI instruction after non-PHI", v2);
    v2 = v135;
    v5 = *(unsigned __int16 *)(v135 + 68);
    goto LABEL_10;
  }
  v5 = *(unsigned __int16 *)(v2 + 68);
LABEL_10:
  if ( (unsigned int)(v5 - 1) <= 1 )
  {
    if ( (*(_DWORD *)(v2 + 40) & 0xFFFFFFu) <= 1 )
    {
      sub_2EF06E0((__int64)a1, "Too few operands on inline asm", v2);
      v2 = v135;
      goto LABEL_33;
    }
    v6 = *(_QWORD *)(v2 + 32);
    if ( *(_BYTE *)v6 != 9 )
    {
      sub_2EF06E0((__int64)a1, "Asm string must be an external symbol", v2);
      v6 = *(_QWORD *)(v2 + 32);
    }
    if ( *(_BYTE *)(v6 + 40) != 1 )
    {
      sub_2EF06E0((__int64)a1, "Asm flags must be an immediate", v2);
      v6 = *(_QWORD *)(v2 + 32);
    }
    if ( *(_QWORD *)(v6 + 64) > 0x3Fu )
      sub_2EF0A60((__int64)a1, "Unknown asm flags", v6 + 40, 1u, 0);
    v7 = *(_DWORD *)(v2 + 40) & 0xFFFFFF;
    if ( v7 <= 2 )
    {
      v8 = 2;
LABEL_115:
      if ( v7 >= v8 )
        goto LABEL_31;
    }
    else
    {
      v8 = 2;
      while ( 1 )
      {
        v9 = *(_QWORD *)(v2 + 32) + 40LL * v8;
        if ( *(_BYTE *)v9 != 1 )
          break;
        v8 += (((unsigned int)*(_QWORD *)(v9 + 24) >> 3) & 0x1FFF) + 1;
        if ( v7 <= v8 )
          goto LABEL_115;
      }
      if ( v7 >= v8 )
      {
LABEL_23:
        if ( v7 > v8 )
        {
          v10 = *(_QWORD *)(v2 + 32);
          v11 = v8;
          if ( *(_BYTE *)(v10 + 40LL * v8) == 14 )
          {
            if ( ++v8 >= v7 )
            {
              if ( *(_WORD *)(v2 + 68) == 2 )
              {
                v129 = *(_QWORD *)(v2 + 24);
                goto LABEL_129;
              }
              goto LABEL_32;
            }
            v11 = v8;
          }
          v12 = 40 * v11;
          while ( 1 )
          {
            v13 = (_BYTE *)(v12 + v10);
            if ( *v13 || (v13[3] & 0x20) == 0 )
              sub_2EF0A60((__int64)a1, "Expected implicit register after groups", (__int64)v13, v8, 0);
            ++v8;
            v12 += 40;
            if ( v8 == v7 )
              break;
            v10 = *(_QWORD *)(v2 + 32);
          }
        }
LABEL_31:
        if ( *(_WORD *)(v2 + 68) == 2 )
        {
          v7 = *(_DWORD *)(v2 + 40) & 0xFFFFFF;
          v129 = *(_QWORD *)(v2 + 24);
          if ( v7 != 2 )
          {
            v10 = *(_QWORD *)(v2 + 32);
LABEL_129:
            v59 = 2;
            while ( 1 )
            {
              v60 = v10 + 40LL * v59;
              if ( *(_BYTE *)v60 == 4 )
              {
                if ( !*(_QWORD *)(v60 + 24) )
                {
                  sub_2EF0A60((__int64)a1, "INLINEASM_BR indirect target does not exist", v10 + 40LL * v59, v59, 0);
                  break;
                }
                v125 = *(_QWORD *)(v60 + 24);
                v61 = sub_2E322C0(v129, v125);
                v62 = v125;
                if ( !v61 )
                {
                  sub_2EF0A60((__int64)a1, "INLINEASM_BR indirect target missing from successor list", v60, v59, 0);
                  v62 = v125;
                }
                if ( !sub_2E32290(v62, v129) )
                  sub_2EF0A60((__int64)a1, "INLINEASM_BR indirect target predecessor list missing parent", v60, v59, 0);
              }
              if ( v7 == ++v59 )
                break;
              v10 = *(_QWORD *)(v2 + 32);
            }
          }
        }
LABEL_32:
        v2 = v135;
        goto LABEL_33;
      }
    }
    sub_2EF06E0((__int64)a1, "Missing operands in last group", v2);
    v7 = *(_DWORD *)(v2 + 40) & 0xFFFFFF;
    goto LABEL_23;
  }
LABEL_33:
  v14 = *(_DWORD *)(v2 + 44);
  v15 = a1[6];
  if ( (v14 & 4) != 0 || (v14 & 8) == 0 )
    v16 = (*(_QWORD *)(*(_QWORD *)(v2 + 16) + 24LL) >> 9) & 1LL;
  else
    LOBYTE(v16) = sub_2E88A90(v2, 512, 1);
  if ( !(_BYTE)v16 )
    goto LABEL_38;
  v17 = *(__int64 (**)())(*(_QWORD *)v15 + 536LL);
  if ( v17 == sub_2EEE480 )
    goto LABEL_38;
  v97 = ((__int64 (__fastcall *)(__int64, __int64))v17)(v15, v2);
  v18 = v135;
  v19 = v135;
  if ( v97 )
  {
    v98 = *(_QWORD *)(v135 + 32);
    if ( *(_BYTE *)v98 || (*(_BYTE *)(v98 + 3) & 0x10) == 0 )
    {
      sub_2EF06E0((__int64)a1, "Unspillable Terminator does not define a reg", v135);
      v18 = v135;
      v98 = *(_QWORD *)(v135 + 32);
      v19 = v135;
    }
    v99 = *(_DWORD *)(v98 + 8);
    if ( v99 < 0 && (*(_BYTE *)(a1[4] + 344) & 2) == 0 )
    {
      v100 = *(_QWORD *)(*(_QWORD *)(a1[8] + 56) + 16LL * (v99 & 0x7FFFFFFF) + 8);
      if ( v100 )
      {
        while ( (*(_BYTE *)(v100 + 3) & 0x10) != 0 || (*(_BYTE *)(v100 + 4) & 8) != 0 )
        {
          v100 = *(_QWORD *)(v100 + 32);
          if ( !v100 )
            goto LABEL_39;
        }
        v101 = 0;
        while ( 1 )
        {
          v100 = *(_QWORD *)(v100 + 32);
          if ( !v100 )
            break;
          while ( (*(_BYTE *)(v100 + 3) & 0x10) == 0 && (*(_BYTE *)(v100 + 4) & 8) == 0 )
          {
            v100 = *(_QWORD *)(v100 + 32);
            ++v101;
            if ( !v100 )
              goto LABEL_190;
          }
        }
LABEL_190:
        if ( v101 )
        {
          sub_2EF06E0((__int64)a1, "Unspillable Terminator expected to have at most one use!", v18);
LABEL_38:
          v18 = v135;
          v19 = v135;
        }
      }
    }
  }
LABEL_39:
  if ( (unsigned __int16)(*(_WORD *)(v18 + 68) - 14) <= 1u
    && (*(_DWORD *)(v18 + 40) & 0xFFFFFF) == 4
    && !*(_QWORD *)(v18 + 56) )
  {
    sub_2EF06E0((__int64)a1, "Missing DebugLoc for debug instruction", v18);
    v18 = v135;
    v19 = v135;
  }
  if ( (*(_BYTE *)(*(_QWORD *)(v18 + 16) + 24LL) & 0x10) != 0 && *(_DWORD *)(v18 + 64) )
  {
    sub_2EF06E0((__int64)a1, "Metadata instruction should not have a value tracking number", v18);
    v18 = v135;
    v19 = v135;
  }
  v20 = *(_QWORD *)(v18 + 48);
  v21 = (int *)(v20 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v20 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    v22 = v20 & 7;
    if ( v22 )
    {
      if ( v22 != 3 )
        goto LABEL_69;
      v23 = v21 + 4;
      v24 = 2LL * *v21;
    }
    else
    {
      *(_QWORD *)(v18 + 48) = v21;
      v23 = (int *)(v18 + 48);
      v24 = 2;
    }
    v25 = &v23[v24];
    while ( v25 != v23 )
    {
      while ( 1 )
      {
        v26 = *(_QWORD *)v23;
        v27 = *(_WORD *)(*(_QWORD *)v23 + 32LL);
        if ( (v27 & 1) != 0 )
          break;
        if ( (v27 & 2) == 0 )
          goto LABEL_50;
LABEL_60:
        if ( (unsigned int)*(unsigned __int16 *)(v18 + 68) - 1 <= 1 )
        {
          v28 = *(_DWORD *)(*(_QWORD *)(v18 + 32) + 64LL);
          goto LABEL_62;
        }
LABEL_63:
        v31 = *(_DWORD *)(v18 + 44);
        if ( (v31 & 4) != 0 || (v31 & 8) == 0 )
        {
          v32 = (*(_QWORD *)(*(_QWORD *)(v18 + 16) + 24LL) >> 20) & 1LL;
        }
        else
        {
          LOBYTE(v32) = sub_2E88A90(v18, 0x100000, 1);
          v18 = v135;
        }
        v19 = v18;
        if ( !(_BYTE)v32 )
        {
          sub_2EF06E0((__int64)a1, "Missing mayStore flag", v18);
          v18 = v135;
          goto LABEL_68;
        }
LABEL_50:
        v23 += 2;
        if ( v25 == v23 )
          goto LABEL_69;
      }
      if ( (unsigned int)*(unsigned __int16 *)(v18 + 68) - 1 > 1
        || (LOBYTE(v28) = *(_QWORD *)(*(_QWORD *)(v18 + 32) + 64LL), (v28 & 8) == 0) )
      {
        v29 = *(_DWORD *)(v18 + 44);
        if ( (v29 & 4) != 0 || (v29 & 8) == 0 )
        {
          v30 = (*(_QWORD *)(*(_QWORD *)(v18 + 16) + 24LL) >> 19) & 1LL;
        }
        else
        {
          LOBYTE(v30) = sub_2E88A90(v18, 0x80000, 1);
          v18 = v135;
          v19 = v135;
        }
        if ( !(_BYTE)v30 )
        {
          sub_2EF06E0((__int64)a1, "Missing mayLoad flag", v18);
          v18 = v135;
          v19 = v135;
        }
        if ( (*(_WORD *)(v26 + 32) & 2) == 0 )
          goto LABEL_50;
        goto LABEL_60;
      }
      if ( (v27 & 2) == 0 )
        goto LABEL_50;
LABEL_62:
      if ( (v28 & 0x10) == 0 )
        goto LABEL_63;
LABEL_68:
      v23 += 2;
      v19 = v18;
    }
  }
LABEL_69:
  v33 = a1[80];
  if ( !v33 )
    goto LABEL_76;
  v34 = *(_QWORD *)(v33 + 32);
  v35 = *(_QWORD *)(v34 + 128);
  v36 = *(_DWORD *)(v34 + 144);
  if ( v36 )
  {
    v37 = v36 - 1;
    v38 = (v36 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
    v39 = *(_QWORD *)(v35 + 16LL * v38);
    if ( v39 == v18 )
    {
LABEL_72:
      v40 = *(_WORD *)(v18 + 68);
      if ( (unsigned __int16)(v40 - 14) <= 4u || v40 == 24 )
      {
        sub_2EF06E0((__int64)a1, "Debug instruction has a slot index", v18);
        v19 = v135;
      }
      else if ( (*(_BYTE *)(v18 + 44) & 4) != 0 )
      {
        sub_2EF06E0((__int64)a1, "Instruction inside bundle has a slot index", v18);
        v19 = v135;
      }
      goto LABEL_76;
    }
    v57 = 1;
    while ( v39 != -4096 )
    {
      v38 = v37 & (v57 + v38);
      v39 = *(_QWORD *)(v35 + 16LL * v38);
      if ( v39 == v18 )
        goto LABEL_72;
      ++v57;
    }
  }
  v58 = *(_WORD *)(v18 + 68);
  if ( (unsigned __int16)(v58 - 14) > 4u && v58 != 24 && (*(_BYTE *)(v18 + 44) & 4) == 0 )
  {
    sub_2EF06E0((__int64)a1, "Missing slot index", v18);
    v19 = v135;
  }
LABEL_76:
  if ( (unsigned __int16)(*(_WORD *)v133 - 50) <= 0xFFu )
  {
    sub_2EF30A0((__int64)a1, v19);
    return;
  }
  v41 = a1[6];
  v138[0] = 0;
  v138[1] = 0;
  v42 = *(__int64 (**)())(*(_QWORD *)v41 + 1208LL);
  if ( v42 != sub_2EEE490 )
  {
    if ( !((unsigned __int8 (__fastcall *)(__int64, __int64, char **))v42)(v41, v19, v138) )
      sub_2EF06E0((__int64)a1, v138[0], v135);
    v19 = v135;
  }
  v43 = *(_WORD *)(v19 + 68);
  if ( v43 == 20 )
  {
    v63 = *(_DWORD **)(v19 + 32);
    v64 = a1[8];
    v65 = v63[2];
    v66 = v63[12];
    if ( v65 >= 0 || (v117 = v65 & 0x7FFFFFFF, (unsigned int)v117 >= *(_DWORD *)(v64 + 464)) )
    {
      v67 = 0;
      v68 = 0;
      v69 = 0;
      v70 = 0;
    }
    else
    {
      v118 = *(_QWORD *)(*(_QWORD *)(v64 + 456) + 8 * v117);
      v119 = v118;
      v70 = v118 & 1;
      v68 = (v118 & 4) != 0;
      v67 = v118 >> 3;
      v69 = (v119 & 2) != 0;
    }
    v136 = (8 * v67) | (4LL * v68) | v70 | (2LL * v69);
    if ( v66 >= 0 || (v114 = v66 & 0x7FFFFFFF, (unsigned int)v114 >= *(_DWORD *)(v64 + 464)) )
    {
      v71 = 0;
      v72 = 0;
      v73 = 0;
      v74 = 0;
    }
    else
    {
      v115 = *(_QWORD *)(*(_QWORD *)(v64 + 456) + 8 * v114);
      v116 = v115;
      v74 = v115 & 1;
      v72 = (v115 & 4) != 0;
      v71 = v115 >> 3;
      v73 = (v116 & 2) != 0;
    }
    v75 = (8 * v71) | (4LL * v72) | v74 | (2LL * v73);
    v137 = v75;
    v76 = v136 & 0xFFFFFFFFFFFFFFF9LL;
    if ( (v75 & 0xFFFFFFFFFFFFFFF9LL) != 0 )
    {
      if ( v76 )
      {
        if ( (((unsigned __int8)v136 ^ (unsigned __int8)v75) & 7) != 0 || ((v136 ^ v75) & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        {
          sub_2EF06E0((__int64)a1, "Copy Instruction is illegal with mismatching types", v19);
          v77 = sub_904010(a1[2], "Def = ");
          sub_34B2640(&v136, v77);
          v78 = sub_904010(v77, ", Src = ");
          sub_34B2640(&v137, v78);
          sub_A51310(v78, 0xAu);
        }
        return;
      }
    }
    else if ( !v76 )
    {
      return;
    }
    v79 = sub_2FF6F50(a1[7], (unsigned int)v66, v64);
    v81 = v80;
    v139 = v79;
    v82 = a1[8];
    v83 = v79;
    v140 = v81;
    v134 = v81;
    v84 = sub_2FF6F50(a1[7], (unsigned int)v65, v82);
    v86 = v66 - 1;
    v141 = (__int64 *)v84;
    v87 = v84;
    v142 = v85;
    v88 = (char)v85;
    if ( (unsigned int)(v66 - 1) > 0x3FFFFFFE )
    {
      if ( (unsigned int)(v65 - 1) <= 0x3FFFFFFE )
      {
        if ( (v137 & 0xFFFFFFFFFFFFFFF9LL) == 0 )
          goto LABEL_202;
        goto LABEL_229;
      }
    }
    else
    {
      if ( (v136 & 0xFFFFFFFFFFFFFFF9LL) != 0 )
      {
        v127 = v84;
        v131 = (char)v85;
        v120 = sub_2FF6620(a1[7], (unsigned int)v66);
        v88 = v131;
        v87 = v127;
        v86 = v66 - 1;
        if ( v120 )
        {
          v134 = 0;
          v83 = *(unsigned int *)(*(_QWORD *)(a1[7] + 312)
                                + 16LL
                                * (*(unsigned __int16 *)(*(_QWORD *)v120 + 24LL)
                                 + *(_DWORD *)(a1[7] + 328)
                                 * (unsigned int)((__int64)(*(_QWORD *)(a1[7] + 288) - *(_QWORD *)(a1[7] + 280)) >> 3)));
        }
      }
      if ( (unsigned int)(v65 - 1) <= 0x3FFFFFFE )
      {
        if ( (v137 & 0xFFFFFFFFFFFFFFF9LL) == 0 )
        {
LABEL_155:
          if ( v65 < 0 && v86 <= 0x3FFFFFFE && v88 )
            goto LABEL_158;
LABEL_202:
          if ( v66 < 0 && v134 && !v88 )
            return;
          goto LABEL_159;
        }
LABEL_229:
        v124 = v86;
        v128 = v87;
        v132 = v88;
        v121 = sub_2FF6620(a1[7], (unsigned int)v65);
        v88 = v132;
        v87 = v128;
        v86 = v124;
        if ( v121 )
        {
          v88 = 0;
          v87 = *(unsigned int *)(*(_QWORD *)(a1[7] + 312)
                                + 16LL
                                * (*(unsigned __int16 *)(*(_QWORD *)v121 + 24LL)
                                 + *(_DWORD *)(a1[7] + 328)
                                 * (unsigned int)((__int64)(*(_QWORD *)(a1[7] + 288) - *(_QWORD *)(a1[7] + 280)) >> 3)));
          goto LABEL_202;
        }
        goto LABEL_155;
      }
      if ( v65 < 0 && v88 )
      {
LABEL_158:
        if ( !v134 )
          return;
      }
    }
LABEL_159:
    if ( v83 && v87 && (v87 != v83 || v134 != v88) && (*v63 & 0xFFF00) == 0 && (v63[10] & 0xFFF00) == 0 )
    {
      v126 = v87;
      v130 = v88;
      sub_2EF06E0((__int64)a1, "Copy Instruction is illegal with mismatching sizes", v135);
      v89 = sub_904010(a1[2], "Def Size = ");
      v90 = v126;
      v91 = v89;
      if ( v130 )
      {
        v123 = *(_QWORD *)(v89 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(v91 + 24) - v123) <= 8 )
        {
          sub_CB6200(v91, "vscale x ", 9u);
          v90 = v126;
        }
        else
        {
          *(_BYTE *)(v123 + 8) = 32;
          *(_QWORD *)v123 = 0x7820656C61637376LL;
          *(_QWORD *)(v91 + 32) += 9LL;
        }
      }
      sub_CB59D0(v91, v90);
      v92 = sub_904010(v91, ", Src Size = ");
      v93 = v92;
      if ( v134 )
      {
        v122 = *(_QWORD *)(v92 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(v93 + 24) - v122) <= 8 )
        {
          sub_CB6200(v93, "vscale x ", 9u);
        }
        else
        {
          *(_BYTE *)(v122 + 8) = 32;
          *(_QWORD *)v122 = 0x7820656C61637376LL;
          *(_QWORD *)(v93 + 32) += 9LL;
        }
      }
      sub_CB59D0(v93, v83);
      sub_A51310(v93, 0xAu);
    }
    return;
  }
  if ( v43 > 0x14u )
  {
    if ( v43 == 32 )
    {
      v139 = v19;
      LODWORD(v140) = *(unsigned __int8 *)(*(_QWORD *)(v19 + 16) + 9LL) + (unsigned int)sub_2E88FE0(v19);
      v50 = *(_QWORD *)(v135 + 32);
      if ( *(_BYTE *)(v50 + 40LL * (unsigned int)v140) == 1
        && *(_BYTE *)(v50 + 40LL * (unsigned int)(v140 + 1)) == 1
        && (v104 = 40LL * (unsigned int)(v140 + 2), *(_BYTE *)(v50 + v104) == 1) )
      {
        v142 = a1;
        v141 = &v135;
        v105 = 0;
        sub_2EF0A00((__int64)&v141, *(_DWORD *)(*(_QWORD *)(v139 + 32) + v104 + 24) + v140 + 5);
        sub_2EF0A00(
          (__int64)&v141,
          *(_DWORD *)(*(_QWORD *)(v139 + 32) + 40LL * (unsigned int)(v140 + 2) + 24) + v140 + 7);
        sub_2EF0A00(
          (__int64)&v141,
          *(_DWORD *)(*(_QWORD *)(v139 + 32) + 40LL * (unsigned int)(v140 + 2) + 24) + v140 + 9);
        v106 = sub_2FC8910(&v139);
        sub_2EF0A00((__int64)&v141, v106);
        v107 = sub_2FC89B0(&v139);
        sub_2EF0A00((__int64)&v141, v107);
        v108 = sub_2FC8A10(&v139);
        sub_2EF0A00((__int64)&v141, v108);
        v109 = sub_2FC8970(&v139);
        v110 = sub_2FC89B0(&v139) - 2;
        while ( 1 )
        {
          v112 = v135;
          if ( v105 >= (unsigned int)sub_2E88FE0(v135) + *(unsigned __int8 *)(*(_QWORD *)(v112 + 16) + 9LL) )
            break;
          v113 = *(_QWORD *)(v135 + 32) + 40LL * v105;
          if ( *(_BYTE *)v113 || (*(_BYTE *)(v113 + 3) & 0x10) == 0 || (*(_WORD *)(v113 + 2) & 0xFF0) == 0 )
          {
            sub_2EF06E0((__int64)a1, "STATEPOINT defs expected to be tied", v135);
            return;
          }
          v111 = sub_2E89F40(v135, v105);
          if ( v109 > v111 || v110 < v111 )
          {
            sub_2EF06E0((__int64)a1, "STATEPOINT def tied to non-gc operand", v135);
            return;
          }
          ++v105;
        }
      }
      else
      {
        sub_2EF06E0((__int64)a1, "meta operands to STATEPOINT not constant!", v135);
      }
    }
  }
  else if ( v43 == 9 )
  {
    v94 = *(_QWORD *)(v19 + 32);
    v95 = a1[7];
    if ( ((*(_DWORD *)(v94 + 80) >> 8) & 0xFFF) != 0 )
    {
      v96 = sub_2FF7530(v95, (*(_DWORD *)(v94 + 80) >> 8) & 0xFFF);
    }
    else
    {
      v141 = (__int64 *)sub_2FF6F50(v95, *(unsigned int *)(v94 + 88), a1[8]);
      v142 = v103;
      v96 = sub_CA1930(&v141);
    }
    if ( v96 > (unsigned int)sub_2FF7530(a1[7], *(_QWORD *)(*(_QWORD *)(v135 + 32) + 144LL)) )
      sub_2EF06E0(
        (__int64)a1,
        "INSERT_SUBREG expected inserted value to have equal or lesser size than the subreg it was inserted into",
        v135);
  }
  else if ( v43 == 19 )
  {
    v44 = *(_DWORD *)(v19 + 40) & 0xFFFFFF;
    if ( (*(_DWORD *)(v19 + 40) & 1) != 0 )
    {
      if ( v44 != 1 )
      {
        for ( i = 1; i != v44; i += 2 )
        {
          v47 = *(_QWORD *)(v19 + 32);
          v48 = i + 1;
          v49 = v47 + 40 * v48;
          if ( *(_BYTE *)(v47 + 40LL * i) )
            sub_2EF0A60((__int64)a1, "Invalid register operand for REG_SEQUENCE", v47 + 40LL * i, i, 0);
          if ( *(_BYTE *)v49 != 1 || (v46 = *(_QWORD *)(v49 + 24)) == 0 || *(unsigned int *)(a1[7] + 96) <= v46 )
            sub_2EF0A60((__int64)a1, "Invalid subregister index operand for REG_SEQUENCE", v49, v48, 0);
          v19 = v135;
        }
      }
      v102 = *(_DWORD **)(v19 + 32);
      if ( (unsigned int)(v102[2] - 1) <= 0x3FFFFFFE )
      {
        sub_2EF06E0((__int64)a1, "REG_SEQUENCE does not support physical register results", v19);
        v19 = v135;
        v102 = *(_DWORD **)(v135 + 32);
      }
      if ( (*v102 & 0xFFF00) != 0 )
        sub_2EF06E0((__int64)a1, "Invalid subreg result for REG_SEQUENCE", v19);
    }
    else
    {
      sub_2EF06E0((__int64)a1, "Invalid number of operands for REG_SEQUENCE", v19);
    }
  }
}
