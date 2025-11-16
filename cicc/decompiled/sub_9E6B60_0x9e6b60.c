// Function: sub_9E6B60
// Address: 0x9e6b60
//
__int64 *__fastcall sub_9E6B60(__int64 *a1, __int64 a2, __int64 *a3, unsigned __int64 a4)
{
  __int64 v5; // r15
  const char *v6; // rax
  const char *v8; // rbx
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rsi
  char v13; // al
  __int64 v14; // r12
  unsigned int v15; // r14d
  __int64 v16; // r9
  unsigned int v17; // ecx
  __int64 v18; // rax
  __int64 v19; // rbx
  unsigned int v20; // esi
  __int64 v21; // r9
  __int64 v22; // r8
  __int64 v23; // rdi
  int v24; // r15d
  __int64 *v25; // rax
  unsigned int v26; // r11d
  _QWORD *v27; // rdx
  __int64 v28; // rcx
  unsigned int *v29; // rax
  __int64 v30; // rdi
  __int64 v31; // rax
  bool v32; // al
  __int64 v33; // rcx
  __int64 v34; // rdx
  unsigned __int64 v35; // rsi
  unsigned int v36; // r12d
  unsigned int v37; // r15d
  unsigned int *v38; // rbx
  __int64 v39; // rax
  __int64 v40; // rcx
  unsigned __int64 v41; // rax
  __int64 v42; // r12
  unsigned int v43; // eax
  unsigned __int8 v44; // di
  __int64 v45; // rdx
  __int64 v46; // rax
  unsigned __int64 v47; // rdx
  char v48; // dl
  __int64 v49; // rcx
  __int64 v50; // rdx
  __int64 v51; // rcx
  const char *v52; // rax
  unsigned int v53; // eax
  __int64 v54; // rsi
  __int64 v55; // rdx
  __int64 v56; // rax
  int v57; // eax
  char v58; // cl
  unsigned int v59; // eax
  __int64 v60; // rsi
  __int64 v61; // rax
  bool v62; // zf
  __int64 v63; // rax
  __int64 v64; // rsi
  __int64 v65; // rsi
  char v66; // dl
  __int64 v67; // rcx
  char v68; // dl
  unsigned int v69; // esi
  __int64 v70; // rbx
  __int64 v71; // r10
  __int64 *v72; // rdx
  __int64 v73; // r8
  int v74; // r11d
  __int64 *v75; // rcx
  int v76; // eax
  int v77; // eax
  int v78; // ecx
  int v79; // edx
  __int64 v80; // rdx
  char v81; // cl
  __int64 v82; // rax
  _BYTE *v83; // rsi
  int v84; // r11d
  __int64 v85; // rdi
  int v86; // r11d
  __int64 v87; // r10
  unsigned int v88; // ecx
  __int64 *v89; // rsi
  int v90; // r11d
  int v91; // r11d
  __int64 v92; // r10
  unsigned int v93; // ecx
  unsigned int v94; // [rsp+8h] [rbp-D8h]
  __int64 v95; // [rsp+8h] [rbp-D8h]
  __int64 v96; // [rsp+10h] [rbp-D0h]
  unsigned __int64 v97; // [rsp+10h] [rbp-D0h]
  __int64 v98; // [rsp+18h] [rbp-C8h]
  unsigned __int64 v99; // [rsp+20h] [rbp-C0h]
  __int64 *v100; // [rsp+28h] [rbp-B8h]
  unsigned int v101; // [rsp+30h] [rbp-B0h]
  unsigned int v103; // [rsp+48h] [rbp-98h] BYREF
  unsigned int v104; // [rsp+4Ch] [rbp-94h] BYREF
  __int64 v105; // [rsp+50h] [rbp-90h] BYREF
  __int64 v106; // [rsp+58h] [rbp-88h] BYREF
  __int64 v107; // [rsp+60h] [rbp-80h] BYREF
  __int128 v108; // [rsp+68h] [rbp-78h]
  __int64 (__fastcall *v109)(__int64, unsigned int *); // [rsp+78h] [rbp-68h]
  __int64 v110[2]; // [rsp+80h] [rbp-60h] BYREF
  _QWORD v111[2]; // [rsp+90h] [rbp-50h] BYREF
  __int16 v112; // [rsp+A0h] [rbp-40h]

  v100 = a3;
  v99 = a4;
  v98 = a2 + 8;
  if ( *(_BYTE *)(a2 + 392) )
  {
    v5 = a3[1];
    if ( (unsigned __int64)(*a3 + v5) > *(_QWORD *)(a2 + 384) )
    {
LABEL_3:
      HIBYTE(v112) = 1;
      v6 = "Invalid record";
LABEL_4:
      v110[0] = (__int64)v6;
      LOBYTE(v112) = 3;
      sub_9C81F0(a1, v98, (__int64)v110);
      return a1;
    }
    v8 = (const char *)(*(_QWORD *)(a2 + 376) + *a3);
    v99 = a4 - 2;
    v100 = a3 + 2;
  }
  else
  {
    v5 = 0;
    v8 = byte_3F871B3;
  }
  if ( v99 <= 7 )
    goto LABEL_3;
  v10 = *v100;
  v103 = *v100;
  v11 = sub_9CAD80((_QWORD *)a2, v10);
  v12 = v11;
  if ( !v11 )
    goto LABEL_3;
  v13 = *(_BYTE *)(v11 + 8);
  if ( v13 == 14 )
  {
    v103 = sub_9C2A90(a2, v103, 0);
    v56 = sub_9CAD80((_QWORD *)a2, v103);
    v12 = v56;
    if ( !v56 )
    {
      HIBYTE(v112) = 1;
      v6 = "Missing element type for old-style function";
      goto LABEL_4;
    }
    v13 = *(_BYTE *)(v56 + 8);
  }
  if ( v13 != 13 )
  {
    HIBYTE(v112) = 1;
    v6 = "Invalid type for value";
    goto LABEL_4;
  }
  v14 = v100[1];
  v15 = v14 & 0xFFFFFC00;
  if ( (v14 & 0xFFFFFC00) != 0 )
  {
    HIBYTE(v112) = 1;
    v6 = "Invalid calling convention ID";
    goto LABEL_4;
  }
  v16 = *(_QWORD *)(a2 + 440);
  v17 = *(_DWORD *)(v16 + 320);
  if ( v99 > 0x10 )
    v17 = *((_DWORD *)v100 + 32);
  v94 = v17;
  v110[0] = (__int64)v8;
  v96 = v16;
  v112 = 261;
  v110[1] = v5;
  v18 = sub_BD2DA0(136);
  v19 = v18;
  if ( v18 )
    sub_B2C3B0(v18, v12, 0, v94, v110, v96);
  v20 = *(_DWORD *)(a2 + 640);
  v105 = v19;
  v21 = a2 + 616;
  v101 = v103;
  if ( !v20 )
  {
    ++*(_QWORD *)(a2 + 616);
    goto LABEL_155;
  }
  v22 = v20 - 1;
  v23 = *(_QWORD *)(a2 + 624);
  v24 = 1;
  v25 = 0;
  v26 = v22 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
  v27 = (_QWORD *)(v23 + 16LL * v26);
  v28 = *v27;
  if ( v19 == *v27 )
  {
LABEL_18:
    v29 = (unsigned int *)(v27 + 1);
    goto LABEL_19;
  }
  while ( v28 != -4096 )
  {
    if ( v28 == -8192 && !v25 )
      v25 = v27;
    v26 = v22 & (v24 + v26);
    v27 = (_QWORD *)(v23 + 16LL * v26);
    v28 = *v27;
    if ( v19 == *v27 )
      goto LABEL_18;
    ++v24;
  }
  v78 = *(_DWORD *)(a2 + 632);
  if ( !v25 )
    v25 = v27;
  ++*(_QWORD *)(a2 + 616);
  v79 = v78 + 1;
  if ( 4 * (v78 + 1) >= 3 * v20 )
  {
LABEL_155:
    sub_9E07A0(a2 + 616, 2 * v20);
    v84 = *(_DWORD *)(a2 + 640);
    if ( v84 )
    {
      v85 = v105;
      v86 = v84 - 1;
      v87 = *(_QWORD *)(a2 + 624);
      v79 = *(_DWORD *)(a2 + 632) + 1;
      v88 = v86 & (((unsigned int)v105 >> 9) ^ ((unsigned int)v105 >> 4));
      v25 = (__int64 *)(v87 + 16LL * v88);
      v19 = *v25;
      if ( v105 == *v25 )
        goto LABEL_140;
      v22 = 1;
      v89 = 0;
      while ( v19 != -4096 )
      {
        if ( !v89 && v19 == -8192 )
          v89 = v25;
        v21 = (unsigned int)(v22 + 1);
        v22 = v88 + (unsigned int)v22;
        v88 = v86 & v22;
        v25 = (__int64 *)(v87 + 16LL * (v86 & (unsigned int)v22));
        v19 = *v25;
        if ( v105 == *v25 )
          goto LABEL_140;
        v22 = (unsigned int)v21;
      }
LABEL_159:
      v19 = v85;
      if ( v89 )
        v25 = v89;
      goto LABEL_140;
    }
LABEL_197:
    ++*(_DWORD *)(a2 + 632);
LABEL_198:
    BUG();
  }
  if ( v20 - *(_DWORD *)(a2 + 636) - v79 <= v20 >> 3 )
  {
    sub_9E07A0(a2 + 616, v20);
    v90 = *(_DWORD *)(a2 + 640);
    if ( v90 )
    {
      v85 = v105;
      v91 = v90 - 1;
      v92 = *(_QWORD *)(a2 + 624);
      v22 = 1;
      v79 = *(_DWORD *)(a2 + 632) + 1;
      v89 = 0;
      v93 = v91 & (((unsigned int)v105 >> 9) ^ ((unsigned int)v105 >> 4));
      v25 = (__int64 *)(v92 + 16LL * v93);
      v19 = *v25;
      if ( v105 == *v25 )
        goto LABEL_140;
      while ( v19 != -4096 )
      {
        if ( v19 == -8192 && !v89 )
          v89 = v25;
        v21 = (unsigned int)(v22 + 1);
        v22 = v93 + (unsigned int)v22;
        v93 = v91 & v22;
        v25 = (__int64 *)(v92 + 16LL * (v91 & (unsigned int)v22));
        v19 = *v25;
        if ( v105 == *v25 )
          goto LABEL_140;
        v22 = (unsigned int)v21;
      }
      goto LABEL_159;
    }
    goto LABEL_197;
  }
LABEL_140:
  *(_DWORD *)(a2 + 632) = v79;
  if ( *v25 != -4096 )
    --*(_DWORD *)(a2 + 636);
  *v25 = v19;
  v29 = (unsigned int *)(v25 + 1);
  *v29 = 0;
LABEL_19:
  *v29 = v101;
  *(_WORD *)(v105 + 2) = *(_WORD *)(v105 + 2) & 0xC00F | (16 * v14);
  v30 = v105;
  v97 = v100[3];
  v31 = (unsigned int)(v97 - 1);
  v95 = v100[2];
  if ( (unsigned int)v31 > 0x12 )
  {
    *(_BYTE *)(v105 + 32) &= 0xF0u;
    v32 = 1;
LABEL_21:
    if ( (*(_BYTE *)(v30 + 32) & 0x30) == 0 || !v32 )
      goto LABEL_23;
    goto LABEL_75;
  }
  v57 = dword_3F22240[v31];
  v58 = v57 & 0xF;
  if ( (unsigned int)(v57 - 7) > 1 )
  {
    *(_BYTE *)(v105 + 32) = v58 | *(_BYTE *)(v105 + 32) & 0xF0;
    v32 = v58 != 9;
    goto LABEL_21;
  }
  *(_WORD *)(v105 + 32) = *(_WORD *)(v105 + 32) & 0xFCC0 | v57 & 0xF;
LABEL_75:
  *(_BYTE *)(v30 + 33) |= 0x40u;
LABEL_23:
  v33 = *(_QWORD *)(a2 + 1480);
  v34 = 0;
  v35 = (unsigned int)v100[4] - 1;
  if ( v35 < (*(_QWORD *)(a2 + 1488) - v33) >> 3 )
    v34 = *(_QWORD *)(v33 + 8 * v35);
  *(_QWORD *)(v30 + 120) = v34;
  if ( *(_BYTE *)(a2 + 2000) )
  {
    v62 = *(_QWORD *)(a2 + 1984) == 0;
    v110[0] = a2;
    v111[1] = sub_9C2B30;
    v111[0] = sub_9C29A0;
    v109 = sub_9CB9F0;
    *((_QWORD *)&v108 + 1) = sub_9C29D0;
    v107 = a2;
    v106 = v30;
    v104 = v103;
    if ( v62 )
      sub_4263D6(v30, v35, v34);
    (*(void (__fastcall **)(__int64, __int64 *, unsigned int *, __int64 *, __int64 *, __int64))(a2 + 1992))(
      a2 + 1968,
      &v106,
      &v104,
      &v107,
      v110,
      v21);
    if ( *((_QWORD *)&v108 + 1) )
      (*((void (__fastcall **)(__int64 *, __int64 *, __int64))&v108 + 1))(&v107, &v107, 3);
    if ( v111[0] )
      ((void (__fastcall *)(__int64 *, __int64 *, __int64))v111[0])(v110, v110, 3);
    v30 = v105;
  }
  if ( *(_QWORD *)(v30 + 104) )
  {
    do
    {
      v36 = v15;
      v37 = 81;
      ++v15;
      v38 = (unsigned int *)&unk_3F222A0;
      if ( !(unsigned __int8)sub_B2D640(v30, v36, 81, v33, v22, v21) )
        goto LABEL_29;
LABEL_28:
      v110[0] = sub_B2D8D0(v105, v36, v37);
      if ( sub_A72A60(v110) )
        goto LABEL_29;
      sub_B2D580(v105, v36, v37);
      v53 = sub_9C2A90(a2, v103, v15);
      v54 = sub_9CAE40((_QWORD *)a2, v53);
      if ( !v54 )
      {
        HIBYTE(v112) = 1;
        v52 = "Missing param element type for attribute upgrade";
        goto LABEL_61;
      }
      switch ( v37 )
      {
        case 'S':
          v55 = sub_A77E50(*(_QWORD *)(a2 + 432), v54);
          break;
        case 'U':
          v55 = sub_A77E40(*(_QWORD *)(a2 + 432), v54);
          break;
        case 'Q':
          v55 = sub_A77E30(*(_QWORD *)(a2 + 432), v54);
          break;
        default:
          goto LABEL_198;
      }
      sub_B2D410(v105, v36, v55);
LABEL_29:
      while ( 1 )
      {
        ++v38;
        v30 = v105;
        if ( &unk_3F222AC == (_UNKNOWN *)v38 )
          break;
        v37 = *v38;
        if ( (unsigned __int8)sub_B2D640(v105, v36, *v38, v33, v22, v21) )
          goto LABEL_28;
      }
    }
    while ( v15 != *(_QWORD *)(v105 + 104) );
    if ( ((*(_WORD *)(v105 + 2) >> 4) & 0x3FF) == 0x53
      && v15
      && !(unsigned __int8)sub_B2D640(v105, 0, 81, v33, v22, v21) )
    {
      v59 = sub_9C2A90(a2, v103, 1u);
      v60 = sub_9CAE40((_QWORD *)a2, v59);
      if ( v60 )
      {
        v61 = sub_A77E30(*(_QWORD *)(a2 + 432), v60);
        sub_B2D410(v105, 0, v61);
        goto LABEL_37;
      }
      HIBYTE(v112) = 1;
      v52 = "Missing param element type for x86_intrcc upgrade";
      goto LABEL_61;
    }
  }
LABEL_37:
  LOWORD(v104) = 0;
  sub_9C88F0(v110, a2, v100[5], &v104, v22);
  if ( (v110[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v110[0] & 0xFFFFFFFFFFFFFFFELL | 1;
    return a1;
  }
  if ( BYTE1(v104) )
    sub_B2F770(v105, (unsigned __int8)v104);
  v39 = v100[6];
  if ( v39 )
  {
    v40 = *(_QWORD *)(a2 + 480);
    v41 = v39 - 1;
    if ( v41 >= (*(_QWORD *)(a2 + 488) - v40) >> 5 )
      goto LABEL_88;
    sub_B31A00(v105, *(_QWORD *)(v40 + 32 * v41), *(_QWORD *)(v40 + 32 * v41 + 8));
  }
  v42 = v105;
  v43 = (*(_BYTE *)(v105 + 32) & 0xF) - 7;
  v44 = *(_BYTE *)(v105 + 32) & 0xF;
  if ( v43 > 1 )
  {
    v66 = 1;
    v67 = v100[7];
    if ( (_DWORD)v67 != 1 )
      v66 = 2 * ((_DWORD)v67 == 2);
    v68 = (16 * v66) | *(_BYTE *)(v105 + 32) & 0xCF;
    *(_BYTE *)(v105 + 32) = v68;
    if ( (v68 & 0x30) != 0 && v44 != 9 )
      *(_BYTE *)(v42 + 33) |= 0x40u;
  }
  if ( v99 == 8 )
  {
    *(_BYTE *)(v42 + 32) &= 0x3Fu;
    v107 = v42;
    v108 = 0;
    goto LABEL_116;
  }
  v45 = v100[8];
  if ( v45 )
  {
    v46 = *(_QWORD *)(a2 + 504);
    v47 = v45 - 1;
    if ( v47 < (*(_QWORD *)(a2 + 512) - v46) >> 5 )
    {
      v110[0] = (__int64)v111;
      sub_9C36C0(v110, *(_BYTE **)(v46 + 32 * v47), *(_QWORD *)(v46 + 32 * v47) + *(_QWORD *)(v46 + 32 * v47 + 8));
      sub_B2EBE0(v42, v110);
      if ( (_QWORD *)v110[0] != v111 )
        j_j___libc_free_0(v110[0], v111[0] + 1LL);
      v42 = v105;
      v44 = *(_BYTE *)(v105 + 32) & 0xF;
      v43 = v44 - 7;
      goto LABEL_50;
    }
LABEL_88:
    HIBYTE(v112) = 1;
    v52 = "Invalid ID";
    goto LABEL_61;
  }
LABEL_50:
  if ( v99 == 9 )
  {
    v48 = 0;
  }
  else
  {
    v48 = 2;
    v49 = v100[9];
    if ( (_DWORD)v49 != 1 )
      v48 = (_DWORD)v49 == 2;
  }
  *(_BYTE *)(v42 + 32) = (v48 << 6) | *(_BYTE *)(v42 + 32) & 0x3F;
  v107 = v42;
  v108 = 0;
  if ( v99 > 0xA )
    DWORD2(v108) = v100[10];
  if ( v99 > 0xB )
  {
    if ( v43 > 1 )
    {
      v80 = v100[11];
      v81 = 1;
      if ( (_DWORD)v80 != 1 )
        v81 = 2 * ((_DWORD)v80 == 2);
      *(_BYTE *)(v42 + 33) = v81 | *(_BYTE *)(v42 + 33) & 0xFC;
    }
    if ( v99 != 12 )
    {
      v50 = v100[12];
      if ( (_DWORD)v50 )
      {
        v51 = *(_QWORD *)(a2 + 824);
        if ( (*(_QWORD *)(a2 + 832) - v51) >> 3 < (unsigned __int64)(unsigned int)v50 )
        {
          HIBYTE(v112) = 1;
          v52 = "Invalid function comdat ID";
LABEL_61:
          v110[0] = (__int64)v52;
          LOBYTE(v112) = 3;
          sub_9C81F0(a1, v98, (__int64)v110);
          return a1;
        }
        sub_B2F990(v42, *(_QWORD *)(v51 + 8LL * (unsigned int)(v50 - 1)));
        v42 = v105;
        v44 = *(_BYTE *)(v105 + 32) & 0xF;
        v43 = v44 - 7;
      }
      goto LABEL_90;
    }
    goto LABEL_170;
  }
LABEL_116:
  if ( v43 <= 1 )
  {
    if ( v97 > 0xB )
      goto LABEL_92;
    goto LABEL_118;
  }
  if ( (_DWORD)v97 == 5 )
  {
    *(_BYTE *)(v42 + 33) = *(_BYTE *)(v42 + 33) & 0xFC | 1;
    goto LABEL_90;
  }
  if ( (_DWORD)v97 == 6 )
  {
    *(_BYTE *)(v42 + 33) = *(_BYTE *)(v42 + 33) & 0xFC | 2;
    goto LABEL_90;
  }
LABEL_170:
  if ( v97 > 0xB )
  {
LABEL_90:
    if ( v99 > 0xD )
      DWORD1(v108) = v100[13];
    goto LABEL_92;
  }
LABEL_118:
  if ( ((1LL << v97) & 0xC12) == 0 )
    goto LABEL_90;
  v69 = *(_DWORD *)(a2 + 872);
  v106 = v42;
  if ( !v69 )
  {
    ++*(_QWORD *)(a2 + 848);
    v110[0] = 0;
    goto LABEL_191;
  }
  v70 = *(_QWORD *)(a2 + 856);
  LODWORD(v71) = (v69 - 1) & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
  v72 = (__int64 *)(v70 + 8LL * (unsigned int)v71);
  v73 = *v72;
  if ( *v72 != v42 )
  {
    v74 = 1;
    v75 = 0;
    while ( v73 != -4096 )
    {
      if ( v73 == -8192 && !v75 )
        v75 = v72;
      v71 = (v69 - 1) & ((_DWORD)v71 + v74);
      v72 = (__int64 *)(v70 + 8 * v71);
      v73 = *v72;
      if ( *v72 == v42 )
        goto LABEL_92;
      ++v74;
    }
    v76 = *(_DWORD *)(a2 + 864);
    if ( v75 )
      v72 = v75;
    ++*(_QWORD *)(a2 + 848);
    v77 = v76 + 1;
    v110[0] = (__int64)v72;
    if ( 4 * v77 < 3 * v69 )
    {
      if ( v69 - *(_DWORD *)(a2 + 868) - v77 > v69 >> 3 )
      {
LABEL_127:
        *(_DWORD *)(a2 + 864) = v77;
        if ( *v72 != -4096 )
          --*(_DWORD *)(a2 + 868);
        *v72 = v42;
        v42 = v105;
        v44 = *(_BYTE *)(v105 + 32) & 0xF;
        v43 = v44 - 7;
        goto LABEL_92;
      }
      sub_9E6990(a2 + 848, v69);
LABEL_189:
      sub_9D28C0(a2 + 848, &v106, v110);
      v42 = v106;
      v72 = (__int64 *)v110[0];
      v77 = *(_DWORD *)(a2 + 864) + 1;
      goto LABEL_127;
    }
LABEL_191:
    sub_9E6990(a2 + 848, 2 * v69);
    goto LABEL_189;
  }
LABEL_92:
  if ( v99 > 0xE )
    LODWORD(v108) = v100[14];
  if ( v99 > 0xF )
    *(_BYTE *)(v42 + 33) = ((*((_DWORD *)v100 + 30) == 1) << 6) | *(_BYTE *)(v42 + 33) & 0xBF;
  if ( v43 <= 1 || (*(_BYTE *)(v42 + 32) & 0x30) != 0 && v44 != 9 )
    *(_BYTE *)(v42 + 33) |= 0x40u;
  if ( v99 > 0x12 )
  {
    v63 = *(_QWORD *)(a2 + 376);
    if ( v63 )
    {
      v64 = v100[17];
      if ( (unsigned __int64)(v64 + v100[18]) <= *(_QWORD *)(a2 + 384) )
      {
        sub_B30D10(v42, v63 + v64);
        v42 = v105;
      }
    }
  }
  LODWORD(v106) = sub_9E2F80(a2, *(_QWORD *)(v42 + 8), (int *)&v103, 1);
  v110[0] = v105;
  sub_9C9EA0((__int64 *)(a2 + 744), v110, &v106);
  if ( (_QWORD)v108 || DWORD2(v108) )
  {
    v65 = *(_QWORD *)(a2 + 1464);
    if ( v65 == *(_QWORD *)(a2 + 1472) )
    {
      sub_9C2BD0(a2 + 1456, (_BYTE *)v65, (const __m128i *)&v107);
    }
    else
    {
      if ( v65 )
      {
        *(__m128i *)v65 = _mm_loadu_si128((const __m128i *)&v107);
        *(_QWORD *)(v65 + 16) = *((_QWORD *)&v108 + 1);
        v65 = *(_QWORD *)(a2 + 1464);
      }
      *(_QWORD *)(a2 + 1464) = v65 + 24;
    }
  }
  if ( !v95 )
  {
    v82 = v105;
    *(_WORD *)(v105 + 34) |= 0x800u;
    v83 = *(_BYTE **)(a2 + 1584);
    if ( v83 == *(_BYTE **)(a2 + 1592) )
    {
      sub_9CC5C0(a2 + 1576, v83, &v105);
    }
    else
    {
      if ( v83 )
      {
        *(_QWORD *)v83 = v82;
        v83 = *(_BYTE **)(a2 + 1584);
      }
      *(_QWORD *)(a2 + 1584) = v83 + 8;
    }
    *sub_9DDC30(a2 + 1640, &v105) = 0;
  }
  *a1 = 1;
  return a1;
}
