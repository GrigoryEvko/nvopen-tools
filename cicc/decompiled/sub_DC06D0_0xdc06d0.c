// Function: sub_DC06D0
// Address: 0xdc06d0
//
__int64 __fastcall sub_DC06D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned int v7; // eax
  __int64 v8; // rdx
  __int64 *v9; // rcx
  __int64 v10; // r8
  unsigned int v11; // ebx
  __int16 v12; // ax
  __int64 *v13; // rax
  __int16 v14; // ax
  __int16 v15; // ax
  char v16; // di
  __int64 *v17; // rax
  __int64 *v18; // rdx
  __int64 v19; // rsi
  __int64 v20; // rsi
  __int64 *v21; // rsi
  __int64 v22; // rbx
  __int64 v23; // r14
  __int64 v25; // rbx
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  _QWORD *v29; // rdx
  __int64 v30; // rcx
  __int64 *v31; // rdx
  unsigned int v32; // eax
  const void *v33; // rcx
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 *v36; // r14
  __int64 *v37; // rbx
  __int64 v38; // rax
  int v39; // esi
  unsigned int v40; // ecx
  _QWORD *v41; // rdx
  __int64 v42; // r9
  unsigned int v43; // esi
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 *v46; // r14
  __int64 *v47; // rbx
  __int64 v48; // rax
  __int64 *v49; // r8
  int v50; // esi
  unsigned int v51; // ecx
  __int64 *v52; // rdx
  __int64 v53; // r9
  unsigned int v54; // esi
  __int64 *v55; // rax
  __int64 v56; // rsi
  __int64 *v57; // r8
  int v58; // esi
  unsigned int v59; // edx
  __int64 *v60; // rax
  __int64 v61; // r11
  int v62; // esi
  unsigned int v63; // edx
  _QWORD *v64; // rax
  __int64 v65; // r11
  unsigned int v66; // edx
  unsigned int v67; // ecx
  unsigned int v68; // r8d
  __int64 *v69; // rdx
  unsigned int v70; // edx
  unsigned int v71; // ecx
  __int64 *v72; // rdx
  unsigned int v73; // esi
  unsigned int v74; // esi
  _QWORD *v75; // rax
  __int64 v76; // rsi
  __int64 v77; // rbx
  unsigned int v78; // edx
  const void *v79; // r14
  bool v80; // al
  int v81; // r11d
  __int64 *v82; // r10
  int v83; // r11d
  _QWORD *v84; // r10
  unsigned int v85; // eax
  unsigned int v86; // edx
  unsigned int v87; // edi
  unsigned int v88; // ecx
  __int64 v89; // rdx
  __int64 *v90; // rax
  unsigned int v91; // eax
  unsigned int v92; // edx
  unsigned int v93; // edi
  unsigned int v94; // ecx
  __int64 v95; // rdx
  __int64 *v96; // rax
  unsigned int v97; // eax
  __int64 v98; // rax
  int v99; // r9d
  _QWORD *v100; // rdi
  int v101; // r9d
  __int64 *v102; // rdi
  unsigned int v103; // eax
  __int64 v104; // rax
  unsigned int v106; // [rsp+8h] [rbp-138h]
  int v107; // [rsp+Ch] [rbp-134h]
  __int64 v109; // [rsp+20h] [rbp-120h] BYREF
  unsigned int v110; // [rsp+28h] [rbp-118h]
  __int64 v111; // [rsp+30h] [rbp-110h] BYREF
  unsigned int v112; // [rsp+38h] [rbp-108h]
  __int64 v113; // [rsp+40h] [rbp-100h] BYREF
  const void *v114; // [rsp+48h] [rbp-F8h] BYREF
  unsigned int v115; // [rsp+50h] [rbp-F0h]
  __int64 *v116; // [rsp+60h] [rbp-E0h] BYREF
  const void *v117; // [rsp+68h] [rbp-D8h] BYREF
  unsigned int v118; // [rsp+70h] [rbp-D0h]
  char v119; // [rsp+78h] [rbp-C8h]
  __int64 v120; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v121; // [rsp+88h] [rbp-B8h] BYREF
  __int64 *v122; // [rsp+90h] [rbp-B0h] BYREF
  unsigned int v123; // [rsp+98h] [rbp-A8h]
  _BYTE v124[48]; // [rsp+110h] [rbp-30h] BYREF

  v5 = a3;
  v6 = sub_D95540(a3);
  v7 = sub_D97050(a2, v6);
  v110 = v7;
  v11 = v7;
  if ( v7 > 0x40 )
  {
    sub_C43690((__int64)&v109, 0, 0);
    v112 = v11;
    sub_C43690((__int64)&v111, 1, 0);
  }
  else
  {
    v109 = 0;
    v112 = v7;
    v111 = 1;
  }
  v107 = 8;
  while ( 1 )
  {
    if ( a4 == v5 )
    {
      v97 = v110;
      v110 = 0;
      *(_DWORD *)(a1 + 8) = v97;
      v98 = v109;
      *(_BYTE *)(a1 + 16) = 1;
      *(_QWORD *)a1 = v98;
      goto LABEL_52;
    }
    v12 = *(_WORD *)(v5 + 24);
    if ( *(_WORD *)(a4 + 24) != 8 || v12 != 8 )
      break;
    if ( *(_QWORD *)(a4 + 48) != *(_QWORD *)(v5 + 48)
      || *(_QWORD *)(a4 + 40) != 2
      || *(_QWORD *)(v5 + 40) != 2
      || (v25 = sub_D33D80((_QWORD *)a4, a2, v8, (__int64)v9, v10), v25 != sub_D33D80((_QWORD *)v5, a2, v26, v27, v28)) )
    {
LABEL_51:
      *(_BYTE *)(a1 + 16) = 0;
      goto LABEL_52;
    }
    a4 = **(_QWORD **)(a4 + 32);
    v5 = **(_QWORD **)(v5 + 32);
LABEL_50:
    if ( !--v107 )
      goto LABEL_51;
  }
  if ( v12 != 6 )
    goto LABEL_9;
  if ( *(_QWORD *)(v5 + 40) != 2 )
    goto LABEL_9;
  v29 = *(_QWORD **)(v5 + 32);
  if ( *(_WORD *)(*v29 + 24LL) )
    goto LABEL_9;
  v30 = *(_QWORD *)(*v29 + 32LL);
  v31 = (__int64 *)v29[1];
  v120 = (__int64)v31;
  v32 = *(_DWORD *)(v30 + 32);
  LODWORD(v122) = v32;
  if ( v32 > 0x40 )
  {
    sub_C43780((__int64)&v121, (const void **)(v30 + 24));
    v31 = (__int64 *)v120;
    v32 = (unsigned int)v122;
    v33 = (const void *)v121;
  }
  else
  {
    v33 = *(const void **)(v30 + 24);
  }
  v116 = v31;
  v118 = v32;
  v117 = v33;
  v119 = 1;
  if ( *(_WORD *)(a4 + 24) != 6 )
    goto LABEL_69;
  if ( *(_QWORD *)(a4 + 40) != 2 )
    goto LABEL_69;
  v75 = *(_QWORD **)(a4 + 32);
  if ( *(_WORD *)(*v75 + 24LL) )
    goto LABEL_69;
  v76 = *(_QWORD *)(*v75 + 32LL) + 24LL;
  v113 = v75[1];
  sub_9865C0((__int64)&v114, v76);
  v77 = v113;
  v78 = v115;
  LOBYTE(v123) = 1;
  v79 = v114;
  v120 = v113;
  LODWORD(v122) = v115;
  v121 = (__int64)v114;
  if ( v118 > 0x40 )
  {
    v106 = v115;
    v80 = sub_C43C50((__int64)&v117, (const void **)&v121);
    v78 = v106;
    if ( !v80 )
      goto LABEL_140;
    goto LABEL_146;
  }
  if ( v114 == v117 )
  {
LABEL_146:
    v5 = (__int64)v116;
    sub_C47360((__int64)&v111, (__int64 *)&v117);
    if ( (_BYTE)v123 )
    {
      LOBYTE(v123) = 0;
      sub_969240(&v121);
    }
    if ( v119 )
    {
      v119 = 0;
      sub_969240((__int64 *)&v117);
    }
    a4 = v77;
    goto LABEL_50;
  }
LABEL_140:
  LOBYTE(v123) = 0;
  if ( v78 > 0x40 && v79 )
    j_j___libc_free_0_0(v79);
  if ( v119 )
  {
LABEL_69:
    v119 = 0;
    if ( v118 > 0x40 && v117 )
      j_j___libc_free_0_0(v117);
  }
LABEL_9:
  v120 = 0;
  v13 = (__int64 *)&v122;
  v121 = 1;
  do
  {
    *v13 = -4096;
    v13 += 2;
  }
  while ( v13 != (__int64 *)v124 );
  v14 = *(_WORD *)(v5 + 24);
  if ( v14 == 5 )
  {
    v44 = sub_D960E0(v5);
    v46 = (__int64 *)(v44 + 8 * v45);
    v47 = (__int64 *)v44;
    if ( (__int64 *)v44 == v46 )
      goto LABEL_16;
    while ( 1 )
    {
      v48 = *v47;
      v113 = v48;
      if ( !*(_WORD *)(v48 + 24) )
      {
        sub_C472A0((__int64)&v116, *(_QWORD *)(v48 + 32) + 24LL, &v111);
        sub_C45EE0((__int64)&v109, (__int64 *)&v116);
        if ( (unsigned int)v117 > 0x40 && v116 )
          j_j___libc_free_0_0(v116);
        goto LABEL_98;
      }
      if ( (v121 & 1) != 0 )
      {
        v49 = (__int64 *)&v122;
        v50 = 7;
      }
      else
      {
        v54 = v123;
        v49 = v122;
        if ( !v123 )
        {
          v66 = v121;
          ++v120;
          v116 = 0;
          v67 = ((unsigned int)v121 >> 1) + 1;
          goto LABEL_120;
        }
        v50 = v123 - 1;
      }
      v51 = v50 & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
      v52 = &v49[2 * v51];
      v53 = *v52;
      if ( *v52 != v48 )
        break;
LABEL_103:
      ++*((_DWORD *)v52 + 2);
LABEL_98:
      if ( v46 == ++v47 )
        goto LABEL_16;
    }
    v81 = 1;
    v82 = 0;
    while ( v53 != -4096 )
    {
      if ( v53 == -8192 && !v82 )
        v82 = v52;
      v51 = v50 & (v81 + v51);
      v52 = &v49[2 * v51];
      v53 = *v52;
      if ( v48 == *v52 )
        goto LABEL_103;
      ++v81;
    }
    v68 = 24;
    v54 = 8;
    if ( v82 )
      v52 = v82;
    ++v120;
    v116 = v52;
    v66 = v121;
    v67 = ((unsigned int)v121 >> 1) + 1;
    if ( (v121 & 1) == 0 )
    {
      v54 = v123;
LABEL_120:
      v68 = 3 * v54;
    }
    if ( v68 <= 4 * v67 )
    {
      v54 *= 2;
    }
    else if ( v54 - HIDWORD(v121) - v67 > v54 >> 3 )
    {
LABEL_123:
      LODWORD(v121) = (2 * (v66 >> 1) + 2) | v66 & 1;
      v69 = v116;
      if ( *v116 != -4096 )
        --HIDWORD(v121);
      *v116 = v48;
      *((_DWORD *)v69 + 2) = 0;
      *((_DWORD *)v69 + 2) = 1;
      goto LABEL_98;
    }
    sub_DACF10((__int64)&v120, v54);
    sub_D9F560((__int64)&v120, &v113, &v116);
    v48 = v113;
    v66 = v121;
    goto LABEL_123;
  }
  v113 = v5;
  if ( !v14 )
  {
    sub_C472A0((__int64)&v116, *(_QWORD *)(v5 + 32) + 24LL, &v111);
    sub_C45EE0((__int64)&v109, (__int64 *)&v116);
    if ( (unsigned int)v117 > 0x40 && v116 )
      j_j___libc_free_0_0(v116);
    goto LABEL_16;
  }
  if ( (v121 & 1) != 0 )
  {
    v57 = (__int64 *)&v122;
    v58 = 7;
    goto LABEL_112;
  }
  v74 = v123;
  v57 = v122;
  if ( !v123 )
  {
    v85 = v121;
    ++v120;
    v116 = 0;
    v86 = ((unsigned int)v121 >> 1) + 1;
    goto LABEL_168;
  }
  v58 = v123 - 1;
LABEL_112:
  v59 = v58 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v60 = &v57[2 * v59];
  v61 = *v60;
  if ( v5 == *v60 )
  {
LABEL_113:
    ++*((_DWORD *)v60 + 2);
    goto LABEL_16;
  }
  v101 = 1;
  v102 = 0;
  while ( v61 != -4096 )
  {
    if ( !v102 && v61 == -8192 )
      v102 = v60;
    v59 = v58 & (v101 + v59);
    v60 = &v57[2 * v59];
    v61 = *v60;
    if ( v5 == *v60 )
      goto LABEL_113;
    ++v101;
  }
  v74 = 8;
  if ( v102 )
    v60 = v102;
  ++v120;
  v87 = 24;
  v116 = v60;
  v85 = v121;
  v86 = ((unsigned int)v121 >> 1) + 1;
  if ( (v121 & 1) == 0 )
  {
    v74 = v123;
LABEL_168:
    v87 = 3 * v74;
  }
  if ( v87 <= 4 * v86 )
  {
    v74 *= 2;
    goto LABEL_191;
  }
  v88 = v74 - HIDWORD(v121) - v86;
  v89 = v5;
  if ( v88 <= v74 >> 3 )
  {
LABEL_191:
    sub_DACF10((__int64)&v120, v74);
    sub_D9F560((__int64)&v120, &v113, &v116);
    v89 = v113;
    v85 = v121;
  }
  LODWORD(v121) = (2 * (v85 >> 1) + 2) | v85 & 1;
  v90 = v116;
  if ( *v116 != -4096 )
    --HIDWORD(v121);
  *v116 = v89;
  *((_DWORD *)v90 + 2) = 0;
  *((_DWORD *)v90 + 2) = 1;
LABEL_16:
  v15 = *(_WORD *)(a4 + 24);
  if ( v15 == 5 )
  {
    v34 = sub_D960E0(a4);
    v36 = (__int64 *)(v34 + 8 * v35);
    v37 = (__int64 *)v34;
    if ( (__int64 *)v34 == v36 )
      goto LABEL_21;
    while ( 1 )
    {
      v38 = *v37;
      v113 = v38;
      if ( !*(_WORD *)(v38 + 24) )
      {
        sub_C472A0((__int64)&v116, *(_QWORD *)(v38 + 32) + 24LL, &v111);
        sub_C46B40((__int64)&v109, (__int64 *)&v116);
        if ( (unsigned int)v117 > 0x40 && v116 )
          j_j___libc_free_0_0(v116);
        goto LABEL_85;
      }
      if ( (v121 & 1) != 0 )
      {
        v10 = (__int64)&v122;
        v39 = 7;
      }
      else
      {
        v43 = v123;
        v10 = (__int64)v122;
        if ( !v123 )
        {
          v70 = v121;
          ++v120;
          v116 = 0;
          v71 = ((unsigned int)v121 >> 1) + 1;
          goto LABEL_127;
        }
        v39 = v123 - 1;
      }
      v40 = v39 & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
      v41 = (_QWORD *)(v10 + 16LL * v40);
      v42 = *v41;
      if ( *v41 != v38 )
        break;
LABEL_90:
      --*((_DWORD *)v41 + 2);
LABEL_85:
      if ( v36 == ++v37 )
        goto LABEL_21;
    }
    v83 = 1;
    v84 = 0;
    while ( v42 != -4096 )
    {
      if ( !v84 && v42 == -8192 )
        v84 = v41;
      v40 = v39 & (v83 + v40);
      v41 = (_QWORD *)(v10 + 16LL * v40);
      v42 = *v41;
      if ( v38 == *v41 )
        goto LABEL_90;
      ++v83;
    }
    v10 = 24;
    v43 = 8;
    if ( v84 )
      v41 = v84;
    ++v120;
    v116 = v41;
    v70 = v121;
    v71 = ((unsigned int)v121 >> 1) + 1;
    if ( (v121 & 1) == 0 )
    {
      v43 = v123;
LABEL_127:
      v10 = 3 * v43;
    }
    if ( (unsigned int)v10 <= 4 * v71 )
    {
      v43 *= 2;
    }
    else if ( v43 - HIDWORD(v121) - v71 > v43 >> 3 )
    {
LABEL_130:
      LODWORD(v121) = (2 * (v70 >> 1) + 2) | v70 & 1;
      v72 = v116;
      if ( *v116 != -4096 )
        --HIDWORD(v121);
      *v116 = v38;
      *((_DWORD *)v72 + 2) = 0;
      *((_DWORD *)v72 + 2) = -1;
      goto LABEL_85;
    }
    sub_DACF10((__int64)&v120, v43);
    sub_D9F560((__int64)&v120, &v113, &v116);
    v38 = v113;
    v70 = v121;
    goto LABEL_130;
  }
  v113 = a4;
  if ( !v15 )
  {
    sub_C472A0((__int64)&v116, *(_QWORD *)(a4 + 32) + 24LL, &v111);
    sub_C46B40((__int64)&v109, (__int64 *)&v116);
    if ( (unsigned int)v117 > 0x40 && v116 )
      j_j___libc_free_0_0(v116);
    goto LABEL_21;
  }
  if ( (v121 & 1) != 0 )
  {
    v10 = (__int64)&v122;
    v62 = 7;
    goto LABEL_117;
  }
  v73 = v123;
  v10 = (__int64)v122;
  if ( !v123 )
  {
    v91 = v121;
    ++v120;
    v116 = 0;
    v92 = ((unsigned int)v121 >> 1) + 1;
    goto LABEL_175;
  }
  v62 = v123 - 1;
LABEL_117:
  v63 = v62 & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
  v64 = (_QWORD *)(v10 + 16LL * v63);
  v65 = *v64;
  if ( a4 == *v64 )
  {
LABEL_118:
    --*((_DWORD *)v64 + 2);
    goto LABEL_21;
  }
  v99 = 1;
  v100 = 0;
  while ( v65 != -4096 )
  {
    if ( !v100 && v65 == -8192 )
      v100 = v64;
    v63 = v62 & (v99 + v63);
    v64 = (_QWORD *)(v10 + 16LL * v63);
    v65 = *v64;
    if ( a4 == *v64 )
      goto LABEL_118;
    ++v99;
  }
  v73 = 8;
  if ( v100 )
    v64 = v100;
  ++v120;
  v93 = 24;
  v116 = v64;
  v91 = v121;
  v92 = ((unsigned int)v121 >> 1) + 1;
  if ( (v121 & 1) == 0 )
  {
    v73 = v123;
LABEL_175:
    v93 = 3 * v73;
  }
  if ( 4 * v92 >= v93 )
  {
    v73 *= 2;
    goto LABEL_183;
  }
  v94 = v73 - HIDWORD(v121) - v92;
  v95 = a4;
  if ( v94 <= v73 >> 3 )
  {
LABEL_183:
    sub_DACF10((__int64)&v120, v73);
    sub_D9F560((__int64)&v120, &v113, &v116);
    v95 = v113;
    v91 = v121;
  }
  LODWORD(v121) = (2 * (v91 >> 1) + 2) | v91 & 1;
  v96 = v116;
  if ( *v116 != -4096 )
    --HIDWORD(v121);
  *v116 = v95;
  *((_DWORD *)v96 + 2) = 0;
  *((_DWORD *)v96 + 2) = -1;
LABEL_21:
  v16 = v121 & 1;
  if ( !((unsigned int)v121 >> 1) )
  {
    if ( v16 )
    {
      v55 = (__int64 *)&v122;
      v56 = 16;
    }
    else
    {
      v55 = v122;
      v56 = 2LL * v123;
    }
    v17 = &v55[v56];
    v9 = v17;
    goto LABEL_27;
  }
  if ( v16 )
  {
    v9 = (__int64 *)v124;
    v17 = (__int64 *)&v122;
    do
    {
LABEL_24:
      if ( *v17 != -8192 && *v17 != -4096 )
        break;
      v17 += 2;
    }
    while ( v17 != v9 );
LABEL_27:
    if ( !v16 )
    {
      v18 = v122;
      v19 = v123;
      goto LABEL_29;
    }
    v18 = (__int64 *)&v122;
    v20 = 16;
  }
  else
  {
    v19 = v123;
    v18 = v122;
    v17 = v122;
    v9 = &v122[2 * v123];
    if ( v122 != v9 )
      goto LABEL_24;
LABEL_29:
    v20 = 2 * v19;
  }
  v21 = &v18[v20];
  if ( v17 != v21 )
  {
    v22 = 0;
    v23 = 0;
    do
    {
      v8 = *((unsigned int *)v17 + 2);
      if ( (_DWORD)v8 )
      {
        if ( (_DWORD)v8 == 1 )
        {
          if ( v23 )
            goto LABEL_74;
          v23 = *v17;
        }
        else
        {
          if ( (_DWORD)v8 != -1 || v22 )
            goto LABEL_74;
          v22 = *v17;
        }
      }
      for ( v17 += 2; v9 != v17; v17 += 2 )
      {
        v8 = *v17;
        if ( *v17 != -8192 && v8 != -4096 )
          break;
      }
    }
    while ( v17 != v21 );
    if ( v22 == a4 || v23 == v5 )
      goto LABEL_74;
    LOBYTE(v8) = v22 == 0;
    if ( !(v22 | v23) )
      goto LABEL_199;
    if ( !v22 || !v23 )
    {
LABEL_74:
      *(_BYTE *)(a1 + 16) = 0;
      goto LABEL_75;
    }
    if ( !v16 )
      sub_C7D6A0((__int64)v122, 16LL * v123, 8);
    a4 = v22;
    v5 = v23;
    goto LABEL_50;
  }
LABEL_199:
  v103 = v110;
  v110 = 0;
  *(_DWORD *)(a1 + 8) = v103;
  v104 = v109;
  *(_BYTE *)(a1 + 16) = 1;
  *(_QWORD *)a1 = v104;
LABEL_75:
  if ( !v16 )
    sub_C7D6A0((__int64)v122, 16LL * v123, 8);
LABEL_52:
  if ( v112 > 0x40 && v111 )
    j_j___libc_free_0_0(v111);
  if ( v110 > 0x40 && v109 )
    j_j___libc_free_0_0(v109);
  return a1;
}
