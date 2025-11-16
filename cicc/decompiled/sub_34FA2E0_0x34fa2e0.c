// Function: sub_34FA2E0
// Address: 0x34fa2e0
//
__int64 __fastcall sub_34FA2E0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  unsigned int v5; // r15d
  unsigned __int16 v8; // dx
  __int64 *v9; // rdi
  char v11; // r15
  __int64 v12; // rax
  __int64 (__fastcall *v13)(__int64); // rcx
  __int64 (*v14)(); // rax
  int v15; // r10d
  bool v16; // r11
  __int64 *v17; // rbx
  __int64 v18; // r8
  bool v19; // r15
  __int64 *v20; // r13
  __int64 v21; // r9
  __int64 v22; // rax
  unsigned __int64 v23; // rdx
  unsigned int v24; // r12d
  __int64 v25; // rdx
  unsigned __int8 v26; // cl
  char *v27; // rdi
  __int64 v28; // r12
  char v29; // r15
  __int64 v30; // r13
  __int64 *v31; // rdi
  __int64 v32; // r12
  unsigned __int64 v33; // rbx
  __int64 v34; // r9
  __int64 v35; // r10
  __int64 v36; // r8
  __int64 v37; // r14
  unsigned __int64 v38; // r12
  __int64 v39; // r15
  _QWORD *v40; // r8
  __int64 v41; // r13
  signed int v42; // ecx
  unsigned __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // r13
  __int64 v46; // rax
  char v47; // al
  __int64 v48; // rdi
  __int64 v49; // r12
  _QWORD *v50; // r13
  __int64 (*v51)(); // rax
  __int64 v52; // rcx
  __int64 v53; // rdx
  __int64 v54; // rdi
  unsigned int v55; // esi
  __int64 *v56; // rax
  __int64 v57; // r8
  __int64 v58; // rdx
  unsigned int v59; // ecx
  __int64 v60; // rdx
  __int64 v61; // rcx
  _BYTE *v62; // rax
  unsigned __int64 v63; // r14
  __int64 i; // rbx
  unsigned int v65; // ebx
  __int64 v66; // r14
  __int64 v67; // rax
  unsigned int *v68; // rdi
  __int64 v69; // r13
  unsigned int *v70; // r14
  unsigned int *v71; // rdx
  int v72; // eax
  __int64 v73; // r9
  __int64 v74; // rbx
  unsigned int v75; // r15d
  __int64 v76; // r12
  __int64 v77; // rcx
  __int64 v78; // r9
  __int64 v79; // rdx
  unsigned int *v80; // rdx
  unsigned int *v81; // r12
  unsigned int *v82; // rbx
  unsigned int v83; // edx
  unsigned int v84; // esi
  __int64 v85; // rax
  __int64 v86; // rax
  _BYTE *v87; // rax
  __int64 v88; // rax
  __int64 v89; // rdx
  unsigned __int64 v90; // r8
  unsigned __int64 v91; // rbx
  int v92; // eax
  int v93; // r10d
  __int64 v94; // rax
  _QWORD *v95; // [rsp+0h] [rbp-160h]
  __int64 v96; // [rsp+0h] [rbp-160h]
  __int64 v97; // [rsp+8h] [rbp-158h]
  unsigned int v98; // [rsp+8h] [rbp-158h]
  __int64 v99; // [rsp+8h] [rbp-158h]
  int v100; // [rsp+10h] [rbp-150h]
  _QWORD *v101; // [rsp+10h] [rbp-150h]
  __int64 v102; // [rsp+10h] [rbp-150h]
  char v103; // [rsp+10h] [rbp-150h]
  unsigned int v104; // [rsp+18h] [rbp-148h]
  int v105; // [rsp+20h] [rbp-140h]
  __int64 v106; // [rsp+28h] [rbp-138h]
  unsigned __int8 v107; // [rsp+28h] [rbp-138h]
  char *v108; // [rsp+28h] [rbp-138h]
  char v109; // [rsp+30h] [rbp-130h]
  int v110; // [rsp+30h] [rbp-130h]
  unsigned __int64 v111; // [rsp+30h] [rbp-130h]
  __int64 v112; // [rsp+30h] [rbp-130h]
  __int64 v113; // [rsp+30h] [rbp-130h]
  int v114; // [rsp+38h] [rbp-128h]
  __int64 *v115; // [rsp+40h] [rbp-120h] BYREF
  __int64 v116; // [rsp+48h] [rbp-118h]
  __int32 v117; // [rsp+5Ch] [rbp-104h] BYREF
  __int64 v118; // [rsp+60h] [rbp-100h] BYREF
  __int64 v119; // [rsp+68h] [rbp-F8h] BYREF
  __int64 v120; // [rsp+70h] [rbp-F0h] BYREF
  _BYTE *v121; // [rsp+80h] [rbp-E0h]
  __int64 v122; // [rsp+88h] [rbp-D8h]
  _QWORD v123[6]; // [rsp+90h] [rbp-D0h] BYREF
  char *v124; // [rsp+C0h] [rbp-A0h] BYREF
  __int64 v125; // [rsp+C8h] [rbp-98h]
  _BYTE v126[32]; // [rsp+D0h] [rbp-90h] BYREF
  unsigned int *v127; // [rsp+F0h] [rbp-70h] BYREF
  __int64 v128; // [rsp+F8h] [rbp-68h]
  _BYTE v129[96]; // [rsp+100h] [rbp-60h] BYREF

  v115 = a2;
  v116 = a3;
  if ( !a3 )
    return 0;
  v4 = *a2;
  v5 = 0;
  if ( a2[2 * a3 - 2] != *a2 || (*(_BYTE *)(v4 + 44) & 0xC) != 0 )
    return v5;
  v8 = *(_WORD *)(v4 + 68);
  v9 = *(__int64 **)(a1 + 48);
  v11 = 1;
  v12 = *v9;
  if ( v8 != 20 )
  {
    v13 = *(__int64 (__fastcall **)(__int64))(v12 + 520);
    v11 = 0;
    if ( v13 != sub_2DCA430 )
    {
      ((void (__fastcall *)(unsigned int **, __int64 *, __int64))v13)(&v127, v9, *a2);
      v8 = *(_WORD *)(v4 + 68);
      v11 = v129[0];
      v12 = **(_QWORD **)(a1 + 48);
    }
  }
  v14 = *(__int64 (**)())(v12 + 584);
  v15 = v8;
  if ( v14 == sub_2FDC590 )
    goto LABEL_9;
  v114 = v8;
  v47 = v14();
  v15 = v114;
  v16 = v47;
  if ( !v47 )
  {
    v8 = *(_WORD *)(v4 + 68);
LABEL_9:
    v16 = v8 == 32 || ((v8 - 26) & 0xFFFD) == 0;
  }
  v17 = v115;
  v124 = v126;
  v125 = 0x800000000LL;
  if ( v115 == &v115[2 * v116] )
    return 0;
  v109 = v11;
  v18 = a4;
  v19 = v16;
  v106 = a1;
  v20 = &v115[2 * v116];
  v21 = 0;
  do
  {
    v24 = *((_DWORD *)v17 + 2);
    v25 = *(_QWORD *)(v4 + 32) + 40LL * v24;
    v26 = *(_BYTE *)(v25 + 3) & 0x10;
    if ( v26 || (*(_BYTE *)(v25 + 4) & 1) == 0 && (*(_BYTE *)(v25 + 4) & 2) == 0 || (*(_WORD *)(v25 + 2) & 0xFF0) != 0 )
    {
      if ( (*(_BYTE *)(v25 + 3) & 0x20) != 0 )
      {
        v21 = *(unsigned int *)(v25 + 8);
      }
      else
      {
        if ( !v19 && (*(_DWORD *)v25 & 0xFFF00) != 0 || v26 && v18 )
          goto LABEL_31;
        if ( v15 == 32 || *(_BYTE *)v25 | v26 || (*(_WORD *)(v25 + 2) & 0xFF0) == 0 )
        {
          v22 = (unsigned int)v125;
          v23 = (unsigned int)v125 + 1LL;
          if ( v23 > HIDWORD(v125) )
          {
            v97 = v18;
            v100 = v15;
            v104 = v21;
            sub_C8D5F0((__int64)&v124, v126, v23, 4u, v18, v21);
            v22 = (unsigned int)v125;
            v18 = v97;
            v15 = v100;
            v21 = v104;
          }
          *(_DWORD *)&v124[4 * v22] = v24;
          LODWORD(v125) = v125 + 1;
        }
      }
    }
    v17 += 2;
  }
  while ( v17 != v20 );
  v105 = v21;
  v28 = v18;
  v29 = v109;
  v30 = v106;
  if ( !(_DWORD)v125 )
  {
LABEL_31:
    v27 = v124;
    v5 = 0;
    goto LABEL_32;
  }
  v110 = v15;
  sub_34F5E30(&v120, (__int64 *)v4, *(_QWORD *)(v4 + 24));
  v127 = (unsigned int *)v129;
  v128 = 0x600000000LL;
  if ( v110 != 32 )
    goto LABEL_36;
  v108 = &v124[4 * (unsigned int)v125];
  if ( v124 == v108 )
    goto LABEL_36;
  v99 = v30;
  v69 = v4;
  v70 = (unsigned int *)v124;
  v103 = v29;
  v96 = v28;
  do
  {
    v74 = *v70;
    v75 = *v70;
    v76 = 40 * v74;
    v112 = 40 * v74 + *(_QWORD *)(v69 + 32);
    if ( (*(_WORD *)(v112 + 2) & 0xFF0) == 0 )
      goto LABEL_114;
    v77 = (unsigned int)sub_2E89F40(v69, v74);
    v79 = (unsigned int)v128;
    v72 = v128;
    if ( (*(_BYTE *)(v112 + 3) & 0x10) == 0 )
    {
      if ( (unsigned int)v128 < (unsigned __int64)HIDWORD(v128) )
      {
        v71 = &v127[2 * (unsigned int)v128];
        if ( v71 )
        {
          *v71 = v77;
          v71[1] = v74;
          v72 = v128;
        }
        goto LABEL_110;
      }
      v90 = (unsigned int)v128 + 1LL;
      v91 = (unsigned int)v77 | (unsigned __int64)(v74 << 32);
      if ( HIDWORD(v128) >= v90 )
      {
LABEL_149:
        *(_QWORD *)&v127[2 * v79] = v91;
        LODWORD(v128) = v128 + 1;
        goto LABEL_111;
      }
LABEL_151:
      sub_C8D5F0((__int64)&v127, v129, v90, 8u, v90, v78);
      v79 = (unsigned int)v128;
      goto LABEL_149;
    }
    if ( (unsigned int)v128 >= (unsigned __int64)HIDWORD(v128) )
    {
      v90 = (unsigned int)v128 + 1LL;
      v91 = (v77 << 32) | v74;
      if ( HIDWORD(v128) >= v90 )
        goto LABEL_149;
      goto LABEL_151;
    }
    v80 = &v127[2 * (unsigned int)v128];
    if ( v80 )
    {
      *v80 = v74;
      v80[1] = v77;
      v72 = v128;
    }
LABEL_110:
    LODWORD(v128) = v72 + 1;
LABEL_111:
    v73 = v76 + *(_QWORD *)(v69 + 32);
    if ( !*(_BYTE *)v73 && (*(_WORD *)(v73 + 2) & 0xFF0) != 0 )
    {
      v113 = v76 + *(_QWORD *)(v69 + 32);
      v85 = *(_QWORD *)(v69 + 32) + 40LL * (unsigned int)sub_2E89F40(v69, v75);
      *(_WORD *)(v85 + 2) &= 0xF00Fu;
      *(_WORD *)(v113 + 2) &= 0xF00Fu;
    }
LABEL_114:
    ++v70;
  }
  while ( v108 != (char *)v70 );
  v4 = v69;
  v29 = v103;
  v30 = v99;
  v28 = v96;
LABEL_36:
  v31 = *(__int64 **)(v30 + 48);
  if ( v28 )
    v32 = sub_2FDFD70(v31, v4, v124, (unsigned int)v125, v28);
  else
    v32 = sub_2FDF650(
            v31,
            v4,
            v124,
            (unsigned int)v125,
            *(_DWORD *)(v30 + 80),
            *(_QWORD *)(v30 + 16),
            *(_QWORD *)(v30 + 32));
  if ( v32 )
  {
    v33 = v4;
    if ( (*(_BYTE *)(v4 + 44) & 4) != 0 )
    {
      do
        v33 = *(_QWORD *)v33 & 0xFFFFFFFFFFFFFFF8LL;
      while ( (*(_BYTE *)(v33 + 44) & 4) != 0 );
    }
    v34 = *(_QWORD *)(v4 + 24) + 48LL;
    while ( 1 )
    {
      v35 = *(_QWORD *)(v33 + 32);
      v36 = v35 + 40LL * (*(_DWORD *)(v33 + 40) & 0xFFFFFF);
      if ( v35 != v36 )
        break;
      v33 = *(_QWORD *)(v33 + 8);
      if ( v34 == v33 )
        break;
      if ( (*(_BYTE *)(v33 + 44) & 4) == 0 )
      {
        v33 = *(_QWORD *)(v4 + 24) + 48LL;
        break;
      }
    }
    v111 = v4;
    v37 = v32;
    v38 = v34;
    v107 = v29;
    v39 = v36;
    v40 = (_QWORD *)v30;
    v41 = v35;
    while ( v41 != v39 )
    {
      while ( 1 )
      {
        if ( !*(_BYTE *)v41 )
        {
          v42 = *(_DWORD *)(v41 + 8);
          if ( v42 > 0
            && (*(_QWORD *)(*(_QWORD *)(v40[5] + 384LL) + 8LL * ((unsigned int)v42 >> 6)) & (1LL << v42)) == 0
            && (*(_BYTE *)(v41 + 3) & 0x10) != 0 )
          {
            v101 = v40;
            v98 = *(_DWORD *)(v41 + 8);
            v43 = sub_2E92660(v37, v42, v40[7]);
            v40 = v101;
            if ( (v43 & 0xFF0000) == 0 )
            {
              v95 = v101;
              v102 = v101[2];
              v44 = sub_2DF8360(*(_QWORD *)(v102 + 32), v111, 0);
              sub_2E14ED0(v102, v98, v44 & 0xFFFFFFFFFFFFFFF8LL | 4);
              v40 = v95;
            }
          }
        }
        v45 = v41 + 40;
        v46 = v39;
        if ( v45 == v39 )
          break;
        v39 = v45;
LABEL_60:
        v41 = v39;
        v39 = v46;
      }
      while ( 1 )
      {
        v33 = *(_QWORD *)(v33 + 8);
        if ( v38 == v33 )
          break;
        if ( (*(_BYTE *)(v33 + 44) & 4) == 0 )
        {
          v33 = v38;
          break;
        }
        v39 = *(_QWORD *)(v33 + 32);
        v46 = v39 + 40LL * (*(_DWORD *)(v33 + 40) & 0xFFFFFF);
        if ( v39 != v46 )
          goto LABEL_60;
      }
      v41 = v39;
      v39 = v46;
    }
    v48 = v40[6];
    v49 = v37;
    v50 = v40;
    v5 = v107;
    v51 = *(__int64 (**)())(*(_QWORD *)v48 + 120LL);
    if ( v51 != sub_2F4C0B0
      && ((unsigned int (__fastcall *)(__int64, unsigned __int64, __int32 *))v51)(v48, v111, &v117) )
    {
      sub_34FA050((__int64)(v50 + 57), v111, v117);
    }
    v52 = *(_QWORD *)(v50[2] + 32LL);
    v53 = *(unsigned int *)(v52 + 144);
    v54 = *(_QWORD *)(v52 + 128);
    if ( (_DWORD)v53 )
    {
      v55 = (v53 - 1) & (((unsigned int)v111 >> 9) ^ ((unsigned int)v111 >> 4));
      v56 = (__int64 *)(v54 + 16LL * v55);
      v57 = *v56;
      if ( v111 == *v56 )
      {
LABEL_66:
        if ( v56 != (__int64 *)(v54 + 16 * v53) )
        {
          v58 = v56[1];
          *(_QWORD *)((v58 & 0xFFFFFFFFFFFFFFF8LL) + 16) = v37;
          *v56 = -8192;
          --*(_DWORD *)(v52 + 136);
          ++*(_DWORD *)(v52 + 140);
          v119 = v58;
          v118 = v37;
          sub_2FADAE0((__int64)v123, v52 + 120, &v118, &v119);
        }
      }
      else
      {
        v92 = 1;
        while ( v57 != -4096 )
        {
          v93 = v92 + 1;
          v94 = ((_DWORD)v53 - 1) & (v55 + v92);
          v55 = v94;
          v56 = (__int64 *)(v54 + 16 * v94);
          v57 = *v56;
          if ( v111 == *v56 )
            goto LABEL_66;
          v92 = v93;
        }
      }
    }
    if ( sub_2E88ED0(v111, 0) )
    {
      v86 = sub_2E88D60(v111);
      sub_2E7E910(v86, v111, v37);
    }
    if ( *(_DWORD *)(v111 + 64) )
    {
      v59 = *((_DWORD *)v115 + 2);
      if ( v59 )
      {
        sub_2E798E0(v50[1], v111, v37, v59);
      }
      else
      {
        v123[0] = v50;
        v123[1] = v37;
        v123[2] = v111;
        v123[3] = &v115;
        v60 = *(_QWORD *)(v111 + 32);
        v61 = v60 + 40LL * *((unsigned int *)v115 + 2);
        if ( v116 == 1 )
        {
          if ( (*(_BYTE *)(v61 + 3) & 0x10) != 0 )
            goto LABEL_153;
        }
        else if ( v116 == 2
               && (*(_BYTE *)(v61 + 3) & 0x10) != 0
               && (*(_WORD *)(v60 + 42) & 0xFF0) != 0
               && *(_DWORD *)(v61 + 8) == *(_DWORD *)(v60 + 48) )
        {
LABEL_153:
          sub_34F4E40((__int64)v123);
        }
      }
    }
    sub_2E88E20(v111);
    v62 = v121;
    if ( v121 == (_BYTE *)(v120 + 48) )
    {
      v63 = *(_QWORD *)(v120 + 56);
    }
    else
    {
      if ( !v121 )
        BUG();
      if ( (*v121 & 4) == 0 && (v121[44] & 8) != 0 )
      {
        do
          v62 = (_BYTE *)*((_QWORD *)v62 + 1);
        while ( (v62[44] & 8) != 0 );
      }
      v63 = *((_QWORD *)v62 + 1);
    }
    for ( i = v122; i != v63; v63 = *(_QWORD *)(v63 + 8) )
    {
      if ( v49 != v63 )
      {
        sub_2E192D0(*(_QWORD *)(v50[2] + 32LL), v63, 0);
        if ( !v63 )
          BUG();
      }
      if ( (*(_BYTE *)v63 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v63 + 44) & 8) != 0 )
          v63 = *(_QWORD *)(v63 + 8);
      }
    }
    if ( v105 && (*(_DWORD *)(v49 + 40) & 0xFFFFFF) != 0 )
    {
      v65 = (*(_DWORD *)(v49 + 40) & 0xFFFFFF) - 1;
      v66 = 40LL * v65;
      while ( 1 )
      {
        v67 = v66 + *(_QWORD *)(v49 + 32);
        if ( *(_BYTE *)v67 || (*(_BYTE *)(v67 + 3) & 0x20) == 0 )
          break;
        if ( v105 == *(_DWORD *)(v67 + 8) )
          sub_2E8A650(v49, v65);
        v66 -= 40;
        if ( !v65 )
          break;
        --v65;
      }
    }
    if ( !v107 || *((_DWORD *)v115 + 2) )
      goto LABEL_101;
    v87 = v121;
    if ( v121 == (_BYTE *)(v120 + 48) )
    {
      v88 = *(_QWORD *)(v120 + 56);
    }
    else
    {
      if ( !v121 )
        BUG();
      if ( (*v121 & 4) == 0 && (v121[44] & 8) != 0 )
      {
        do
          v87 = (_BYTE *)*((_QWORD *)v87 + 1);
        while ( (v87[44] & 8) != 0 );
      }
      v88 = *((_QWORD *)v87 + 1);
    }
    if ( v122 == v88 )
      goto LABEL_147;
    v89 = 0;
    do
    {
      if ( !v88 )
        BUG();
      if ( (*(_BYTE *)v88 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v88 + 44) & 8) != 0 )
          v88 = *(_QWORD *)(v88 + 8);
      }
      v88 = *(_QWORD *)(v88 + 8);
      ++v89;
    }
    while ( v122 != v88 );
    if ( v89 == 1 )
    {
LABEL_147:
      sub_34F9990((__int64)(v50 + 57), v49, *((_DWORD *)v50 + 20), *((_DWORD *)v50 + 21));
      v68 = v127;
    }
    else
    {
LABEL_101:
      v68 = v127;
      v5 = 1;
    }
  }
  else
  {
    v68 = v127;
    v81 = &v127[2 * (unsigned int)v128];
    v82 = v127;
    if ( v81 != v127 )
    {
      do
      {
        v83 = v82[1];
        v84 = *v82;
        v82 += 2;
        sub_2E89ED0(v4, v84, v83);
      }
      while ( v81 != v82 );
      v68 = v127;
    }
    v5 = 0;
  }
  if ( v68 != (unsigned int *)v129 )
    _libc_free((unsigned __int64)v68);
  v27 = v124;
LABEL_32:
  if ( v27 != v126 )
    _libc_free((unsigned __int64)v27);
  return v5;
}
