// Function: sub_37E0750
// Address: 0x37e0750
//
void __fastcall sub_37E0750(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, unsigned __int64 a5, __int64 a6)
{
  unsigned __int64 *v7; // rax
  __int64 v8; // rdx
  unsigned __int64 *v9; // r13
  __int64 v10; // r12
  int v11; // eax
  unsigned int v12; // r14d
  int v13; // eax
  int v14; // ecx
  unsigned __int64 v15; // rcx
  __int64 v16; // r14
  __int64 v17; // rbx
  int v18; // eax
  __int64 v19; // rdi
  int v20; // edx
  __int64 v21; // rax
  int i; // esi
  __int64 v23; // rcx
  int v24; // edx
  unsigned __int64 v25; // r12
  __int64 j; // rbx
  __int64 v27; // r11
  __int64 v28; // r12
  __int64 v29; // r14
  __int64 v30; // r11
  int v31; // ecx
  unsigned __int64 v32; // rax
  int v33; // r10d
  __int64 v34; // rdx
  __int64 v35; // rbx
  __int64 v36; // rax
  char v37; // si
  unsigned int v38; // ecx
  int v39; // edx
  unsigned int v40; // eax
  int v41; // edx
  _DWORD *v42; // rax
  __int64 v43; // rbx
  __int64 v44; // r11
  __int64 k; // rbx
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // rdi
  __int64 v49; // r10
  unsigned __int64 *v50; // rsi
  unsigned __int64 v51; // r12
  int v52; // eax
  __int64 v53; // rdx
  int v54; // ecx
  __int64 v55; // rdi
  unsigned __int64 v56; // rax
  int v57; // ebx
  unsigned int v58; // r13d
  __int64 v59; // rsi
  int v60; // eax
  __int64 v61; // rdx
  __int64 v62; // r9
  unsigned int v63; // ecx
  _DWORD *v64; // rax
  _DWORD *v65; // rsi
  unsigned int v66; // esi
  __int64 v67; // rax
  __int64 v68; // r15
  __int64 v69; // r14
  unsigned __int64 *v70; // r12
  _QWORD *v71; // rax
  __int64 v72; // rdi
  __int64 v73; // rdx
  int v74; // ecx
  unsigned int v75; // edx
  __int64 v76; // rax
  _QWORD *v77; // rcx
  __int64 v78; // rsi
  __int64 v79; // rcx
  int v80; // ecx
  _QWORD *v81; // r9
  unsigned int v82; // esi
  __int64 v83; // rax
  unsigned __int64 *v85; // rbx
  unsigned __int64 *v86; // r12
  __int64 v87; // rax
  __int64 v88; // r8
  unsigned int v90; // r13d
  unsigned int v91; // edx
  __int64 v92; // rbx
  int v93; // eax
  char v94; // cl
  __int64 v95; // rdi
  int v96; // esi
  unsigned int v97; // r9d
  _DWORD *v98; // rax
  int v99; // r8d
  int v100; // eax
  unsigned int v101; // r13d
  unsigned int v102; // r8d
  unsigned int v103; // esi
  int v104; // r13d
  _QWORD *v105; // r9
  unsigned __int64 v106; // rdi
  __int64 v107; // rdx
  unsigned __int64 v108; // rdi
  unsigned __int64 v109; // r10
  unsigned int v112; // esi
  unsigned int v113; // edx
  int v114; // eax
  _DWORD *v115; // rax
  __int64 v116; // rdi
  int v117; // r9d
  unsigned int v118; // ecx
  int v119; // esi
  int v120; // eax
  int v121; // r10d
  _DWORD *v122; // rdx
  int v123; // [rsp+18h] [rbp-9F8h]
  int v124; // [rsp+1Ch] [rbp-9F4h]
  unsigned __int64 v125; // [rsp+28h] [rbp-9E8h]
  __int64 v126; // [rsp+28h] [rbp-9E8h]
  __int64 v127; // [rsp+30h] [rbp-9E0h]
  int v128; // [rsp+30h] [rbp-9E0h]
  unsigned int v129; // [rsp+38h] [rbp-9D8h]
  _DWORD *v130; // [rsp+38h] [rbp-9D8h]
  __int64 v131; // [rsp+40h] [rbp-9D0h]
  unsigned int v132; // [rsp+48h] [rbp-9C8h]
  __int64 v133; // [rsp+50h] [rbp-9C0h]
  int v134; // [rsp+50h] [rbp-9C0h]
  _DWORD *v136; // [rsp+68h] [rbp-9A8h] BYREF
  _DWORD *v137; // [rsp+70h] [rbp-9A0h] BYREF
  __int64 v138; // [rsp+78h] [rbp-998h]
  void *v139; // [rsp+80h] [rbp-990h] BYREF
  __int64 v140; // [rsp+88h] [rbp-988h]
  _DWORD v141[16]; // [rsp+90h] [rbp-980h] BYREF
  unsigned __int64 *v142; // [rsp+D0h] [rbp-940h] BYREF
  __int64 v143; // [rsp+D8h] [rbp-938h]
  _BYTE v144[2352]; // [rsp+E0h] [rbp-930h] BYREF

  v142 = (unsigned __int64 *)v144;
  v143 = 0x2000000000LL;
  v123 = a4;
  v131 = (unsigned int)a4;
  if ( !(_DWORD)a4 )
  {
    v12 = (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1 + 16) + 16LL) + 31) >> 5;
    goto LABEL_27;
  }
  if ( (unsigned int)a4 <= 0x20uLL )
  {
    v7 = (unsigned __int64 *)v144;
    v8 = 9LL * (unsigned int)a4;
    v9 = (unsigned __int64 *)&v144[v8 * 8];
    do
    {
LABEL_4:
      if ( v7 )
      {
        v7[8] = 0;
        *v7 = (unsigned __int64)(v7 + 2);
        *((_DWORD *)v7 + 2) = 0;
        *((_DWORD *)v7 + 3) = 6;
        *((_OWORD *)v7 + 1) = 0;
        *((_OWORD *)v7 + 2) = 0;
        *((_OWORD *)v7 + 3) = 0;
      }
      v7 += 9;
    }
    while ( v9 != v7 );
    v10 = (__int64)v142;
    v9 = &v142[v8];
    goto LABEL_8;
  }
  v17 = (unsigned int)a4;
  sub_2FD0B90((__int64)&v142, (unsigned int)a4, (__int64)a3, a4, a5, a6);
  v10 = (__int64)v142;
  v8 = 9 * v17;
  v9 = &v142[9 * v17];
  v7 = &v142[9 * (unsigned int)v143];
  if ( v7 != v9 )
    goto LABEL_4;
LABEL_8:
  LODWORD(v143) = v123;
  v11 = *(_DWORD *)(*(_QWORD *)(a1 + 16) + 16LL);
  v12 = (unsigned int)(v11 + 31) >> 5;
  if ( (unsigned __int64 *)v10 != v9 )
  {
    v132 = (unsigned int)(v11 + 31) >> 5;
    while ( 1 )
    {
      v14 = *(_DWORD *)(v10 + 64) & 0x3F;
      if ( v14 )
        *(_QWORD *)(*(_QWORD *)v10 + 8LL * *(unsigned int *)(v10 + 8) - 8) |= -1LL << v14;
      v15 = *(unsigned int *)(v10 + 8);
      *(_DWORD *)(v10 + 64) = v11;
      a5 = (unsigned int)(v11 + 63) >> 6;
      if ( a5 != v15 )
      {
        if ( a5 < v15 )
        {
          *(_DWORD *)(v10 + 8) = (unsigned int)(v11 + 63) >> 6;
        }
        else
        {
          v16 = a5 - v15;
          if ( a5 > *(unsigned int *)(v10 + 12) )
          {
            sub_C8D5F0(v10, (const void *)(v10 + 16), a5, 8u, a5, a6);
            v15 = *(unsigned int *)(v10 + 8);
          }
          if ( 8 * v16 )
          {
            memset((void *)(*(_QWORD *)v10 + 8 * v15), 255, 8 * v16);
            LODWORD(v15) = *(_DWORD *)(v10 + 8);
          }
          v11 = *(_DWORD *)(v10 + 64);
          *(_DWORD *)(v10 + 8) = v16 + v15;
        }
      }
      v13 = v11 & 0x3F;
      if ( v13 )
        *(_QWORD *)(*(_QWORD *)v10 + 8LL * *(unsigned int *)(v10 + 8) - 8) &= ~(-1LL << v13);
      v10 += 72;
      if ( v9 == (unsigned __int64 *)v10 )
        break;
      v11 = *(_DWORD *)(*(_QWORD *)(a1 + 16) + 16LL);
    }
    v12 = v132;
  }
LABEL_27:
  v127 = a2 + 320;
  v133 = *(_QWORD *)(a2 + 328);
  if ( v133 == a2 + 320 )
    goto LABEL_75;
  do
  {
    v18 = *(_DWORD *)(v133 + 24);
    *(_DWORD *)(a1 + 420) = 1;
    *(_DWORD *)(a1 + 416) = v18;
    *(_DWORD *)(*(_QWORD *)(a1 + 408) + 304LL) = 0;
    v19 = *(_QWORD *)(a1 + 408);
    v20 = *(_DWORD *)(a1 + 416);
    a5 = *(unsigned int *)(v19 + 40);
    *(_DWORD *)(v19 + 280) = v20;
    if ( (_DWORD)a5 )
    {
      v21 = 0;
      for ( i = v20; ; i = *(_DWORD *)(v19 + 280) )
      {
        v23 = *(_QWORD *)(v19 + 32) + 8 * v21;
        *(_QWORD *)v23 = i & 0xFFFFF | *(_QWORD *)v23 & 0xFFFFFF0000000000LL;
        v24 = v21++;
        *(_DWORD *)(v23 + 4) = v24 << 8;
        if ( a5 == v21 )
          break;
      }
    }
    v25 = *(_QWORD *)(v133 + 56);
    for ( j = v133 + 48; j != v25; v25 = *(_QWORD *)(v25 + 8) )
    {
      while ( 1 )
      {
        sub_37E06C0(a1, v25, 0, 0);
        if ( (unsigned __int16)(*(_WORD *)(v25 + 68) - 14) <= 2u )
          sub_37D4430(a1, v25);
        if ( *(_DWORD *)(v25 + 64) )
        {
          v39 = *(_DWORD *)(a1 + 420);
          v139 = (void *)*(unsigned int *)(v25 + 64);
          v140 = v25;
          v141[0] = v39;
          sub_37BE540((_QWORD *)(a1 + 728), (unsigned __int64 *)&v139);
        }
        ++*(_DWORD *)(a1 + 420);
        if ( (*(_BYTE *)v25 & 4) == 0 )
          break;
        v25 = *(_QWORD *)(v25 + 8);
        if ( j == v25 )
          goto LABEL_41;
      }
      while ( (*(_BYTE *)(v25 + 44) & 8) != 0 )
        v25 = *(_QWORD *)(v25 + 8);
    }
LABEL_41:
    v27 = *(_QWORD *)(a1 + 408);
    v28 = 0;
    if ( !*(_DWORD *)(v27 + 40) )
      goto LABEL_62;
    v129 = v12;
    v29 = *(_QWORD *)(a1 + 408);
    v30 = *(unsigned int *)(v27 + 40);
    do
    {
      while ( 1 )
      {
        v34 = *(_QWORD *)(v29 + 32) + 8 * v28;
        if ( (*(_QWORD *)v34 & 0xFFFFF00000LL) == 0 && *(_DWORD *)(v34 + 4) >> 8 == v28 )
          goto LABEL_46;
        v35 = *a3 + 80LL * *(unsigned int *)(a1 + 416);
        v36 = *(_QWORD *)v34;
        LODWORD(v139) = v28;
        v140 = v36;
        v37 = *(_BYTE *)(v35 + 8) & 1;
        if ( v37 )
        {
          a5 = v35 + 16;
          v31 = 3;
        }
        else
        {
          v38 = *(_DWORD *)(v35 + 24);
          a5 = *(_QWORD *)(v35 + 16);
          if ( !v38 )
          {
            v137 = 0;
            v40 = *(_DWORD *)(v35 + 8);
            ++*(_QWORD *)v35;
            v41 = (v40 >> 1) + 1;
            goto LABEL_56;
          }
          v31 = v38 - 1;
        }
        a6 = v31 & (unsigned int)v28;
        v32 = a5 + 16 * a6;
        v33 = *(_DWORD *)v32;
        if ( *(_DWORD *)v32 != (_DWORD)v28 )
          break;
LABEL_45:
        *(_QWORD *)(v32 + 8) = *(_QWORD *)v34;
LABEL_46:
        if ( v30 == ++v28 )
          goto LABEL_61;
      }
      v124 = 1;
      v125 = 0;
      while ( v33 != -1 )
      {
        if ( v33 == -2 )
        {
          if ( v125 )
            v32 = v125;
          v125 = v32;
        }
        a6 = v31 & (unsigned int)(v124 + a6);
        v32 = a5 + 16LL * (unsigned int)a6;
        v33 = *(_DWORD *)v32;
        if ( *(_DWORD *)v32 == (_DWORD)v28 )
          goto LABEL_45;
        ++v124;
      }
      if ( v125 )
        v32 = v125;
      v137 = (_DWORD *)v32;
      v40 = *(_DWORD *)(v35 + 8);
      ++*(_QWORD *)v35;
      v41 = (v40 >> 1) + 1;
      if ( !v37 )
      {
        v38 = *(_DWORD *)(v35 + 24);
LABEL_56:
        if ( 4 * v41 >= 3 * v38 )
          goto LABEL_98;
        goto LABEL_57;
      }
      v38 = 4;
      if ( (unsigned int)(4 * v41) >= 0xC )
      {
LABEL_98:
        v126 = v30;
        v66 = 2 * v38;
        goto LABEL_99;
      }
LABEL_57:
      if ( v38 - *(_DWORD *)(v35 + 12) - v41 > v38 >> 3 )
        goto LABEL_58;
      v126 = v30;
      v66 = v38;
LABEL_99:
      sub_37D4A50(v35, v66);
      sub_37BFF80(v35, (int *)&v139, &v137);
      v40 = *(_DWORD *)(v35 + 8);
      v30 = v126;
LABEL_58:
      *(_DWORD *)(v35 + 8) = (2 * (v40 >> 1) + 2) | v40 & 1;
      v42 = v137;
      if ( *v137 != -1 )
        --*(_DWORD *)(v35 + 12);
      ++v28;
      *v42 = (_DWORD)v139;
      *((_QWORD *)v42 + 1) = v140;
    }
    while ( v30 != v28 );
LABEL_61:
    v12 = v129;
    v27 = *(_QWORD *)(a1 + 408);
LABEL_62:
    v43 = 16LL * *(unsigned int *)(v27 + 304);
    v44 = *(_QWORD *)(v27 + 296);
    for ( k = v44 + v43; k != v44; v44 += 16 )
    {
      a5 = (unsigned __int64)&v142[9 * *(unsigned int *)(a1 + 416)];
      v46 = *(_QWORD *)(*(_QWORD *)v44 + 24LL);
      a6 = (unsigned int)(*(_DWORD *)(a5 + 64) + 31) >> 5;
      if ( (unsigned int)a6 > v12 )
        a6 = v12;
      if ( (unsigned int)a6 <= 1 )
      {
        LODWORD(v48) = 0;
      }
      else
      {
        v47 = 0;
        v48 = ((unsigned int)(a6 - 2) >> 1) + 1;
        v49 = 8 * v48;
        do
        {
          v50 = (unsigned __int64 *)(v47 + *(_QWORD *)a5);
          v51 = *v50 & ~(unsigned __int64)(unsigned int)~*(_DWORD *)(v46 + v47);
          v52 = *(_DWORD *)(v46 + v47 + 4);
          v47 += 8;
          *v50 = v51 & ~((unsigned __int64)(unsigned int)~v52 << 32);
        }
        while ( v49 != v47 );
        v46 += v49;
        a6 &= 1u;
      }
      if ( (_DWORD)a6 )
      {
        v53 = v46 + 4;
        v54 = 0;
        v55 = 8LL * (unsigned int)v48;
        a6 = v53;
        while ( 1 )
        {
          v56 = (unsigned __int64)(unsigned int)~*(_DWORD *)(v53 - 4) << v54;
          v54 += 32;
          *(_QWORD *)(v55 + *(_QWORD *)a5) &= ~v56;
          if ( v53 == a6 )
            break;
          v53 += 4;
        }
      }
    }
    v133 = *(_QWORD *)(v133 + 8);
  }
  while ( v127 != v133 );
LABEL_75:
  v57 = *(_DWORD *)(*(_QWORD *)(a1 + 16) + 16LL);
  v139 = v141;
  v58 = (unsigned int)(v57 + 63) >> 6;
  v140 = 0x600000000LL;
  if ( v58 > 6 )
  {
    sub_C8D5F0((__int64)&v139, v141, v58, 8u, a5, a6);
    memset(v139, 0, 8LL * v58);
    LODWORD(v140) = (unsigned int)(v57 + 63) >> 6;
  }
  else
  {
    if ( v58 && 8LL * v58 )
      memset(v141, 0, 8LL * v58);
    LODWORD(v140) = (unsigned int)(v57 + 63) >> 6;
  }
  v59 = *(_QWORD *)(a1 + 408);
  v141[12] = v57;
  v60 = *(_DWORD *)(v59 + 40);
  if ( v60 )
  {
    v61 = 0;
    v62 = 4LL * (unsigned int)(v60 - 1);
    while ( 1 )
    {
      v63 = *(_DWORD *)(*(_QWORD *)(v59 + 88) + v61);
      if ( v63 >= *(_DWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
        break;
      if ( *(_QWORD *)(v59 + 200) )
      {
        v87 = *(_QWORD *)(v59 + 176);
        if ( v87 )
        {
          v88 = v59 + 168;
          do
          {
            if ( v63 > *(_DWORD *)(v87 + 32) )
            {
              v87 = *(_QWORD *)(v87 + 24);
            }
            else
            {
              v88 = v87;
              v87 = *(_QWORD *)(v87 + 16);
            }
          }
          while ( v87 );
          if ( v88 != v59 + 168 && v63 >= *(_DWORD *)(v88 + 32) )
            break;
        }
      }
      else
      {
        v64 = *(_DWORD **)(v59 + 112);
        v65 = &v64[*(unsigned int *)(v59 + 120)];
        if ( v64 != v65 )
        {
          while ( v63 != *v64 )
          {
            if ( v65 == ++v64 )
              goto LABEL_102;
          }
          if ( v65 != v64 )
            break;
        }
      }
LABEL_102:
      *((_QWORD *)v139 + (v63 >> 6)) |= 1LL << v63;
      if ( v62 == v61 )
        goto LABEL_103;
LABEL_90:
      v59 = *(_QWORD *)(a1 + 408);
      v61 += 4;
    }
    if ( v62 == v61 )
      goto LABEL_103;
    goto LABEL_90;
  }
LABEL_103:
  if ( v123 )
  {
    v67 = a1;
    v68 = 0;
    v69 = v67;
    while ( 1 )
    {
      v70 = &v142[9 * v68];
      v71 = (_QWORD *)*v70;
      v72 = *((unsigned int *)v70 + 2);
      v73 = *v70 + 8 * v72;
      if ( *v70 != v73 )
      {
        do
        {
          *v71 = ~*v71;
          ++v71;
        }
        while ( (_QWORD *)v73 != v71 );
        v72 = *((unsigned int *)v70 + 2);
      }
      v74 = v70[8] & 0x3F;
      if ( v74 )
      {
        *(_QWORD *)(*v70 + 8 * v72 - 8) &= ~(-1LL << v74);
        LODWORD(v72) = *((_DWORD *)v70 + 2);
      }
      v75 = v72;
      if ( (unsigned int)v140 <= (unsigned int)v72 )
        v75 = v140;
      v76 = 0;
      if ( v75 )
      {
        do
        {
          v77 = (_QWORD *)(v76 + *v70);
          v78 = *(_QWORD *)((char *)v139 + v76);
          v76 += 8;
          *v77 &= v78;
        }
        while ( 8LL * v75 != v76 );
      }
      for ( ; v75 != (_DWORD)v72; *(_QWORD *)(*v70 + 8 * v79) = 0 )
        v79 = v75++;
      v80 = *((_DWORD *)v70 + 16);
      if ( v80 )
      {
        v81 = (_QWORD *)*v70;
        v82 = (unsigned int)(v80 - 1) >> 6;
        v83 = 0;
        while ( 1 )
        {
          _RDX = v81[v83];
          if ( v82 == (_DWORD)v83 )
            _RDX = (0xFFFFFFFFFFFFFFFFLL >> -(char)v80) & v81[v83];
          if ( _RDX )
            break;
          if ( ++v83 == v82 + 1 )
            goto LABEL_122;
        }
        __asm { tzcnt   rdx, rdx }
        v90 = ((_DWORD)v83 << 6) + _RDX;
        if ( v90 != -1 )
          break;
      }
LABEL_122:
      if ( v131 == ++v68 )
        goto LABEL_123;
    }
    v134 = v68 & 0xFFFFF;
    while ( 2 )
    {
      v91 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v69 + 408) + 64LL) + 4LL * v90);
      v92 = *a3 + 80 * v68;
      LODWORD(v137) = v91;
      LODWORD(v138) = v134 | v138 & 0xFFF00000;
      v93 = (unsigned __int8)((v138 & 0xFFFFFF00000FFFFFLL) >> 32);
      LODWORD(v138) = v138 & 0xFFFFF | 0x100000;
      HIDWORD(v138) = (v91 << 8) | v93;
      v94 = *(_BYTE *)(v92 + 8) & 1;
      if ( v94 )
      {
        v95 = v92 + 16;
        v96 = 3;
        goto LABEL_144;
      }
      v112 = *(_DWORD *)(v92 + 24);
      v95 = *(_QWORD *)(v92 + 16);
      if ( v112 )
      {
        v96 = v112 - 1;
LABEL_144:
        v97 = v96 & v91;
        v98 = (_DWORD *)(v95 + 16LL * (v96 & v91));
        v99 = *v98;
        if ( v91 == *v98 )
        {
LABEL_145:
          if ( (v98[2] & 0xFFFFF) == (_DWORD)v68 && (*((_QWORD *)v98 + 1) & 0xFFFFF00000LL) == 0 )
          {
            v98[2] = v134 | v98[2] & 0xFFF00000;
            *((_QWORD *)v98 + 1) = *((_QWORD *)v98 + 1) & 0xFFFFFF00000FFFFFLL | 0x100000;
            v98[3] = v91 << 8;
          }
LABEL_146:
          v100 = *((_DWORD *)v70 + 16);
          v101 = v90 + 1;
          if ( v100 == v101 )
            goto LABEL_122;
          v102 = v101 >> 6;
          v103 = (unsigned int)(v100 - 1) >> 6;
          if ( v101 >> 6 > v103 )
            goto LABEL_122;
          v104 = v101 & 0x3F;
          v105 = (_QWORD *)*v70;
          v106 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v104);
          if ( v104 == 0 )
            v106 = 0;
          v107 = v102;
          v108 = ~v106;
          v109 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v100;
          while ( 1 )
          {
            _RAX = v105[v107];
            if ( v102 == (_DWORD)v107 )
              _RAX = v108 & v105[v107];
            if ( (_DWORD)v107 == v103 )
              _RAX &= v109;
            if ( _RAX )
              break;
            if ( v103 < (unsigned int)++v107 )
              goto LABEL_122;
          }
          __asm { tzcnt   rax, rax }
          v90 = _RAX + ((_DWORD)v107 << 6);
          if ( v90 == -1 )
            goto LABEL_122;
          continue;
        }
        v128 = 1;
        v130 = 0;
        while ( v99 != -1 )
        {
          if ( v99 == -2 )
          {
            if ( v130 )
              v98 = v130;
            v130 = v98;
          }
          v97 = v96 & (v128 + v97);
          v98 = (_DWORD *)(v95 + 16LL * v97);
          v99 = *v98;
          if ( v91 == *v98 )
            goto LABEL_145;
          ++v128;
        }
        v113 = *(_DWORD *)(v92 + 8);
        if ( v130 )
          v98 = v130;
        v136 = v98;
        ++*(_QWORD *)v92;
        v114 = (v113 >> 1) + 1;
        if ( v94 )
        {
          v112 = 4;
          if ( (unsigned int)(4 * v114) >= 0xC )
          {
LABEL_176:
            sub_37D4A50(v92, 2 * v112);
            if ( (*(_BYTE *)(v92 + 8) & 1) != 0 )
            {
              v116 = v92 + 16;
              v117 = 3;
LABEL_178:
              v118 = v117 & (unsigned int)v137;
              v115 = (_DWORD *)(v116 + 16LL * (v117 & (unsigned int)v137));
              v119 = *v115;
              if ( (_DWORD)v137 == *v115 )
              {
LABEL_179:
                v136 = v115;
                v113 = *(_DWORD *)(v92 + 8);
              }
              else
              {
                v121 = 1;
                v122 = 0;
                while ( v119 != -1 )
                {
                  if ( !v122 && v119 == -2 )
                    v122 = v115;
                  v118 = v117 & (v121 + v118);
                  v115 = (_DWORD *)(v116 + 16LL * v118);
                  v119 = *v115;
                  if ( (_DWORD)v137 == *v115 )
                    goto LABEL_179;
                  ++v121;
                }
                if ( !v122 )
                  v122 = v115;
                v136 = v122;
                v115 = v122;
                v113 = *(_DWORD *)(v92 + 8);
              }
            }
            else
            {
              v120 = *(_DWORD *)(v92 + 24);
              v116 = *(_QWORD *)(v92 + 16);
              v117 = v120 - 1;
              if ( v120 )
                goto LABEL_178;
              v136 = 0;
              v113 = *(_DWORD *)(v92 + 8);
              v115 = 0;
            }
LABEL_165:
            *(_DWORD *)(v92 + 8) = (2 * (v113 >> 1) + 2) | v113 & 1;
            if ( *v115 != -1 )
              --*(_DWORD *)(v92 + 12);
            *v115 = (_DWORD)v137;
            *((_QWORD *)v115 + 1) = v138;
            goto LABEL_146;
          }
LABEL_163:
          if ( v112 - *(_DWORD *)(v92 + 12) - v114 <= v112 >> 3 )
          {
            sub_37D4A50(v92, v112);
            sub_37BFF80(v92, (int *)&v137, &v136);
            v115 = v136;
            v113 = *(_DWORD *)(v92 + 8);
          }
          else
          {
            v115 = v136;
          }
          goto LABEL_165;
        }
        v112 = *(_DWORD *)(v92 + 24);
      }
      else
      {
        v136 = 0;
        v113 = *(_DWORD *)(v92 + 8);
        ++*(_QWORD *)v92;
        v114 = (v113 >> 1) + 1;
      }
      break;
    }
    if ( 4 * v114 >= 3 * v112 )
      goto LABEL_176;
    goto LABEL_163;
  }
LABEL_123:
  if ( v139 != v141 )
    _libc_free((unsigned __int64)v139);
  v85 = v142;
  v86 = &v142[9 * (unsigned int)v143];
  if ( v142 != v86 )
  {
    do
    {
      v86 -= 9;
      if ( (unsigned __int64 *)*v86 != v86 + 2 )
        _libc_free(*v86);
    }
    while ( v85 != v86 );
    v86 = v142;
  }
  if ( v86 != (unsigned __int64 *)v144 )
    _libc_free((unsigned __int64)v86);
}
