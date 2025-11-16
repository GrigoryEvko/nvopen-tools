// Function: sub_16A3760
// Address: 0x16a3760
//
void __fastcall sub_16A3760(__int64 a1, __int64 a2, unsigned int a3, unsigned int a4, char a5)
{
  char v6; // al
  int v7; // r13d
  int v8; // r12d
  __int64 v9; // rax
  unsigned int v10; // r15d
  unsigned int v12; // ecx
  int v14; // r12d
  unsigned __int64 v15; // rcx
  unsigned int v16; // edx
  unsigned int v17; // eax
  unsigned int v18; // edx
  unsigned int v19; // r8d
  char v20; // cl
  unsigned int v21; // r15d
  __int64 v22; // rax
  unsigned __int64 v23; // rcx
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rdx
  unsigned int v29; // eax
  unsigned int v30; // r8d
  char v31; // r15
  unsigned int i; // ebx
  unsigned __int64 v33; // rdx
  __int64 v34; // rdx
  int v35; // r15d
  unsigned int v36; // ebx
  unsigned int v37; // eax
  _BYTE *v38; // r8
  unsigned int v39; // ecx
  int v40; // ebx
  __int64 v41; // rax
  unsigned int v42; // r15d
  _BYTE *v43; // r12
  bool v44; // cf
  __int64 v45; // rax
  int v46; // r12d
  __int64 v47; // rax
  char v48; // bl
  int v49; // edx
  char v50; // cl
  unsigned int v51; // r8d
  int v52; // ebx
  unsigned int v53; // r8d
  __int64 v54; // rdx
  int v55; // r13d
  unsigned int v56; // r12d
  _BYTE *v57; // rbx
  char v58; // al
  unsigned __int64 v59; // r13
  _BYTE *v60; // rdx
  __int64 v61; // rax
  unsigned int v62; // r8d
  __int64 v63; // rdx
  int v64; // ebx
  _BYTE *v65; // r12
  __int64 v66; // rax
  __int64 v67; // rax
  int v68; // r12d
  __int64 v69; // rax
  int v70; // r15d
  int v71; // r12d
  unsigned int v72; // r13d
  _BYTE *v73; // rbx
  int v74; // ebx
  __int64 v75; // rax
  int j; // ebx
  _BYTE *v77; // r12
  __int64 v78; // rdi
  _BYTE *v79; // rsi
  __int64 v80; // rdx
  size_t v81; // rdx
  int v82; // ebx
  __int64 v83; // rdx
  unsigned int v84; // edx
  unsigned int v85; // eax
  size_t v86; // r12
  __int64 v87; // rdi
  __int64 v88; // rsi
  __int64 v89; // rdi
  size_t v90; // r13
  int v91; // r8d
  unsigned int v92; // eax
  unsigned int v93; // eax
  int v94; // r13d
  unsigned int v95; // [rsp+8h] [rbp-1C8h]
  unsigned int v96; // [rsp+10h] [rbp-1C0h]
  unsigned int v97; // [rsp+10h] [rbp-1C0h]
  unsigned int v98; // [rsp+10h] [rbp-1C0h]
  unsigned int v99; // [rsp+10h] [rbp-1C0h]
  int v100; // [rsp+10h] [rbp-1C0h]
  unsigned int v103; // [rsp+20h] [rbp-1B0h]
  int v104; // [rsp+20h] [rbp-1B0h]
  int v105; // [rsp+20h] [rbp-1B0h]
  int v107; // [rsp+28h] [rbp-1A8h]
  int v108; // [rsp+28h] [rbp-1A8h]
  char v109; // [rsp+3Fh] [rbp-191h] BYREF
  unsigned __int64 v110; // [rsp+40h] [rbp-190h] BYREF
  unsigned int v111; // [rsp+48h] [rbp-188h]
  unsigned __int64 v112; // [rsp+50h] [rbp-180h] BYREF
  unsigned int v113; // [rsp+58h] [rbp-178h]
  __int64 *v114; // [rsp+60h] [rbp-170h] BYREF
  unsigned int v115; // [rsp+68h] [rbp-168h]
  _BYTE *v116; // [rsp+70h] [rbp-160h] BYREF
  __int64 v117; // [rsp+78h] [rbp-158h]
  _BYTE v118[16]; // [rsp+80h] [rbp-150h] BYREF
  void *dest; // [rsp+90h] [rbp-140h] BYREF
  __int64 v120; // [rsp+98h] [rbp-138h]
  _BYTE v121[304]; // [rsp+A0h] [rbp-130h] BYREF

  v6 = *(_BYTE *)(a1 + 18) & 7;
  switch ( v6 )
  {
    case 1:
      v27 = *(unsigned int *)(a2 + 8);
      if ( (unsigned __int64)*(unsigned int *)(a2 + 12) - v27 <= 2 )
      {
        sub_16CD150(a2, a2 + 16, v27 + 3, 1);
        v27 = *(unsigned int *)(a2 + 8);
      }
      v28 = *(_QWORD *)a2 + v27;
      *(_WORD *)v28 = 24910;
      *(_BYTE *)(v28 + 2) = 78;
      *(_DWORD *)(a2 + 8) += 3;
      return;
    case 3:
      if ( (*(_BYTE *)(a1 + 18) & 8) != 0 )
      {
        LOBYTE(dest) = 45;
        sub_16A3710(a2, &dest);
      }
      if ( a4 )
      {
        LOBYTE(dest) = 48;
        sub_16A3710(a2, &dest);
      }
      else
      {
        v24 = *(unsigned int *)(a2 + 8);
        v25 = *(unsigned int *)(a2 + 12) - v24;
        if ( a5 )
        {
          if ( v25 <= 5 )
          {
            sub_16CD150(a2, a2 + 16, v24 + 6, 1);
            v24 = *(unsigned int *)(a2 + 8);
          }
          v26 = *(_QWORD *)a2 + v24;
          *(_DWORD *)v26 = 1160785456;
          *(_WORD *)(v26 + 4) = 12331;
          *(_DWORD *)(a2 + 8) += 6;
        }
        else
        {
          if ( v25 <= 2 )
          {
            sub_16CD150(a2, a2 + 16, v24 + 3, 1);
            v24 = *(unsigned int *)(a2 + 8);
          }
          v66 = *(_QWORD *)a2 + v24;
          *(_WORD *)v66 = 11824;
          *(_BYTE *)(v66 + 2) = 48;
          v67 = (unsigned int)(*(_DWORD *)(a2 + 8) + 3);
          *(_DWORD *)(a2 + 8) = v67;
          if ( a3 > 1 )
          {
            v86 = a3 - 1;
            v87 = (unsigned int)v67;
            if ( v86 > *(unsigned int *)(a2 + 12) - (unsigned __int64)(unsigned int)v67 )
            {
              sub_16CD150(a2, a2 + 16, v86 + (unsigned int)v67, 1);
              v87 = *(unsigned int *)(a2 + 8);
            }
            memset((void *)(*(_QWORD *)a2 + v87), 48, v86);
            v67 = (unsigned int)(*(_DWORD *)(a2 + 8) + v86);
            *(_DWORD *)(a2 + 8) = v67;
          }
          if ( (unsigned __int64)*(unsigned int *)(a2 + 12) - v67 <= 3 )
          {
            sub_16CD150(a2, a2 + 16, v67 + 4, 1);
            v67 = *(unsigned int *)(a2 + 8);
          }
          *(_DWORD *)(*(_QWORD *)a2 + v67) = 808463205;
          *(_DWORD *)(a2 + 8) += 4;
        }
      }
      return;
    case 0:
      v22 = *(unsigned int *)(a2 + 8);
      v23 = *(unsigned int *)(a2 + 12) - v22;
      if ( (*(_BYTE *)(a1 + 18) & 8) != 0 )
      {
        if ( v23 <= 3 )
        {
          sub_16CD150(a2, a2 + 16, v22 + 4, 1);
          v22 = *(unsigned int *)(a2 + 8);
        }
        *(_DWORD *)(*(_QWORD *)a2 + v22) = 1718503725;
        *(_DWORD *)(a2 + 8) += 4;
      }
      else
      {
        if ( v23 <= 3 )
        {
          sub_16CD150(a2, a2 + 16, v22 + 4, 1);
          v22 = *(unsigned int *)(a2 + 8);
        }
        *(_DWORD *)(*(_QWORD *)a2 + v22) = 1718503723;
        *(_DWORD *)(a2 + 8) += 4;
      }
      return;
  }
  if ( (*(_BYTE *)(a1 + 18) & 8) != 0 )
  {
    LOBYTE(dest) = 45;
    sub_16A3710(a2, &dest);
  }
  v7 = *(_DWORD *)(*(_QWORD *)a1 + 4LL);
  v8 = *(__int16 *)(a1 + 16) - (v7 - 1);
  v9 = sub_16984A0(a1);
  sub_16A50F0(&v110, *(unsigned int *)(*(_QWORD *)a1 + 4LL), v9, (unsigned int)(v7 + 63) >> 6);
  if ( !a3 )
    a3 = 59 * *(_DWORD *)(*(_QWORD *)a1 + 4LL) / 0xC4u + 2;
  v10 = v111;
  if ( v111 > 0x40 )
  {
    v29 = sub_16A58A0(&v110);
    v14 = v29 + v8;
    sub_16A8110(&v110, v29);
    if ( !v14 )
    {
LABEL_47:
      v10 = v111;
      if ( v111 > 0x40 )
      {
        v16 = v10 - sub_16A57B0(&v110);
        goto LABEL_18;
      }
      goto LABEL_15;
    }
LABEL_113:
    v49 = *(_DWORD *)(*(_QWORD *)a1 + 4LL);
    if ( v14 <= 0 )
    {
      v82 = -v14;
      v83 = (-137 * v14 + 136) / 0x3Bu + v49;
      v97 = v83;
      sub_16A5C50(&dest, &v110, v83);
      v84 = v97;
      if ( v111 > 0x40 && v110 )
      {
        j_j___libc_free_0_0(v110);
        v84 = v97;
      }
      v98 = v84;
      v110 = (unsigned __int64)dest;
      v85 = v120;
      LODWORD(v120) = 0;
      v111 = v85;
      sub_135E100((__int64 *)&dest);
      LODWORD(v120) = v98;
      if ( v98 <= 0x40 )
        dest = (void *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v98) & 5);
      else
        sub_16A4EF0(&dest, 5, 0);
      while ( 1 )
      {
        if ( (v82 & 1) != 0 )
          sub_16A7C10(&v110, &dest);
        v82 >>= 1;
        if ( !v82 )
          break;
        sub_16A7C10(&dest, &dest);
      }
      sub_135E100((__int64 *)&dest);
    }
    else
    {
      sub_16A5C50(&dest, &v110, (unsigned int)(v14 + v49));
      if ( v111 > 0x40 && v110 )
        j_j___libc_free_0_0(v110);
      v10 = v120;
      v110 = (unsigned __int64)dest;
      v111 = v120;
      if ( (unsigned int)v120 <= 0x40 )
      {
        if ( v14 == (_DWORD)v120 )
        {
          v110 = 0;
          v14 = 0;
          goto LABEL_64;
        }
        v50 = v14;
        v14 = 0;
        v15 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v120) & ((_QWORD)dest << v50);
        v110 = v15;
        goto LABEL_16;
      }
      v88 = (unsigned int)v14;
      v14 = 0;
      sub_16A7DC0(&v110, v88);
    }
    goto LABEL_47;
  }
  _RAX = v110;
  if ( v110 )
  {
    v12 = v111;
    __asm { tzcnt   rdx, rax }
    if ( (unsigned int)_RDX <= v111 )
      v12 = _RDX;
    v14 = v12 + v8;
    if ( (unsigned int)_RDX < v111 )
    {
      v110 >>= v12;
      goto LABEL_14;
    }
  }
  else
  {
    v14 = v111 + v8;
  }
  v110 = 0;
LABEL_14:
  if ( v14 )
    goto LABEL_113;
LABEL_15:
  v15 = v110;
LABEL_16:
  if ( !v15 )
    goto LABEL_64;
  _BitScanReverse64(&v15, v15);
  v16 = 64 - (v15 ^ 0x3F);
LABEL_18:
  v17 = (196 * a3 + 58) / 0x3B;
  if ( v17 < v16 )
  {
    v18 = v16 - v17;
    if ( 59 * v18 > 0xC3 )
    {
      v115 = v10;
      v19 = 59 * v18 / 0xC4;
      v14 += v19;
      if ( v10 <= 0x40 )
      {
        v20 = v111;
        v114 = (__int64 *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v10) & 1);
        LODWORD(v117) = v111;
      }
      else
      {
        v95 = 59 * v18 / 0xC4;
        sub_16A4EF0(&v114, 1, 0);
        v20 = v111;
        v19 = v95;
        LODWORD(v117) = v111;
        if ( v111 > 0x40 )
        {
          sub_16A4EF0(&v116, 10, 0);
          v19 = v95;
          goto LABEL_23;
        }
      }
      v116 = (_BYTE *)((0xFFFFFFFFFFFFFFFFLL >> -v20) & 0xA);
LABEL_23:
      v21 = v19;
      while ( 1 )
      {
        if ( (v21 & 1) != 0 )
          sub_16A7C10(&v114, &v116);
        v21 >>= 1;
        if ( !v21 )
          break;
        sub_16A7C10(&v116, &v116);
      }
      sub_16A9D70(&dest, &v110, &v114);
      if ( v111 > 0x40 && v110 )
        j_j___libc_free_0_0(v110);
      v110 = (unsigned __int64)dest;
      v111 = v120;
      if ( (unsigned int)v120 > 0x40 )
        sub_16A57B0(&v110);
      sub_16A5A50(&dest, &v110);
      if ( v111 > 0x40 && v110 )
        j_j___libc_free_0_0(v110);
      v110 = (unsigned __int64)dest;
      v111 = v120;
      if ( (unsigned int)v117 > 0x40 && v116 )
        j_j___libc_free_0_0(v116);
      if ( v115 > 0x40 && v114 )
        j_j___libc_free_0_0(v114);
    }
  }
LABEL_64:
  v30 = v111;
  dest = v121;
  v120 = 0x10000000000LL;
  v113 = v111;
  if ( v111 > 0x40 )
  {
    v96 = v111;
    sub_16A4EF0(&v112, 10, 0);
    v115 = v96;
    sub_16A4EF0(&v114, 0, 0);
    v30 = v111;
  }
  else
  {
    v115 = v111;
    v114 = 0;
    v112 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v111) & 0xA;
  }
  v31 = 1;
  for ( i = v30; i <= 0x40; i = v111 )
  {
    v33 = v110;
    if ( !v110 )
      goto LABEL_76;
LABEL_68:
    sub_16ADD10(&v110, &v112, &v110, &v114);
    LODWORD(v34) = (_DWORD)v114;
    if ( v115 > 0x40 )
      v34 = *v114;
    v31 &= (_DWORD)v34 == 0;
    if ( v31 )
    {
      ++v14;
    }
    else
    {
      v47 = (unsigned int)v120;
      v48 = v34 + 48;
      if ( (unsigned int)v120 >= HIDWORD(v120) )
      {
        sub_16CD150(&dest, v121, 0, 1);
        v47 = (unsigned int)v120;
      }
      *((_BYTE *)dest + v47) = v48;
      LODWORD(v120) = v120 + 1;
    }
  }
  if ( i - (unsigned int)sub_16A57B0(&v110) > 0x40 )
    goto LABEL_68;
  v33 = *(_QWORD *)v110;
  if ( *(_QWORD *)v110 )
    goto LABEL_68;
LABEL_76:
  v35 = v120;
  v36 = v14;
  if ( a3 < (unsigned int)v120 )
  {
    v37 = v120 - a3;
    if ( *((char *)dest + (unsigned int)v120 - a3 - 1) > 52 )
    {
      while ( 1 )
      {
        v38 = (char *)dest + v37;
        if ( *v38 != 57 )
          break;
        if ( (_DWORD)v120 == ++v37 )
          goto LABEL_208;
      }
      v39 = v37 + v14;
      ++*v38;
      v14 += v37;
      if ( v37 != v35 )
      {
        v40 = v120 - v37;
        if ( (unsigned int)v120 != (unsigned __int64)v37 )
        {
          v99 = v39;
          v92 = (unsigned int)memmove(dest, (char *)dest + v37, (unsigned int)v120 - (unsigned __int64)v37);
          v39 = v99;
          v40 = v92 + v40 - (_DWORD)dest;
        }
        LODWORD(v120) = v40;
        v35 = v40;
        v36 = v39;
        goto LABEL_85;
      }
LABEL_208:
      v36 += v37;
      LODWORD(v120) = 0;
      v14 = v36;
      if ( !HIDWORD(v120) )
      {
        sub_16CD150(&dest, v121, 0, 1);
        v33 = (unsigned int)v120;
      }
      *((_BYTE *)dest + v33) = 49;
      v35 = v120 + 1;
      LODWORD(v120) = v120 + 1;
    }
    else
    {
      v78 = v37;
      v79 = (char *)dest + v37;
      if ( (unsigned int)v120 > v37 )
      {
        do
        {
          if ( *v79 != 48 )
          {
            v78 = v37;
            goto LABEL_175;
          }
          ++v37;
          ++v79;
        }
        while ( (_DWORD)v120 != v37 );
        v80 = v37;
        v79 = (char *)dest + v37;
        v78 = v37;
      }
      else
      {
LABEL_175:
        v80 = (unsigned int)v120;
      }
      v36 = v14 + v37;
      v14 += v37;
      v81 = v80 - v78;
      if ( v81 )
      {
        v100 = v81;
        v93 = (unsigned int)memmove(dest, v79, v81);
        LODWORD(v81) = v100 + v93 - (_DWORD)dest;
      }
      LODWORD(v120) = v81;
      v35 = v81;
    }
  }
LABEL_85:
  if ( !a4 )
    goto LABEL_121;
  if ( v14 < 0 )
  {
    v51 = v35 - 1;
    v52 = v35 - 1 + v36;
    if ( v52 < 0 && -v52 > a4 )
      goto LABEL_122;
    v68 = v35 + v14;
    if ( v68 <= 0 )
    {
      LOBYTE(v116) = 48;
      sub_16A3710(a2, &v116);
      LOBYTE(v116) = 46;
      sub_16A3710(a2, &v116);
      if ( v68 != 0 )
      {
        v94 = 1;
        do
        {
          ++v94;
          LOBYTE(v116) = 48;
          sub_16A3710(a2, &v116);
        }
        while ( 1 - v68 != v94 );
      }
      v74 = 0;
    }
    else
    {
      v105 = v68;
      v69 = *(unsigned int *)(a2 + 8);
      v107 = v35;
      v70 = v68;
      v71 = 0;
      v72 = v51;
      do
      {
        v73 = (char *)dest + v72 - v71;
        if ( *(_DWORD *)(a2 + 12) <= (unsigned int)v69 )
        {
          sub_16CD150(a2, a2 + 16, 0, 1);
          v69 = *(unsigned int *)(a2 + 8);
        }
        ++v71;
        *(_BYTE *)(*(_QWORD *)a2 + v69) = *v73;
        v69 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
        *(_DWORD *)(a2 + 8) = v69;
      }
      while ( v71 != v70 );
      LOBYTE(v116) = 46;
      v35 = v107;
      v74 = v105;
      sub_16A3710(a2, &v116);
    }
    if ( v35 != v74 )
    {
      v75 = *(unsigned int *)(a2 + 8);
      for ( j = v74 + 1; ; ++j )
      {
        v77 = (char *)dest + (unsigned int)(v35 - j);
        if ( (unsigned int)v75 >= *(_DWORD *)(a2 + 12) )
        {
          sub_16CD150(a2, a2 + 16, 0, 1);
          v75 = *(unsigned int *)(a2 + 8);
        }
        *(_BYTE *)(*(_QWORD *)a2 + v75) = *v77;
        v75 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
        *(_DWORD *)(a2 + 8) = v75;
        if ( v35 == j )
          break;
      }
    }
    sub_135E100((__int64 *)&v114);
    sub_135E100((__int64 *)&v112);
    if ( dest != v121 )
      _libc_free((unsigned __int64)dest);
    sub_135E100((__int64 *)&v110);
  }
  else
  {
    if ( a4 >= v36 && v36 + v35 <= a3 )
    {
      if ( v35 )
      {
        v41 = *(unsigned int *)(a2 + 8);
        v42 = v35 - 1;
        do
        {
          v43 = (char *)dest + v42;
          if ( (unsigned int)v41 >= *(_DWORD *)(a2 + 12) )
          {
            sub_16CD150(a2, a2 + 16, 0, 1);
            v41 = *(unsigned int *)(a2 + 8);
          }
          *(_BYTE *)(*(_QWORD *)a2 + v41) = *v43;
          v41 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
          *(_DWORD *)(a2 + 8) = v41;
          v44 = v42-- == 0;
        }
        while ( !v44 );
      }
      if ( v36 )
      {
        v45 = *(unsigned int *)(a2 + 8);
        v46 = 0;
        do
        {
          if ( (unsigned int)v45 >= *(_DWORD *)(a2 + 12) )
          {
            sub_16CD150(a2, a2 + 16, 0, 1);
            v45 = *(unsigned int *)(a2 + 8);
          }
          ++v46;
          *(_BYTE *)(*(_QWORD *)a2 + v45) = 48;
          v45 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
          *(_DWORD *)(a2 + 8) = v45;
        }
        while ( v36 != v46 );
      }
      goto LABEL_99;
    }
LABEL_121:
    v51 = v35 - 1;
    v52 = v35 - 1 + v36;
LABEL_122:
    v103 = v51;
    sub_16A3710(a2, (_BYTE *)dest + v51);
    LOBYTE(v116) = 46;
    sub_16A3710(a2, &v116);
    v53 = v103;
    if ( v35 == 1 && a5 )
    {
      LOBYTE(v116) = 48;
      sub_16A3710(a2, &v116);
      v58 = 69;
    }
    else
    {
      v54 = *(unsigned int *)(a2 + 8);
      if ( v35 == 1 )
        goto LABEL_199;
      v104 = v52;
      v55 = 1;
      v56 = v53;
      do
      {
        v57 = (char *)dest + v56 - v55;
        if ( (unsigned int)v54 >= *(_DWORD *)(a2 + 12) )
        {
          sub_16CD150(a2, a2 + 16, 0, 1);
          v54 = *(unsigned int *)(a2 + 8);
        }
        ++v55;
        *(_BYTE *)(*(_QWORD *)a2 + v54) = *v57;
        v54 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
        *(_DWORD *)(a2 + 8) = v54;
      }
      while ( v55 != v35 );
      v52 = v104;
      v58 = 69;
      if ( !a5 )
      {
        v58 = 101;
        if ( a3 > v56 )
        {
LABEL_199:
          v89 = (unsigned int)v54;
          v90 = a3 + 1 - v35;
          v91 = a3 + 1 - v35;
          if ( v90 > *(unsigned int *)(a2 + 12) - (unsigned __int64)(unsigned int)v54 )
          {
            sub_16CD150(a2, a2 + 16, v90 + (unsigned int)v54, 1);
            v89 = *(unsigned int *)(a2 + 8);
            v91 = a3 + 1 - v35;
            LODWORD(v54) = *(_DWORD *)(a2 + 8);
          }
          if ( a3 + 1 != v35 )
          {
            v108 = v91;
            memset((void *)(*(_QWORD *)a2 + v89), 48, v90);
            LODWORD(v54) = *(_DWORD *)(a2 + 8);
            v91 = v108;
          }
          v58 = 101;
          *(_DWORD *)(a2 + 8) = v91 + v54;
        }
      }
    }
    LOBYTE(v116) = v58;
    sub_16A3710(a2, &v116);
    if ( v52 < 0 )
    {
      LOBYTE(v116) = 45;
      v52 = -v52;
    }
    else
    {
      LOBYTE(v116) = 43;
    }
    sub_16A3710(a2, &v116);
    v59 = (unsigned int)v52;
    v117 = 0x600000000LL;
    v60 = v118;
    v116 = v118;
    v61 = 0;
    LOBYTE(v62) = v52 % 0xAu + 48;
    while ( 1 )
    {
      v60[v61] = v62;
      v61 = (unsigned int)(v117 + 1);
      v59 = (3435973837u * v59) >> 35;
      LODWORD(v117) = v117 + 1;
      if ( !(_DWORD)v59 )
        break;
      v62 = (unsigned int)v59 % 0xA + 48;
      if ( HIDWORD(v117) <= (unsigned int)v61 )
      {
        sub_16CD150(&v116, v118, 0, 1);
        v61 = (unsigned int)v117;
        LOBYTE(v62) = (unsigned int)v59 % 0xA + 48;
      }
      v60 = v116;
      v59 = (unsigned int)v59;
    }
    if ( a5 != 1 && (unsigned int)v61 <= 1 )
    {
      v109 = 48;
      sub_16A3710((__int64)&v116, &v109);
      LODWORD(v61) = v117;
    }
    if ( (_DWORD)v61 )
    {
      v63 = *(unsigned int *)(a2 + 8);
      v64 = v61 - 1;
      do
      {
        v65 = &v116[v64];
        if ( *(_DWORD *)(a2 + 12) <= (unsigned int)v63 )
        {
          sub_16CD150(a2, a2 + 16, 0, 1);
          v63 = *(unsigned int *)(a2 + 8);
        }
        *(_BYTE *)(*(_QWORD *)a2 + v63) = *v65;
        v63 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
        *(_DWORD *)(a2 + 8) = v63;
        v44 = v64-- == 0;
      }
      while ( !v44 );
    }
    if ( v116 != v118 )
      _libc_free((unsigned __int64)v116);
LABEL_99:
    if ( v115 > 0x40 && v114 )
      j_j___libc_free_0_0(v114);
    if ( v113 > 0x40 && v112 )
      j_j___libc_free_0_0(v112);
    if ( dest != v121 )
      _libc_free((unsigned __int64)dest);
    if ( v111 > 0x40 )
    {
      if ( v110 )
        j_j___libc_free_0_0(v110);
    }
  }
}
