// Function: sub_C32110
// Address: 0xc32110
//
__int64 __fastcall sub_C32110(
        __int64 *a1,
        __int64 p_dest,
        int a3,
        _QWORD **a4,
        unsigned int a5,
        unsigned int a6,
        char a7)
{
  unsigned int v9; // r9d
  unsigned int v10; // r14d
  unsigned int v11; // eax
  int v12; // r12d
  unsigned int v13; // edx
  unsigned int v14; // eax
  unsigned int v15; // edx
  unsigned int v16; // ebx
  unsigned int v17; // edx
  char v18; // bl
  unsigned int v19; // r15d
  unsigned __int64 v20; // r8
  __int64 v21; // rdx
  signed __int64 v22; // rdx
  unsigned int v24; // ebx
  int v25; // r14d
  unsigned int v26; // eax
  _BYTE *v27; // rdi
  signed __int64 v28; // rax
  __int64 result; // rax
  unsigned int v30; // r14d
  char v31; // r12
  bool v32; // cf
  int v33; // r12d
  void *v34; // rax
  __int64 v35; // rax
  unsigned __int64 v36; // rax
  unsigned int v37; // ebx
  unsigned int v38; // edx
  bool v39; // cc
  char v40; // r15
  signed __int64 v41; // rdx
  unsigned int v42; // eax
  signed int v43; // ebx
  char v44; // r12
  __int64 v45; // rdx
  __int64 v46; // rdx
  __int64 v47; // rcx
  unsigned __int64 v48; // rdx
  __int64 v49; // rdx
  __int64 v50; // rdi
  int v51; // r12d
  unsigned int v52; // ebx
  char v53; // r13
  char v54; // r12
  __int64 v55; // rax
  unsigned __int64 v56; // r12
  unsigned __int64 v57; // rdx
  unsigned __int64 v58; // rdx
  unsigned int v59; // ebx
  char v60; // r12
  unsigned int v61; // r13d
  signed __int64 v62; // rbx
  unsigned int v64; // ecx
  __int64 v66; // rax
  int v67; // r12d
  __int64 v68; // rdx
  int v69; // ebx
  int v70; // r13d
  int v71; // r14d
  char v72; // r12
  __int64 v73; // rcx
  int v74; // ebx
  int i; // ebx
  char v76; // r12
  __int64 v77; // rdi
  __int64 v78; // r14
  _BYTE *v79; // rax
  int v80; // ebx
  unsigned int v81; // edx
  int v82; // eax
  unsigned __int64 v83; // rdx
  size_t v84; // r13
  _BYTE *v85; // rax
  int v86; // r13d
  __int64 v87; // rax
  __int64 v88; // rcx
  int v89; // ebx
  unsigned __int64 v90; // rdx
  unsigned __int64 v91; // [rsp+0h] [rbp-1C0h]
  unsigned int v93; // [rsp+18h] [rbp-1A8h]
  unsigned int v94; // [rsp+18h] [rbp-1A8h]
  unsigned int v95; // [rsp+20h] [rbp-1A0h]
  signed int v96; // [rsp+20h] [rbp-1A0h]
  int v97; // [rsp+20h] [rbp-1A0h]
  unsigned __int64 v98; // [rsp+20h] [rbp-1A0h]
  unsigned int v99; // [rsp+20h] [rbp-1A0h]
  unsigned __int64 v100; // [rsp+20h] [rbp-1A0h]
  unsigned __int64 v101; // [rsp+20h] [rbp-1A0h]
  unsigned __int64 v102; // [rsp+20h] [rbp-1A0h]
  unsigned int v103; // [rsp+20h] [rbp-1A0h]
  unsigned __int64 v104; // [rsp+20h] [rbp-1A0h]
  int v106; // [rsp+28h] [rbp-198h]
  unsigned __int64 v107; // [rsp+28h] [rbp-198h]
  unsigned __int64 v108; // [rsp+28h] [rbp-198h]
  unsigned __int64 v109; // [rsp+28h] [rbp-198h]
  unsigned __int64 v110; // [rsp+28h] [rbp-198h]
  unsigned __int64 v111; // [rsp+28h] [rbp-198h]
  unsigned __int64 v112; // [rsp+28h] [rbp-198h]
  __int64 v113; // [rsp+30h] [rbp-190h] BYREF
  unsigned int v114; // [rsp+38h] [rbp-188h]
  __int64 v115; // [rsp+40h] [rbp-180h] BYREF
  unsigned int v116; // [rsp+48h] [rbp-178h]
  __int64 v117; // [rsp+50h] [rbp-170h] BYREF
  unsigned __int64 v118; // [rsp+58h] [rbp-168h]
  unsigned __int64 v119; // [rsp+60h] [rbp-160h]
  char v120[8]; // [rsp+68h] [rbp-158h] BYREF
  void *dest; // [rsp+70h] [rbp-150h] BYREF
  signed __int64 v122; // [rsp+78h] [rbp-148h]
  unsigned __int64 v123; // [rsp+80h] [rbp-140h]
  _BYTE v124[312]; // [rsp+88h] [rbp-138h] BYREF

  v9 = *((_DWORD *)a4 + 2);
  v10 = v9;
  if ( (_BYTE)p_dest )
  {
    v66 = a1[1];
    if ( v66 + 1 > (unsigned __int64)a1[2] )
    {
      p_dest = (__int64)(a1 + 3);
      v103 = *((_DWORD *)a4 + 2);
      sub_C8D290(a1, a1 + 3, v66 + 1, 1);
      v66 = a1[1];
      v9 = v103;
    }
    *(_BYTE *)(*a1 + v66) = 45;
    ++a1[1];
    v10 = *((_DWORD *)a4 + 2);
  }
  if ( !a5 )
    a5 = (int)(59 * v9) / 196 + 2;
  if ( v10 <= 0x40 )
  {
    _RAX = (unsigned __int64)*a4;
    if ( *a4 )
    {
      v64 = v10;
      __asm { tzcnt   rdx, rax }
      if ( (unsigned int)_RDX <= v10 )
        v64 = _RDX;
      v12 = v64 + a3;
      if ( (unsigned int)_RDX < v10 )
      {
        *a4 = (_QWORD *)(_RAX >> v64);
        goto LABEL_135;
      }
    }
    else
    {
      v12 = v10 + a3;
    }
    *a4 = 0;
LABEL_135:
    if ( !v12 )
      goto LABEL_136;
    goto LABEL_57;
  }
  v95 = v9;
  v11 = sub_C44590(a4);
  v12 = v11 + a3;
  p_dest = v11;
  sub_C482E0(a4, v11);
  if ( !v12 )
    goto LABEL_6;
  v9 = v95;
LABEL_57:
  if ( v12 > 0 )
  {
    p_dest = (__int64)a4;
    sub_C449B0(&dest, a4, v9 + v12);
    if ( *((_DWORD *)a4 + 2) > 0x40u && *a4 )
      j_j___libc_free_0_0(*a4);
    v34 = dest;
    v10 = v122;
    *a4 = dest;
    *((_DWORD *)a4 + 2) = v10;
    if ( v10 > 0x40 )
    {
      p_dest = (unsigned int)v12;
      v12 = 0;
      sub_C47690(a4, p_dest);
      goto LABEL_6;
    }
    v35 = (_QWORD)v34 << v12;
    if ( v12 == v10 )
      v35 = 0;
    if ( !v10 )
    {
      *a4 = 0;
      v12 = 0;
      goto LABEL_10;
    }
    v36 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v10) & v35;
    v12 = 0;
    *a4 = (_QWORD *)v36;
    goto LABEL_137;
  }
  p_dest = (__int64)a4;
  v80 = -v12;
  v99 = (-137 * v12 + 136) / 0x3Bu + v9;
  sub_C449B0(&dest, a4, v99);
  v81 = v99;
  if ( *((_DWORD *)a4 + 2) > 0x40u && *a4 )
  {
    j_j___libc_free_0_0(*a4);
    v81 = v99;
  }
  *a4 = dest;
  v82 = v122;
  LODWORD(v122) = v81;
  *((_DWORD *)a4 + 2) = v82;
  if ( v81 > 0x40 )
  {
    p_dest = 5;
    sub_C43690(&dest, 5, 0);
  }
  else
  {
    dest = (void *)5;
  }
  while ( (v80 & 1) == 0 )
  {
    v80 >>= 1;
    if ( !v80 )
      goto LABEL_181;
LABEL_178:
    p_dest = (__int64)&dest;
    sub_C47360(&dest, &dest);
  }
  p_dest = (__int64)&dest;
  sub_C47360(a4, &dest);
  v80 >>= 1;
  if ( v80 )
    goto LABEL_178;
LABEL_181:
  if ( (unsigned int)v122 > 0x40 && dest )
    j_j___libc_free_0_0(dest);
LABEL_6:
  v10 = *((_DWORD *)a4 + 2);
  if ( v10 > 0x40 )
  {
    v13 = v10 - sub_C444A0(a4);
    goto LABEL_8;
  }
LABEL_136:
  v36 = (unsigned __int64)*a4;
LABEL_137:
  if ( !v36 )
    goto LABEL_10;
  _BitScanReverse64(&v36, v36);
  v13 = 64 - (v36 ^ 0x3F);
LABEL_8:
  v14 = (196 * a5 + 58) / 0x3B;
  if ( v14 >= v13 )
    goto LABEL_10;
  v15 = 59 * (v13 - v14);
  if ( v15 <= 0xC3 )
    goto LABEL_10;
  v116 = v10;
  v37 = v15 / 0xC4;
  v12 += v15 / 0xC4;
  if ( v10 <= 0x40 )
  {
    v115 = 1;
    LODWORD(v118) = *((_DWORD *)a4 + 2);
LABEL_68:
    v117 = 10;
    goto LABEL_71;
  }
  sub_C43690(&v115, 1, 0);
  LODWORD(v118) = *((_DWORD *)a4 + 2);
  if ( (unsigned int)v118 <= 0x40 )
    goto LABEL_68;
  sub_C43690(&v117, 10, 0);
LABEL_71:
  while ( 2 )
  {
    if ( (v37 & 1) == 0 )
    {
      v37 >>= 1;
      if ( !v37 )
        break;
      goto LABEL_70;
    }
    sub_C47360(&v115, &v117);
    v37 >>= 1;
    if ( v37 )
    {
LABEL_70:
      sub_C47360(&v117, &v117);
      continue;
    }
    break;
  }
  sub_C4A1D0(&dest, a4, &v115);
  if ( *((_DWORD *)a4 + 2) > 0x40u && *a4 )
    j_j___libc_free_0_0(*a4);
  v38 = v122;
  *a4 = dest;
  *((_DWORD *)a4 + 2) = v38;
  if ( v38 > 0x40 )
    sub_C444A0(a4);
  p_dest = (__int64)a4;
  sub_C44740(&dest, a4);
  if ( *((_DWORD *)a4 + 2) > 0x40u && *a4 )
    j_j___libc_free_0_0(*a4);
  v39 = (unsigned int)v118 <= 0x40;
  *a4 = dest;
  *((_DWORD *)a4 + 2) = v122;
  if ( !v39 && v117 )
    j_j___libc_free_0_0(v117);
  if ( v116 > 0x40 && v115 )
    j_j___libc_free_0_0(v115);
LABEL_10:
  v16 = *((_DWORD *)a4 + 2);
  v122 = 0;
  dest = v124;
  v123 = 256;
  if ( v16 <= 3 )
  {
    p_dest = (__int64)a4;
    sub_C449B0(&v117, a4, 4);
    if ( *((_DWORD *)a4 + 2) > 0x40u && *a4 )
      j_j___libc_free_0_0(*a4);
    v17 = v118;
    v16 = 4;
    v114 = 4;
    *a4 = (_QWORD *)v117;
    *((_DWORD *)a4 + 2) = v17;
  }
  else
  {
    v114 = v16;
    if ( v16 > 0x40 )
    {
      sub_C43690(&v113, 10, 0);
      p_dest = 0;
      v116 = v16;
      sub_C43690(&v115, 0, 0);
      v17 = *((_DWORD *)a4 + 2);
      goto LABEL_13;
    }
    v17 = v16;
  }
  v113 = 10;
  v116 = v16;
  v115 = 0;
LABEL_13:
  v18 = 1;
  v19 = v17;
  while ( 2 )
  {
    if ( v19 <= 0x40 )
    {
      v20 = (unsigned __int64)*a4;
      if ( !*a4 )
        break;
      goto LABEL_15;
    }
    if ( v19 - (unsigned int)sub_C444A0(a4) > 0x40 || (v20 = **a4) != 0 )
    {
LABEL_15:
      p_dest = (__int64)&v113;
      sub_C4BFE0(a4, &v113, a4, &v115);
      LODWORD(v21) = v115;
      if ( v116 > 0x40 )
        v21 = *(_QWORD *)v115;
      v18 &= (_DWORD)v21 == 0;
      if ( v18 )
      {
        ++v12;
      }
      else
      {
        v40 = v21 + 48;
        v41 = v122;
        if ( v122 + 1 > v123 )
        {
          p_dest = (__int64)v124;
          sub_C8D290(&dest, v124, v122 + 1, 1);
          v41 = v122;
        }
        *((_BYTE *)dest + v41) = v40;
        ++v122;
      }
      v19 = *((_DWORD *)a4 + 2);
      continue;
    }
    break;
  }
  LODWORD(v22) = v122;
  v24 = v12;
  v25 = v122;
  if ( a5 < (unsigned int)v122 )
  {
    v26 = v122 - a5;
    p_dest = (unsigned int)v122 - a5 - 1;
    if ( *((char *)dest + p_dest) > 52 )
    {
      while ( 1 )
      {
        if ( (_DWORD)v122 == v26 )
          goto LABEL_29;
        p_dest = v26;
        v27 = (char *)dest + v26;
        if ( *v27 != 57 )
          break;
        ++v26;
      }
      v61 = v12 + v26;
      ++*v27;
      v12 += v26;
      if ( v25 == v26 )
      {
LABEL_29:
        v122 = 0;
        v24 += v26;
        v28 = 0;
        v12 = v24;
        if ( !v123 )
        {
          p_dest = (__int64)v124;
          v104 = v20;
          sub_C8D290(&dest, v124, 1, 1);
          v28 = v122;
          v20 = v104;
        }
        *((_BYTE *)dest + v28) = 49;
        v22 = v122 + 1;
        goto LABEL_32;
      }
      v22 = 0;
      v62 = v122 - v26;
      if ( v122 != v26 )
      {
        v100 = v20;
        p_dest = (__int64)dest + v26;
        v85 = memmove(dest, (const void *)p_dest, v122 - v26);
        v20 = v100;
        v22 = &v85[v62] - (_BYTE *)dest;
      }
      v122 = v22;
      v25 = v22;
      v24 = v61;
    }
    else
    {
      v77 = v26;
      p_dest = (__int64)dest + v26;
      if ( (unsigned int)v122 > v26 )
      {
        do
        {
          if ( *(_BYTE *)p_dest != 48 )
          {
            v77 = v26;
            goto LABEL_167;
          }
          ++v26;
          ++p_dest;
        }
        while ( (_DWORD)v122 != v26 );
        v77 = v26;
        p_dest = (__int64)dest + v26;
      }
LABEL_167:
      v24 = v12 + v26;
      v78 = v122 - v77;
      v12 += v26;
      v22 = 0;
      if ( v122 != v77 )
      {
        v98 = v20;
        v79 = memmove(dest, (const void *)p_dest, v122 - v77);
        v20 = v98;
        v22 = &v79[v78] - (_BYTE *)dest;
      }
LABEL_32:
      v122 = v22;
      v25 = v22;
    }
  }
  if ( !a6 )
    goto LABEL_92;
  if ( v12 < 0 )
  {
    v42 = v25 - 1;
    v43 = v25 - 1 + v24;
    if ( v43 < 0 && -v43 > a6 )
      goto LABEL_93;
    v67 = v22 + v12;
    v68 = a1[1];
    v69 = v67;
    if ( v67 <= 0 )
    {
      v86 = 1 - v67;
      if ( v68 + 1 > (unsigned __int64)a1[2] )
      {
        p_dest = (__int64)(a1 + 3);
        sub_C8D290(a1, a1 + 3, v68 + 1, 1);
        v68 = a1[1];
      }
      *(_BYTE *)(*a1 + v68) = 48;
      v87 = a1[1];
      v88 = v87 + 1;
      a1[1] = v87 + 1;
      if ( v87 + 2 > (unsigned __int64)a1[2] )
      {
        p_dest = (__int64)(a1 + 3);
        sub_C8D290(a1, a1 + 3, v87 + 2, 1);
        v88 = a1[1];
      }
      v89 = 1;
      *(_BYTE *)(*a1 + v88) = 46;
      result = a1[1] + 1;
      for ( a1[1] = result; v86 != v89; a1[1] = result )
      {
        if ( result + 1 > (unsigned __int64)a1[2] )
        {
          p_dest = (__int64)(a1 + 3);
          sub_C8D290(a1, a1 + 3, result + 1, 1);
          result = a1[1];
        }
        ++v89;
        *(_BYTE *)(*a1 + result) = 48;
        result = a1[1] + 1;
      }
      v74 = 0;
    }
    else
    {
      v106 = v67;
      v70 = 0;
      p_dest = (__int64)(a1 + 3);
      v97 = v25;
      v71 = v25 - 1;
      do
      {
        v72 = *((_BYTE *)dest + (unsigned int)(v71 - v70));
        if ( v68 + 1 > (unsigned __int64)a1[2] )
        {
          sub_C8D290(a1, p_dest, v68 + 1, 1);
          v68 = a1[1];
        }
        ++v70;
        *(_BYTE *)(*a1 + v68) = v72;
        v73 = a1[1];
        v68 = v73 + 1;
        a1[1] = v73 + 1;
      }
      while ( v70 != v69 );
      v74 = v106;
      v25 = v97;
      if ( v73 + 2 > (unsigned __int64)a1[2] )
      {
        p_dest = (__int64)(a1 + 3);
        sub_C8D290(a1, a1 + 3, v73 + 2, 1);
        v68 = a1[1];
      }
      result = *a1;
      *(_BYTE *)(*a1 + v68) = 46;
      ++a1[1];
    }
    if ( v74 != v25 )
    {
      result = a1[1];
      for ( i = v74 + 1; ; ++i )
      {
        v76 = *((_BYTE *)dest + (unsigned int)(v25 - i));
        if ( result + 1 > (unsigned __int64)a1[2] )
        {
          p_dest = (__int64)(a1 + 3);
          sub_C8D290(a1, a1 + 3, result + 1, 1);
          result = a1[1];
        }
        *(_BYTE *)(*a1 + result) = v76;
        result = a1[1] + 1;
        a1[1] = result;
        if ( v25 == i )
          break;
      }
    }
  }
  else
  {
    if ( a6 >= v24 )
    {
      result = v24 + v25;
      if ( (unsigned int)result <= a5 )
      {
        if ( v25 )
        {
          result = a1[1];
          v30 = v25 - 1;
          do
          {
            v31 = *((_BYTE *)dest + v30);
            if ( result + 1 > (unsigned __int64)a1[2] )
            {
              p_dest = (__int64)(a1 + 3);
              sub_C8D290(a1, a1 + 3, result + 1, 1);
              result = a1[1];
            }
            *(_BYTE *)(*a1 + result) = v31;
            result = a1[1] + 1;
            a1[1] = result;
            v32 = v30-- == 0;
          }
          while ( !v32 );
        }
        if ( v24 )
        {
          result = a1[1];
          v33 = 0;
          do
          {
            if ( result + 1 > (unsigned __int64)a1[2] )
            {
              p_dest = (__int64)(a1 + 3);
              sub_C8D290(a1, a1 + 3, result + 1, 1);
              result = a1[1];
            }
            ++v33;
            *(_BYTE *)(*a1 + result) = 48;
            result = a1[1] + 1;
            a1[1] = result;
          }
          while ( v33 != v24 );
        }
        goto LABEL_47;
      }
    }
LABEL_92:
    v42 = v25 - 1;
    v43 = v25 - 1 + v24;
LABEL_93:
    v44 = *((_BYTE *)dest + v42);
    v45 = a1[1];
    if ( v45 + 1 > (unsigned __int64)a1[2] )
    {
      p_dest = (__int64)(a1 + 3);
      v94 = v42;
      v102 = v20;
      sub_C8D290(a1, a1 + 3, v45 + 1, 1);
      v45 = a1[1];
      v42 = v94;
      v20 = v102;
    }
    *(_BYTE *)(*a1 + v45) = v44;
    v46 = a1[1];
    v47 = v46 + 1;
    v48 = v46 + 2;
    a1[1] = v47;
    if ( v48 > a1[2] )
    {
      p_dest = (__int64)(a1 + 3);
      v93 = v42;
      v101 = v20;
      sub_C8D290(a1, a1 + 3, v48, 1);
      v47 = a1[1];
      v42 = v93;
      v20 = v101;
    }
    *(_BYTE *)(*a1 + v47) = 46;
    v49 = a1[1];
    v50 = v49 + 1;
    a1[1] = v49 + 1;
    if ( v25 == 1 )
    {
      if ( !a7 )
        goto LABEL_105;
      v83 = v49 + 2;
      if ( v83 > a1[2] )
      {
        p_dest = (__int64)(a1 + 3);
        v111 = v20;
        sub_C8D290(a1, a1 + 3, v83, 1);
        v50 = a1[1];
        v20 = v111;
      }
      v54 = 69;
      *(_BYTE *)(*a1 + v50) = 48;
      v50 = a1[1] + 1;
      a1[1] = v50;
    }
    else
    {
      v96 = v43;
      v51 = 1;
      v52 = v42;
      p_dest = (__int64)(a1 + 3);
      do
      {
        v53 = *((_BYTE *)dest + v52 - v51);
        if ( v50 + 1 > (unsigned __int64)a1[2] )
        {
          v91 = v20;
          sub_C8D290(a1, p_dest, v50 + 1, 1);
          v50 = a1[1];
          v20 = v91;
        }
        ++v51;
        *(_BYTE *)(*a1 + v50) = v53;
        v50 = a1[1] + 1;
        a1[1] = v50;
      }
      while ( v51 != v25 );
      v42 = v52;
      v54 = 69;
      v43 = v96;
      if ( !a7 )
      {
LABEL_105:
        v54 = 101;
        if ( a5 > v42 )
        {
          v84 = a5 + 1 - v25;
          if ( a1[2] < v84 + v50 )
          {
            p_dest = (__int64)(a1 + 3);
            v112 = v20;
            sub_C8D290(a1, a1 + 3, v84 + v50, 1);
            v50 = a1[1];
            v20 = v112;
          }
          if ( v84 )
          {
            p_dest = 48;
            v108 = v20;
            memset((void *)(*a1 + v50), 48, v84);
            v50 = a1[1];
            v20 = v108;
          }
          v50 += v84;
          v54 = 101;
          a1[1] = v50;
        }
      }
    }
    if ( v50 + 1 > (unsigned __int64)a1[2] )
    {
      p_dest = (__int64)(a1 + 3);
      v107 = v20;
      sub_C8D290(a1, a1 + 3, v50 + 1, 1);
      v50 = a1[1];
      v20 = v107;
    }
    *(_BYTE *)(*a1 + v50) = v54;
    v55 = a1[1];
    a1[1] = v55 + 1;
    if ( v43 < 0 )
    {
      if ( a1[2] < (unsigned __int64)(v55 + 2) )
      {
        p_dest = (__int64)(a1 + 3);
        v43 = -v43;
        v109 = v20;
        sub_C8D290(a1, a1 + 3, v55 + 2, 1);
        v20 = v109;
        *(_BYTE *)(*a1 + a1[1]) = 45;
      }
      else
      {
        v43 = -v43;
        *(_BYTE *)(*a1 + v55 + 1) = 45;
      }
      ++a1[1];
    }
    else
    {
      if ( a1[2] < (unsigned __int64)(v55 + 2) )
      {
        p_dest = (__int64)(a1 + 3);
        v110 = v20;
        sub_C8D290(a1, a1 + 3, v55 + 2, 1);
        v20 = v110;
        *(_BYTE *)(*a1 + a1[1]) = 43;
      }
      else
      {
        *(_BYTE *)(*a1 + v55 + 1) = 43;
      }
      ++a1[1];
    }
    v56 = (unsigned int)v43;
    v118 = 0;
    v117 = (__int64)v120;
    v119 = 6;
    result = (__int64)v120;
    LOBYTE(v43) = v43 % 0xAu + 48;
    while ( 1 )
    {
      *(_BYTE *)(result + v20) = v43;
      v58 = v118;
      v20 = v118 + 1;
      v56 = (3435973837u * v56) >> 35;
      ++v118;
      if ( !(_DWORD)v56 )
        break;
      v57 = v58 + 2;
      v43 = (unsigned int)v56 % 0xA + 48;
      if ( v57 > v119 )
      {
        p_dest = (__int64)v120;
        sub_C8D290(&v117, v120, v57, 1);
        v20 = v118;
      }
      result = v117;
      v56 = (unsigned int)v56;
    }
    if ( v20 <= 1 && a7 != 1 )
    {
      v90 = v58 + 2;
      if ( v90 > v119 )
      {
        p_dest = (__int64)v120;
        sub_C8D290(&v117, v120, v90, 1);
        v20 = v118;
      }
      *(_BYTE *)(v117 + v20) = 48;
      result = v118;
      LODWORD(v20) = ++v118;
    }
    if ( (_DWORD)v20 )
    {
      result = a1[1];
      v59 = v20 - 1;
      do
      {
        v60 = *(_BYTE *)(v117 + v59);
        if ( result + 1 > (unsigned __int64)a1[2] )
        {
          p_dest = (__int64)(a1 + 3);
          sub_C8D290(a1, a1 + 3, result + 1, 1);
          result = a1[1];
        }
        *(_BYTE *)(*a1 + result) = v60;
        result = a1[1] + 1;
        a1[1] = result;
        v32 = v59-- == 0;
      }
      while ( !v32 );
    }
    if ( (char *)v117 != v120 )
      result = _libc_free(v117, p_dest);
  }
LABEL_47:
  if ( v116 > 0x40 && v115 )
    result = j_j___libc_free_0_0(v115);
  if ( v114 > 0x40 && v113 )
    result = j_j___libc_free_0_0(v113);
  if ( dest != v124 )
    return _libc_free(dest, p_dest);
  return result;
}
