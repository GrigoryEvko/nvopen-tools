// Function: sub_107CBF0
// Address: 0x107cbf0
//
__int64 __fastcall sub_107CBF0(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // r8
  __int64 v5; // r9
  int v6; // eax
  __int64 v7; // rsi
  __int64 v8; // rdx
  _QWORD *v9; // rax
  _QWORD *i; // rdx
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rdi
  int v14; // r15d
  unsigned int v15; // eax
  _QWORD *v16; // rbx
  __int64 v17; // rdx
  __int64 v18; // r14
  _QWORD *v19; // r13
  __int64 v20; // rdi
  int v21; // r14d
  _QWORD *v22; // r15
  __int64 v23; // rcx
  unsigned int v24; // edx
  __int64 v25; // r13
  _QWORD *v26; // rbx
  __int64 v27; // rdx
  __int64 v28; // rdi
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  _QWORD *v33; // r13
  _QWORD *v34; // rbx
  _QWORD *v35; // rdi
  __int64 v36; // rax
  __int64 v37; // r13
  __int64 v38; // rbx
  __int64 v39; // rdi
  int v40; // eax
  __int64 v41; // rdx
  _QWORD *v42; // rax
  _QWORD *m; // rdx
  unsigned int v45; // ecx
  unsigned int v46; // eax
  _QWORD *v47; // rdi
  int v48; // ebx
  unsigned __int64 v49; // rax
  unsigned __int64 v50; // rdi
  _QWORD *v51; // rax
  __int64 v52; // rdx
  _QWORD *n; // rdx
  unsigned int v54; // ecx
  unsigned int v55; // eax
  _QWORD *v56; // rdi
  int v57; // ebx
  _QWORD *v58; // rax
  __int64 v59; // rdi
  _QWORD *v60; // rdi
  __int64 v61; // rdx
  int v62; // ebx
  unsigned int v63; // r14d
  unsigned int v64; // eax
  unsigned __int64 v65; // rdx
  unsigned __int64 v66; // rax
  unsigned int v67; // edx
  int v68; // ebx
  unsigned int v69; // r15d
  unsigned int v70; // eax
  _QWORD *v71; // rdi
  unsigned __int64 v72; // rax
  unsigned __int64 v73; // rdi
  _QWORD *v74; // rax
  __int64 v75; // rdx
  _QWORD *k; // rdx
  unsigned __int64 v77; // rax
  unsigned __int64 v78; // rdi
  _QWORD *v79; // rax
  __int64 v80; // rdx
  _QWORD *j; // rdx
  _QWORD *v82; // rax
  _QWORD *v83; // rax
  _QWORD v84[2]; // [rsp+10h] [rbp-70h] BYREF
  char v85; // [rsp+20h] [rbp-60h] BYREF
  _QWORD v86[2]; // [rsp+28h] [rbp-58h] BYREF
  _BYTE v87[16]; // [rsp+38h] [rbp-48h] BYREF
  __int64 v88; // [rsp+48h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 120);
  if ( v2 != *(_QWORD *)(a1 + 128) )
    *(_QWORD *)(a1 + 128) = v2;
  v3 = *(_QWORD *)(a1 + 144);
  if ( v3 != *(_QWORD *)(a1 + 152) )
    *(_QWORD *)(a1 + 152) = v3;
  sub_107C4D0(a1 + 168);
  sub_107C4D0(a1 + 232);
  sub_107C4D0(a1 + 264);
  sub_107C4D0(a1 + 200);
  v6 = *(_DWORD *)(a1 + 312);
  ++*(_QWORD *)(a1 + 296);
  if ( !v6 )
  {
    v7 = *(unsigned int *)(a1 + 316);
    if ( !(_DWORD)v7 )
      goto LABEL_11;
    v8 = *(unsigned int *)(a1 + 320);
    if ( (unsigned int)v8 > 0x40 )
    {
      v7 = 32LL * (unsigned int)v8;
      sub_C7D6A0(*(_QWORD *)(a1 + 304), v7, 8);
      *(_QWORD *)(a1 + 304) = 0;
      *(_QWORD *)(a1 + 312) = 0;
      *(_DWORD *)(a1 + 320) = 0;
      goto LABEL_11;
    }
    goto LABEL_8;
  }
  v54 = 4 * v6;
  v7 = 64;
  v8 = *(unsigned int *)(a1 + 320);
  if ( (unsigned int)(4 * v6) < 0x40 )
    v54 = 64;
  if ( (unsigned int)v8 <= v54 )
  {
LABEL_8:
    v9 = *(_QWORD **)(a1 + 304);
    for ( i = &v9[4 * v8]; i != v9; v9 += 4 )
      *v9 = -4096;
    *(_QWORD *)(a1 + 312) = 0;
    goto LABEL_11;
  }
  v55 = v6 - 1;
  if ( v55 )
  {
    _BitScanReverse(&v55, v55);
    v56 = *(_QWORD **)(a1 + 304);
    v57 = 1 << (33 - (v55 ^ 0x1F));
    if ( v57 < 64 )
      v57 = 64;
    if ( (_DWORD)v8 == v57 )
    {
      *(_QWORD *)(a1 + 312) = 0;
      v58 = &v56[4 * (unsigned int)v8];
      do
      {
        if ( v56 )
          *v56 = -4096;
        v56 += 4;
      }
      while ( v58 != v56 );
      goto LABEL_11;
    }
  }
  else
  {
    v56 = *(_QWORD **)(a1 + 304);
    v57 = 64;
  }
  sub_C7D6A0((__int64)v56, 32LL * (unsigned int)v8, 8);
  v7 = 8;
  v77 = ((((((((4 * v57 / 3u + 1) | ((unsigned __int64)(4 * v57 / 3u + 1) >> 1)) >> 2)
           | (4 * v57 / 3u + 1)
           | ((unsigned __int64)(4 * v57 / 3u + 1) >> 1)) >> 4)
         | (((4 * v57 / 3u + 1) | ((unsigned __int64)(4 * v57 / 3u + 1) >> 1)) >> 2)
         | (4 * v57 / 3u + 1)
         | ((unsigned __int64)(4 * v57 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v57 / 3u + 1) | ((unsigned __int64)(4 * v57 / 3u + 1) >> 1)) >> 2)
         | (4 * v57 / 3u + 1)
         | ((unsigned __int64)(4 * v57 / 3u + 1) >> 1)) >> 4)
       | (((4 * v57 / 3u + 1) | ((unsigned __int64)(4 * v57 / 3u + 1) >> 1)) >> 2)
       | (4 * v57 / 3u + 1)
       | ((unsigned __int64)(4 * v57 / 3u + 1) >> 1)) >> 16;
  v78 = (v77
       | (((((((4 * v57 / 3u + 1) | ((unsigned __int64)(4 * v57 / 3u + 1) >> 1)) >> 2)
           | (4 * v57 / 3u + 1)
           | ((unsigned __int64)(4 * v57 / 3u + 1) >> 1)) >> 4)
         | (((4 * v57 / 3u + 1) | ((unsigned __int64)(4 * v57 / 3u + 1) >> 1)) >> 2)
         | (4 * v57 / 3u + 1)
         | ((unsigned __int64)(4 * v57 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v57 / 3u + 1) | ((unsigned __int64)(4 * v57 / 3u + 1) >> 1)) >> 2)
         | (4 * v57 / 3u + 1)
         | ((unsigned __int64)(4 * v57 / 3u + 1) >> 1)) >> 4)
       | (((4 * v57 / 3u + 1) | ((unsigned __int64)(4 * v57 / 3u + 1) >> 1)) >> 2)
       | (4 * v57 / 3u + 1)
       | ((unsigned __int64)(4 * v57 / 3u + 1) >> 1))
      + 1;
  *(_DWORD *)(a1 + 320) = v78;
  v79 = (_QWORD *)sub_C7D670(32 * v78, 8);
  v80 = *(unsigned int *)(a1 + 320);
  *(_QWORD *)(a1 + 312) = 0;
  *(_QWORD *)(a1 + 304) = v79;
  for ( j = &v79[4 * v80]; j != v79; v79 += 4 )
  {
    if ( v79 )
      *v79 = -4096;
  }
LABEL_11:
  v11 = *(_QWORD *)(a1 + 328);
  if ( v11 != *(_QWORD *)(a1 + 336) )
    *(_QWORD *)(a1 + 336) = v11;
  v12 = *(_QWORD *)(a1 + 352);
  *(_QWORD *)(a1 + 352) = 0;
  if ( v12 )
  {
    v7 = 32;
    j_j___libc_free_0(v12, 32);
  }
  v13 = *(_QWORD *)(a1 + 360);
  *(_QWORD *)(a1 + 360) = 0;
  if ( v13 )
  {
    v7 = 32;
    j_j___libc_free_0(v13, 32);
  }
  v14 = *(_DWORD *)(a1 + 384);
  ++*(_QWORD *)(a1 + 368);
  if ( v14 || *(_DWORD *)(a1 + 388) )
  {
    v15 = 4 * v14;
    v16 = *(_QWORD **)(a1 + 376);
    v17 = *(unsigned int *)(a1 + 392);
    v18 = 32 * v17;
    if ( (unsigned int)(4 * v14) < 0x40 )
      v15 = 64;
    v19 = &v16[(unsigned __int64)v18 / 8];
    if ( (unsigned int)v17 <= v15 )
    {
      for ( ; v16 != v19; v16 += 4 )
      {
        if ( *v16 != -4096 )
        {
          if ( *v16 != -8192 )
          {
            v20 = v16[1];
            if ( v20 )
            {
              v7 = v16[3] - v20;
              j_j___libc_free_0(v20, v7);
            }
          }
          *v16 = -4096;
        }
      }
      goto LABEL_29;
    }
    while ( 1 )
    {
      while ( *v16 == -8192 )
      {
LABEL_84:
        v16 += 4;
        if ( v16 == v19 )
          goto LABEL_101;
      }
      if ( *v16 != -4096 )
      {
        v59 = v16[1];
        if ( v59 )
        {
          v7 = v16[3] - v59;
          j_j___libc_free_0(v59, v7);
        }
        goto LABEL_84;
      }
      v16 += 4;
      if ( v16 == v19 )
      {
LABEL_101:
        v67 = *(_DWORD *)(a1 + 392);
        if ( v14 )
        {
          v68 = 64;
          v69 = v14 - 1;
          if ( v69 )
          {
            _BitScanReverse(&v70, v69);
            v68 = 1 << (33 - (v70 ^ 0x1F));
            if ( v68 < 64 )
              v68 = 64;
          }
          v71 = *(_QWORD **)(a1 + 376);
          if ( v67 == v68 )
          {
            *(_QWORD *)(a1 + 384) = 0;
            v83 = &v71[4 * v67];
            do
            {
              if ( v71 )
                *v71 = -4096;
              v71 += 4;
            }
            while ( v83 != v71 );
          }
          else
          {
            sub_C7D6A0((__int64)v71, v18, 8);
            v7 = 8;
            v72 = ((((((((4 * v68 / 3u + 1) | ((unsigned __int64)(4 * v68 / 3u + 1) >> 1)) >> 2)
                     | (4 * v68 / 3u + 1)
                     | ((unsigned __int64)(4 * v68 / 3u + 1) >> 1)) >> 4)
                   | (((4 * v68 / 3u + 1) | ((unsigned __int64)(4 * v68 / 3u + 1) >> 1)) >> 2)
                   | (4 * v68 / 3u + 1)
                   | ((unsigned __int64)(4 * v68 / 3u + 1) >> 1)) >> 8)
                 | (((((4 * v68 / 3u + 1) | ((unsigned __int64)(4 * v68 / 3u + 1) >> 1)) >> 2)
                   | (4 * v68 / 3u + 1)
                   | ((unsigned __int64)(4 * v68 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v68 / 3u + 1) | ((unsigned __int64)(4 * v68 / 3u + 1) >> 1)) >> 2)
                 | (4 * v68 / 3u + 1)
                 | ((unsigned __int64)(4 * v68 / 3u + 1) >> 1)) >> 16;
            v73 = (v72
                 | (((((((4 * v68 / 3u + 1) | ((unsigned __int64)(4 * v68 / 3u + 1) >> 1)) >> 2)
                     | (4 * v68 / 3u + 1)
                     | ((unsigned __int64)(4 * v68 / 3u + 1) >> 1)) >> 4)
                   | (((4 * v68 / 3u + 1) | ((unsigned __int64)(4 * v68 / 3u + 1) >> 1)) >> 2)
                   | (4 * v68 / 3u + 1)
                   | ((unsigned __int64)(4 * v68 / 3u + 1) >> 1)) >> 8)
                 | (((((4 * v68 / 3u + 1) | ((unsigned __int64)(4 * v68 / 3u + 1) >> 1)) >> 2)
                   | (4 * v68 / 3u + 1)
                   | ((unsigned __int64)(4 * v68 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v68 / 3u + 1) | ((unsigned __int64)(4 * v68 / 3u + 1) >> 1)) >> 2)
                 | (4 * v68 / 3u + 1)
                 | ((unsigned __int64)(4 * v68 / 3u + 1) >> 1))
                + 1;
            *(_DWORD *)(a1 + 392) = v73;
            v74 = (_QWORD *)sub_C7D670(32 * v73, 8);
            v75 = *(unsigned int *)(a1 + 392);
            *(_QWORD *)(a1 + 384) = 0;
            *(_QWORD *)(a1 + 376) = v74;
            for ( k = &v74[4 * v75]; k != v74; v74 += 4 )
            {
              if ( v74 )
                *v74 = -4096;
            }
          }
          break;
        }
        if ( v67 )
        {
          v7 = v18;
          sub_C7D6A0(*(_QWORD *)(a1 + 376), v18, 8);
          *(_QWORD *)(a1 + 376) = 0;
          *(_QWORD *)(a1 + 384) = 0;
          *(_DWORD *)(a1 + 392) = 0;
          break;
        }
LABEL_29:
        *(_QWORD *)(a1 + 384) = 0;
        break;
      }
    }
  }
  v21 = *(_DWORD *)(a1 + 448);
  ++*(_QWORD *)(a1 + 432);
  if ( v21 || *(_DWORD *)(a1 + 452) )
  {
    v7 = 64;
    v22 = *(_QWORD **)(a1 + 440);
    v23 = *(unsigned int *)(a1 + 456);
    v24 = 4 * v21;
    v25 = 72 * v23;
    if ( (unsigned int)(4 * v21) < 0x40 )
      v24 = 64;
    v26 = &v22[(unsigned __int64)v25 / 8];
    if ( (unsigned int)v23 <= v24 )
    {
      v27 = 0x100000000LL;
      v84[0] = &v85;
      v86[0] = v87;
      v84[1] = 0x100000000LL;
      v86[1] = 0x400000000LL;
      v88 = 0x100000000LL;
      if ( v26 == v22 )
      {
        *(_QWORD *)(a1 + 448) = 0;
      }
      else
      {
        do
        {
          sub_10774E0((__int64)v22, (__int64)v84, v27, v23, v4, v5);
          v28 = (__int64)(v22 + 3);
          v7 = (__int64)v86;
          v22 += 9;
          sub_10774E0(v28, (__int64)v86, v29, v30, v31, v32);
          *((_DWORD *)v22 - 4) = v88;
          v27 = HIDWORD(v88);
          *((_DWORD *)v22 - 3) = HIDWORD(v88);
        }
        while ( v22 != v26 );
        *(_QWORD *)(a1 + 448) = 0;
        if ( (_BYTE *)v86[0] != v87 )
          _libc_free(v86[0], v86);
      }
      if ( (char *)v84[0] != &v85 )
        _libc_free(v84[0], v7);
      goto LABEL_41;
    }
    do
    {
      v60 = (_QWORD *)v22[3];
      if ( v60 != v22 + 5 )
        _libc_free(v60, 64);
      if ( (_QWORD *)*v22 != v22 + 2 )
        _libc_free(*v22, 64);
      v22 += 9;
    }
    while ( v22 != v26 );
    v61 = *(unsigned int *)(a1 + 456);
    if ( v21 )
    {
      v62 = 64;
      v63 = v21 - 1;
      if ( v63 )
      {
        _BitScanReverse(&v64, v63);
        v23 = 33 - (v64 ^ 0x1F);
        v62 = 1 << (33 - (v64 ^ 0x1F));
        if ( v62 < 64 )
          v62 = 64;
      }
      if ( (_DWORD)v61 != v62 )
      {
        sub_C7D6A0(*(_QWORD *)(a1 + 440), v25, 8);
        v7 = 8;
        v65 = ((((((((4 * v62 / 3u + 1) | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 2)
                 | (4 * v62 / 3u + 1)
                 | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 4)
               | (((4 * v62 / 3u + 1) | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 2)
               | (4 * v62 / 3u + 1)
               | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 8)
             | (((((4 * v62 / 3u + 1) | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 2)
               | (4 * v62 / 3u + 1)
               | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 4)
             | (((4 * v62 / 3u + 1) | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 2)
             | (4 * v62 / 3u + 1)
             | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 16;
        v66 = (v65
             | (((((((4 * v62 / 3u + 1) | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 2)
                 | (4 * v62 / 3u + 1)
                 | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 4)
               | (((4 * v62 / 3u + 1) | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 2)
               | (4 * v62 / 3u + 1)
               | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 8)
             | (((((4 * v62 / 3u + 1) | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 2)
               | (4 * v62 / 3u + 1)
               | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 4)
             | (((4 * v62 / 3u + 1) | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1)) >> 2)
             | (4 * v62 / 3u + 1)
             | ((unsigned __int64)(4 * v62 / 3u + 1) >> 1))
            + 1;
        *(_DWORD *)(a1 + 456) = v66;
        *(_QWORD *)(a1 + 440) = sub_C7D670(72 * v66, 8);
      }
    }
    else if ( (_DWORD)v61 )
    {
      v7 = v25;
      sub_C7D6A0(*(_QWORD *)(a1 + 440), v25, 8);
      *(_QWORD *)(a1 + 440) = 0;
      *(_QWORD *)(a1 + 448) = 0;
      *(_DWORD *)(a1 + 456) = 0;
      goto LABEL_41;
    }
    sub_107CAD0(a1 + 432, (char **)v7, v61, v23, v4, v5);
  }
LABEL_41:
  v33 = *(_QWORD **)(a1 + 464);
  v34 = &v33[8 * (unsigned __int64)*(unsigned int *)(a1 + 472)];
  while ( v33 != v34 )
  {
    while ( 1 )
    {
      v34 -= 8;
      v35 = (_QWORD *)v34[3];
      if ( v35 != v34 + 5 )
        _libc_free(v35, v7);
      if ( (_QWORD *)*v34 == v34 + 2 )
        break;
      _libc_free(*v34, v7);
      if ( v33 == v34 )
        goto LABEL_47;
    }
  }
LABEL_47:
  v36 = *(unsigned int *)(a1 + 744);
  v37 = *(_QWORD *)(a1 + 736);
  *(_DWORD *)(a1 + 472) = 0;
  v38 = v37 + 80 * v36;
  while ( v37 != v38 )
  {
    v38 -= 80;
    v39 = *(_QWORD *)(v38 + 48);
    if ( v39 != v38 + 72 )
      _libc_free(v39, v7);
  }
  v40 = *(_DWORD *)(a1 + 416);
  ++*(_QWORD *)(a1 + 400);
  *(_DWORD *)(a1 + 744) = 0;
  if ( !v40 )
  {
    if ( !*(_DWORD *)(a1 + 420) )
      goto LABEL_57;
    v41 = *(unsigned int *)(a1 + 424);
    if ( (unsigned int)v41 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 408), 16LL * (unsigned int)v41, 8);
      *(_QWORD *)(a1 + 408) = 0;
      *(_QWORD *)(a1 + 416) = 0;
      *(_DWORD *)(a1 + 424) = 0;
      goto LABEL_57;
    }
    goto LABEL_54;
  }
  v45 = 4 * v40;
  v41 = *(unsigned int *)(a1 + 424);
  if ( (unsigned int)(4 * v40) < 0x40 )
    v45 = 64;
  if ( v45 >= (unsigned int)v41 )
  {
LABEL_54:
    v42 = *(_QWORD **)(a1 + 408);
    for ( m = &v42[2 * v41]; m != v42; v42 += 2 )
      *v42 = -4096;
    *(_QWORD *)(a1 + 416) = 0;
    goto LABEL_57;
  }
  v46 = v40 - 1;
  if ( !v46 )
  {
    v47 = *(_QWORD **)(a1 + 408);
    v48 = 64;
LABEL_65:
    sub_C7D6A0((__int64)v47, 16LL * (unsigned int)v41, 8);
    v49 = ((((((((4 * v48 / 3u + 1) | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 2)
             | (4 * v48 / 3u + 1)
             | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 4)
           | (((4 * v48 / 3u + 1) | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 2)
           | (4 * v48 / 3u + 1)
           | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v48 / 3u + 1) | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 2)
           | (4 * v48 / 3u + 1)
           | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 4)
         | (((4 * v48 / 3u + 1) | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 2)
         | (4 * v48 / 3u + 1)
         | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 16;
    v50 = (v49
         | (((((((4 * v48 / 3u + 1) | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 2)
             | (4 * v48 / 3u + 1)
             | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 4)
           | (((4 * v48 / 3u + 1) | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 2)
           | (4 * v48 / 3u + 1)
           | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v48 / 3u + 1) | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 2)
           | (4 * v48 / 3u + 1)
           | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 4)
         | (((4 * v48 / 3u + 1) | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 2)
         | (4 * v48 / 3u + 1)
         | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 424) = v50;
    v51 = (_QWORD *)sub_C7D670(16 * v50, 8);
    v52 = *(unsigned int *)(a1 + 424);
    *(_QWORD *)(a1 + 416) = 0;
    *(_QWORD *)(a1 + 408) = v51;
    for ( n = &v51[2 * v52]; n != v51; v51 += 2 )
    {
      if ( v51 )
        *v51 = -4096;
    }
    goto LABEL_57;
  }
  _BitScanReverse(&v46, v46);
  v47 = *(_QWORD **)(a1 + 408);
  v48 = 1 << (33 - (v46 ^ 0x1F));
  if ( v48 < 64 )
    v48 = 64;
  if ( (_DWORD)v41 != v48 )
    goto LABEL_65;
  *(_QWORD *)(a1 + 416) = 0;
  v82 = &v47[2 * (unsigned int)v41];
  do
  {
    if ( v47 )
      *v47 = -4096;
    v47 += 2;
  }
  while ( v82 != v47 );
LABEL_57:
  *(_QWORD *)(a1 + 1072) = 0;
  *(_DWORD *)(a1 + 1080) = 0;
  return sub_E8EB90(a1);
}
