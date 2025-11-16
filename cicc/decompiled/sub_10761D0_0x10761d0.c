// Function: sub_10761D0
// Address: 0x10761d0
//
__int64 __fastcall sub_10761D0(__int64 a1, __int64 a2)
{
  int v3; // r15d
  _QWORD *v4; // rbx
  unsigned int v5; // eax
  __int64 v6; // rdx
  __int64 v7; // r14
  _QWORD *v8; // r13
  __int64 v9; // rdi
  int v10; // eax
  __int64 v11; // rdx
  _QWORD *v12; // rax
  _QWORD *j; // rdx
  __int64 v14; // rax
  __int64 v15; // rax
  int v16; // eax
  __int64 v17; // rdx
  _QWORD *v18; // rax
  _QWORD *n; // rdx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // r13
  __int64 v24; // rbx
  __int64 v25; // rdi
  _QWORD *v26; // rax
  _QWORD *v27; // r14
  _QWORD *v28; // r13
  _QWORD *v29; // rbx
  _QWORD *v30; // r15
  unsigned int v32; // ecx
  unsigned int v33; // eax
  _QWORD *v34; // rdi
  int v35; // ebx
  _QWORD *v36; // rax
  unsigned int v37; // ecx
  unsigned int v38; // eax
  _QWORD *v39; // rdi
  int v40; // ebx
  _QWORD *v41; // rax
  __int64 v42; // rdi
  unsigned int v43; // edx
  int v44; // ebx
  unsigned int v45; // eax
  _QWORD *v46; // rdi
  unsigned __int64 v47; // rdx
  unsigned __int64 v48; // rax
  _QWORD *v49; // rax
  __int64 v50; // rdx
  _QWORD *i; // rdx
  unsigned __int64 v52; // rdx
  unsigned __int64 v53; // rax
  _QWORD *v54; // rax
  __int64 v55; // rdx
  _QWORD *m; // rdx
  unsigned __int64 v57; // rdx
  unsigned __int64 v58; // rax
  _QWORD *v59; // rax
  __int64 v60; // rdx
  _QWORD *k; // rdx
  _QWORD *v62; // rax
  _QWORD *v63; // [rsp+8h] [rbp-38h]

  v3 = *(_DWORD *)(a1 + 128);
  ++*(_QWORD *)(a1 + 112);
  if ( !v3 && !*(_DWORD *)(a1 + 132) )
    goto LABEL_14;
  v4 = *(_QWORD **)(a1 + 120);
  v5 = 4 * v3;
  v6 = *(unsigned int *)(a1 + 136);
  v7 = 32 * v6;
  if ( (unsigned int)(4 * v3) < 0x40 )
    v5 = 64;
  v8 = &v4[(unsigned __int64)v7 / 8];
  if ( (unsigned int)v6 <= v5 )
  {
    for ( ; v4 != v8; v4 += 4 )
    {
      if ( *v4 != -4096 )
      {
        if ( *v4 != -8192 )
        {
          v9 = v4[1];
          if ( v9 )
          {
            a2 = v4[3] - v9;
            j_j___libc_free_0(v9, a2);
          }
        }
        *v4 = -4096;
      }
    }
    goto LABEL_13;
  }
  do
  {
    while ( *v4 == -4096 )
    {
LABEL_77:
      v4 += 4;
      if ( v4 == v8 )
        goto LABEL_81;
    }
    if ( *v4 != -8192 )
    {
      v42 = v4[1];
      if ( v42 )
      {
        a2 = v4[3] - v42;
        j_j___libc_free_0(v42, a2);
      }
      goto LABEL_77;
    }
    v4 += 4;
  }
  while ( v4 != v8 );
LABEL_81:
  v43 = *(_DWORD *)(a1 + 136);
  if ( !v3 )
  {
    if ( v43 )
    {
      a2 = v7;
      sub_C7D6A0(*(_QWORD *)(a1 + 120), v7, 8);
      *(_QWORD *)(a1 + 120) = 0;
      *(_QWORD *)(a1 + 128) = 0;
      *(_DWORD *)(a1 + 136) = 0;
      goto LABEL_14;
    }
LABEL_13:
    *(_QWORD *)(a1 + 128) = 0;
    goto LABEL_14;
  }
  v44 = 64;
  if ( v3 != 1 )
  {
    _BitScanReverse(&v45, v3 - 1);
    v44 = 1 << (33 - (v45 ^ 0x1F));
    if ( v44 < 64 )
      v44 = 64;
  }
  v46 = *(_QWORD **)(a1 + 120);
  if ( v43 == v44 )
  {
    *(_QWORD *)(a1 + 128) = 0;
    v62 = &v46[4 * v43];
    do
    {
      if ( v46 )
        *v46 = -4096;
      v46 += 4;
    }
    while ( v62 != v46 );
  }
  else
  {
    sub_C7D6A0((__int64)v46, v7, 8);
    a2 = 8;
    v47 = ((((((((4 * v44 / 3u + 1) | ((unsigned __int64)(4 * v44 / 3u + 1) >> 1)) >> 2)
             | (4 * v44 / 3u + 1)
             | ((unsigned __int64)(4 * v44 / 3u + 1) >> 1)) >> 4)
           | (((4 * v44 / 3u + 1) | ((unsigned __int64)(4 * v44 / 3u + 1) >> 1)) >> 2)
           | (4 * v44 / 3u + 1)
           | ((unsigned __int64)(4 * v44 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v44 / 3u + 1) | ((unsigned __int64)(4 * v44 / 3u + 1) >> 1)) >> 2)
           | (4 * v44 / 3u + 1)
           | ((unsigned __int64)(4 * v44 / 3u + 1) >> 1)) >> 4)
         | (((4 * v44 / 3u + 1) | ((unsigned __int64)(4 * v44 / 3u + 1) >> 1)) >> 2)
         | (4 * v44 / 3u + 1)
         | ((unsigned __int64)(4 * v44 / 3u + 1) >> 1)) >> 16;
    v48 = (v47
         | (((((((4 * v44 / 3u + 1) | ((unsigned __int64)(4 * v44 / 3u + 1) >> 1)) >> 2)
             | (4 * v44 / 3u + 1)
             | ((unsigned __int64)(4 * v44 / 3u + 1) >> 1)) >> 4)
           | (((4 * v44 / 3u + 1) | ((unsigned __int64)(4 * v44 / 3u + 1) >> 1)) >> 2)
           | (4 * v44 / 3u + 1)
           | ((unsigned __int64)(4 * v44 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v44 / 3u + 1) | ((unsigned __int64)(4 * v44 / 3u + 1) >> 1)) >> 2)
           | (4 * v44 / 3u + 1)
           | ((unsigned __int64)(4 * v44 / 3u + 1) >> 1)) >> 4)
         | (((4 * v44 / 3u + 1) | ((unsigned __int64)(4 * v44 / 3u + 1) >> 1)) >> 2)
         | (4 * v44 / 3u + 1)
         | ((unsigned __int64)(4 * v44 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 136) = v48;
    v49 = (_QWORD *)sub_C7D670(32 * v48, 8);
    v50 = *(unsigned int *)(a1 + 136);
    *(_QWORD *)(a1 + 128) = 0;
    *(_QWORD *)(a1 + 120) = v49;
    for ( i = &v49[4 * v50]; i != v49; v49 += 4 )
    {
      if ( v49 )
        *v49 = -4096;
    }
  }
LABEL_14:
  v10 = *(_DWORD *)(a1 + 184);
  ++*(_QWORD *)(a1 + 168);
  if ( v10 )
  {
    v37 = 4 * v10;
    a2 = 64;
    v11 = *(unsigned int *)(a1 + 192);
    if ( (unsigned int)(4 * v10) < 0x40 )
      v37 = 64;
    if ( v37 >= (unsigned int)v11 )
    {
LABEL_17:
      v12 = *(_QWORD **)(a1 + 176);
      for ( j = &v12[2 * v11]; j != v12; v12 += 2 )
        *v12 = -4096;
      *(_QWORD *)(a1 + 184) = 0;
      goto LABEL_20;
    }
    v38 = v10 - 1;
    if ( v38 )
    {
      _BitScanReverse(&v38, v38);
      v39 = *(_QWORD **)(a1 + 176);
      v40 = 1 << (33 - (v38 ^ 0x1F));
      if ( v40 < 64 )
        v40 = 64;
      if ( (_DWORD)v11 == v40 )
      {
        *(_QWORD *)(a1 + 184) = 0;
        v41 = &v39[2 * (unsigned int)v11];
        do
        {
          if ( v39 )
            *v39 = -4096;
          v39 += 2;
        }
        while ( v41 != v39 );
        goto LABEL_20;
      }
    }
    else
    {
      v39 = *(_QWORD **)(a1 + 176);
      v40 = 64;
    }
    sub_C7D6A0((__int64)v39, 16 * v11, 8);
    a2 = 8;
    v57 = ((((((((4 * v40 / 3u + 1) | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 2)
             | (4 * v40 / 3u + 1)
             | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 4)
           | (((4 * v40 / 3u + 1) | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 2)
           | (4 * v40 / 3u + 1)
           | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v40 / 3u + 1) | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 2)
           | (4 * v40 / 3u + 1)
           | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 4)
         | (((4 * v40 / 3u + 1) | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 2)
         | (4 * v40 / 3u + 1)
         | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 16;
    v58 = (v57
         | (((((((4 * v40 / 3u + 1) | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 2)
             | (4 * v40 / 3u + 1)
             | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 4)
           | (((4 * v40 / 3u + 1) | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 2)
           | (4 * v40 / 3u + 1)
           | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v40 / 3u + 1) | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 2)
           | (4 * v40 / 3u + 1)
           | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 4)
         | (((4 * v40 / 3u + 1) | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 2)
         | (4 * v40 / 3u + 1)
         | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 192) = v58;
    v59 = (_QWORD *)sub_C7D670(16 * v58, 8);
    v60 = *(unsigned int *)(a1 + 192);
    *(_QWORD *)(a1 + 184) = 0;
    *(_QWORD *)(a1 + 176) = v59;
    for ( k = &v59[2 * v60]; k != v59; v59 += 2 )
    {
      if ( v59 )
        *v59 = -4096;
    }
  }
  else if ( *(_DWORD *)(a1 + 188) )
  {
    v11 = *(unsigned int *)(a1 + 192);
    if ( (unsigned int)v11 <= 0x40 )
      goto LABEL_17;
    a2 = 16 * v11;
    sub_C7D6A0(*(_QWORD *)(a1 + 176), 16 * v11, 8);
    *(_QWORD *)(a1 + 176) = 0;
    *(_QWORD *)(a1 + 184) = 0;
    *(_DWORD *)(a1 + 192) = 0;
  }
LABEL_20:
  v14 = *(_QWORD *)(a1 + 144);
  if ( v14 != *(_QWORD *)(a1 + 152) )
    *(_QWORD *)(a1 + 152) = v14;
  v15 = *(_QWORD *)(a1 + 200);
  if ( v15 != *(_QWORD *)(a1 + 208) )
    *(_QWORD *)(a1 + 208) = v15;
  v16 = *(_DWORD *)(a1 + 240);
  ++*(_QWORD *)(a1 + 224);
  if ( v16 )
  {
    v32 = 4 * v16;
    a2 = 64;
    v17 = *(unsigned int *)(a1 + 248);
    if ( (unsigned int)(4 * v16) < 0x40 )
      v32 = 64;
    if ( v32 < (unsigned int)v17 )
    {
      v33 = v16 - 1;
      if ( v33 )
      {
        _BitScanReverse(&v33, v33);
        v34 = *(_QWORD **)(a1 + 232);
        v35 = 1 << (33 - (v33 ^ 0x1F));
        if ( v35 < 64 )
          v35 = 64;
        if ( v35 == (_DWORD)v17 )
        {
          *(_QWORD *)(a1 + 240) = 0;
          v36 = &v34[2 * (unsigned int)v35];
          do
          {
            if ( v34 )
              *v34 = -4096;
            v34 += 2;
          }
          while ( v36 != v34 );
          goto LABEL_30;
        }
      }
      else
      {
        v34 = *(_QWORD **)(a1 + 232);
        v35 = 64;
      }
      sub_C7D6A0((__int64)v34, 16 * v17, 8);
      a2 = 8;
      v52 = ((((((((4 * v35 / 3u + 1) | ((unsigned __int64)(4 * v35 / 3u + 1) >> 1)) >> 2)
               | (4 * v35 / 3u + 1)
               | ((unsigned __int64)(4 * v35 / 3u + 1) >> 1)) >> 4)
             | (((4 * v35 / 3u + 1) | ((unsigned __int64)(4 * v35 / 3u + 1) >> 1)) >> 2)
             | (4 * v35 / 3u + 1)
             | ((unsigned __int64)(4 * v35 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v35 / 3u + 1) | ((unsigned __int64)(4 * v35 / 3u + 1) >> 1)) >> 2)
             | (4 * v35 / 3u + 1)
             | ((unsigned __int64)(4 * v35 / 3u + 1) >> 1)) >> 4)
           | (((4 * v35 / 3u + 1) | ((unsigned __int64)(4 * v35 / 3u + 1) >> 1)) >> 2)
           | (4 * v35 / 3u + 1)
           | ((unsigned __int64)(4 * v35 / 3u + 1) >> 1)) >> 16;
      v53 = (v52
           | (((((((4 * v35 / 3u + 1) | ((unsigned __int64)(4 * v35 / 3u + 1) >> 1)) >> 2)
               | (4 * v35 / 3u + 1)
               | ((unsigned __int64)(4 * v35 / 3u + 1) >> 1)) >> 4)
             | (((4 * v35 / 3u + 1) | ((unsigned __int64)(4 * v35 / 3u + 1) >> 1)) >> 2)
             | (4 * v35 / 3u + 1)
             | ((unsigned __int64)(4 * v35 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v35 / 3u + 1) | ((unsigned __int64)(4 * v35 / 3u + 1) >> 1)) >> 2)
             | (4 * v35 / 3u + 1)
             | ((unsigned __int64)(4 * v35 / 3u + 1) >> 1)) >> 4)
           | (((4 * v35 / 3u + 1) | ((unsigned __int64)(4 * v35 / 3u + 1) >> 1)) >> 2)
           | (4 * v35 / 3u + 1)
           | ((unsigned __int64)(4 * v35 / 3u + 1) >> 1))
          + 1;
      *(_DWORD *)(a1 + 248) = v53;
      v54 = (_QWORD *)sub_C7D670(16 * v53, 8);
      v55 = *(unsigned int *)(a1 + 248);
      *(_QWORD *)(a1 + 240) = 0;
      *(_QWORD *)(a1 + 232) = v54;
      for ( m = &v54[2 * v55]; m != v54; v54 += 2 )
      {
        if ( v54 )
          *v54 = -4096;
      }
      goto LABEL_30;
    }
LABEL_27:
    v18 = *(_QWORD **)(a1 + 232);
    for ( n = &v18[2 * v17]; n != v18; v18 += 2 )
      *v18 = -4096;
    *(_QWORD *)(a1 + 240) = 0;
  }
  else if ( *(_DWORD *)(a1 + 244) )
  {
    v17 = *(unsigned int *)(a1 + 248);
    if ( (unsigned int)v17 <= 0x40 )
      goto LABEL_27;
    a2 = 16 * v17;
    sub_C7D6A0(*(_QWORD *)(a1 + 232), 16 * v17, 8);
    *(_QWORD *)(a1 + 232) = 0;
    *(_QWORD *)(a1 + 240) = 0;
    *(_DWORD *)(a1 + 248) = 0;
  }
LABEL_30:
  *(_DWORD *)(a1 + 264) = 0;
  sub_C0C1A0(a1 + 272);
  v20 = *(_QWORD *)(a1 + 320);
  if ( v20 != *(_QWORD *)(a1 + 328) )
    *(_QWORD *)(a1 + 328) = v20;
  v21 = *(_QWORD *)(a1 + 344);
  if ( v21 != *(_QWORD *)(a1 + 352) )
    *(_QWORD *)(a1 + 352) = v21;
  v22 = *(_QWORD *)(a1 + 368);
  if ( v22 != *(_QWORD *)(a1 + 376) )
    *(_QWORD *)(a1 + 376) = v22;
  v23 = *(_QWORD *)(a1 + 400);
  v24 = v23 + 48LL * *(unsigned int *)(a1 + 408);
  while ( v23 != v24 )
  {
    while ( 1 )
    {
      v24 -= 48;
      v25 = *(_QWORD *)(v24 + 8);
      if ( v25 == v24 + 24 )
        break;
      _libc_free(v25, a2);
      if ( v23 == v24 )
        goto LABEL_40;
    }
  }
LABEL_40:
  v26 = *(_QWORD **)(a1 + 2024);
  v27 = *(_QWORD **)(a1 + 2032);
  *(_DWORD *)(a1 + 408) = 0;
  *(_QWORD *)(a1 + 392) = 0;
  *(_DWORD *)(a1 + 1960) = 0;
  v28 = v26;
  *(_QWORD *)(a1 + 1972) = 0;
  *(_QWORD *)(a1 + 1980) = 0;
  *(_DWORD *)(a1 + 1996) = 0;
  *(_QWORD *)(a1 + 2008) = 0;
  *(_QWORD *)(a1 + 2016) = 0;
  v63 = v26;
  if ( v26 != v27 )
  {
    do
    {
      v29 = (_QWORD *)v28[1];
      v30 = (_QWORD *)*v28;
      if ( v29 != (_QWORD *)*v28 )
      {
        do
        {
          if ( (_QWORD *)*v30 != v30 + 2 )
            j_j___libc_free_0(*v30, v30[2] + 1LL);
          v30 += 4;
        }
        while ( v29 != v30 );
        v30 = (_QWORD *)*v28;
      }
      if ( v30 )
        j_j___libc_free_0(v30, v28[2] - (_QWORD)v30);
      v28 += 3;
    }
    while ( v27 != v28 );
    *(_QWORD *)(a1 + 2032) = v63;
  }
  return sub_E8EB90(a1);
}
