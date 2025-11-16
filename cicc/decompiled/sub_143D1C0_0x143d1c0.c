// Function: sub_143D1C0
// Address: 0x143d1c0
//
void __fastcall sub_143D1C0(__int64 a1)
{
  int v2; // eax
  __int64 v3; // rdx
  _QWORD *v4; // rax
  _QWORD *i; // rdx
  int v6; // r14d
  __int64 v7; // rbx
  unsigned int v8; // eax
  __int64 v9; // r13
  unsigned __int64 v10; // rdi
  int v11; // r14d
  __int64 v12; // rbx
  unsigned int v13; // eax
  __int64 v14; // r13
  unsigned __int64 v15; // rdi
  unsigned int v16; // ecx
  _QWORD *v17; // rdi
  unsigned int v18; // eax
  int v19; // eax
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rax
  int v22; // ebx
  __int64 v23; // r13
  _QWORD *v24; // rax
  __int64 v25; // rdx
  _QWORD *j; // rdx
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // rdi
  __int64 v29; // rdx
  __int64 v30; // rdx
  int v31; // ebx
  unsigned int v32; // r14d
  unsigned int v33; // eax
  _DWORD *v34; // rdi
  unsigned __int64 v35; // rdx
  unsigned __int64 v36; // rax
  _DWORD *v37; // rax
  __int64 v38; // rdx
  _DWORD *k; // rdx
  int v40; // ebx
  unsigned int v41; // r14d
  unsigned int v42; // eax
  _DWORD *v43; // rdi
  unsigned __int64 v44; // rdx
  unsigned __int64 v45; // rax
  _DWORD *v46; // rax
  __int64 v47; // rdx
  _DWORD *m; // rdx
  _QWORD *v49; // rax
  _DWORD *v50; // rax
  _DWORD *v51; // rax

  v2 = *(_DWORD *)(a1 + 24);
  ++*(_QWORD *)(a1 + 8);
  if ( !v2 )
  {
    if ( !*(_DWORD *)(a1 + 28) )
      goto LABEL_7;
    v3 = *(unsigned int *)(a1 + 32);
    if ( (unsigned int)v3 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 16));
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      *(_DWORD *)(a1 + 32) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v16 = 4 * v2;
  v3 = *(unsigned int *)(a1 + 32);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v16 = 64;
  if ( (unsigned int)v3 <= v16 )
  {
LABEL_4:
    v4 = *(_QWORD **)(a1 + 16);
    for ( i = &v4[2 * v3]; i != v4; v4 += 2 )
      *v4 = -8;
    *(_QWORD *)(a1 + 24) = 0;
    goto LABEL_7;
  }
  v17 = *(_QWORD **)(a1 + 16);
  v18 = v2 - 1;
  if ( !v18 )
  {
    v23 = 2048;
    v22 = 128;
LABEL_44:
    j___libc_free_0(v17);
    *(_DWORD *)(a1 + 32) = v22;
    v24 = (_QWORD *)sub_22077B0(v23);
    v25 = *(unsigned int *)(a1 + 32);
    *(_QWORD *)(a1 + 24) = 0;
    *(_QWORD *)(a1 + 16) = v24;
    for ( j = &v24[2 * v25]; j != v24; v24 += 2 )
    {
      if ( v24 )
        *v24 = -8;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v18, v18);
  v19 = 1 << (33 - (v18 ^ 0x1F));
  if ( v19 < 64 )
    v19 = 64;
  if ( (_DWORD)v3 != v19 )
  {
    v20 = (((4 * v19 / 3u + 1) | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 2)
        | (4 * v19 / 3u + 1)
        | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)
        | (((((4 * v19 / 3u + 1) | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 2)
          | (4 * v19 / 3u + 1)
          | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 4);
    v21 = (v20 >> 8) | v20;
    v22 = (v21 | (v21 >> 16)) + 1;
    v23 = 16 * ((v21 | (v21 >> 16)) + 1);
    goto LABEL_44;
  }
  *(_QWORD *)(a1 + 24) = 0;
  v49 = &v17[2 * (unsigned int)v3];
  do
  {
    if ( v17 )
      *v17 = -8;
    v17 += 2;
  }
  while ( v49 != v17 );
LABEL_7:
  v6 = *(_DWORD *)(a1 + 56);
  ++*(_QWORD *)(a1 + 40);
  if ( !v6 && !*(_DWORD *)(a1 + 60) )
    goto LABEL_21;
  v7 = *(_QWORD *)(a1 + 48);
  v8 = 4 * v6;
  v9 = v7 + 80LL * *(unsigned int *)(a1 + 64);
  if ( (unsigned int)(4 * v6) < 0x40 )
    v8 = 64;
  if ( *(_DWORD *)(a1 + 64) <= v8 )
  {
    while ( v7 != v9 )
    {
      if ( *(_DWORD *)v7 != -1 )
      {
        if ( *(_DWORD *)v7 != -2 )
        {
          v10 = *(_QWORD *)(v7 + 24);
          if ( v10 != *(_QWORD *)(v7 + 16) )
            _libc_free(v10);
        }
        *(_DWORD *)v7 = -1;
      }
      v7 += 80;
    }
LABEL_20:
    *(_QWORD *)(a1 + 56) = 0;
    goto LABEL_21;
  }
  do
  {
    if ( *(_DWORD *)v7 <= 0xFFFFFFFD )
    {
      v27 = *(_QWORD *)(v7 + 24);
      if ( v27 != *(_QWORD *)(v7 + 16) )
        _libc_free(v27);
    }
    v7 += 80;
  }
  while ( v7 != v9 );
  v30 = *(unsigned int *)(a1 + 64);
  if ( !v6 )
  {
    if ( (_DWORD)v30 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 48));
      *(_QWORD *)(a1 + 48) = 0;
      *(_QWORD *)(a1 + 56) = 0;
      *(_DWORD *)(a1 + 64) = 0;
      goto LABEL_21;
    }
    goto LABEL_20;
  }
  v31 = 64;
  v32 = v6 - 1;
  if ( v32 )
  {
    _BitScanReverse(&v33, v32);
    v31 = 1 << (33 - (v33 ^ 0x1F));
    if ( v31 < 64 )
      v31 = 64;
  }
  v34 = *(_DWORD **)(a1 + 48);
  if ( (_DWORD)v30 == v31 )
  {
    *(_QWORD *)(a1 + 56) = 0;
    v51 = &v34[20 * v30];
    do
    {
      if ( v34 )
        *v34 = -1;
      v34 += 20;
    }
    while ( v51 != v34 );
  }
  else
  {
    j___libc_free_0(v34);
    v35 = ((((((((4 * v31 / 3u + 1) | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 2)
             | (4 * v31 / 3u + 1)
             | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 4)
           | (((4 * v31 / 3u + 1) | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 2)
           | (4 * v31 / 3u + 1)
           | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v31 / 3u + 1) | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 2)
           | (4 * v31 / 3u + 1)
           | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 4)
         | (((4 * v31 / 3u + 1) | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 2)
         | (4 * v31 / 3u + 1)
         | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 16;
    v36 = (v35
         | (((((((4 * v31 / 3u + 1) | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 2)
             | (4 * v31 / 3u + 1)
             | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 4)
           | (((4 * v31 / 3u + 1) | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 2)
           | (4 * v31 / 3u + 1)
           | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v31 / 3u + 1) | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 2)
           | (4 * v31 / 3u + 1)
           | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 4)
         | (((4 * v31 / 3u + 1) | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1)) >> 2)
         | (4 * v31 / 3u + 1)
         | ((unsigned __int64)(4 * v31 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 64) = v36;
    v37 = (_DWORD *)sub_22077B0(80 * v36);
    v38 = *(unsigned int *)(a1 + 64);
    *(_QWORD *)(a1 + 56) = 0;
    *(_QWORD *)(a1 + 48) = v37;
    for ( k = &v37[20 * v38]; k != v37; v37 += 20 )
    {
      if ( v37 )
        *v37 = -1;
    }
  }
LABEL_21:
  v11 = *(_DWORD *)(a1 + 88);
  ++*(_QWORD *)(a1 + 72);
  if ( v11 || *(_DWORD *)(a1 + 92) )
  {
    v12 = *(_QWORD *)(a1 + 80);
    v13 = 4 * v11;
    v14 = v12 + 80LL * *(unsigned int *)(a1 + 96);
    if ( (unsigned int)(4 * v11) < 0x40 )
      v13 = 64;
    if ( *(_DWORD *)(a1 + 96) <= v13 )
    {
      while ( v12 != v14 )
      {
        if ( *(_DWORD *)v12 != -1 )
        {
          if ( *(_DWORD *)v12 != -2 )
          {
            v15 = *(_QWORD *)(v12 + 24);
            if ( v15 != *(_QWORD *)(v12 + 16) )
              _libc_free(v15);
          }
          *(_DWORD *)v12 = -1;
        }
        v12 += 80;
      }
LABEL_34:
      *(_QWORD *)(a1 + 88) = 0;
      return;
    }
    do
    {
      if ( *(_DWORD *)v12 <= 0xFFFFFFFD )
      {
        v28 = *(_QWORD *)(v12 + 24);
        if ( v28 != *(_QWORD *)(v12 + 16) )
          _libc_free(v28);
      }
      v12 += 80;
    }
    while ( v14 != v12 );
    v29 = *(unsigned int *)(a1 + 96);
    if ( !v11 )
    {
      if ( (_DWORD)v29 )
      {
        j___libc_free_0(*(_QWORD *)(a1 + 80));
        *(_QWORD *)(a1 + 80) = 0;
        *(_QWORD *)(a1 + 88) = 0;
        *(_DWORD *)(a1 + 96) = 0;
        return;
      }
      goto LABEL_34;
    }
    v40 = 64;
    v41 = v11 - 1;
    if ( v41 )
    {
      _BitScanReverse(&v42, v41);
      v40 = 1 << (33 - (v42 ^ 0x1F));
      if ( v40 < 64 )
        v40 = 64;
    }
    v43 = *(_DWORD **)(a1 + 80);
    if ( (_DWORD)v29 == v40 )
    {
      *(_QWORD *)(a1 + 88) = 0;
      v50 = &v43[20 * v29];
      do
      {
        if ( v43 )
          *v43 = -1;
        v43 += 20;
      }
      while ( v50 != v43 );
    }
    else
    {
      j___libc_free_0(v43);
      v44 = ((((((((4 * v40 / 3u + 1) | ((unsigned __int64)(4 * v40 / 3u + 1) >> 1)) >> 2)
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
      v45 = (v44
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
      *(_DWORD *)(a1 + 96) = v45;
      v46 = (_DWORD *)sub_22077B0(80 * v45);
      v47 = *(unsigned int *)(a1 + 96);
      *(_QWORD *)(a1 + 88) = 0;
      *(_QWORD *)(a1 + 80) = v46;
      for ( m = &v46[20 * v47]; m != v46; v46 += 20 )
      {
        if ( v46 )
          *v46 = -1;
      }
    }
  }
}
