// Function: sub_22D5150
// Address: 0x22d5150
//
__int64 __fastcall sub_22D5150(__int64 a1)
{
  int v2; // eax
  __int64 v3; // rdx
  _QWORD *v4; // rax
  _QWORD *i; // rdx
  int v6; // r15d
  __int64 v7; // rbx
  unsigned int v8; // eax
  __int64 v9; // r14
  __int64 v10; // r13
  unsigned __int64 v11; // rdi
  int v12; // r15d
  __int64 result; // rax
  unsigned int *v14; // rbx
  __int64 v15; // r14
  unsigned int *v16; // r13
  unsigned __int64 v17; // rdi
  unsigned int v18; // ecx
  unsigned int v19; // eax
  _QWORD *v20; // rdi
  int v21; // ebx
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // rdi
  _QWORD *v24; // rax
  __int64 v25; // rdx
  _QWORD *j; // rdx
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // rdi
  __int64 v29; // rdx
  int v30; // ebx
  unsigned int v31; // r15d
  unsigned int v32; // eax
  _DWORD *v33; // rdi
  unsigned __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 m; // rdx
  int v38; // edx
  __int64 v39; // rbx
  unsigned int v40; // r15d
  unsigned int v41; // eax
  _DWORD *v42; // rdi
  unsigned __int64 v43; // rax
  __int64 v44; // rax
  _DWORD *v45; // rax
  __int64 v46; // rdx
  _DWORD *k; // rdx
  _QWORD *v48; // rax
  _DWORD *v49; // rax

  v2 = *(_DWORD *)(a1 + 24);
  ++*(_QWORD *)(a1 + 8);
  if ( !v2 )
  {
    if ( !*(_DWORD *)(a1 + 28) )
      goto LABEL_7;
    v3 = *(unsigned int *)(a1 + 32);
    if ( (unsigned int)v3 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 16), 16LL * (unsigned int)v3, 8);
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      *(_DWORD *)(a1 + 32) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v18 = 4 * v2;
  v3 = *(unsigned int *)(a1 + 32);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v18 = 64;
  if ( (unsigned int)v3 <= v18 )
  {
LABEL_4:
    v4 = *(_QWORD **)(a1 + 16);
    for ( i = &v4[2 * v3]; i != v4; v4 += 2 )
      *v4 = -4096;
    *(_QWORD *)(a1 + 24) = 0;
    goto LABEL_7;
  }
  v19 = v2 - 1;
  if ( !v19 )
  {
    v20 = *(_QWORD **)(a1 + 16);
    v21 = 64;
LABEL_44:
    sub_C7D6A0((__int64)v20, 16LL * (unsigned int)v3, 8);
    v22 = ((((((((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
             | (4 * v21 / 3u + 1)
             | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 4)
           | (((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
           | (4 * v21 / 3u + 1)
           | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
           | (4 * v21 / 3u + 1)
           | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 4)
         | (((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
         | (4 * v21 / 3u + 1)
         | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 16;
    v23 = (v22
         | (((((((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
             | (4 * v21 / 3u + 1)
             | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 4)
           | (((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
           | (4 * v21 / 3u + 1)
           | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
           | (4 * v21 / 3u + 1)
           | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 4)
         | (((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
         | (4 * v21 / 3u + 1)
         | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 32) = v23;
    v24 = (_QWORD *)sub_C7D670(16 * v23, 8);
    v25 = *(unsigned int *)(a1 + 32);
    *(_QWORD *)(a1 + 24) = 0;
    *(_QWORD *)(a1 + 16) = v24;
    for ( j = &v24[2 * v25]; j != v24; v24 += 2 )
    {
      if ( v24 )
        *v24 = -4096;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v19, v19);
  v20 = *(_QWORD **)(a1 + 16);
  v21 = 1 << (33 - (v19 ^ 0x1F));
  if ( v21 < 64 )
    v21 = 64;
  if ( v21 != (_DWORD)v3 )
    goto LABEL_44;
  *(_QWORD *)(a1 + 24) = 0;
  v48 = &v20[2 * (unsigned int)v21];
  do
  {
    if ( v20 )
      *v20 = -4096;
    v20 += 2;
  }
  while ( v48 != v20 );
LABEL_7:
  v6 = *(_DWORD *)(a1 + 56);
  ++*(_QWORD *)(a1 + 40);
  if ( v6 || *(_DWORD *)(a1 + 60) )
  {
    v7 = *(_QWORD *)(a1 + 48);
    v8 = 4 * v6;
    v9 = 88LL * *(unsigned int *)(a1 + 64);
    if ( (unsigned int)(4 * v6) < 0x40 )
      v8 = 64;
    v10 = v7 + v9;
    if ( *(_DWORD *)(a1 + 64) <= v8 )
    {
      for ( ; v7 != v10; v7 += 88 )
      {
        if ( *(_DWORD *)v7 != -1 )
        {
          if ( *(_DWORD *)v7 != -2 )
          {
            v11 = *(_QWORD *)(v7 + 40);
            if ( v11 != v7 + 56 )
              _libc_free(v11);
            sub_C7D6A0(*(_QWORD *)(v7 + 16), 8LL * *(unsigned int *)(v7 + 32), 8);
          }
          *(_DWORD *)v7 = -1;
        }
      }
LABEL_20:
      *(_QWORD *)(a1 + 56) = 0;
      goto LABEL_21;
    }
    do
    {
      if ( *(_DWORD *)v7 <= 0xFFFFFFFD )
      {
        v27 = *(_QWORD *)(v7 + 40);
        if ( v27 != v7 + 56 )
          _libc_free(v27);
        sub_C7D6A0(*(_QWORD *)(v7 + 16), 8LL * *(unsigned int *)(v7 + 32), 8);
      }
      v7 += 88;
    }
    while ( v7 != v10 );
    v38 = *(_DWORD *)(a1 + 64);
    if ( !v6 )
    {
      if ( v38 )
      {
        sub_C7D6A0(*(_QWORD *)(a1 + 48), v9, 8);
        *(_QWORD *)(a1 + 48) = 0;
        *(_QWORD *)(a1 + 56) = 0;
        *(_DWORD *)(a1 + 64) = 0;
        goto LABEL_21;
      }
      goto LABEL_20;
    }
    v39 = 64;
    v40 = v6 - 1;
    if ( v40 )
    {
      _BitScanReverse(&v41, v40);
      v39 = (unsigned int)(1 << (33 - (v41 ^ 0x1F)));
      if ( (int)v39 < 64 )
        v39 = 64;
    }
    v42 = *(_DWORD **)(a1 + 48);
    if ( (_DWORD)v39 == v38 )
    {
      *(_QWORD *)(a1 + 56) = 0;
      v49 = &v42[22 * v39];
      do
      {
        if ( v42 )
          *v42 = -1;
        v42 += 22;
      }
      while ( v49 != v42 );
    }
    else
    {
      sub_C7D6A0((__int64)v42, v9, 8);
      v43 = (((((((4 * (int)v39 / 3u + 1) | ((unsigned __int64)(4 * (int)v39 / 3u + 1) >> 1)) >> 2)
              | (4 * (int)v39 / 3u + 1)
              | ((unsigned __int64)(4 * (int)v39 / 3u + 1) >> 1)) >> 4)
            | (((4 * (int)v39 / 3u + 1) | ((unsigned __int64)(4 * (int)v39 / 3u + 1) >> 1)) >> 2)
            | (4 * (int)v39 / 3u + 1)
            | ((unsigned __int64)(4 * (int)v39 / 3u + 1) >> 1)) >> 8)
          | (((((4 * (int)v39 / 3u + 1) | ((unsigned __int64)(4 * (int)v39 / 3u + 1) >> 1)) >> 2)
            | (4 * (int)v39 / 3u + 1)
            | ((unsigned __int64)(4 * (int)v39 / 3u + 1) >> 1)) >> 4)
          | (((4 * (int)v39 / 3u + 1) | ((unsigned __int64)(4 * (int)v39 / 3u + 1) >> 1)) >> 2)
          | (4 * (int)v39 / 3u + 1)
          | ((unsigned __int64)(4 * (int)v39 / 3u + 1) >> 1);
      v44 = ((v43 >> 16) | v43) + 1;
      *(_DWORD *)(a1 + 64) = v44;
      v45 = (_DWORD *)sub_C7D670(88 * v44, 8);
      v46 = *(unsigned int *)(a1 + 64);
      *(_QWORD *)(a1 + 56) = 0;
      *(_QWORD *)(a1 + 48) = v45;
      for ( k = &v45[22 * v46]; k != v45; v45 += 22 )
      {
        if ( v45 )
          *v45 = -1;
      }
    }
  }
LABEL_21:
  v12 = *(_DWORD *)(a1 + 88);
  ++*(_QWORD *)(a1 + 72);
  if ( !v12 )
  {
    result = *(unsigned int *)(a1 + 92);
    if ( !(_DWORD)result )
      return result;
  }
  v14 = *(unsigned int **)(a1 + 80);
  result = (unsigned int)(4 * v12);
  v15 = 88LL * *(unsigned int *)(a1 + 96);
  if ( (unsigned int)result < 0x40 )
    result = 64;
  v16 = &v14[(unsigned __int64)v15 / 4];
  if ( *(_DWORD *)(a1 + 96) <= (unsigned int)result )
  {
    while ( v14 != v16 )
    {
      result = *v14;
      if ( (_DWORD)result != -1 )
      {
        if ( (_DWORD)result != -2 )
        {
          v17 = *((_QWORD *)v14 + 5);
          if ( (unsigned int *)v17 != v14 + 14 )
            _libc_free(v17);
          result = sub_C7D6A0(*((_QWORD *)v14 + 2), 8LL * v14[8], 8);
        }
        *v14 = -1;
      }
      v14 += 22;
    }
LABEL_35:
    *(_QWORD *)(a1 + 88) = 0;
    return result;
  }
  do
  {
    if ( *v14 <= 0xFFFFFFFD )
    {
      v28 = *((_QWORD *)v14 + 5);
      if ( (unsigned int *)v28 != v14 + 14 )
        _libc_free(v28);
      result = sub_C7D6A0(*((_QWORD *)v14 + 2), 8LL * v14[8], 8);
    }
    v14 += 22;
  }
  while ( v14 != v16 );
  v29 = *(unsigned int *)(a1 + 96);
  if ( !v12 )
  {
    if ( (_DWORD)v29 )
    {
      result = sub_C7D6A0(*(_QWORD *)(a1 + 80), v15, 8);
      *(_QWORD *)(a1 + 80) = 0;
      *(_QWORD *)(a1 + 88) = 0;
      *(_DWORD *)(a1 + 96) = 0;
      return result;
    }
    goto LABEL_35;
  }
  v30 = 64;
  v31 = v12 - 1;
  if ( v31 )
  {
    _BitScanReverse(&v32, v31);
    v30 = 1 << (33 - (v32 ^ 0x1F));
    if ( v30 < 64 )
      v30 = 64;
  }
  v33 = *(_DWORD **)(a1 + 80);
  if ( (_DWORD)v29 == v30 )
  {
    *(_QWORD *)(a1 + 88) = 0;
    result = (__int64)&v33[22 * v29];
    do
    {
      if ( v33 )
        *v33 = -1;
      v33 += 22;
    }
    while ( (_DWORD *)result != v33 );
  }
  else
  {
    sub_C7D6A0((__int64)v33, v15, 8);
    v34 = (((((((4 * v30 / 3u + 1) | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 2)
            | (4 * v30 / 3u + 1)
            | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 4)
          | (((4 * v30 / 3u + 1) | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 2)
          | (4 * v30 / 3u + 1)
          | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 8)
        | (((((4 * v30 / 3u + 1) | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 2)
          | (4 * v30 / 3u + 1)
          | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 4)
        | (((4 * v30 / 3u + 1) | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 2)
        | (4 * v30 / 3u + 1)
        | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1);
    v35 = ((v34 >> 16) | v34) + 1;
    *(_DWORD *)(a1 + 96) = v35;
    result = sub_C7D670(88 * v35, 8);
    v36 = *(unsigned int *)(a1 + 96);
    *(_QWORD *)(a1 + 88) = 0;
    *(_QWORD *)(a1 + 80) = result;
    for ( m = result + 88 * v36; m != result; result += 88 )
    {
      if ( result )
        *(_DWORD *)result = -1;
    }
  }
  return result;
}
