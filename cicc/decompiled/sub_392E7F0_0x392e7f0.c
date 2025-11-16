// Function: sub_392E7F0
// Address: 0x392e7f0
//
void __fastcall sub_392E7F0(__int64 a1)
{
  int v2; // r14d
  _QWORD *v3; // rbx
  unsigned int v4; // eax
  __int64 v5; // rdx
  _QWORD *v6; // r13
  unsigned __int64 v7; // rdi
  int v8; // eax
  __int64 v9; // rdx
  _QWORD *v10; // rax
  _QWORD *j; // rdx
  unsigned int v12; // ecx
  _QWORD *v13; // rdi
  unsigned int v14; // eax
  int v15; // eax
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rax
  int v18; // ebx
  unsigned __int64 v19; // r13
  _QWORD *v20; // rax
  __int64 v21; // rdx
  _QWORD *k; // rdx
  unsigned __int64 v23; // rdi
  int v24; // edx
  int v25; // ebx
  unsigned int v26; // r14d
  unsigned int v27; // eax
  _QWORD *v28; // rdi
  unsigned __int64 v29; // rax
  unsigned __int64 v30; // rdi
  _QWORD *v31; // rax
  __int64 v32; // rdx
  _QWORD *i; // rdx
  _QWORD *v34; // rax
  _QWORD *v35; // rax

  v2 = *(_DWORD *)(a1 + 32);
  ++*(_QWORD *)(a1 + 16);
  if ( !v2 && !*(_DWORD *)(a1 + 36) )
    goto LABEL_14;
  v3 = *(_QWORD **)(a1 + 24);
  v4 = 4 * v2;
  v5 = *(unsigned int *)(a1 + 40);
  v6 = &v3[4 * v5];
  if ( (unsigned int)(4 * v2) < 0x40 )
    v4 = 64;
  if ( (unsigned int)v5 <= v4 )
  {
    for ( ; v3 != v6; v3 += 4 )
    {
      if ( *v3 != -8 )
      {
        if ( *v3 != -16 )
        {
          v7 = v3[1];
          if ( v7 )
            j_j___libc_free_0(v7);
        }
        *v3 = -8;
      }
    }
    goto LABEL_13;
  }
  do
  {
    while ( *v3 == -16 )
    {
LABEL_36:
      v3 += 4;
      if ( v3 == v6 )
        goto LABEL_40;
    }
    if ( *v3 != -8 )
    {
      v23 = v3[1];
      if ( v23 )
        j_j___libc_free_0(v23);
      goto LABEL_36;
    }
    v3 += 4;
  }
  while ( v3 != v6 );
LABEL_40:
  v24 = *(_DWORD *)(a1 + 40);
  if ( !v2 )
  {
    if ( v24 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 24));
      *(_QWORD *)(a1 + 24) = 0;
      *(_QWORD *)(a1 + 32) = 0;
      *(_DWORD *)(a1 + 40) = 0;
      goto LABEL_14;
    }
LABEL_13:
    *(_QWORD *)(a1 + 32) = 0;
    goto LABEL_14;
  }
  v25 = 64;
  v26 = v2 - 1;
  if ( v26 )
  {
    _BitScanReverse(&v27, v26);
    v25 = 1 << (33 - (v27 ^ 0x1F));
    if ( v25 < 64 )
      v25 = 64;
  }
  v28 = *(_QWORD **)(a1 + 24);
  if ( v25 == v24 )
  {
    *(_QWORD *)(a1 + 32) = 0;
    v35 = &v28[4 * (unsigned int)v25];
    do
    {
      if ( v28 )
        *v28 = -8;
      v28 += 4;
    }
    while ( v35 != v28 );
  }
  else
  {
    j___libc_free_0((unsigned __int64)v28);
    v29 = ((((((((4 * v25 / 3u + 1) | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 2)
             | (4 * v25 / 3u + 1)
             | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 4)
           | (((4 * v25 / 3u + 1) | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 2)
           | (4 * v25 / 3u + 1)
           | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v25 / 3u + 1) | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 2)
           | (4 * v25 / 3u + 1)
           | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 4)
         | (((4 * v25 / 3u + 1) | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 2)
         | (4 * v25 / 3u + 1)
         | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 16;
    v30 = (v29
         | (((((((4 * v25 / 3u + 1) | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 2)
             | (4 * v25 / 3u + 1)
             | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 4)
           | (((4 * v25 / 3u + 1) | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 2)
           | (4 * v25 / 3u + 1)
           | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v25 / 3u + 1) | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 2)
           | (4 * v25 / 3u + 1)
           | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 4)
         | (((4 * v25 / 3u + 1) | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1)) >> 2)
         | (4 * v25 / 3u + 1)
         | ((unsigned __int64)(4 * v25 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 40) = v30;
    v31 = (_QWORD *)sub_22077B0(32 * v30);
    v32 = *(unsigned int *)(a1 + 40);
    *(_QWORD *)(a1 + 32) = 0;
    *(_QWORD *)(a1 + 24) = v31;
    for ( i = &v31[4 * v32]; i != v31; v31 += 4 )
    {
      if ( v31 )
        *v31 = -8;
    }
  }
LABEL_14:
  v8 = *(_DWORD *)(a1 + 64);
  ++*(_QWORD *)(a1 + 48);
  if ( v8 )
  {
    v12 = 4 * v8;
    v9 = *(unsigned int *)(a1 + 72);
    if ( (unsigned int)(4 * v8) < 0x40 )
      v12 = 64;
    if ( v12 >= (unsigned int)v9 )
    {
LABEL_17:
      v10 = *(_QWORD **)(a1 + 56);
      for ( j = &v10[2 * v9]; j != v10; v10 += 2 )
        *v10 = -8;
      *(_QWORD *)(a1 + 64) = 0;
      return;
    }
    v13 = *(_QWORD **)(a1 + 56);
    v14 = v8 - 1;
    if ( v14 )
    {
      _BitScanReverse(&v14, v14);
      v15 = 1 << (33 - (v14 ^ 0x1F));
      if ( v15 < 64 )
        v15 = 64;
      if ( (_DWORD)v9 == v15 )
      {
        *(_QWORD *)(a1 + 64) = 0;
        v34 = &v13[2 * (unsigned int)v9];
        do
        {
          if ( v13 )
            *v13 = -8;
          v13 += 2;
        }
        while ( v34 != v13 );
        return;
      }
      v16 = (((4 * v15 / 3u + 1) | ((unsigned __int64)(4 * v15 / 3u + 1) >> 1)) >> 2)
          | (4 * v15 / 3u + 1)
          | ((unsigned __int64)(4 * v15 / 3u + 1) >> 1)
          | (((((4 * v15 / 3u + 1) | ((unsigned __int64)(4 * v15 / 3u + 1) >> 1)) >> 2)
            | (4 * v15 / 3u + 1)
            | ((unsigned __int64)(4 * v15 / 3u + 1) >> 1)) >> 4);
      v17 = (v16 >> 8) | v16;
      v18 = (v17 | (v17 >> 16)) + 1;
      v19 = 16 * ((v17 | (v17 >> 16)) + 1);
    }
    else
    {
      v19 = 2048;
      v18 = 128;
    }
    j___libc_free_0((unsigned __int64)v13);
    *(_DWORD *)(a1 + 72) = v18;
    v20 = (_QWORD *)sub_22077B0(v19);
    v21 = *(unsigned int *)(a1 + 72);
    *(_QWORD *)(a1 + 64) = 0;
    *(_QWORD *)(a1 + 56) = v20;
    for ( k = &v20[2 * v21]; k != v20; v20 += 2 )
    {
      if ( v20 )
        *v20 = -8;
    }
    return;
  }
  if ( *(_DWORD *)(a1 + 68) )
  {
    v9 = *(unsigned int *)(a1 + 72);
    if ( (unsigned int)v9 <= 0x40 )
      goto LABEL_17;
    j___libc_free_0(*(_QWORD *)(a1 + 56));
    *(_QWORD *)(a1 + 56) = 0;
    *(_QWORD *)(a1 + 64) = 0;
    *(_DWORD *)(a1 + 72) = 0;
  }
}
