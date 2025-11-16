// Function: sub_2E5E230
// Address: 0x2e5e230
//
__int64 __fastcall sub_2E5E230(__int64 a1)
{
  unsigned __int64 *v2; // r12
  unsigned __int64 v3; // r15
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int64 *v6; // rbx
  unsigned __int64 *v7; // r14
  unsigned __int64 v8; // rdi
  int v9; // eax
  __int64 v10; // rdx
  _QWORD *v11; // rax
  _QWORD *i; // rdx
  int v13; // eax
  __int64 result; // rax
  __int64 v15; // rdx
  __int64 k; // rdx
  unsigned int v17; // ecx
  unsigned int v18; // eax
  _QWORD *v19; // rdi
  int v20; // ebx
  unsigned int v21; // ecx
  unsigned int v22; // eax
  _QWORD *v23; // rdi
  int v24; // ebx
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rdi
  _QWORD *v27; // rax
  __int64 v28; // rdx
  _QWORD *j; // rdx
  unsigned __int64 v30; // rax
  unsigned __int64 v31; // rdi
  __int64 v32; // rdx
  __int64 m; // rdx
  _QWORD *v34; // rax
  unsigned __int64 *v35; // [rsp+0h] [rbp-40h]
  unsigned __int64 *v36; // [rsp+8h] [rbp-38h]

  v35 = *(unsigned __int64 **)(a1 + 72);
  v36 = *(unsigned __int64 **)(a1 + 80);
  if ( v35 != v36 )
  {
    v2 = *(unsigned __int64 **)(a1 + 72);
    do
    {
      v3 = *v2;
      if ( *v2 )
      {
        v4 = *(_QWORD *)(v3 + 176);
        if ( v4 != v3 + 192 )
          _libc_free(v4);
        v5 = *(_QWORD *)(v3 + 88);
        if ( v5 != v3 + 104 )
          _libc_free(v5);
        sub_C7D6A0(*(_QWORD *)(v3 + 64), 8LL * *(unsigned int *)(v3 + 80), 8);
        v6 = *(unsigned __int64 **)(v3 + 40);
        v7 = *(unsigned __int64 **)(v3 + 32);
        if ( v6 != v7 )
        {
          do
          {
            if ( *v7 )
              sub_2E5DCD0(*v7);
            ++v7;
          }
          while ( v6 != v7 );
          v7 = *(unsigned __int64 **)(v3 + 32);
        }
        if ( v7 )
          j_j___libc_free_0((unsigned __int64)v7);
        v8 = *(_QWORD *)(v3 + 8);
        if ( v8 != v3 + 24 )
          _libc_free(v8);
        j_j___libc_free_0(v3);
      }
      ++v2;
    }
    while ( v36 != v2 );
    *(_QWORD *)(a1 + 80) = v35;
  }
  v9 = *(_DWORD *)(a1 + 24);
  ++*(_QWORD *)(a1 + 8);
  if ( !v9 )
  {
    if ( !*(_DWORD *)(a1 + 28) )
      goto LABEL_26;
    v10 = *(unsigned int *)(a1 + 32);
    if ( (unsigned int)v10 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 16), 16LL * (unsigned int)v10, 8);
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      *(_DWORD *)(a1 + 32) = 0;
      goto LABEL_26;
    }
    goto LABEL_23;
  }
  v21 = 4 * v9;
  v10 = *(unsigned int *)(a1 + 32);
  if ( (unsigned int)(4 * v9) < 0x40 )
    v21 = 64;
  if ( v21 >= (unsigned int)v10 )
  {
LABEL_23:
    v11 = *(_QWORD **)(a1 + 16);
    for ( i = &v11[2 * v10]; i != v11; v11 += 2 )
      *v11 = -4096;
    *(_QWORD *)(a1 + 24) = 0;
    goto LABEL_26;
  }
  v22 = v9 - 1;
  if ( !v22 )
  {
    v23 = *(_QWORD **)(a1 + 16);
    v24 = 64;
LABEL_52:
    sub_C7D6A0((__int64)v23, 16LL * (unsigned int)v10, 8);
    v25 = ((((((((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
             | (4 * v24 / 3u + 1)
             | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 4)
           | (((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
           | (4 * v24 / 3u + 1)
           | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
           | (4 * v24 / 3u + 1)
           | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 4)
         | (((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
         | (4 * v24 / 3u + 1)
         | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 16;
    v26 = (v25
         | (((((((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
             | (4 * v24 / 3u + 1)
             | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 4)
           | (((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
           | (4 * v24 / 3u + 1)
           | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
           | (4 * v24 / 3u + 1)
           | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 4)
         | (((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
         | (4 * v24 / 3u + 1)
         | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 32) = v26;
    v27 = (_QWORD *)sub_C7D670(16 * v26, 8);
    v28 = *(unsigned int *)(a1 + 32);
    *(_QWORD *)(a1 + 24) = 0;
    *(_QWORD *)(a1 + 16) = v27;
    for ( j = &v27[2 * v28]; j != v27; v27 += 2 )
    {
      if ( v27 )
        *v27 = -4096;
    }
    goto LABEL_26;
  }
  _BitScanReverse(&v22, v22);
  v23 = *(_QWORD **)(a1 + 16);
  v24 = 1 << (33 - (v22 ^ 0x1F));
  if ( v24 < 64 )
    v24 = 64;
  if ( v24 != (_DWORD)v10 )
    goto LABEL_52;
  *(_QWORD *)(a1 + 24) = 0;
  v34 = &v23[2 * (unsigned int)v24];
  do
  {
    if ( v23 )
      *v23 = -4096;
    v23 += 2;
  }
  while ( v34 != v23 );
LABEL_26:
  v13 = *(_DWORD *)(a1 + 56);
  ++*(_QWORD *)(a1 + 40);
  if ( !v13 )
  {
    result = *(unsigned int *)(a1 + 60);
    if ( !(_DWORD)result )
      return result;
    v15 = *(unsigned int *)(a1 + 64);
    if ( (unsigned int)v15 > 0x40 )
    {
      result = sub_C7D6A0(*(_QWORD *)(a1 + 48), 16LL * (unsigned int)v15, 8);
      *(_QWORD *)(a1 + 48) = 0;
      *(_QWORD *)(a1 + 56) = 0;
      *(_DWORD *)(a1 + 64) = 0;
      return result;
    }
    goto LABEL_29;
  }
  v17 = 4 * v13;
  v15 = *(unsigned int *)(a1 + 64);
  if ( (unsigned int)(4 * v13) < 0x40 )
    v17 = 64;
  if ( v17 >= (unsigned int)v15 )
  {
LABEL_29:
    result = *(_QWORD *)(a1 + 48);
    for ( k = result + 16 * v15; k != result; result += 16 )
      *(_QWORD *)result = -4096;
    *(_QWORD *)(a1 + 56) = 0;
    return result;
  }
  v18 = v13 - 1;
  if ( v18 )
  {
    _BitScanReverse(&v18, v18);
    v19 = *(_QWORD **)(a1 + 48);
    v20 = 1 << (33 - (v18 ^ 0x1F));
    if ( v20 < 64 )
      v20 = 64;
    if ( v20 == (_DWORD)v15 )
    {
      *(_QWORD *)(a1 + 56) = 0;
      result = (__int64)&v19[2 * (unsigned int)v20];
      do
      {
        if ( v19 )
          *v19 = -4096;
        v19 += 2;
      }
      while ( (_QWORD *)result != v19 );
      return result;
    }
  }
  else
  {
    v19 = *(_QWORD **)(a1 + 48);
    v20 = 64;
  }
  sub_C7D6A0((__int64)v19, 16LL * (unsigned int)v15, 8);
  v30 = ((((((((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
           | (4 * v20 / 3u + 1)
           | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 4)
         | (((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
         | (4 * v20 / 3u + 1)
         | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
         | (4 * v20 / 3u + 1)
         | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 4)
       | (((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
       | (4 * v20 / 3u + 1)
       | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 16;
  v31 = (v30
       | (((((((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
           | (4 * v20 / 3u + 1)
           | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 4)
         | (((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
         | (4 * v20 / 3u + 1)
         | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
         | (4 * v20 / 3u + 1)
         | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 4)
       | (((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
       | (4 * v20 / 3u + 1)
       | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1))
      + 1;
  *(_DWORD *)(a1 + 64) = v31;
  result = sub_C7D670(16 * v31, 8);
  v32 = *(unsigned int *)(a1 + 64);
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 48) = result;
  for ( m = result + 16 * v32; m != result; result += 16 )
  {
    if ( result )
      *(_QWORD *)result = -4096;
  }
  return result;
}
