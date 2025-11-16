// Function: sub_E38360
// Address: 0xe38360
//
__int64 __fastcall sub_E38360(__int64 a1, __int64 a2)
{
  __int64 *v3; // r12
  __int64 v4; // r15
  __int64 v5; // rdi
  __int64 v6; // rdi
  __int64 v7; // rsi
  __int64 *v8; // rbx
  __int64 *v9; // r14
  __int64 v10; // rdi
  int v11; // eax
  __int64 v12; // rdx
  _QWORD *v13; // rax
  _QWORD *i; // rdx
  int v15; // eax
  __int64 result; // rax
  __int64 v17; // rdx
  __int64 k; // rdx
  unsigned int v19; // ecx
  unsigned int v20; // eax
  _QWORD *v21; // rdi
  int v22; // ebx
  unsigned int v23; // ecx
  unsigned int v24; // eax
  _QWORD *v25; // rdi
  int v26; // ebx
  unsigned __int64 v27; // rdx
  unsigned __int64 v28; // rax
  _QWORD *v29; // rax
  __int64 v30; // rdx
  _QWORD *j; // rdx
  unsigned __int64 v32; // rdx
  unsigned __int64 v33; // rax
  __int64 v34; // rdx
  __int64 m; // rdx
  _QWORD *v36; // rax
  __int64 *v37; // [rsp+0h] [rbp-40h]
  __int64 *v38; // [rsp+8h] [rbp-38h]

  v37 = *(__int64 **)(a1 + 72);
  v38 = *(__int64 **)(a1 + 80);
  if ( v37 != v38 )
  {
    v3 = *(__int64 **)(a1 + 72);
    do
    {
      v4 = *v3;
      if ( *v3 )
      {
        v5 = *(_QWORD *)(v4 + 176);
        if ( v5 != v4 + 192 )
          _libc_free(v5, a2);
        v6 = *(_QWORD *)(v4 + 88);
        if ( v6 != v4 + 104 )
          _libc_free(v6, a2);
        v7 = 8LL * *(unsigned int *)(v4 + 80);
        sub_C7D6A0(*(_QWORD *)(v4 + 64), v7, 8);
        v8 = *(__int64 **)(v4 + 40);
        v9 = *(__int64 **)(v4 + 32);
        if ( v8 != v9 )
        {
          do
          {
            if ( *v9 )
              sub_E38110(*v9, v7);
            ++v9;
          }
          while ( v8 != v9 );
          v9 = *(__int64 **)(v4 + 32);
        }
        if ( v9 )
        {
          v7 = *(_QWORD *)(v4 + 48) - (_QWORD)v9;
          j_j___libc_free_0(v9, v7);
        }
        v10 = *(_QWORD *)(v4 + 8);
        if ( v10 != v4 + 24 )
          _libc_free(v10, v7);
        a2 = 224;
        j_j___libc_free_0(v4, 224);
      }
      ++v3;
    }
    while ( v38 != v3 );
    *(_QWORD *)(a1 + 80) = v37;
  }
  v11 = *(_DWORD *)(a1 + 24);
  ++*(_QWORD *)(a1 + 8);
  if ( !v11 )
  {
    if ( !*(_DWORD *)(a1 + 28) )
      goto LABEL_26;
    v12 = *(unsigned int *)(a1 + 32);
    if ( (unsigned int)v12 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 16), 16LL * (unsigned int)v12, 8);
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      *(_DWORD *)(a1 + 32) = 0;
      goto LABEL_26;
    }
    goto LABEL_23;
  }
  v23 = 4 * v11;
  v12 = *(unsigned int *)(a1 + 32);
  if ( (unsigned int)(4 * v11) < 0x40 )
    v23 = 64;
  if ( v23 >= (unsigned int)v12 )
  {
LABEL_23:
    v13 = *(_QWORD **)(a1 + 16);
    for ( i = &v13[2 * v12]; i != v13; v13 += 2 )
      *v13 = -4096;
    *(_QWORD *)(a1 + 24) = 0;
    goto LABEL_26;
  }
  v24 = v11 - 1;
  if ( !v24 )
  {
    v25 = *(_QWORD **)(a1 + 16);
    v26 = 64;
LABEL_52:
    sub_C7D6A0((__int64)v25, 16LL * *(unsigned int *)(a1 + 32), 8);
    v27 = ((((((((4 * v26 / 3u + 1) | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 2)
             | (4 * v26 / 3u + 1)
             | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 4)
           | (((4 * v26 / 3u + 1) | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 2)
           | (4 * v26 / 3u + 1)
           | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v26 / 3u + 1) | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 2)
           | (4 * v26 / 3u + 1)
           | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 4)
         | (((4 * v26 / 3u + 1) | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 2)
         | (4 * v26 / 3u + 1)
         | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 16;
    v28 = (v27
         | (((((((4 * v26 / 3u + 1) | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 2)
             | (4 * v26 / 3u + 1)
             | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 4)
           | (((4 * v26 / 3u + 1) | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 2)
           | (4 * v26 / 3u + 1)
           | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v26 / 3u + 1) | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 2)
           | (4 * v26 / 3u + 1)
           | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 4)
         | (((4 * v26 / 3u + 1) | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 2)
         | (4 * v26 / 3u + 1)
         | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 32) = v28;
    v29 = (_QWORD *)sub_C7D670(16 * v28, 8);
    v30 = *(unsigned int *)(a1 + 32);
    *(_QWORD *)(a1 + 24) = 0;
    *(_QWORD *)(a1 + 16) = v29;
    for ( j = &v29[2 * v30]; j != v29; v29 += 2 )
    {
      if ( v29 )
        *v29 = -4096;
    }
    goto LABEL_26;
  }
  _BitScanReverse(&v24, v24);
  v25 = *(_QWORD **)(a1 + 16);
  v26 = 1 << (33 - (v24 ^ 0x1F));
  if ( v26 < 64 )
    v26 = 64;
  if ( v26 != (_DWORD)v12 )
    goto LABEL_52;
  *(_QWORD *)(a1 + 24) = 0;
  v36 = &v25[2 * (unsigned int)v26];
  do
  {
    if ( v25 )
      *v25 = -4096;
    v25 += 2;
  }
  while ( v36 != v25 );
LABEL_26:
  v15 = *(_DWORD *)(a1 + 56);
  ++*(_QWORD *)(a1 + 40);
  if ( !v15 )
  {
    result = *(unsigned int *)(a1 + 60);
    if ( !(_DWORD)result )
      return result;
    v17 = *(unsigned int *)(a1 + 64);
    if ( (unsigned int)v17 > 0x40 )
    {
      result = sub_C7D6A0(*(_QWORD *)(a1 + 48), 16LL * (unsigned int)v17, 8);
      *(_QWORD *)(a1 + 48) = 0;
      *(_QWORD *)(a1 + 56) = 0;
      *(_DWORD *)(a1 + 64) = 0;
      return result;
    }
    goto LABEL_29;
  }
  v19 = 4 * v15;
  v17 = *(unsigned int *)(a1 + 64);
  if ( (unsigned int)(4 * v15) < 0x40 )
    v19 = 64;
  if ( v19 >= (unsigned int)v17 )
  {
LABEL_29:
    result = *(_QWORD *)(a1 + 48);
    for ( k = result + 16 * v17; k != result; result += 16 )
      *(_QWORD *)result = -4096;
    *(_QWORD *)(a1 + 56) = 0;
    return result;
  }
  v20 = v15 - 1;
  if ( v20 )
  {
    _BitScanReverse(&v20, v20);
    v21 = *(_QWORD **)(a1 + 48);
    v22 = 1 << (33 - (v20 ^ 0x1F));
    if ( v22 < 64 )
      v22 = 64;
    if ( v22 == (_DWORD)v17 )
    {
      *(_QWORD *)(a1 + 56) = 0;
      result = (__int64)&v21[2 * (unsigned int)v22];
      do
      {
        if ( v21 )
          *v21 = -4096;
        v21 += 2;
      }
      while ( (_QWORD *)result != v21 );
      return result;
    }
  }
  else
  {
    v21 = *(_QWORD **)(a1 + 48);
    v22 = 64;
  }
  sub_C7D6A0((__int64)v21, 16 * v17, 8);
  v32 = ((((((((4 * v22 / 3u + 1) | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1)) >> 2)
           | (4 * v22 / 3u + 1)
           | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1)) >> 4)
         | (((4 * v22 / 3u + 1) | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1)) >> 2)
         | (4 * v22 / 3u + 1)
         | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v22 / 3u + 1) | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1)) >> 2)
         | (4 * v22 / 3u + 1)
         | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1)) >> 4)
       | (((4 * v22 / 3u + 1) | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1)) >> 2)
       | (4 * v22 / 3u + 1)
       | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1)) >> 16;
  v33 = (v32
       | (((((((4 * v22 / 3u + 1) | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1)) >> 2)
           | (4 * v22 / 3u + 1)
           | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1)) >> 4)
         | (((4 * v22 / 3u + 1) | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1)) >> 2)
         | (4 * v22 / 3u + 1)
         | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v22 / 3u + 1) | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1)) >> 2)
         | (4 * v22 / 3u + 1)
         | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1)) >> 4)
       | (((4 * v22 / 3u + 1) | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1)) >> 2)
       | (4 * v22 / 3u + 1)
       | ((unsigned __int64)(4 * v22 / 3u + 1) >> 1))
      + 1;
  *(_DWORD *)(a1 + 64) = v33;
  result = sub_C7D670(16 * v33, 8);
  v34 = *(unsigned int *)(a1 + 64);
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 48) = result;
  for ( m = result + 16 * v34; m != result; result += 16 )
  {
    if ( result )
      *(_QWORD *)result = -4096;
  }
  return result;
}
