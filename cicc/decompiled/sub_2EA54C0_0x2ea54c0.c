// Function: sub_2EA54C0
// Address: 0x2ea54c0
//
_QWORD *__fastcall sub_2EA54C0(__int64 a1, __int64 a2)
{
  int v3; // eax
  __int64 v4; // rdx
  _QWORD *v5; // rax
  _QWORD *i; // rdx
  _QWORD *result; // rax
  _QWORD *v8; // r12
  __int64 v9; // r15
  __int64 *v10; // rbx
  __int64 *v11; // r14
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rax
  unsigned int v15; // eax
  __int64 v16; // rdx
  char v17; // al
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  __int64 *v20; // rbx
  __int64 *k; // r12
  __int64 v22; // rsi
  __int64 v23; // rdi
  __int64 v24; // rdx
  unsigned int v25; // ecx
  unsigned int v26; // eax
  _QWORD *v27; // rdi
  int v28; // ebx
  unsigned __int64 v29; // rdx
  unsigned __int64 v30; // rax
  _QWORD *v31; // rax
  __int64 v32; // rdx
  _QWORD *j; // rdx
  __int64 v34; // rcx
  __int64 *v35; // rbx
  __int64 *v36; // r14
  __int64 v37; // rdi
  unsigned int v38; // ecx
  __int64 v39; // rsi
  _QWORD *v40; // rax
  _QWORD *v41; // [rsp+8h] [rbp-38h]

  v3 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v3 )
  {
    if ( !*(_DWORD *)(a1 + 20) )
      goto LABEL_7;
    v4 = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)v4 > 0x40 )
    {
      a2 = 16 * v4;
      sub_C7D6A0(*(_QWORD *)(a1 + 8), 16 * v4, 8);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v25 = 4 * v3;
  a2 = 64;
  v4 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)(4 * v3) < 0x40 )
    v25 = 64;
  if ( (unsigned int)v4 <= v25 )
  {
LABEL_4:
    v5 = *(_QWORD **)(a1 + 8);
    for ( i = &v5[2 * v4]; i != v5; v5 += 2 )
      *v5 = -4096;
    *(_QWORD *)(a1 + 16) = 0;
    goto LABEL_7;
  }
  v26 = v3 - 1;
  if ( !v26 )
  {
    v27 = *(_QWORD **)(a1 + 8);
    v28 = 64;
LABEL_43:
    sub_C7D6A0((__int64)v27, 16 * v4, 8);
    a2 = 8;
    v29 = ((((((((4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 2)
             | (4 * v28 / 3u + 1)
             | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 4)
           | (((4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 2)
           | (4 * v28 / 3u + 1)
           | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 2)
           | (4 * v28 / 3u + 1)
           | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 4)
         | (((4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 2)
         | (4 * v28 / 3u + 1)
         | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 16;
    v30 = (v29
         | (((((((4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 2)
             | (4 * v28 / 3u + 1)
             | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 4)
           | (((4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 2)
           | (4 * v28 / 3u + 1)
           | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 2)
           | (4 * v28 / 3u + 1)
           | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 4)
         | (((4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 2)
         | (4 * v28 / 3u + 1)
         | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 24) = v30;
    v31 = (_QWORD *)sub_C7D670(16 * v30, 8);
    v32 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = v31;
    for ( j = &v31[2 * v32]; j != v31; v31 += 2 )
    {
      if ( v31 )
        *v31 = -4096;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v26, v26);
  v27 = *(_QWORD **)(a1 + 8);
  v28 = 1 << (33 - (v26 ^ 0x1F));
  if ( v28 < 64 )
    v28 = 64;
  if ( (_DWORD)v4 != v28 )
    goto LABEL_43;
  *(_QWORD *)(a1 + 16) = 0;
  v40 = &v27[2 * (unsigned int)v4];
  do
  {
    if ( v27 )
      *v27 = -4096;
    v27 += 2;
  }
  while ( v40 != v27 );
LABEL_7:
  result = *(_QWORD **)(a1 + 40);
  v8 = *(_QWORD **)(a1 + 32);
  v41 = result;
  if ( v8 != result )
  {
    do
    {
      v9 = *v8;
      v10 = *(__int64 **)(*v8 + 16LL);
      if ( *(__int64 **)(*v8 + 8LL) == v10 )
      {
        *(_BYTE *)(v9 + 152) = 1;
      }
      else
      {
        v11 = *(__int64 **)(*v8 + 8LL);
        do
        {
          v12 = *v11++;
          sub_2EA4EF0(v12, a2);
        }
        while ( v10 != v11 );
        *(_BYTE *)(v9 + 152) = 1;
        v13 = *(_QWORD *)(v9 + 8);
        if ( v13 != *(_QWORD *)(v9 + 16) )
          *(_QWORD *)(v9 + 16) = v13;
      }
      v14 = *(_QWORD *)(v9 + 32);
      if ( v14 != *(_QWORD *)(v9 + 40) )
        *(_QWORD *)(v9 + 40) = v14;
      ++*(_QWORD *)(v9 + 56);
      if ( *(_BYTE *)(v9 + 84) )
      {
        *(_QWORD *)v9 = 0;
      }
      else
      {
        v15 = 4 * (*(_DWORD *)(v9 + 76) - *(_DWORD *)(v9 + 80));
        v16 = *(unsigned int *)(v9 + 72);
        if ( v15 < 0x20 )
          v15 = 32;
        if ( (unsigned int)v16 > v15 )
        {
          sub_C8C990(v9 + 56, a2);
        }
        else
        {
          a2 = 0xFFFFFFFFLL;
          memset(*(void **)(v9 + 64), -1, 8 * v16);
        }
        v17 = *(_BYTE *)(v9 + 84);
        *(_QWORD *)v9 = 0;
        if ( !v17 )
          _libc_free(*(_QWORD *)(v9 + 64));
      }
      v18 = *(_QWORD *)(v9 + 32);
      if ( v18 )
      {
        a2 = *(_QWORD *)(v9 + 48) - v18;
        j_j___libc_free_0(v18);
      }
      v19 = *(_QWORD *)(v9 + 8);
      if ( v19 )
      {
        a2 = *(_QWORD *)(v9 + 24) - v19;
        j_j___libc_free_0(v19);
      }
      ++v8;
    }
    while ( v41 != v8 );
    result = *(_QWORD **)(a1 + 32);
    if ( *(_QWORD **)(a1 + 40) != result )
      *(_QWORD *)(a1 + 40) = result;
  }
  v20 = *(__int64 **)(a1 + 120);
  for ( k = &v20[2 * *(unsigned int *)(a1 + 128)]; k != v20; result = (_QWORD *)sub_C7D6A0(v23, v22, 16) )
  {
    v22 = v20[1];
    v23 = *v20;
    v20 += 2;
  }
  *(_DWORD *)(a1 + 128) = 0;
  v24 = *(unsigned int *)(a1 + 80);
  if ( (_DWORD)v24 )
  {
    *(_QWORD *)(a1 + 136) = 0;
    result = *(_QWORD **)(a1 + 72);
    v34 = *result;
    v35 = &result[v24];
    v36 = result + 1;
    *(_QWORD *)(a1 + 56) = *result;
    *(_QWORD *)(a1 + 64) = v34 + 4096;
    if ( v35 != result + 1 )
    {
      while ( 1 )
      {
        v37 = *v36;
        v38 = (unsigned int)(v36 - result) >> 7;
        v39 = 4096LL << v38;
        if ( v38 >= 0x1E )
          v39 = 0x40000000000LL;
        ++v36;
        result = (_QWORD *)sub_C7D6A0(v37, v39, 16);
        if ( v35 == v36 )
          break;
        result = *(_QWORD **)(a1 + 72);
      }
    }
    *(_DWORD *)(a1 + 80) = 1;
  }
  return result;
}
