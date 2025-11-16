// Function: sub_2FAD1F0
// Address: 0x2fad1f0
//
__int64 *__fastcall sub_2FAD1F0(__int64 a1)
{
  int v2; // eax
  __int64 v3; // rdx
  _QWORD *v4; // rax
  _QWORD *i; // rdx
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 *v8; // rbx
  __int64 *result; // rax
  __int64 *v10; // r12
  __int64 v11; // rsi
  __int64 v12; // rdi
  __int64 v13; // rdx
  unsigned int v14; // ecx
  unsigned int v15; // eax
  _QWORD *v16; // rdi
  int v17; // ebx
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rdi
  _QWORD *v20; // rax
  __int64 v21; // rdx
  _QWORD *j; // rdx
  __int64 v23; // rcx
  __int64 *v24; // rbx
  __int64 *v25; // r14
  __int64 v26; // rdi
  unsigned int v27; // ecx
  __int64 v28; // rsi
  _QWORD *v29; // rax

  v2 = *(_DWORD *)(a1 + 136);
  ++*(_QWORD *)(a1 + 120);
  if ( !v2 )
  {
    if ( !*(_DWORD *)(a1 + 140) )
      goto LABEL_7;
    v3 = *(unsigned int *)(a1 + 144);
    if ( (unsigned int)v3 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 128), 16LL * (unsigned int)v3, 8);
      *(_QWORD *)(a1 + 128) = 0;
      *(_QWORD *)(a1 + 136) = 0;
      *(_DWORD *)(a1 + 144) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v14 = 4 * v2;
  v3 = *(unsigned int *)(a1 + 144);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v14 = 64;
  if ( (unsigned int)v3 <= v14 )
  {
LABEL_4:
    v4 = *(_QWORD **)(a1 + 128);
    for ( i = &v4[2 * v3]; i != v4; v4 += 2 )
      *v4 = -4096;
    *(_QWORD *)(a1 + 136) = 0;
    goto LABEL_7;
  }
  v15 = v2 - 1;
  if ( !v15 )
  {
    v16 = *(_QWORD **)(a1 + 128);
    v17 = 64;
LABEL_18:
    sub_C7D6A0((__int64)v16, 16LL * (unsigned int)v3, 8);
    v18 = ((((((((4 * v17 / 3u + 1) | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 2)
             | (4 * v17 / 3u + 1)
             | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 4)
           | (((4 * v17 / 3u + 1) | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 2)
           | (4 * v17 / 3u + 1)
           | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v17 / 3u + 1) | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 2)
           | (4 * v17 / 3u + 1)
           | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 4)
         | (((4 * v17 / 3u + 1) | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 2)
         | (4 * v17 / 3u + 1)
         | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 16;
    v19 = (v18
         | (((((((4 * v17 / 3u + 1) | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 2)
             | (4 * v17 / 3u + 1)
             | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 4)
           | (((4 * v17 / 3u + 1) | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 2)
           | (4 * v17 / 3u + 1)
           | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v17 / 3u + 1) | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 2)
           | (4 * v17 / 3u + 1)
           | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 4)
         | (((4 * v17 / 3u + 1) | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 2)
         | (4 * v17 / 3u + 1)
         | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 144) = v19;
    v20 = (_QWORD *)sub_C7D670(16 * v19, 8);
    v21 = *(unsigned int *)(a1 + 144);
    *(_QWORD *)(a1 + 136) = 0;
    *(_QWORD *)(a1 + 128) = v20;
    for ( j = &v20[2 * v21]; j != v20; v20 += 2 )
    {
      if ( v20 )
        *v20 = -4096;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v15, v15);
  v16 = *(_QWORD **)(a1 + 128);
  v17 = 1 << (33 - (v15 ^ 0x1F));
  if ( v17 < 64 )
    v17 = 64;
  if ( (_DWORD)v3 != v17 )
    goto LABEL_18;
  *(_QWORD *)(a1 + 136) = 0;
  v29 = &v16[2 * (unsigned int)v3];
  do
  {
    if ( v16 )
      *v16 = -4096;
    v16 += 2;
  }
  while ( v29 != v16 );
LABEL_7:
  v6 = *(_QWORD *)(a1 + 96);
  v7 = *(unsigned int *)(a1 + 72);
  *(_DWORD *)(a1 + 160) = 0;
  v8 = *(__int64 **)(a1 + 64);
  *(_QWORD *)(a1 + 104) = a1 + 96;
  *(_DWORD *)(a1 + 304) = 0;
  result = (__int64 *)((a1 + 96) | v6 & 7);
  v10 = &v8[2 * v7];
  for ( *(_QWORD *)(a1 + 96) = result; v10 != v8; result = (__int64 *)sub_C7D6A0(v12, v11, 16) )
  {
    v11 = v8[1];
    v12 = *v8;
    v8 += 2;
  }
  v13 = *(unsigned int *)(a1 + 24);
  *(_DWORD *)(a1 + 72) = 0;
  if ( (_DWORD)v13 )
  {
    result = *(__int64 **)(a1 + 16);
    *(_QWORD *)(a1 + 80) = 0;
    v23 = *result;
    v24 = &result[v13];
    v25 = result + 1;
    *(_QWORD *)a1 = *result;
    *(_QWORD *)(a1 + 8) = v23 + 4096;
    if ( v24 != result + 1 )
    {
      while ( 1 )
      {
        v26 = *v25;
        v27 = (unsigned int)(v25 - result) >> 7;
        v28 = 4096LL << v27;
        if ( v27 >= 0x1E )
          v28 = 0x40000000000LL;
        ++v25;
        result = (__int64 *)sub_C7D6A0(v26, v28, 16);
        if ( v24 == v25 )
          break;
        result = *(__int64 **)(a1 + 16);
      }
    }
    *(_DWORD *)(a1 + 24) = 1;
  }
  return result;
}
