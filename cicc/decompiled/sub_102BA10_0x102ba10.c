// Function: sub_102BA10
// Address: 0x102ba10
//
__int64 __fastcall sub_102BA10(__int64 a1)
{
  int v2; // eax
  __int64 result; // rax
  __int64 v4; // rdx
  __int64 i; // rdx
  __int64 *v6; // rbx
  __int64 *k; // r13
  __int64 v8; // rsi
  __int64 v9; // rdi
  __int64 v10; // rdx
  unsigned int v11; // ecx
  unsigned int v12; // eax
  _QWORD *v13; // rdi
  int v14; // ebx
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // rax
  __int64 v17; // rdx
  __int64 j; // rdx
  __int64 v19; // rcx
  __int64 *v20; // rbx
  __int64 *v21; // r14
  __int64 v22; // rdi
  unsigned int v23; // ecx
  __int64 v24; // rsi

  v2 = *(_DWORD *)(a1 + 304);
  ++*(_QWORD *)(a1 + 288);
  if ( !v2 )
  {
    result = *(unsigned int *)(a1 + 308);
    if ( !(_DWORD)result )
      goto LABEL_7;
    v4 = *(unsigned int *)(a1 + 312);
    if ( (unsigned int)v4 > 0x40 )
    {
      result = sub_C7D6A0(*(_QWORD *)(a1 + 296), 24 * v4, 8);
      *(_QWORD *)(a1 + 296) = 0;
      *(_QWORD *)(a1 + 304) = 0;
      *(_DWORD *)(a1 + 312) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v11 = 4 * v2;
  v4 = *(unsigned int *)(a1 + 312);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v11 = 64;
  if ( (unsigned int)v4 <= v11 )
  {
LABEL_4:
    result = *(_QWORD *)(a1 + 296);
    for ( i = result + 24 * v4; i != result; result += 24 )
      *(_QWORD *)result = -4096;
    *(_QWORD *)(a1 + 304) = 0;
    goto LABEL_7;
  }
  v12 = v2 - 1;
  if ( !v12 )
  {
    v13 = *(_QWORD **)(a1 + 296);
    v14 = 64;
LABEL_18:
    sub_C7D6A0((__int64)v13, 24 * v4, 8);
    v15 = ((((((((4 * v14 / 3u + 1) | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 2)
             | (4 * v14 / 3u + 1)
             | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 4)
           | (((4 * v14 / 3u + 1) | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 2)
           | (4 * v14 / 3u + 1)
           | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v14 / 3u + 1) | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 2)
           | (4 * v14 / 3u + 1)
           | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 4)
         | (((4 * v14 / 3u + 1) | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 2)
         | (4 * v14 / 3u + 1)
         | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 16;
    v16 = (v15
         | (((((((4 * v14 / 3u + 1) | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 2)
             | (4 * v14 / 3u + 1)
             | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 4)
           | (((4 * v14 / 3u + 1) | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 2)
           | (4 * v14 / 3u + 1)
           | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v14 / 3u + 1) | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 2)
           | (4 * v14 / 3u + 1)
           | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 4)
         | (((4 * v14 / 3u + 1) | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 2)
         | (4 * v14 / 3u + 1)
         | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 312) = v16;
    result = sub_C7D670(24 * v16, 8);
    v17 = *(unsigned int *)(a1 + 312);
    *(_QWORD *)(a1 + 304) = 0;
    *(_QWORD *)(a1 + 296) = result;
    for ( j = result + 24 * v17; j != result; result += 24 )
    {
      if ( result )
        *(_QWORD *)result = -4096;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v12, v12);
  v13 = *(_QWORD **)(a1 + 296);
  v14 = 1 << (33 - (v12 ^ 0x1F));
  if ( v14 < 64 )
    v14 = 64;
  if ( (_DWORD)v4 != v14 )
    goto LABEL_18;
  *(_QWORD *)(a1 + 304) = 0;
  result = (__int64)&v13[3 * v4];
  do
  {
    if ( v13 )
      *v13 = -4096;
    v13 += 3;
  }
  while ( (_QWORD *)result != v13 );
LABEL_7:
  v6 = *(__int64 **)(a1 + 384);
  for ( k = &v6[2 * *(unsigned int *)(a1 + 392)]; k != v6; result = sub_C7D6A0(v9, v8, 16) )
  {
    v8 = v6[1];
    v9 = *v6;
    v6 += 2;
  }
  v10 = *(unsigned int *)(a1 + 344);
  *(_DWORD *)(a1 + 392) = 0;
  if ( (_DWORD)v10 )
  {
    result = *(_QWORD *)(a1 + 336);
    *(_QWORD *)(a1 + 400) = 0;
    v19 = *(_QWORD *)result;
    v20 = (__int64 *)(result + 8 * v10);
    v21 = (__int64 *)(result + 8);
    *(_QWORD *)(a1 + 320) = *(_QWORD *)result;
    *(_QWORD *)(a1 + 328) = v19 + 4096;
    if ( v20 != (__int64 *)(result + 8) )
    {
      while ( 1 )
      {
        v22 = *v21;
        v23 = (unsigned int)(((__int64)v21 - result) >> 3) >> 7;
        v24 = 4096LL << v23;
        if ( v23 >= 0x1E )
          v24 = 0x40000000000LL;
        ++v21;
        result = sub_C7D6A0(v22, v24, 16);
        if ( v20 == v21 )
          break;
        result = *(_QWORD *)(a1 + 336);
      }
    }
    *(_DWORD *)(a1 + 344) = 1;
  }
  return result;
}
