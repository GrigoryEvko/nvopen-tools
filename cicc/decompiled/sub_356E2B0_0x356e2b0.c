// Function: sub_356E2B0
// Address: 0x356e2b0
//
__int64 __fastcall sub_356E2B0(__int64 *a1, int a2)
{
  __int64 v3; // r12
  int v5; // eax
  unsigned int v6; // ecx
  __int64 v7; // rdx
  _QWORD *v8; // rax
  _QWORD *i; // rdx
  __int64 v10; // rdx
  unsigned int v11; // esi
  __int64 v12; // rdi
  __int64 v13; // rcx
  unsigned __int64 v14; // rsi
  bool v15; // dl
  bool v16; // al
  unsigned __int8 v17; // di
  __int64 result; // rax
  unsigned __int64 v19; // rsi
  char v20; // al
  __int64 v21; // rax
  unsigned int v22; // eax
  _QWORD *v23; // rdi
  int v24; // r14d
  _QWORD *v25; // rax
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rdi
  _QWORD *v28; // rax
  __int64 v29; // rdx
  _QWORD *j; // rdx

  v3 = *a1;
  if ( *a1 )
  {
    v5 = *(_DWORD *)(v3 + 16);
    ++*(_QWORD *)v3;
    if ( !v5 )
    {
      if ( !*(_DWORD *)(v3 + 20) )
        goto LABEL_9;
      v7 = *(unsigned int *)(v3 + 24);
      if ( (unsigned int)v7 > 0x40 )
      {
        sub_C7D6A0(*(_QWORD *)(v3 + 8), 16LL * (unsigned int)v7, 8);
        *(_QWORD *)(v3 + 8) = 0;
        *(_QWORD *)(v3 + 16) = 0;
        *(_DWORD *)(v3 + 24) = 0;
        goto LABEL_9;
      }
      goto LABEL_6;
    }
    v6 = 4 * v5;
    v7 = *(unsigned int *)(v3 + 24);
    if ( (unsigned int)(4 * v5) < 0x40 )
      v6 = 64;
    if ( v6 >= (unsigned int)v7 )
    {
LABEL_6:
      v8 = *(_QWORD **)(v3 + 8);
      for ( i = &v8[2 * v7]; i != v8; v8 += 2 )
        *v8 = -4096;
      *(_QWORD *)(v3 + 16) = 0;
      goto LABEL_9;
    }
    v22 = v5 - 1;
    if ( v22 )
    {
      _BitScanReverse(&v22, v22);
      v23 = *(_QWORD **)(v3 + 8);
      v24 = 1 << (33 - (v22 ^ 0x1F));
      if ( v24 < 64 )
        v24 = 64;
      if ( v24 == (_DWORD)v7 )
      {
        *(_QWORD *)(v3 + 16) = 0;
        v25 = &v23[2 * (unsigned int)v24];
        do
        {
          if ( v23 )
            *v23 = -4096;
          v23 += 2;
        }
        while ( v25 != v23 );
        goto LABEL_9;
      }
    }
    else
    {
      v23 = *(_QWORD **)(v3 + 8);
      v24 = 64;
    }
    sub_C7D6A0((__int64)v23, 16LL * (unsigned int)v7, 8);
    v26 = ((((((((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
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
    v27 = (v26
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
    *(_DWORD *)(v3 + 24) = v27;
    v28 = (_QWORD *)sub_C7D670(16 * v27, 8);
    v29 = *(unsigned int *)(v3 + 24);
    *(_QWORD *)(v3 + 16) = 0;
    *(_QWORD *)(v3 + 8) = v28;
    for ( j = &v28[2 * v29]; j != v28; v28 += 2 )
    {
      if ( v28 )
        *v28 = -4096;
    }
  }
  else
  {
    v21 = sub_22077B0(0x20u);
    if ( v21 )
    {
      *(_QWORD *)v21 = 0;
      *(_QWORD *)(v21 + 8) = 0;
      *(_QWORD *)(v21 + 16) = 0;
      *(_DWORD *)(v21 + 24) = 0;
    }
    *a1 = v21;
  }
LABEL_9:
  v10 = a1[5];
  v11 = a2 & 0x7FFFFFFF;
  v12 = a2 & 0x7FFFFFFF;
  v13 = *(_QWORD *)(*(_QWORD *)(v10 + 56) + 16 * v12);
  if ( a2 >= 0 || v11 >= *(_DWORD *)(v10 + 464) )
  {
    v14 = 0;
    v15 = 0;
    v16 = 0;
    v17 = 0;
  }
  else
  {
    v19 = *(_QWORD *)(*(_QWORD *)(v10 + 456) + 8 * v12);
    v20 = v19;
    v17 = v19 & 1;
    v15 = (v19 & 4) != 0;
    v14 = v19 >> 3;
    v16 = (v20 & 2) != 0;
  }
  a1[1] = v13;
  result = (8 * v14) | (4LL * v15) | v17 | (2LL * v16);
  a1[2] = result;
  return result;
}
