// Function: sub_2B2F0D0
// Address: 0x2b2f0d0
//
__int64 __fastcall sub_2B2F0D0(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rcx
  int v4; // edx
  __int64 v5; // rdi
  int v6; // r8d
  unsigned int v7; // esi
  __int64 *v8; // rdx
  __int64 v9; // r9
  __int64 v10; // rdx
  int v11; // ecx
  __int64 v12; // rax
  int v13; // eax
  __int64 result; // rax
  __int64 v15; // rdx
  __int64 i; // rdx
  unsigned int v17; // ecx
  unsigned int v18; // eax
  _QWORD *v19; // rdi
  __int64 v20; // r12
  int v21; // edx
  int v22; // r10d
  unsigned __int64 v23; // rdx
  unsigned __int64 v24; // rax
  __int64 v25; // rdx
  __int64 j; // rdx

  v2 = *(_QWORD *)(a1 + 160);
  while ( *(_QWORD *)(a1 + 168) != v2 )
  {
    v3 = *(_QWORD *)(v2 + 40);
    if ( *(_QWORD *)a1 == v3 )
    {
      v4 = *(_DWORD *)(a1 + 104);
      v5 = *(_QWORD *)(a1 + 88);
      if ( v4 )
      {
        v6 = v4 - 1;
        v7 = (v4 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
        v8 = (__int64 *)(v5 + 16LL * v7);
        v9 = *v8;
        if ( *v8 == v2 )
        {
LABEL_5:
          v10 = v8[1];
          if ( v10 && *(_DWORD *)(v10 + 136) == *(_DWORD *)(a1 + 204) )
          {
            v11 = *(_DWORD *)(v10 + 144);
            *(_BYTE *)(v10 + 152) = 0;
            *(_DWORD *)(v10 + 148) = v11;
            v3 = *(_QWORD *)(v2 + 40);
          }
        }
        else
        {
          v21 = 1;
          while ( v9 != -4096 )
          {
            v22 = v21 + 1;
            v7 = v6 & (v21 + v7);
            v8 = (__int64 *)(v5 + 16LL * v7);
            v9 = *v8;
            if ( *v8 == v2 )
              goto LABEL_5;
            v21 = v22;
          }
        }
      }
    }
    v12 = *(_QWORD *)(v2 + 32);
    if ( v12 == v3 + 48 || !v12 )
      v2 = 0;
    else
      v2 = v12 - 24;
  }
  v13 = *(_DWORD *)(a1 + 128);
  ++*(_QWORD *)(a1 + 112);
  if ( !v13 )
  {
    result = *(unsigned int *)(a1 + 132);
    if ( !(_DWORD)result )
      goto LABEL_18;
    v15 = *(unsigned int *)(a1 + 136);
    if ( (unsigned int)v15 > 0x40 )
    {
      result = sub_C7D6A0(*(_QWORD *)(a1 + 120), 8 * v15, 8);
      *(_QWORD *)(a1 + 120) = 0;
      *(_QWORD *)(a1 + 128) = 0;
      *(_DWORD *)(a1 + 136) = 0;
      goto LABEL_18;
    }
    goto LABEL_15;
  }
  v17 = 4 * v13;
  v15 = *(unsigned int *)(a1 + 136);
  if ( (unsigned int)(4 * v13) < 0x40 )
    v17 = 64;
  if ( v17 >= (unsigned int)v15 )
  {
LABEL_15:
    result = *(_QWORD *)(a1 + 120);
    for ( i = result + 8 * v15; i != result; result += 8 )
      *(_QWORD *)result = -4096;
    *(_QWORD *)(a1 + 128) = 0;
    goto LABEL_18;
  }
  v18 = v13 - 1;
  if ( v18 )
  {
    _BitScanReverse(&v18, v18);
    v19 = *(_QWORD **)(a1 + 120);
    v20 = (unsigned int)(1 << (33 - (v18 ^ 0x1F)));
    if ( (int)v20 < 64 )
      v20 = 64;
    if ( (_DWORD)v20 == (_DWORD)v15 )
    {
      *(_QWORD *)(a1 + 128) = 0;
      result = (__int64)&v19[v20];
      do
      {
        if ( v19 )
          *v19 = -4096;
        ++v19;
      }
      while ( (_QWORD *)result != v19 );
      goto LABEL_18;
    }
  }
  else
  {
    v19 = *(_QWORD **)(a1 + 120);
    LODWORD(v20) = 64;
  }
  sub_C7D6A0((__int64)v19, 8 * v15, 8);
  v23 = ((((((((4 * (int)v20 / 3u + 1) | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v20 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v20 / 3u + 1) | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v20 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 8)
       | (((((4 * (int)v20 / 3u + 1) | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v20 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 4)
       | (((4 * (int)v20 / 3u + 1) | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 2)
       | (4 * (int)v20 / 3u + 1)
       | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 16;
  v24 = (v23
       | (((((((4 * (int)v20 / 3u + 1) | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v20 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v20 / 3u + 1) | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v20 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 8)
       | (((((4 * (int)v20 / 3u + 1) | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v20 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 4)
       | (((4 * (int)v20 / 3u + 1) | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 2)
       | (4 * (int)v20 / 3u + 1)
       | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1))
      + 1;
  *(_DWORD *)(a1 + 136) = v24;
  result = sub_C7D670(8 * v24, 8);
  v25 = *(unsigned int *)(a1 + 136);
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 120) = result;
  for ( j = result + 8 * v25; j != result; result += 8 )
  {
    if ( result )
      *(_QWORD *)result = -4096;
  }
LABEL_18:
  *(_DWORD *)(a1 + 152) = 0;
  return result;
}
