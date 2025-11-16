// Function: sub_3186DF0
// Address: 0x3186df0
//
__int64 __fastcall sub_3186DF0(__int64 a1)
{
  int v2; // r15d
  __int64 result; // rax
  __int64 *v4; // rbx
  __int64 v5; // rdx
  __int64 v6; // r14
  __int64 *v7; // r13
  __int64 v8; // rdi
  __int64 v9; // rdi
  unsigned int v10; // edx
  int v11; // ebx
  unsigned int v12; // r15d
  unsigned int v13; // eax
  _QWORD *v14; // rdi
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 i; // rdx

  v2 = *(_DWORD *)(a1 + 104);
  ++*(_QWORD *)(a1 + 88);
  if ( !v2 )
  {
    result = *(unsigned int *)(a1 + 108);
    if ( !(_DWORD)result )
      return result;
  }
  v4 = *(__int64 **)(a1 + 96);
  result = (unsigned int)(4 * v2);
  v5 = *(unsigned int *)(a1 + 112);
  v6 = 16 * v5;
  if ( (unsigned int)result < 0x40 )
    result = 64;
  v7 = &v4[(unsigned __int64)v6 / 8];
  if ( (unsigned int)v5 <= (unsigned int)result )
  {
    for ( ; v4 != v7; v4 += 2 )
    {
      result = *v4;
      if ( *v4 != -4096 )
      {
        if ( result != -8192 )
        {
          v8 = v4[1];
          if ( v8 )
            result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v8 + 8LL))(v8);
        }
        *v4 = -4096;
      }
    }
LABEL_13:
    *(_QWORD *)(a1 + 104) = 0;
    return result;
  }
  do
  {
    while ( 1 )
    {
      result = *v4;
      if ( *v4 != -8192 )
        break;
LABEL_17:
      v4 += 2;
      if ( v4 == v7 )
        goto LABEL_21;
    }
    if ( result != -4096 )
    {
      v9 = v4[1];
      if ( v9 )
        result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v9 + 8LL))(v9);
      goto LABEL_17;
    }
    v4 += 2;
  }
  while ( v4 != v7 );
LABEL_21:
  v10 = *(_DWORD *)(a1 + 112);
  if ( !v2 )
  {
    if ( v10 )
    {
      result = sub_C7D6A0(*(_QWORD *)(a1 + 96), v6, 8);
      *(_QWORD *)(a1 + 96) = 0;
      *(_QWORD *)(a1 + 104) = 0;
      *(_DWORD *)(a1 + 112) = 0;
      return result;
    }
    goto LABEL_13;
  }
  v11 = 64;
  v12 = v2 - 1;
  if ( v12 )
  {
    _BitScanReverse(&v13, v12);
    v11 = 1 << (33 - (v13 ^ 0x1F));
    if ( v11 < 64 )
      v11 = 64;
  }
  v14 = *(_QWORD **)(a1 + 96);
  if ( v10 == v11 )
  {
    *(_QWORD *)(a1 + 104) = 0;
    result = (__int64)&v14[2 * v10];
    do
    {
      if ( v14 )
        *v14 = -4096;
      v14 += 2;
    }
    while ( (_QWORD *)result != v14 );
  }
  else
  {
    sub_C7D6A0((__int64)v14, v6, 8);
    v15 = ((((((((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
             | (4 * v11 / 3u + 1)
             | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 4)
           | (((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
           | (4 * v11 / 3u + 1)
           | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
           | (4 * v11 / 3u + 1)
           | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 4)
         | (((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
         | (4 * v11 / 3u + 1)
         | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 16;
    v16 = (v15
         | (((((((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
             | (4 * v11 / 3u + 1)
             | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 4)
           | (((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
           | (4 * v11 / 3u + 1)
           | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
           | (4 * v11 / 3u + 1)
           | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 4)
         | (((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
         | (4 * v11 / 3u + 1)
         | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 112) = v16;
    result = sub_C7D670(16 * v16, 8);
    v17 = *(unsigned int *)(a1 + 112);
    *(_QWORD *)(a1 + 104) = 0;
    *(_QWORD *)(a1 + 96) = result;
    for ( i = result + 16 * v17; i != result; result += 16 )
    {
      if ( result )
        *(_QWORD *)result = -4096;
    }
  }
  return result;
}
