// Function: sub_1440C10
// Address: 0x1440c10
//
__int64 __fastcall sub_1440C10(__int64 a1)
{
  int v2; // r14d
  __int64 result; // rax
  __int64 *v4; // rbx
  __int64 v5; // rdx
  __int64 *v6; // r13
  __int64 v7; // r14
  __int64 v8; // rdi
  __int64 v9; // rdi
  __int64 v10; // r15
  int v11; // edx
  int v12; // ebx
  unsigned int v13; // r14d
  unsigned int v14; // eax
  _QWORD *v15; // rdi
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rdi
  __int64 v18; // rdx
  __int64 i; // rdx

  v2 = *(_DWORD *)(a1 + 224);
  ++*(_QWORD *)(a1 + 208);
  if ( !v2 )
  {
    result = *(unsigned int *)(a1 + 228);
    if ( !(_DWORD)result )
      goto LABEL_17;
  }
  v4 = *(__int64 **)(a1 + 216);
  result = (unsigned int)(4 * v2);
  v5 = *(unsigned int *)(a1 + 232);
  v6 = &v4[2 * v5];
  if ( (unsigned int)result < 0x40 )
    result = 64;
  if ( (unsigned int)v5 <= (unsigned int)result )
  {
    while ( v6 != v4 )
    {
      result = *v4;
      if ( *v4 != -8 )
      {
        if ( result != -16 )
        {
          v7 = v4[1];
          if ( v7 )
          {
            v8 = *(_QWORD *)(v7 + 24);
            if ( v8 )
              j_j___libc_free_0(v8, *(_QWORD *)(v7 + 40) - v8);
            result = j_j___libc_free_0(v7, 56);
          }
        }
        *v4 = -8;
      }
      v4 += 2;
    }
LABEL_16:
    *(_QWORD *)(a1 + 224) = 0;
    goto LABEL_17;
  }
  do
  {
    while ( 1 )
    {
      result = *v4;
      if ( *v4 != -8 && result != -16 )
        break;
LABEL_21:
      v4 += 2;
      if ( v6 == v4 )
        goto LABEL_26;
    }
    v10 = v4[1];
    if ( v10 )
    {
      v9 = *(_QWORD *)(v10 + 24);
      if ( v9 )
        j_j___libc_free_0(v9, *(_QWORD *)(v10 + 40) - v9);
      result = j_j___libc_free_0(v10, 56);
      goto LABEL_21;
    }
    v4 += 2;
  }
  while ( v6 != v4 );
LABEL_26:
  v11 = *(_DWORD *)(a1 + 232);
  if ( !v2 )
  {
    if ( v11 )
    {
      result = j___libc_free_0(*(_QWORD *)(a1 + 216));
      *(_QWORD *)(a1 + 216) = 0;
      *(_QWORD *)(a1 + 224) = 0;
      *(_DWORD *)(a1 + 232) = 0;
      goto LABEL_17;
    }
    goto LABEL_16;
  }
  v12 = 64;
  v13 = v2 - 1;
  if ( v13 )
  {
    _BitScanReverse(&v14, v13);
    v12 = 1 << (33 - (v14 ^ 0x1F));
    if ( v12 < 64 )
      v12 = 64;
  }
  v15 = *(_QWORD **)(a1 + 216);
  if ( v12 == v11 )
  {
    *(_QWORD *)(a1 + 224) = 0;
    result = (__int64)&v15[2 * (unsigned int)v12];
    do
    {
      if ( v15 )
        *v15 = -8;
      v15 += 2;
    }
    while ( (_QWORD *)result != v15 );
  }
  else
  {
    j___libc_free_0(v15);
    v16 = ((((((((4 * v12 / 3u + 1) | ((unsigned __int64)(4 * v12 / 3u + 1) >> 1)) >> 2)
             | (4 * v12 / 3u + 1)
             | ((unsigned __int64)(4 * v12 / 3u + 1) >> 1)) >> 4)
           | (((4 * v12 / 3u + 1) | ((unsigned __int64)(4 * v12 / 3u + 1) >> 1)) >> 2)
           | (4 * v12 / 3u + 1)
           | ((unsigned __int64)(4 * v12 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v12 / 3u + 1) | ((unsigned __int64)(4 * v12 / 3u + 1) >> 1)) >> 2)
           | (4 * v12 / 3u + 1)
           | ((unsigned __int64)(4 * v12 / 3u + 1) >> 1)) >> 4)
         | (((4 * v12 / 3u + 1) | ((unsigned __int64)(4 * v12 / 3u + 1) >> 1)) >> 2)
         | (4 * v12 / 3u + 1)
         | ((unsigned __int64)(4 * v12 / 3u + 1) >> 1)) >> 16;
    v17 = (v16
         | (((((((4 * v12 / 3u + 1) | ((unsigned __int64)(4 * v12 / 3u + 1) >> 1)) >> 2)
             | (4 * v12 / 3u + 1)
             | ((unsigned __int64)(4 * v12 / 3u + 1) >> 1)) >> 4)
           | (((4 * v12 / 3u + 1) | ((unsigned __int64)(4 * v12 / 3u + 1) >> 1)) >> 2)
           | (4 * v12 / 3u + 1)
           | ((unsigned __int64)(4 * v12 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v12 / 3u + 1) | ((unsigned __int64)(4 * v12 / 3u + 1) >> 1)) >> 2)
           | (4 * v12 / 3u + 1)
           | ((unsigned __int64)(4 * v12 / 3u + 1) >> 1)) >> 4)
         | (((4 * v12 / 3u + 1) | ((unsigned __int64)(4 * v12 / 3u + 1) >> 1)) >> 2)
         | (4 * v12 / 3u + 1)
         | ((unsigned __int64)(4 * v12 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 232) = v17;
    result = sub_22077B0(16 * v17);
    v18 = *(unsigned int *)(a1 + 232);
    *(_QWORD *)(a1 + 224) = 0;
    *(_QWORD *)(a1 + 216) = result;
    for ( i = result + 16 * v18; i != result; result += 16 )
    {
      if ( result )
        *(_QWORD *)result = -8;
    }
  }
LABEL_17:
  *(_DWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 240) = 0;
  *(_QWORD *)(a1 + 248) = 0;
  *(_BYTE *)(a1 + 256) = 0;
  *(_DWORD *)(a1 + 260) = 0;
  return result;
}
