// Function: sub_B73010
// Address: 0xb73010
//
__int64 __fastcall sub_B73010(__int64 a1)
{
  int v2; // r15d
  __int64 result; // rax
  unsigned int *v4; // rbx
  __int64 v5; // rdx
  __int64 v6; // r12
  unsigned int *v7; // r14
  __int64 v8; // r12
  __int64 v9; // rdi
  __int64 v10; // r8
  __int64 v11; // rdi
  int v12; // edx
  int v13; // ebx
  unsigned int v14; // r15d
  unsigned int v15; // eax
  _DWORD *v16; // rdi
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rdi
  __int64 v19; // rdx
  __int64 i; // rdx
  __int64 v21; // [rsp+8h] [rbp-38h]
  __int64 v22; // [rsp+8h] [rbp-38h]

  v2 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( v2 || (result = *(unsigned int *)(a1 + 20), (_DWORD)result) )
  {
    result = (unsigned int)(4 * v2);
    v4 = *(unsigned int **)(a1 + 8);
    v5 = *(unsigned int *)(a1 + 24);
    v6 = 16 * v5;
    if ( (unsigned int)result < 0x40 )
      result = 64;
    v7 = &v4[(unsigned __int64)v6 / 4];
    if ( (unsigned int)v5 <= (unsigned int)result )
    {
      for ( ; v4 != v7; v4 += 4 )
      {
        result = *v4;
        if ( (_DWORD)result != -1 )
        {
          if ( (_DWORD)result != -2 )
          {
            v8 = *((_QWORD *)v4 + 1);
            if ( v8 )
            {
              if ( *(_DWORD *)(v8 + 32) > 0x40u )
              {
                v9 = *(_QWORD *)(v8 + 24);
                if ( v9 )
                  j_j___libc_free_0_0(v9);
              }
              sub_BD7260(v8);
              result = sub_BD2DD0(v8);
            }
          }
          *v4 = -1;
        }
      }
LABEL_16:
      *(_QWORD *)(a1 + 16) = 0;
      return result;
    }
    do
    {
      if ( *v4 <= 0xFFFFFFFD )
      {
        v10 = *((_QWORD *)v4 + 1);
        if ( v10 )
        {
          if ( *(_DWORD *)(v10 + 32) > 0x40u )
          {
            v11 = *(_QWORD *)(v10 + 24);
            if ( v11 )
            {
              v21 = *((_QWORD *)v4 + 1);
              j_j___libc_free_0_0(v11);
              v10 = v21;
            }
          }
          v22 = v10;
          sub_BD7260(v10);
          result = sub_BD2DD0(v22);
        }
      }
      v4 += 4;
    }
    while ( v4 != v7 );
    v12 = *(_DWORD *)(a1 + 24);
    if ( !v2 )
    {
      if ( v12 )
      {
        result = sub_C7D6A0(*(_QWORD *)(a1 + 8), v6, 8);
        *(_QWORD *)(a1 + 8) = 0;
        *(_QWORD *)(a1 + 16) = 0;
        *(_DWORD *)(a1 + 24) = 0;
        return result;
      }
      goto LABEL_16;
    }
    v13 = 64;
    v14 = v2 - 1;
    if ( v14 )
    {
      _BitScanReverse(&v15, v14);
      v13 = 1 << (33 - (v15 ^ 0x1F));
      if ( v13 < 64 )
        v13 = 64;
    }
    v16 = *(_DWORD **)(a1 + 8);
    if ( v12 == v13 )
    {
      *(_QWORD *)(a1 + 16) = 0;
      result = (__int64)&v16[4 * v12];
      do
      {
        if ( v16 )
          *v16 = -1;
        v16 += 4;
      }
      while ( (_DWORD *)result != v16 );
    }
    else
    {
      sub_C7D6A0(v16, v6, 8);
      v17 = ((((((((4 * v13 / 3u + 1) | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 2)
               | (4 * v13 / 3u + 1)
               | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 4)
             | (((4 * v13 / 3u + 1) | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 2)
             | (4 * v13 / 3u + 1)
             | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v13 / 3u + 1) | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 2)
             | (4 * v13 / 3u + 1)
             | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 4)
           | (((4 * v13 / 3u + 1) | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 2)
           | (4 * v13 / 3u + 1)
           | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 16;
      v18 = (v17
           | (((((((4 * v13 / 3u + 1) | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 2)
               | (4 * v13 / 3u + 1)
               | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 4)
             | (((4 * v13 / 3u + 1) | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 2)
             | (4 * v13 / 3u + 1)
             | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v13 / 3u + 1) | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 2)
             | (4 * v13 / 3u + 1)
             | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 4)
           | (((4 * v13 / 3u + 1) | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 2)
           | (4 * v13 / 3u + 1)
           | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1))
          + 1;
      *(_DWORD *)(a1 + 24) = v18;
      result = sub_C7D670(16 * v18, 8);
      v19 = *(unsigned int *)(a1 + 24);
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 8) = result;
      for ( i = result + 16 * v19; i != result; result += 16 )
      {
        if ( result )
          *(_DWORD *)result = -1;
      }
    }
  }
  return result;
}
