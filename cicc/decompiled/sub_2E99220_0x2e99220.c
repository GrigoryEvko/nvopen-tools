// Function: sub_2E99220
// Address: 0x2e99220
//
__int64 __fastcall sub_2E99220(__int64 a1)
{
  int v2; // ecx
  __int64 result; // rax
  __int64 v4; // r12
  __int64 v5; // r13
  __int64 v6; // r15
  __int64 v7; // rsi
  __int64 v8; // rbx
  __int64 v9; // r13
  unsigned __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v13; // rax
  unsigned __int64 v14; // rdi
  int v15; // edx
  __int64 v16; // rbx
  unsigned int v17; // ecx
  unsigned int v18; // eax
  _QWORD *v19; // rdi
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // rax
  __int64 v22; // rdx
  __int64 i; // rdx
  __int64 v24; // [rsp+0h] [rbp-40h]
  int v25; // [rsp+Ch] [rbp-34h]
  int v26; // [rsp+Ch] [rbp-34h]

  v2 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( v2 || (result = *(unsigned int *)(a1 + 20), (_DWORD)result) )
  {
    v4 = *(_QWORD *)(a1 + 8);
    result = (unsigned int)(4 * v2);
    v5 = 40LL * *(unsigned int *)(a1 + 24);
    if ( (unsigned int)result < 0x40 )
      result = 64;
    v6 = v4 + v5;
    if ( *(_DWORD *)(a1 + 24) <= (unsigned int)result )
    {
      for ( ; v4 != v6; v4 += 40 )
      {
        result = *(_QWORD *)v4;
        if ( *(_QWORD *)v4 != -4096 )
        {
          if ( result != -8192 )
          {
            v7 = *(unsigned int *)(v4 + 32);
            if ( (_DWORD)v7 )
            {
              v8 = *(_QWORD *)(v4 + 16);
              v9 = v8 + 32 * v7;
              do
              {
                while ( 1 )
                {
                  if ( *(_DWORD *)v8 <= 0xFFFFFFFD )
                  {
                    v10 = *(_QWORD *)(v8 + 8);
                    if ( v10 )
                      break;
                  }
                  v8 += 32;
                  if ( v9 == v8 )
                    goto LABEL_15;
                }
                v8 += 32;
                j_j___libc_free_0(v10);
              }
              while ( v9 != v8 );
LABEL_15:
              v7 = *(unsigned int *)(v4 + 32);
            }
            result = sub_C7D6A0(*(_QWORD *)(v4 + 16), 32 * v7, 8);
          }
          *(_QWORD *)v4 = -4096;
        }
      }
LABEL_19:
      *(_QWORD *)(a1 + 16) = 0;
      return result;
    }
    do
    {
      result = *(_QWORD *)v4;
      if ( *(_QWORD *)v4 != -8192 && result != -4096 )
      {
        v11 = *(unsigned int *)(v4 + 32);
        if ( (_DWORD)v11 )
        {
          v12 = *(_QWORD *)(v4 + 16);
          v13 = v12 + 32 * v11;
          do
          {
            if ( *(_DWORD *)v12 <= 0xFFFFFFFD )
            {
              v14 = *(_QWORD *)(v12 + 8);
              if ( v14 )
              {
                v24 = v13;
                v26 = v2;
                j_j___libc_free_0(v14);
                v13 = v24;
                v2 = v26;
              }
            }
            v12 += 32;
          }
          while ( v13 != v12 );
          LODWORD(v11) = *(_DWORD *)(v4 + 32);
        }
        v25 = v2;
        result = sub_C7D6A0(*(_QWORD *)(v4 + 16), 32LL * (unsigned int)v11, 8);
        v2 = v25;
      }
      v4 += 40;
    }
    while ( v4 != v6 );
    v15 = *(_DWORD *)(a1 + 24);
    if ( !v2 )
    {
      if ( v15 )
      {
        result = sub_C7D6A0(*(_QWORD *)(a1 + 8), v5, 8);
        *(_QWORD *)(a1 + 8) = 0;
        *(_QWORD *)(a1 + 16) = 0;
        *(_DWORD *)(a1 + 24) = 0;
        return result;
      }
      goto LABEL_19;
    }
    v16 = 64;
    v17 = v2 - 1;
    if ( v17 )
    {
      _BitScanReverse(&v18, v17);
      v16 = (unsigned int)(1 << (33 - (v18 ^ 0x1F)));
      if ( (int)v16 < 64 )
        v16 = 64;
    }
    v19 = *(_QWORD **)(a1 + 8);
    if ( (_DWORD)v16 == v15 )
    {
      *(_QWORD *)(a1 + 16) = 0;
      result = (__int64)&v19[5 * v16];
      do
      {
        if ( v19 )
          *v19 = -4096;
        v19 += 5;
      }
      while ( (_QWORD *)result != v19 );
    }
    else
    {
      sub_C7D6A0((__int64)v19, v5, 8);
      v20 = ((((((((4 * (int)v16 / 3u + 1) | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v16 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 4)
             | (((4 * (int)v16 / 3u + 1) | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v16 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 8)
           | (((((4 * (int)v16 / 3u + 1) | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v16 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v16 / 3u + 1) | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v16 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 16;
      v21 = (v20
           | (((((((4 * (int)v16 / 3u + 1) | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v16 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 4)
             | (((4 * (int)v16 / 3u + 1) | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v16 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 8)
           | (((((4 * (int)v16 / 3u + 1) | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v16 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v16 / 3u + 1) | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v16 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v16 / 3u + 1) >> 1))
          + 1;
      *(_DWORD *)(a1 + 24) = v21;
      result = sub_C7D670(40 * v21, 8);
      v22 = *(unsigned int *)(a1 + 24);
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 8) = result;
      for ( i = result + 40 * v22; i != result; result += 40 )
      {
        if ( result )
          *(_QWORD *)result = -4096;
      }
    }
  }
  return result;
}
