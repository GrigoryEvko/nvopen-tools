// Function: sub_2099180
// Address: 0x2099180
//
__int64 __fastcall sub_2099180(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 v4; // r13
  unsigned __int64 v5; // rdi
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // rbx
  __int64 i; // rdx
  __int64 v10; // rax
  unsigned __int64 v11; // rsi
  int v12; // edi
  int v13; // r8d
  int v14; // edi
  __int64 v15; // r9
  __m128i *v16; // r14
  int v17; // r11d
  unsigned int j; // r10d
  __m128i *v19; // rdx
  unsigned int v20; // edx
  __int64 v21; // rcx
  __int64 k; // rdx
  __int32 v23; // ecx

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = ((((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
            | (unsigned int)(a2 - 1)
            | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
          | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
        | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 16)
      | (((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
      | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
      | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
      | (unsigned int)(a2 - 1)
      | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1))
     + 1;
  if ( (unsigned int)v5 < 0x40 )
    LODWORD(v5) = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = sub_22077B0(32LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = v4 + 32 * v3;
    for ( i = result + 32 * v7; i != result; result += 32 )
    {
      if ( result )
      {
        *(_QWORD *)result = 0;
        *(_DWORD *)(result + 8) = -1;
      }
    }
    if ( v8 != v4 )
    {
      v10 = v4;
      do
      {
        while ( 1 )
        {
          v11 = *(_QWORD *)v10;
          if ( *(_QWORD *)v10 || *(_DWORD *)(v10 + 8) <= 0xFFFFFFFD )
            break;
          v10 += 32;
          if ( v8 == v10 )
            return j___libc_free_0(v4);
        }
        v12 = *(_DWORD *)(a1 + 24);
        if ( !v12 )
        {
          MEMORY[0] = *(_QWORD *)v10;
          MEMORY[8] = *(_DWORD *)(v10 + 8);
          BUG();
        }
        v13 = *(_DWORD *)(v10 + 8);
        v14 = v12 - 1;
        v15 = *(_QWORD *)(a1 + 8);
        v16 = 0;
        v17 = 1;
        for ( j = v14 & (v13 + ((v11 >> 9) ^ (v11 >> 4))); ; j = v14 & v20 )
        {
          v19 = (__m128i *)(v15 + 32LL * j);
          if ( v11 == v19->m128i_i64[0] && v13 == v19->m128i_i32[2] )
            break;
          if ( !v19->m128i_i64[0] )
          {
            v23 = v19->m128i_i32[2];
            if ( v23 == -1 )
            {
              if ( v16 )
                v19 = v16;
              break;
            }
            if ( !v16 && v23 == -2 )
              v16 = (__m128i *)(v15 + 32LL * j);
          }
          v20 = j + v17++;
        }
        v21 = *(_QWORD *)v10;
        v10 += 32;
        v19->m128i_i64[0] = v21;
        v19->m128i_i32[2] = *(_DWORD *)(v10 - 24);
        v19[1] = _mm_loadu_si128((const __m128i *)(v10 - 16));
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v8 != v10 );
    }
    return j___libc_free_0(v4);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = result + 32LL * *(unsigned int *)(a1 + 24); k != result; result += 32 )
    {
      if ( result )
      {
        *(_QWORD *)result = 0;
        *(_DWORD *)(result + 8) = -1;
      }
    }
  }
  return result;
}
