// Function: sub_18FA270
// Address: 0x18fa270
//
__int64 __fastcall sub_18FA270(__int64 a1, int a2)
{
  __int64 v3; // r12
  const __m128i *v4; // r15
  unsigned __int64 v5; // rdi
  __int64 result; // rax
  __int64 v7; // rdx
  const __m128i *v8; // r9
  __int64 i; // rdx
  const __m128i *v10; // rax
  __int64 v11; // r8
  unsigned __int8 v12; // di
  int v13; // esi
  __int64 v14; // r11
  int v15; // esi
  __int64 v16; // r12
  int v17; // r14d
  __int64 v18; // r15
  unsigned int j; // ecx
  __int64 v20; // rdx
  __int64 v21; // r13
  __m128i v22; // xmm0
  __int64 v23; // rdx
  __int64 k; // rdx
  unsigned int v25; // ecx
  const __m128i *v26; // [rsp+8h] [rbp-38h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(const __m128i **)(a1 + 8);
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
    v8 = &v4[2 * v3];
    for ( i = result + 32 * v7; i != result; result += 32 )
    {
      if ( result )
      {
        *(_BYTE *)result = 0;
        *(_QWORD *)(result + 8) = 0;
        *(_QWORD *)(result + 16) = 0;
      }
    }
    v10 = v4;
    if ( v8 != v4 )
    {
      v26 = v4;
      do
      {
        while ( 1 )
        {
          v11 = v10->m128i_i64[1];
          v12 = v10->m128i_i8[0];
          if ( v11 || v10[1].m128i_i64[0] )
            break;
          v10 += 2;
          if ( v8 == v10 )
            goto LABEL_22;
        }
        v13 = *(_DWORD *)(a1 + 24);
        if ( !v13 )
        {
          MEMORY[0] = _mm_loadu_si128(v10);
          MEMORY[0x10] = v10[1].m128i_i64[0];
          BUG();
        }
        v14 = v10[1].m128i_i64[0];
        v15 = v13 - 1;
        v16 = *(_QWORD *)(a1 + 8);
        v17 = 1;
        v18 = 0;
        for ( j = v15 & (v12 ^ v14 ^ v11); ; j = v15 & v25 )
        {
          v20 = v16 + 32LL * j;
          v21 = *(_QWORD *)(v20 + 8);
          if ( v12 == *(_BYTE *)v20 && v21 == v11 && v14 == *(_QWORD *)(v20 + 16) )
            break;
          if ( *(_BYTE *)v20 )
          {
            if ( !v21 && !(*(_QWORD *)(v20 + 16) | v18) )
              v18 = v16 + 32LL * j;
          }
          else if ( !v21 && !*(_QWORD *)(v20 + 16) )
          {
            if ( v18 )
              v20 = v18;
            break;
          }
          v25 = v17 + j;
          ++v17;
        }
        v22 = _mm_loadu_si128(v10);
        v10 += 2;
        *(__m128i *)v20 = v22;
        *(_QWORD *)(v20 + 16) = v10[-1].m128i_i64[0];
        *(_QWORD *)(v20 + 24) = v10[-1].m128i_i64[1];
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v8 != v10 );
LABEL_22:
      v4 = v26;
    }
    return j___libc_free_0(v4);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = result + 32 * v23; k != result; result += 32 )
    {
      if ( result )
      {
        *(_BYTE *)result = 0;
        *(_QWORD *)(result + 8) = 0;
        *(_QWORD *)(result + 16) = 0;
      }
    }
  }
  return result;
}
