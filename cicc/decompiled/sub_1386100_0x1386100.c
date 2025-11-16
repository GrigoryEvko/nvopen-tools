// Function: sub_1386100
// Address: 0x1386100
//
__int64 __fastcall sub_1386100(__int64 a1, int a2)
{
  __int64 v3; // r13
  const __m128i *v4; // r12
  unsigned __int64 v5; // rdi
  __int64 result; // rax
  __int64 v7; // rdx
  const __m128i *v8; // r8
  __int64 i; // rdx
  const __m128i *v10; // rdx
  __int64 v11; // rcx
  int v12; // esi
  __int32 v13; // edi
  int v14; // esi
  __int64 v15; // r9
  __int64 *v16; // r13
  int v17; // r11d
  unsigned __int64 v18; // r10
  unsigned __int64 v19; // r10
  unsigned int j; // eax
  __int64 *v21; // r10
  __int64 v22; // r14
  __m128i v23; // xmm1
  __int64 v24; // rdx
  __int64 k; // rdx
  unsigned int v26; // eax

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
  result = sub_22077B0(16LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[v3];
    for ( i = result + 16 * v7; i != result; result += 16 )
    {
      if ( result )
      {
        *(_QWORD *)result = -8;
        *(_DWORD *)(result + 8) = -1;
      }
    }
    if ( v8 != v4 )
    {
      v10 = v4;
      while ( 1 )
      {
        v11 = v10->m128i_i64[0];
        if ( v10->m128i_i64[0] == -8 )
        {
          if ( v10->m128i_i32[2] != -1 )
            goto LABEL_12;
          if ( v8 == ++v10 )
            return j___libc_free_0(v4);
        }
        else if ( v11 == -16 && v10->m128i_i32[2] == -2 )
        {
          if ( v8 == ++v10 )
            return j___libc_free_0(v4);
        }
        else
        {
LABEL_12:
          v12 = *(_DWORD *)(a1 + 24);
          if ( !v12 )
          {
            MEMORY[0] = _mm_loadu_si128(v10);
            BUG();
          }
          v13 = v10->m128i_i32[2];
          v14 = v12 - 1;
          v15 = *(_QWORD *)(a1 + 8);
          v16 = 0;
          v17 = 1;
          v18 = ((((unsigned int)(37 * v13)
                 | ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32))
                - 1
                - ((unsigned __int64)(unsigned int)(37 * v13) << 32)) >> 22)
              ^ (((unsigned int)(37 * v13)
                | ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32))
               - 1
               - ((unsigned __int64)(unsigned int)(37 * v13) << 32));
          v19 = ((9 * (((v18 - 1 - (v18 << 13)) >> 8) ^ (v18 - 1 - (v18 << 13)))) >> 15)
              ^ (9 * (((v18 - 1 - (v18 << 13)) >> 8) ^ (v18 - 1 - (v18 << 13))));
          for ( j = v14 & (((v19 - 1 - (v19 << 27)) >> 31) ^ (v19 - 1 - ((_DWORD)v19 << 27))); ; j = v14 & v26 )
          {
            v21 = (__int64 *)(v15 + 16LL * j);
            v22 = *v21;
            if ( v11 == *v21 && v13 == *((_DWORD *)v21 + 2) )
              break;
            if ( v22 == -8 )
            {
              if ( *((_DWORD *)v21 + 2) == -1 )
              {
                if ( v16 )
                  v21 = v16;
                break;
              }
            }
            else if ( v22 == -16 && *((_DWORD *)v21 + 2) == -2 && !v16 )
            {
              v16 = (__int64 *)(v15 + 16LL * j);
            }
            v26 = v17 + j;
            ++v17;
          }
          v23 = _mm_loadu_si128(v10++);
          *(__m128i *)v21 = v23;
          ++*(_DWORD *)(a1 + 16);
          if ( v8 == v10 )
            return j___libc_free_0(v4);
        }
      }
    }
    return j___libc_free_0(v4);
  }
  else
  {
    v24 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = result + 16 * v24; k != result; result += 16 )
    {
      if ( result )
      {
        *(_QWORD *)result = -8;
        *(_DWORD *)(result + 8) = -1;
      }
    }
  }
  return result;
}
