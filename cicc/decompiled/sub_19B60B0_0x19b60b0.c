// Function: sub_19B60B0
// Address: 0x19b60b0
//
__int64 __fastcall sub_19B60B0(__int64 a1, int a2)
{
  __int64 v3; // r12
  const __m128i *v4; // r13
  unsigned __int64 v5; // rdi
  __int64 result; // rax
  __int64 v7; // rdx
  const __m128i *v8; // r10
  __int64 i; // rdx
  const __m128i *j; // rcx
  __int64 v11; // rdi
  int v12; // eax
  int v13; // r9d
  int v14; // r9d
  __int64 v15; // r11
  unsigned __int64 v16; // rsi
  unsigned __int64 v17; // rsi
  unsigned int v18; // edx
  __int64 *v19; // rsi
  __int64 v20; // r12
  int v21; // r8d
  int v22; // r14d
  __int64 *v23; // r15
  __int64 v24; // rdx
  __int64 k; // rdx

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
        *(_DWORD *)(result + 8) = 0;
      }
    }
    if ( v8 != v4 )
    {
      for ( j = v4; v8 != j; ++j )
      {
        while ( 1 )
        {
          v11 = j->m128i_i64[0];
          v12 = (4 * j->m128i_i32[2]) >> 2;
          if ( j->m128i_i64[0] != -8 )
            break;
          if ( v12 )
            goto LABEL_13;
          if ( v8 == ++j )
            return j___libc_free_0(v4);
        }
        if ( v12 || v11 != -16 )
        {
LABEL_13:
          v13 = *(_DWORD *)(a1 + 24);
          if ( !v13 )
          {
            MEMORY[0] = _mm_loadu_si128(j);
            BUG();
          }
          v14 = v13 - 1;
          v15 = *(_QWORD *)(a1 + 8);
          v16 = ((((unsigned int)(37 * v12)
                 | ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32))
                - 1
                - ((unsigned __int64)(unsigned int)(37 * v12) << 32)) >> 22)
              ^ (((unsigned int)(37 * v12)
                | ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32))
               - 1
               - ((unsigned __int64)(unsigned int)(37 * v12) << 32));
          v17 = ((9 * (((v16 - 1 - (v16 << 13)) >> 8) ^ (v16 - 1 - (v16 << 13)))) >> 15)
              ^ (9 * (((v16 - 1 - (v16 << 13)) >> 8) ^ (v16 - 1 - (v16 << 13))));
          v18 = v14 & (((v17 - 1 - (v17 << 27)) >> 31) ^ (v17 - 1 - ((_DWORD)v17 << 27)));
          v19 = (__int64 *)(v15 + 16LL * v18);
          v20 = *v19;
          v21 = (4 * *((_DWORD *)v19 + 2)) >> 2;
          if ( v12 != v21 || v11 != v20 )
          {
            v22 = 1;
            v23 = 0;
            while ( 1 )
            {
              if ( v20 == -8 )
              {
                if ( !v21 )
                {
                  if ( v23 )
                    v19 = v23;
                  break;
                }
              }
              else if ( v20 == -16 && v21 == 0 && !v23 )
              {
                v23 = v19;
              }
              v18 = v14 & (v22 + v18);
              v19 = (__int64 *)(v15 + 16LL * v18);
              v20 = *v19;
              v21 = (4 * *((_DWORD *)v19 + 2)) >> 2;
              if ( v12 == v21 && v11 == v20 )
                break;
              ++v22;
            }
          }
          *(__m128i *)v19 = _mm_loadu_si128(j);
          ++*(_DWORD *)(a1 + 16);
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
        *(_DWORD *)(result + 8) = 0;
      }
    }
  }
  return result;
}
