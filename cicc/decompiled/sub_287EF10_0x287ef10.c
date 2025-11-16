// Function: sub_287EF10
// Address: 0x287ef10
//
__int64 __fastcall sub_287EF10(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  unsigned int v4; // r13d
  __int64 v5; // r12
  unsigned int v6; // edi
  __int64 result; // rax
  __int64 v8; // rdx
  const __m128i *v9; // r10
  __int64 i; // rdx
  const __m128i *v11; // rcx
  __int64 v12; // rdi
  int v13; // eax
  int v14; // r9d
  int v15; // r9d
  __int64 v16; // r11
  unsigned int v17; // edx
  __int64 *v18; // rsi
  __int64 v19; // r12
  int v20; // r8d
  int v21; // r13d
  __int64 *v22; // r14
  __int64 v23; // rdx
  __int64 j; // rdx
  __int64 v25; // [rsp+10h] [rbp-40h]
  __int64 v26; // [rsp+18h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(_DWORD *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  result = sub_C7D670(16LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v26 = 16LL * v4;
    v9 = (const __m128i *)(v5 + v26);
    for ( i = result + 16 * v8; i != result; result += 16 )
    {
      if ( result )
      {
        *(_QWORD *)result = -4096;
        *(_DWORD *)(result + 8) = 0;
      }
    }
    if ( v9 != (const __m128i *)v5 )
    {
      v25 = v5;
      v11 = (const __m128i *)v5;
      while ( 1 )
      {
        while ( 1 )
        {
          v12 = v11->m128i_i64[0];
          v13 = (4 * v11->m128i_i32[2]) >> 2;
          if ( v11->m128i_i64[0] != -4096 )
            break;
          if ( v13 )
            goto LABEL_13;
          if ( v9 == ++v11 )
          {
LABEL_26:
            v5 = v25;
            return sub_C7D6A0(v5, v26, 8);
          }
        }
        if ( v13 || v12 != -8192 )
        {
LABEL_13:
          v14 = *(_DWORD *)(a1 + 24);
          if ( !v14 )
          {
            MEMORY[0] = _mm_loadu_si128(v11);
            BUG();
          }
          v15 = v14 - 1;
          v16 = *(_QWORD *)(a1 + 8);
          v17 = v15
              & (((0xBF58476D1CE4E5B9LL
                 * ((unsigned int)(37 * v13)
                  | ((unsigned __int64)(((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4)) << 32))) >> 31)
               ^ (756364221 * v13));
          v18 = (__int64 *)(v16 + 16LL * v17);
          v19 = *v18;
          v20 = (4 * *((_DWORD *)v18 + 2)) >> 2;
          if ( v13 != v20 || v12 != v19 )
          {
            v21 = 1;
            v22 = 0;
            while ( 1 )
            {
              if ( v19 == -4096 )
              {
                if ( !v20 )
                {
                  if ( v22 )
                    v18 = v22;
                  break;
                }
              }
              else if ( v19 == -8192 && v20 == 0 && !v22 )
              {
                v22 = v18;
              }
              v17 = v15 & (v21 + v17);
              v18 = (__int64 *)(v16 + 16LL * v17);
              v19 = *v18;
              v20 = (4 * *((_DWORD *)v18 + 2)) >> 2;
              if ( v13 == v20 && v12 == v19 )
                break;
              ++v21;
            }
          }
          *(__m128i *)v18 = _mm_loadu_si128(v11);
          ++*(_DWORD *)(a1 + 16);
        }
        if ( v9 == ++v11 )
          goto LABEL_26;
      }
    }
    return sub_C7D6A0(v5, v26, 8);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = result + 16 * v23; j != result; result += 16 )
    {
      if ( result )
      {
        *(_QWORD *)result = -4096;
        *(_DWORD *)(result + 8) = 0;
      }
    }
  }
  return result;
}
