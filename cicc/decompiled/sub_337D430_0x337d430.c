// Function: sub_337D430
// Address: 0x337d430
//
_QWORD *__fastcall sub_337D430(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r13
  unsigned int v6; // eax
  _QWORD *result; // rax
  const __m128i *v8; // r14
  _QWORD *i; // rdx
  const __m128i *v10; // rbx
  __int64 v11; // rax
  int v12; // edx
  int v13; // edx
  __int64 v14; // rdi
  int v15; // r11d
  __int64 *v16; // r10
  unsigned int v17; // ecx
  __int64 *v18; // r12
  __int64 v19; // rsi
  void *v20; // rdi
  unsigned __int32 v21; // r10d
  unsigned __int64 v22; // rdi
  const __m128i *v23; // r11
  const __m128i *v24; // rsi
  size_t v25; // r8
  __int64 v26; // rdx
  _QWORD *j; // rdx
  const __m128i *v28; // [rsp+8h] [rbp-48h]
  __int64 v29; // [rsp+8h] [rbp-48h]
  unsigned __int32 v30; // [rsp+14h] [rbp-3Ch]
  __int32 v31; // [rsp+14h] [rbp-3Ch]
  __int64 v32; // [rsp+18h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
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
  result = (_QWORD *)sub_C7D670(80LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v32 = 80 * v4;
    v8 = (const __m128i *)(v5 + 80 * v4);
    for ( i = &result[10 * *(unsigned int *)(a1 + 24)]; i != result; result += 10 )
    {
      if ( result )
        *result = -4096;
    }
    v10 = (const __m128i *)(v5 + 24);
    if ( v8 != (const __m128i *)v5 )
    {
      while ( 1 )
      {
        v11 = v10[-2].m128i_i64[1];
        if ( v11 != -8192 && v11 != -4096 )
        {
          v12 = *(_DWORD *)(a1 + 24);
          if ( !v12 )
          {
            MEMORY[0] = v10[-2].m128i_i64[1];
            BUG();
          }
          v13 = v12 - 1;
          v14 = *(_QWORD *)(a1 + 8);
          v15 = 1;
          v16 = 0;
          v17 = v13 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v18 = (__int64 *)(v14 + 80LL * v17);
          v19 = *v18;
          if ( v11 != *v18 )
          {
            while ( v19 != -4096 )
            {
              if ( !v16 && v19 == -8192 )
                v16 = v18;
              v17 = v13 & (v15 + v17);
              v18 = (__int64 *)(v14 + 80LL * v17);
              v19 = *v18;
              if ( v11 == *v18 )
                goto LABEL_15;
              ++v15;
            }
            if ( v16 )
              v18 = v16;
          }
LABEL_15:
          *v18 = v11;
          v20 = v18 + 3;
          v18[1] = (__int64)(v18 + 3);
          v18[2] = 0x100000000LL;
          v21 = v10[-1].m128i_u32[2];
          if ( v21 && v18 + 1 != (__int64 *)&v10[-1] )
          {
            v23 = (const __m128i *)v10[-1].m128i_i64[0];
            if ( v10 == v23 )
            {
              v24 = v10;
              v25 = 8;
              if ( v21 == 1 )
                goto LABEL_23;
              v29 = v10[-1].m128i_i64[0];
              v31 = v10[-1].m128i_i32[2];
              sub_C8D5F0((__int64)(v18 + 1), v18 + 3, v21, 8u, 8, (__int64)(v18 + 1));
              v20 = (void *)v18[1];
              v24 = (const __m128i *)v10[-1].m128i_i64[0];
              v21 = v31;
              v23 = (const __m128i *)v29;
              v25 = 8LL * v10[-1].m128i_u32[2];
              if ( v25 )
              {
LABEL_23:
                v28 = v23;
                v30 = v21;
                memcpy(v20, v24, v25);
                *((_DWORD *)v18 + 4) = v30;
                v28[-1].m128i_i32[2] = 0;
              }
              else
              {
                *((_DWORD *)v18 + 4) = v31;
                *(_DWORD *)(v29 - 8) = 0;
              }
            }
            else
            {
              v18[1] = (__int64)v23;
              *((_DWORD *)v18 + 4) = v10[-1].m128i_i32[2];
              *((_DWORD *)v18 + 5) = v10[-1].m128i_i32[3];
              v10[-1].m128i_i64[0] = (__int64)v10;
              v10[-1].m128i_i32[3] = 0;
              v10[-1].m128i_i32[2] = 0;
            }
          }
          v18[4] = v10->m128i_i64[1];
          v18[5] = v10[1].m128i_i64[0];
          v18[6] = v10[1].m128i_i64[1];
          *(__m128i *)(v18 + 7) = _mm_loadu_si128(v10 + 2);
          *((_BYTE *)v18 + 72) = v10[3].m128i_i8[0];
          ++*(_DWORD *)(a1 + 16);
          v22 = v10[-1].m128i_u64[0];
          if ( (const __m128i *)v22 != v10 )
            _libc_free(v22);
        }
        if ( v8 == (const __m128i *)&v10[3].m128i_u64[1] )
          break;
        v10 += 5;
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, v32, 8);
  }
  else
  {
    v26 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[10 * v26]; j != result; result += 10 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
