// Function: sub_B8DC60
// Address: 0xb8dc60
//
_QWORD *__fastcall sub_B8DC60(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r15
  const __m128i *v5; // r13
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r15
  const __m128i *v10; // r14
  _QWORD *i; // rdx
  const __m128i *v12; // rbx
  __m128i **v13; // rdx
  __int64 *v14; // rsi
  __m128i *v15; // rax
  _QWORD *j; // rdx
  __m128i **v17; // [rsp+8h] [rbp-48h]
  __m128i *v18; // [rsp+18h] [rbp-38h] BYREF

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(const __m128i **)(a1 + 8);
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
  result = (_QWORD *)sub_C7D670(32LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    v9 = 32 * v4;
    *(_QWORD *)(a1 + 16) = 0;
    v10 = &v5[(unsigned __int64)v9 / 0x10];
    for ( i = &result[4 * v8]; i != result; result += 4 )
    {
      if ( result )
      {
        *result = -1;
        result[1] = 0;
        result[2] = -1;
        result[3] = 0;
      }
    }
    if ( v10 != v5 )
    {
      v12 = v5;
      v13 = &v18;
      while ( 1 )
      {
        if ( v12->m128i_i64[0] == -1 )
        {
          if ( v12[1].m128i_i64[0] == -1 )
            goto LABEL_11;
LABEL_15:
          v14 = (__int64 *)v12;
          v17 = v13;
          v12 += 2;
          sub_B8D650(a1, v14, v13);
          v15 = v18;
          v13 = v17;
          *v18 = _mm_loadu_si128(v12 - 2);
          v15[1] = _mm_loadu_si128(v12 - 1);
          ++*(_DWORD *)(a1 + 16);
          if ( v10 == v12 )
            return (_QWORD *)sub_C7D6A0(v5, v9, 8);
        }
        else
        {
          if ( v12->m128i_i64[0] != -2 || v12[1].m128i_i64[0] != -2 )
            goto LABEL_15;
LABEL_11:
          v12 += 2;
          if ( v10 == v12 )
            return (_QWORD *)sub_C7D6A0(v5, v9, 8);
        }
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[4 * *(unsigned int *)(a1 + 24)]; j != result; result += 4 )
    {
      if ( result )
      {
        *result = -1;
        result[1] = 0;
        result[2] = -1;
        result[3] = 0;
      }
    }
  }
  return result;
}
