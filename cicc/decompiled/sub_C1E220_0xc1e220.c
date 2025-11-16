// Function: sub_C1E220
// Address: 0xc1e220
//
_QWORD *__fastcall sub_C1E220(__int64 a1, int a2)
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
  unsigned __int64 **v13; // rcx
  __int64 v14; // rax
  __int64 v15; // rdx
  _QWORD *j; // rdx
  unsigned __int64 **v17; // [rsp+8h] [rbp-48h]
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
  result = (_QWORD *)sub_C7D670(16LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    v9 = 16 * v4;
    *(_QWORD *)(a1 + 16) = 0;
    v10 = &v5[(unsigned __int64)v9 / 0x10];
    for ( i = &result[2 * v8]; i != result; result += 2 )
    {
      if ( result )
      {
        *result = 0;
        result[1] = -1;
      }
    }
    if ( v10 != v5 )
    {
      v12 = v5;
      v13 = (unsigned __int64 **)&v18;
      while ( 1 )
      {
        v14 = v12->m128i_i64[1];
        v15 = v12->m128i_i64[0];
        if ( v14 != -1 )
          break;
        if ( v15 )
        {
LABEL_12:
          v17 = v13;
          sub_C1C670(a1, (__int64)v12, v13);
          v13 = v17;
          *v18 = _mm_loadu_si128(v12);
          ++*(_DWORD *)(a1 + 16);
LABEL_13:
          if ( v10 == ++v12 )
            return (_QWORD *)sub_C7D6A0(v5, v9, 8);
        }
        else if ( v10 == ++v12 )
        {
          return (_QWORD *)sub_C7D6A0(v5, v9, 8);
        }
      }
      if ( !v15 && v14 == -2 )
        goto LABEL_13;
      goto LABEL_12;
    }
    return (_QWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[2 * *(unsigned int *)(a1 + 24)]; j != result; result += 2 )
    {
      if ( result )
      {
        *result = 0;
        result[1] = -1;
      }
    }
  }
  return result;
}
