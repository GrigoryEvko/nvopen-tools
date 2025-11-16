// Function: sub_19F1650
// Address: 0x19f1650
//
_QWORD *__fastcall sub_19F1650(__int64 a1, int a2)
{
  __int64 v2; // r14
  __m128i *v3; // r13
  unsigned int v4; // eax
  _QWORD *result; // rax
  __int64 v6; // rdx
  __m128i *v7; // r14
  _QWORD *i; // rdx
  __m128i *v9; // rbx
  _QWORD *j; // rdx
  __m128i *v11; // [rsp+8h] [rbp-38h] BYREF

  v2 = *(unsigned int *)(a1 + 24);
  v3 = *(__m128i **)(a1 + 8);
  v4 = sub_1454B60((unsigned int)(a2 - 1));
  if ( v4 < 0x40 )
    v4 = 64;
  *(_DWORD *)(a1 + 24) = v4;
  result = (_QWORD *)sub_22077B0(16LL * v4);
  *(_QWORD *)(a1 + 8) = result;
  if ( v3 )
  {
    v6 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v7 = &v3[v2];
    for ( i = &result[2 * v6]; i != result; result += 2 )
    {
      if ( result )
      {
        *result = -8;
        result[1] = -8;
      }
    }
    if ( v7 != v3 )
    {
      v9 = v3;
      while ( v9->m128i_i64[0] == -8 )
      {
        if ( v9->m128i_i64[1] == -8 )
        {
          if ( v7 == ++v9 )
            return (_QWORD *)j___libc_free_0(v3);
        }
        else
        {
LABEL_11:
          sub_19E8F30(a1, v9->m128i_i64, (__int64 **)&v11);
          *v11 = _mm_loadu_si128(v9);
          ++*(_DWORD *)(a1 + 16);
LABEL_12:
          if ( v7 == ++v9 )
            return (_QWORD *)j___libc_free_0(v3);
        }
      }
      if ( v9->m128i_i64[0] == -16 && v9->m128i_i64[1] == -16 )
        goto LABEL_12;
      goto LABEL_11;
    }
    return (_QWORD *)j___libc_free_0(v3);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[2 * *(unsigned int *)(a1 + 24)]; j != result; result += 2 )
    {
      if ( result )
      {
        *result = -8;
        result[1] = -8;
      }
    }
  }
  return result;
}
