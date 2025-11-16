// Function: sub_25CFE60
// Address: 0x25cfe60
//
_QWORD *__fastcall sub_25CFE60(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r15
  __int64 v5; // r13
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r15
  __int64 v10; // r14
  _QWORD *i; // rdx
  __int64 v12; // rbx
  __m128i **v13; // rcx
  __m128i *v14; // rax
  _QWORD *j; // rdx
  __m128i **v16; // [rsp+8h] [rbp-48h]
  __m128i *v17; // [rsp+18h] [rbp-38h] BYREF

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
  result = (_QWORD *)sub_C7D670(32LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    v9 = 32 * v4;
    *(_QWORD *)(a1 + 16) = 0;
    v10 = v5 + v9;
    for ( i = &result[4 * v8]; i != result; result += 4 )
    {
      if ( result )
      {
        *result = -1;
        result[1] = 0;
        result[2] = -1;
      }
    }
    if ( v10 != v5 )
    {
      v12 = v5;
      v13 = &v17;
      do
      {
        while ( *(_QWORD *)v12 == -1 )
        {
          if ( *(_QWORD *)(v12 + 16) != -1 )
            goto LABEL_11;
LABEL_12:
          v12 += 32;
          if ( v10 == v12 )
            return (_QWORD *)sub_C7D6A0(v5, v9, 8);
        }
        if ( *(_QWORD *)v12 != -2 || *(_QWORD *)(v12 + 16) != -2 )
        {
LABEL_11:
          v16 = v13;
          sub_25CE2B0(a1, (char **)v12, v13);
          v14 = v17;
          v13 = v16;
          *v17 = _mm_loadu_si128((const __m128i *)v12);
          v14[1].m128i_i64[0] = *(_QWORD *)(v12 + 16);
          v17[1].m128i_i32[2] = *(_DWORD *)(v12 + 24);
          ++*(_DWORD *)(a1 + 16);
          goto LABEL_12;
        }
        v12 += 32;
      }
      while ( v10 != v12 );
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
      }
    }
  }
  return result;
}
