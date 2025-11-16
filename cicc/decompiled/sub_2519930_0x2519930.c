// Function: sub_2519930
// Address: 0x2519930
//
__int64 __fastcall sub_2519930(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // rbx
  __int64 v5; // r12
  unsigned int v6; // edi
  __int64 result; // rax
  __int64 v8; // r12
  __int64 v9; // rdx
  __int64 v10; // rbx
  __int64 v11; // rdx
  __m128i v12; // xmm0
  __int64 v13; // r15
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 j; // rdx
  __m128i v17; // xmm2
  __int64 v18; // [rsp+8h] [rbp-78h]
  __int64 v19; // [rsp+10h] [rbp-70h]
  __int64 v20; // [rsp+18h] [rbp-68h]
  __int64 v21; // [rsp+20h] [rbp-60h]
  __int64 v22; // [rsp+28h] [rbp-58h]
  __int64 v23; // [rsp+30h] [rbp-50h] BYREF
  __m128i i; // [rsp+38h] [rbp-48h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(unsigned int *)(a1 + 24);
  v22 = v4;
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
  result = sub_C7D670(32LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v8 = 32 * v5;
    v9 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v10 = v4 + v8;
    v11 = result + 32 * v9;
    for ( i = _mm_loadu_si128((const __m128i *)&unk_4FEE4D0); v11 != result; result += 32 )
    {
      if ( result )
      {
        v23 = -4096;
        v12 = _mm_loadu_si128((const __m128i *)&v23);
        *(_QWORD *)(result + 16) = i.m128i_i64[1];
        *(__m128i *)result = v12;
      }
    }
    v21 = unk_4FEE4D0;
    v19 = unk_4FEE4D8;
    v18 = qword_4FEE4C0[1];
    v20 = qword_4FEE4C0[0];
    if ( v10 != v22 )
    {
      v13 = v22;
      while ( *(_QWORD *)v13 == -4096 )
      {
        if ( v21 == *(_QWORD *)(v13 + 8) && *(_QWORD *)(v13 + 16) == v19 )
        {
          v13 += 32;
          if ( v10 == v13 )
            return sub_C7D6A0(v22, v8, 8);
        }
        else
        {
LABEL_11:
          sub_2512100(a1, (__int64 *)v13, (__int64 **)&v23);
          v14 = v23;
          *(_QWORD *)v23 = *(_QWORD *)v13;
          *(__m128i *)(v14 + 8) = _mm_loadu_si128((const __m128i *)(v13 + 8));
          *(_QWORD *)(v23 + 24) = *(_QWORD *)(v13 + 24);
          ++*(_DWORD *)(a1 + 16);
LABEL_12:
          v13 += 32;
          if ( v10 == v13 )
            return sub_C7D6A0(v22, v8, 8);
        }
      }
      if ( *(_QWORD *)v13 == -8192 && *(_QWORD *)(v13 + 8) == v20 && *(_QWORD *)(v13 + 16) == v18 )
        goto LABEL_12;
      goto LABEL_11;
    }
    return sub_C7D6A0(v22, v8, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    v15 = 32LL * *(unsigned int *)(a1 + 24);
    i = _mm_loadu_si128((const __m128i *)&unk_4FEE4D0);
    for ( j = result + v15; j != result; result += 32 )
    {
      if ( result )
      {
        v23 = -4096;
        v17 = _mm_loadu_si128((const __m128i *)&v23);
        *(_QWORD *)(result + 16) = i.m128i_i64[1];
        *(__m128i *)result = v17;
      }
    }
  }
  return result;
}
