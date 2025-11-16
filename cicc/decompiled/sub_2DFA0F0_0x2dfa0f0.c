// Function: sub_2DFA0F0
// Address: 0x2dfa0f0
//
__int64 __fastcall sub_2DFA0F0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v3; // rbx
  __int64 v4; // r13
  unsigned int v5; // eax
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // r15
  __int64 v9; // r14
  __int64 i; // rdx
  __int64 v11; // rbx
  __m128i **v12; // rdx
  __int64 v13; // rsi
  __m128i *v14; // rax
  __int64 j; // rdx
  __m128i **v16; // [rsp+8h] [rbp-48h]
  __m128i *v17; // [rsp+18h] [rbp-38h] BYREF

  v2 = (unsigned int)(a2 - 1);
  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
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
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = sub_C7D670(48LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = 48 * v3;
    v9 = v4 + 48 * v3;
    for ( i = result + 48 * v7; i != result; result += 48 )
    {
      if ( result )
      {
        *(_QWORD *)result = 0;
        *(_BYTE *)(result + 24) = 0;
        *(_QWORD *)(result + 32) = 0;
      }
    }
    if ( v9 != v4 )
    {
      v11 = v4;
      v12 = &v17;
      do
      {
        while ( *(_QWORD *)v11
             || *(_BYTE *)(v11 + 24) && (*(_QWORD *)(v11 + 8) || *(_QWORD *)(v11 + 16))
             || *(_QWORD *)(v11 + 32) )
        {
          v13 = v11;
          v16 = v12;
          v11 += 48;
          sub_2DF9F10(a1, v13, v12);
          v14 = v17;
          v12 = v16;
          *v17 = _mm_loadu_si128((const __m128i *)(v11 - 48));
          v14[1] = _mm_loadu_si128((const __m128i *)(v11 - 32));
          v14[2].m128i_i64[0] = *(_QWORD *)(v11 - 16);
          v17[2].m128i_i64[1] = *(_QWORD *)(v11 - 8);
          ++*(_DWORD *)(a1 + 16);
          if ( v9 == v11 )
            return sub_C7D6A0(v4, v8, 8);
        }
        v11 += 48;
      }
      while ( v9 != v11 );
    }
    return sub_C7D6A0(v4, v8, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = result + 48LL * *(unsigned int *)(a1 + 24); j != result; result += 48 )
    {
      if ( result )
      {
        *(_QWORD *)result = 0;
        *(_BYTE *)(result + 24) = 0;
        *(_QWORD *)(result + 32) = 0;
      }
    }
  }
  return result;
}
