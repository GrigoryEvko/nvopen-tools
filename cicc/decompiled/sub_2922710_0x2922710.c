// Function: sub_2922710
// Address: 0x2922710
//
__int64 __fastcall sub_2922710(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r15
  __int64 v5; // r13
  unsigned int v6; // edi
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // r15
  __int64 v10; // r14
  __int64 i; // rdx
  __int64 v12; // rbx
  __int64 *v13; // rdx
  __int64 v14; // rsi
  __m128i *v15; // rax
  __int64 v16; // rax
  unsigned __int64 j; // rdx
  __int64 *v18; // [rsp+8h] [rbp-48h]
  __int64 v19; // [rsp+18h] [rbp-38h] BYREF

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
  result = sub_C7D670((unsigned __int64)v6 << 6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    v9 = v4 << 6;
    *(_QWORD *)(a1 + 16) = 0;
    v10 = v5 + v9;
    for ( i = result + (v8 << 6); i != result; result += 64 )
    {
      if ( result )
      {
        *(_QWORD *)result = 0;
        *(_BYTE *)(result + 24) = 0;
        *(_QWORD *)(result + 32) = 0;
      }
    }
    if ( v10 != v5 )
    {
      v12 = v5;
      v13 = &v19;
      do
      {
        while ( *(_QWORD *)v12
             || *(_BYTE *)(v12 + 24) && (*(_QWORD *)(v12 + 8) || *(_QWORD *)(v12 + 16))
             || *(_QWORD *)(v12 + 32) )
        {
          v14 = v12;
          v18 = v13;
          v12 += 64;
          sub_2921690(a1, v14, v13);
          v15 = (__m128i *)v19;
          v13 = v18;
          *(__m128i *)v19 = _mm_loadu_si128((const __m128i *)(v12 - 64));
          v15[1] = _mm_loadu_si128((const __m128i *)(v12 - 48));
          v15[2].m128i_i64[0] = *(_QWORD *)(v12 - 32);
          v16 = v19;
          *(__m128i *)(v19 + 40) = _mm_loadu_si128((const __m128i *)(v12 - 24));
          *(_QWORD *)(v16 + 56) = *(_QWORD *)(v12 - 8);
          ++*(_DWORD *)(a1 + 16);
          if ( v10 == v12 )
            return sub_C7D6A0(v5, v9, 8);
        }
        v12 += 64;
      }
      while ( v10 != v12 );
    }
    return sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = result + ((unsigned __int64)*(unsigned int *)(a1 + 24) << 6); j != result; result += 64 )
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
