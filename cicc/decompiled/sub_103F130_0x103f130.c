// Function: sub_103F130
// Address: 0x103f130
//
__int64 __fastcall sub_103F130(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v3; // r12
  __int64 v4; // r14
  unsigned int v5; // eax
  __int64 result; // rax
  __int64 v7; // r12
  __int64 i; // rdx
  __int64 v9; // r15
  __int64 v10; // rsi
  __m128i *v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 j; // rdx
  __int64 v15; // [rsp+0h] [rbp-D0h]
  __int64 v16; // [rsp+18h] [rbp-B8h] BYREF
  _QWORD v17[8]; // [rsp+20h] [rbp-B0h] BYREF
  _QWORD v18[14]; // [rsp+60h] [rbp-70h] BYREF

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
  result = sub_C7D670(104LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v15 = 104 * v3;
    v7 = v4 + 104 * v3;
    for ( i = result + 104LL * *(unsigned int *)(a1 + 24); i != result; result += 104 )
    {
      if ( result )
      {
        *(_BYTE *)result = 0;
        *(_QWORD *)(result + 8) = -4096;
        *(_QWORD *)(result + 16) = -3;
        *(_QWORD *)(result + 24) = 0;
        *(_QWORD *)(result + 32) = 0;
        *(_QWORD *)(result + 40) = 0;
        *(_QWORD *)(result + 48) = 0;
      }
    }
    v17[0] = 0;
    v17[1] = -4096;
    v17[2] = -3;
    memset(&v17[3], 0, 32);
    v18[0] = 0;
    v18[1] = -8192;
    v18[2] = -4;
    memset(&v18[3], 0, 32);
    if ( v7 != v4 )
    {
      v9 = v4;
      do
      {
        while ( sub_103B1D0(v9, (__int64)v17) || sub_103B1D0(v9, (__int64)v18) )
        {
          v9 += 104;
          if ( v7 == v9 )
            return sub_C7D6A0(v4, v15, 8);
        }
        v10 = v9;
        v9 += 104;
        sub_103ED40(a1, v10, &v16);
        v11 = (__m128i *)v16;
        *(__m128i *)v16 = _mm_loadu_si128((const __m128i *)(v9 - 104));
        v11[1] = _mm_loadu_si128((const __m128i *)(v9 - 88));
        v11[2] = _mm_loadu_si128((const __m128i *)(v9 - 72));
        v11[3].m128i_i64[0] = *(_QWORD *)(v9 - 56);
        v12 = v16;
        *(__m128i *)(v16 + 56) = _mm_loadu_si128((const __m128i *)(v9 - 48));
        *(__m128i *)(v12 + 72) = _mm_loadu_si128((const __m128i *)(v9 - 32));
        *(__m128i *)(v12 + 88) = _mm_loadu_si128((const __m128i *)(v9 - 16));
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v7 != v9 );
    }
    return sub_C7D6A0(v4, v15, 8);
  }
  else
  {
    v13 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = result + 104 * v13; j != result; result += 104 )
    {
      if ( result )
      {
        *(_BYTE *)result = 0;
        *(_QWORD *)(result + 8) = -4096;
        *(_QWORD *)(result + 16) = -3;
        *(_QWORD *)(result + 24) = 0;
        *(_QWORD *)(result + 32) = 0;
        *(_QWORD *)(result + 40) = 0;
        *(_QWORD *)(result + 48) = 0;
      }
    }
  }
  return result;
}
