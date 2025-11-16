// Function: sub_2D2F0E0
// Address: 0x2d2f0e0
//
__int64 __fastcall sub_2D2F0E0(__int64 a1, int a2)
{
  __int64 v2; // rbx
  __int64 v3; // r14
  unsigned int v4; // eax
  __int64 result; // rax
  __int64 v6; // rcx
  __int64 v7; // r15
  __int64 v8; // r13
  __int64 i; // rdx
  __int64 v10; // rbx
  __int64 *v11; // rdx
  __int64 v12; // rsi
  __m128i *v13; // rax
  bool v14; // al
  __int64 j; // rdx
  __int64 *v16; // [rsp+8h] [rbp-78h]
  __int64 *v17; // [rsp+8h] [rbp-78h]
  _QWORD v18[2]; // [rsp+18h] [rbp-68h] BYREF
  _QWORD v19[11]; // [rsp+28h] [rbp-58h] BYREF

  v2 = *(unsigned int *)(a1 + 24);
  v3 = *(_QWORD *)(a1 + 8);
  v4 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v4 < 0x40 )
    v4 = 64;
  *(_DWORD *)(a1 + 24) = v4;
  result = sub_C7D670(56LL * v4, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v3 )
  {
    v6 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v7 = 56 * v2;
    v8 = v3 + 56 * v2;
    for ( i = result + 56 * v6; i != result; result += 56 )
    {
      if ( result )
      {
        *(_QWORD *)result = 0;
        *(_BYTE *)(result + 24) = 0;
        *(_QWORD *)(result + 32) = 0;
      }
    }
    v18[1] = 0;
    v19[0] = 0;
    v19[1] = 0;
    v19[2] = 1;
    v19[3] = 0;
    if ( v8 != v3 )
    {
      v10 = v3;
      v11 = v18;
      do
      {
        while ( 1 )
        {
          if ( !*(_QWORD *)v10 )
          {
            if ( !*(_BYTE *)(v10 + 24) || (v17 = v11, v14 = sub_2D27C10((_QWORD *)(v10 + 8), v19), v11 = v17, v14) )
            {
              if ( !*(_QWORD *)(v10 + 32) )
                break;
            }
          }
          v12 = v10;
          v16 = v11;
          v10 += 56;
          sub_2D29210(a1, v12, v11);
          v13 = (__m128i *)v18[0];
          v11 = v16;
          *(__m128i *)v18[0] = _mm_loadu_si128((const __m128i *)(v10 - 56));
          v13[1] = _mm_loadu_si128((const __m128i *)(v10 - 40));
          v13[2].m128i_i64[0] = *(_QWORD *)(v10 - 24);
          *(__m128i *)(v18[0] + 40LL) = _mm_loadu_si128((const __m128i *)(v10 - 16));
          ++*(_DWORD *)(a1 + 16);
          if ( v8 == v10 )
            return sub_C7D6A0(v3, v7, 8);
        }
        v10 += 56;
      }
      while ( v8 != v10 );
    }
    return sub_C7D6A0(v3, v7, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = result + 56LL * *(unsigned int *)(a1 + 24); j != result; result += 56 )
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
