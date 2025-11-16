// Function: sub_15CAC60
// Address: 0x15cac60
//
__int64 __fastcall sub_15CAC60(__int64 a1, __m128i *a2)
{
  unsigned int v2; // edx
  __int64 result; // rax
  __int64 v4; // rcx
  __m128i *v5; // rcx
  __int64 v6; // rcx
  __int64 v7; // rdx
  __m128i v8; // xmm0

  v2 = *(_DWORD *)(a1 + 96);
  if ( v2 >= *(_DWORD *)(a1 + 100) )
  {
    sub_14B3F20(a1 + 88, 0);
    v2 = *(_DWORD *)(a1 + 96);
  }
  result = *(_QWORD *)(a1 + 88) + 88LL * v2;
  if ( result )
  {
    *(_QWORD *)result = result + 16;
    if ( (__m128i *)a2->m128i_i64[0] == &a2[1] )
    {
      *(__m128i *)(result + 16) = _mm_loadu_si128(a2 + 1);
    }
    else
    {
      *(_QWORD *)result = a2->m128i_i64[0];
      *(_QWORD *)(result + 16) = a2[1].m128i_i64[0];
    }
    v4 = a2->m128i_i64[1];
    a2->m128i_i64[0] = (__int64)a2[1].m128i_i64;
    a2->m128i_i64[1] = 0;
    *(_QWORD *)(result + 8) = v4;
    a2[1].m128i_i8[0] = 0;
    *(_QWORD *)(result + 32) = result + 48;
    v5 = (__m128i *)a2[2].m128i_i64[0];
    if ( v5 == &a2[3] )
    {
      *(__m128i *)(result + 48) = _mm_loadu_si128(a2 + 3);
    }
    else
    {
      *(_QWORD *)(result + 32) = v5;
      *(_QWORD *)(result + 48) = a2[3].m128i_i64[0];
    }
    v6 = a2[2].m128i_i64[1];
    a2[2].m128i_i64[0] = (__int64)a2[3].m128i_i64;
    a2[2].m128i_i64[1] = 0;
    *(_QWORD *)(result + 40) = v6;
    v7 = a2[5].m128i_i64[0];
    v8 = _mm_loadu_si128(a2 + 4);
    a2[3].m128i_i8[0] = 0;
    *(_QWORD *)(result + 80) = v7;
    *(__m128i *)(result + 64) = v8;
    v2 = *(_DWORD *)(a1 + 96);
  }
  *(_DWORD *)(a1 + 96) = v2 + 1;
  return result;
}
