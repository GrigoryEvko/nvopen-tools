// Function: sub_149DB70
// Address: 0x149db70
//
__int64 __fastcall sub_149DB70(const __m128i *a1, unsigned __int8 (__fastcall *a2)(__m128i *, __int8 *))
{
  __int8 *v2; // rbx
  __int64 v3; // rax
  __m128i v4; // xmm0
  __m128i v5; // xmm1
  int v6; // eax
  __m128i v7; // xmm5
  __int64 result; // rax
  __m128i v9; // [rsp+0h] [rbp-50h] BYREF
  __m128i v10; // [rsp+10h] [rbp-40h] BYREF
  __int64 v11; // [rsp+20h] [rbp-30h]

  v2 = &a1[-3].m128i_i8[8];
  v3 = a1[2].m128i_i64[0];
  v9 = _mm_loadu_si128(a1);
  v11 = v3;
  v10 = _mm_loadu_si128(a1 + 1);
  while ( a2(&v9, v2) )
  {
    v4 = _mm_loadu_si128((const __m128i *)v2);
    v5 = _mm_loadu_si128((const __m128i *)v2 + 1);
    v2 -= 40;
    v6 = *((_DWORD *)v2 + 18);
    *((__m128i *)v2 + 5) = v4;
    *((_DWORD *)v2 + 28) = v6;
    *((__m128i *)v2 + 6) = v5;
  }
  v7 = _mm_loadu_si128(&v10);
  result = (unsigned int)v11;
  *(__m128i *)(v2 + 40) = _mm_loadu_si128(&v9);
  *((_DWORD *)v2 + 18) = result;
  *(__m128i *)(v2 + 56) = v7;
  return result;
}
