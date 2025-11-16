// Function: sub_20C4C90
// Address: 0x20c4c90
//
__m128i *__fastcall sub_20C4C90(__int64 a1, __m128i *a2)
{
  __m128i *result; // rax
  __int64 v3; // rdx
  __int64 v4; // r13
  _BOOL4 v5; // r8d
  __m128i *v6; // rbx
  _BOOL4 v7; // [rsp+Ch] [rbp-34h]

  result = (__m128i *)sub_20C4BF0(a1, (unsigned __int64 *)a2);
  if ( v3 )
  {
    v4 = v3;
    v5 = 1;
    if ( !result && v3 != a1 + 8 )
      v5 = a2->m128i_i64[0] < *(_QWORD *)(v3 + 32);
    v7 = v5;
    v6 = (__m128i *)sub_22077B0(48);
    v6[2] = _mm_loadu_si128(a2);
    sub_220F040(v7, v6, v4, a1 + 8);
    ++*(_QWORD *)(a1 + 40);
    return v6;
  }
  return result;
}
