// Function: sub_7F72E0
// Address: 0x7f72e0
//
__int64 __fastcall sub_7F72E0(__int64 a1, const __m128i *a2, int a3, __int64 a4, int *a5)
{
  __int64 v7; // r12
  _QWORD *v9; // rax
  const __m128i *v10; // rdi
  __int64 result; // rax
  __int64 v12; // [rsp+8h] [rbp-38h]

  v7 = *(_QWORD *)(a1 + 80);
  if ( a3 )
  {
    v12 = *(_QWORD *)(v7 + 88);
    if ( v12 )
    {
      v9 = sub_73A830(1, 5u);
      sub_7E6AB0(v12, (__int64)v9, a5);
    }
  }
  *(__m128i *)(v7 + 8) = _mm_loadu_si128(a2);
  *(__m128i *)(v7 + 24) = _mm_loadu_si128(a2 + 1);
  *(__m128i *)(v7 + 40) = _mm_loadu_si128(a2 + 2);
  *(__m128i *)(v7 + 56) = _mm_loadu_si128(a2 + 3);
  *(_QWORD *)(v7 + 72) = a2[4].m128i_i64[0];
  v10 = (const __m128i *)a2[2].m128i_i64[0];
  if ( v10 )
    *(_QWORD *)(v7 + 40) = sub_7F5340(v10);
  result = *(_QWORD *)(a4 + 48);
  *(_QWORD *)(v7 + 80) = result;
  *(_QWORD *)(a4 + 48) = a1;
  *(_QWORD *)(a4 + 40) = a1;
  return result;
}
