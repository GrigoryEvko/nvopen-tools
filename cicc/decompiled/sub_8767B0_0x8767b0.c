// Function: sub_8767B0
// Address: 0x8767b0
//
int __fastcall sub_8767B0(const __m128i *a1)
{
  __int64 v1; // rbp
  FILE **v2; // rax
  unsigned __int8 *v3; // rsi
  __m128i v5; // [rsp-48h] [rbp-48h] BYREF
  __m128i v6; // [rsp-38h] [rbp-38h]
  __m128i v7; // [rsp-28h] [rbp-28h]
  __m128i v8; // [rsp-18h] [rbp-18h]
  __int64 v9; // [rsp-8h] [rbp-8h]

  v2 = &qword_4D04900;
  if ( qword_4D04900 )
  {
    v9 = v1;
    v6 = _mm_loadu_si128(a1 + 1);
    v3 = (unsigned __int8 *)v6.m128i_i64[1];
    v5 = _mm_loadu_si128(a1);
    v7 = _mm_loadu_si128(a1 + 2);
    v8 = _mm_loadu_si128(a1 + 3);
    if ( !v6.m128i_i64[1] && (v6.m128i_i8[1] & 0x20) == 0 )
    {
      sub_885B10(&v5);
      v3 = (unsigned __int8 *)v6.m128i_i64[1];
    }
    LODWORD(v2) = sub_8754F0(68, v3, (__int64)&v5.m128i_i64[1]);
  }
  return (int)v2;
}
