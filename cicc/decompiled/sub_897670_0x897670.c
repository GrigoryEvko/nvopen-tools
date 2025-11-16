// Function: sub_897670
// Address: 0x897670
//
__int64 sub_897670()
{
  __m128i v0; // xmm1
  __m128i v1; // xmm2
  __m128i v2; // xmm3
  __int64 result; // rax
  __int64 v4[2]; // [rsp+0h] [rbp-40h] BYREF
  __m128i v5; // [rsp+10h] [rbp-30h]
  __m128i v6; // [rsp+20h] [rbp-20h]
  __m128i v7; // [rsp+30h] [rbp-10h]

  v0 = _mm_loadu_si128(&xmmword_4F06660[1]);
  v1 = _mm_loadu_si128(&xmmword_4F06660[2]);
  v2 = _mm_loadu_si128(&xmmword_4F06660[3]);
  v4[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
  v5 = v0;
  v6 = v1;
  v7 = v2;
  v4[1] = *(_QWORD *)&dword_4F077C8;
  result = sub_87A100("initializer_list", 0x10u, v4);
  qword_4F60230 = result;
  return result;
}
