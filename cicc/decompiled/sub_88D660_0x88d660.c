// Function: sub_88D660
// Address: 0x88d660
//
__int64 __fastcall sub_88D660()
{
  __int64 v0; // rbp
  __int64 result; // rax
  __m128i v2; // xmm1
  __m128i v3; // xmm2
  __m128i v4; // xmm3
  __int64 v5[2]; // [rsp-48h] [rbp-48h] BYREF
  __m128i v6; // [rsp-38h] [rbp-38h]
  __m128i v7; // [rsp-28h] [rbp-28h]
  __m128i v8; // [rsp-18h] [rbp-18h]
  __int64 v9; // [rsp-8h] [rbp-8h]

  result = qword_4F60210;
  if ( !qword_4F60210 )
  {
    v9 = v0;
    v2 = _mm_loadu_si128(&xmmword_4F06660[1]);
    v3 = _mm_loadu_si128(&xmmword_4F06660[2]);
    v4 = _mm_loadu_si128(&xmmword_4F06660[3]);
    v5[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
    v6 = v2;
    v7 = v3;
    v8 = v4;
    v5[1] = *(_QWORD *)&dword_4F077C8;
    result = sub_87A100("<invented>", 0xAu, v5);
    qword_4F60210 = result;
  }
  return result;
}
