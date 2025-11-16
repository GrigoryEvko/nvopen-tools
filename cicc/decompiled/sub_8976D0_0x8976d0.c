// Function: sub_8976D0
// Address: 0x8976d0
//
_QWORD *sub_8976D0()
{
  _QWORD *result; // rax
  __m128i v1; // xmm1
  __m128i v2; // xmm2
  __m128i v3; // xmm3
  __int64 v4; // rax
  unsigned __int64 v5; // xmm4_8
  __m128i v6; // xmm5
  __m128i v7; // xmm6
  __m128i v8; // xmm7
  unsigned __int64 v9; // [rsp-68h] [rbp-68h] BYREF
  __int64 v10; // [rsp-60h] [rbp-60h]
  __m128i v11; // [rsp-58h] [rbp-58h]
  __m128i v12; // [rsp-48h] [rbp-48h]
  __m128i v13; // [rsp-38h] [rbp-38h]

  if ( unk_4D04534 )
  {
    qword_4F60220 = 0;
    qword_4F60228 = 0;
    return &qword_4F60228;
  }
  else
  {
    v1 = _mm_loadu_si128(&xmmword_4F06660[1]);
    v2 = _mm_loadu_si128(&xmmword_4F06660[2]);
    v3 = _mm_loadu_si128(&xmmword_4F06660[3]);
    v9 = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
    v11 = v1;
    v12 = v2;
    v13 = v3;
    v10 = *(_QWORD *)&dword_4F077C8;
    v4 = sub_87A100("move", 4u, (__int64 *)&v9);
    v5 = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
    v6 = _mm_loadu_si128(&xmmword_4F06660[1]);
    v7 = _mm_loadu_si128(&xmmword_4F06660[2]);
    qword_4F60228 = v4;
    v8 = _mm_loadu_si128(&xmmword_4F06660[3]);
    v9 = v5;
    v11 = v6;
    v10 = *(_QWORD *)&dword_4F077C8;
    v12 = v7;
    v13 = v8;
    result = (_QWORD *)sub_87A100("forward", 7u, (__int64 *)&v9);
    qword_4F60220 = result;
  }
  return result;
}
