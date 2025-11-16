// Function: sub_87FA90
// Address: 0x87fa90
//
__int64 __fastcall sub_87FA90(__int64 a1, char *a2)
{
  size_t v2; // rax
  __int64 *v3; // rbx
  __int64 result; // rax
  __int64 v5[2]; // [rsp+0h] [rbp-60h] BYREF
  __m128i v6; // [rsp+10h] [rbp-50h]
  __m128i v7; // [rsp+20h] [rbp-40h]
  __m128i v8; // [rsp+30h] [rbp-30h]

  v5[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
  v6 = _mm_loadu_si128(&xmmword_4F06660[1]);
  v7 = _mm_loadu_si128(&xmmword_4F06660[2]);
  v8 = _mm_loadu_si128(&xmmword_4F06660[3]);
  v5[1] = *(_QWORD *)&dword_4F077C8;
  v2 = strlen(a2);
  sub_878540(a2, v2, v5);
  v3 = sub_87EBB0(4u, v5[0], &dword_4F077C8);
  result = sub_877D80(a1, v3);
  v3[11] = a1;
  return result;
}
