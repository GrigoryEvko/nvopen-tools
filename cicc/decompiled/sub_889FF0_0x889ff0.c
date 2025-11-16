// Function: sub_889FF0
// Address: 0x889ff0
//
__int64 __fastcall sub_889FF0(char *a1)
{
  size_t v1; // rax
  unsigned __int64 v2; // rax
  __int64 v4[2]; // [rsp+0h] [rbp-50h] BYREF
  __m128i v5; // [rsp+10h] [rbp-40h]
  __m128i v6; // [rsp+20h] [rbp-30h]
  __m128i v7; // [rsp+30h] [rbp-20h]

  v4[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
  v5 = _mm_loadu_si128(&xmmword_4F06660[1]);
  v6 = _mm_loadu_si128(&xmmword_4F06660[2]);
  v7 = _mm_loadu_si128(&xmmword_4F06660[3]);
  v4[1] = *(_QWORD *)&dword_4F077C8;
  v1 = strlen(a1);
  v2 = sub_87A100(a1, v1, v4);
  return sub_889E70(v2);
}
