// Function: sub_879C70
// Address: 0x879c70
//
__int64 __fastcall sub_879C70(char *src, const char *a2, unsigned int a3)
{
  size_t v4; // rax
  __int64 v6[2]; // [rsp+0h] [rbp-60h] BYREF
  __m128i v7; // [rsp+10h] [rbp-50h]
  __m128i v8; // [rsp+20h] [rbp-40h]
  __m128i v9; // [rsp+30h] [rbp-30h]

  v6[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
  v7 = _mm_loadu_si128(&xmmword_4F06660[1]);
  v8 = _mm_loadu_si128(&xmmword_4F06660[2]);
  v9 = _mm_loadu_si128(&xmmword_4F06660[3]);
  v6[1] = *(_QWORD *)&dword_4F077C8;
  v4 = strlen(src);
  sub_878540(src, v4, v6);
  if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(a2) )
    sub_8AE000(a2);
  return sub_7D2AC0(v6, a2, a3);
}
