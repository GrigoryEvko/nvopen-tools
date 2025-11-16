// Function: sub_5F7E50
// Address: 0x5f7e50
//
__int64 __fastcall sub_5F7E50(char *src, __int64 a2)
{
  __int64 v2; // r15
  size_t v3; // rax
  _QWORD v5[2]; // [rsp+0h] [rbp-2C0h] BYREF
  __m128i v6; // [rsp+10h] [rbp-2B0h]
  __m128i v7; // [rsp+20h] [rbp-2A0h]
  __m128i v8; // [rsp+30h] [rbp-290h]
  _BYTE v9[288]; // [rsp+40h] [rbp-280h] BYREF
  __int64 v10; // [rsp+160h] [rbp-160h]

  v2 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 600);
  v5[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
  v6 = _mm_loadu_si128(&xmmword_4F06660[1]);
  v7 = _mm_loadu_si128(&xmmword_4F06660[2]);
  v8 = _mm_loadu_si128(&xmmword_4F06660[3]);
  v5[1] = unk_4F077C8;
  v3 = strlen(src);
  sub_878540(src, v3);
  sub_5E4C60((__int64)v9, &unk_4F077C8);
  if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(a2) )
    sub_8AE000(a2);
  v10 = a2;
  return sub_5F4F20((__int64)v5, v2, (__int64)v9, dword_4F04C64);
}
