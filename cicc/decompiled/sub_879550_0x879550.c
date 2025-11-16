// Function: sub_879550
// Address: 0x879550
//
__int64 __fastcall sub_879550(char *src, __int64 a2, unsigned int a3)
{
  size_t v4; // rax
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v8[2]; // [rsp+0h] [rbp-60h] BYREF
  __m128i v9; // [rsp+10h] [rbp-50h]
  __m128i v10; // [rsp+20h] [rbp-40h]
  __m128i v11; // [rsp+30h] [rbp-30h]

  v8[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
  v9 = _mm_loadu_si128(&xmmword_4F06660[1]);
  v10 = _mm_loadu_si128(&xmmword_4F06660[2]);
  v11 = _mm_loadu_si128(&xmmword_4F06660[3]);
  v8[1] = *(_QWORD *)&dword_4F077C8;
  v4 = strlen(src);
  sub_878540(src, v4, v8);
  if ( a2 )
    return sub_7D4A40(v8, a2, a3, v5, v6);
  else
    return sub_7D4600(qword_4F07288, v8, a3, v5, v6);
}
