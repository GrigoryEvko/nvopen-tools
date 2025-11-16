// Function: sub_8865A0
// Address: 0x8865a0
//
__int64 __fastcall sub_8865A0(char *src)
{
  _QWORD *v1; // rax
  __int64 v2; // rsi
  __m128i v4; // xmm1
  __m128i v5; // xmm2
  __m128i v6; // xmm3
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9[2]; // [rsp+0h] [rbp-60h] BYREF
  __m128i v10; // [rsp+10h] [rbp-50h]
  __m128i v11; // [rsp+20h] [rbp-40h]
  __m128i v12; // [rsp+30h] [rbp-30h]

  v1 = qword_4D049B8;
  if ( !qword_4D049B8 )
    return 0;
  if ( dword_4F5FFBC )
  {
    if ( *(_QWORD *)(*qword_4D049B8 + 64LL) )
    {
      if ( dword_4D03F98[0] )
      {
        sub_824D70(qword_4F07288, *qword_4D049B8);
        v1 = qword_4D049B8;
      }
    }
  }
  else
  {
    v4 = _mm_loadu_si128(&xmmword_4F06660[1]);
    v5 = _mm_loadu_si128(&xmmword_4F06660[2]);
    v6 = _mm_loadu_si128(&xmmword_4F06660[3]);
    v9[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
    v10 = v4;
    v11 = v5;
    v12 = v6;
    v9[1] = *(_QWORD *)&dword_4F077C8;
    sub_878540("std", 3u, v9);
    sub_886510((__int64)v9);
    if ( (v10.m128i_i8[1] & 0x40) == 0 )
    {
      v10.m128i_i8[0] &= ~0x80u;
      v10.m128i_i64[1] = 0;
    }
    sub_7D4600(*(_QWORD *)(qword_4F04C68[0] + 184LL), v9, 0x200u, v7, v8);
    v1 = qword_4D049B8;
  }
  v2 = v1[11];
  if ( v2 )
    return sub_879550(src, v2, 0);
  else
    return 0;
}
