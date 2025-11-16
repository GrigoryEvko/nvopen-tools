// Function: sub_7AD1A0
// Address: 0x7ad1a0
//
__int64 __fastcall sub_7AD1A0(__int64 a1)
{
  __m128i v1; // xmm1
  __m128i v2; // xmm2
  __m128i v3; // xmm3
  __int64 v4; // rax
  unsigned int v5; // r8d
  char v6; // dl
  __int64 v8; // rax
  _QWORD v9[2]; // [rsp+0h] [rbp-40h] BYREF
  __m128i v10; // [rsp+10h] [rbp-30h]
  __m128i v11; // [rsp+20h] [rbp-20h]
  __m128i v12; // [rsp+30h] [rbp-10h]

  v1 = _mm_loadu_si128(&xmmword_4F06660[1]);
  v2 = _mm_loadu_si128(&xmmword_4F06660[2]);
  v3 = _mm_loadu_si128(&xmmword_4F06660[3]);
  v9[1] = _mm_loadu_si128(xmmword_4F06660).m128i_i64[1];
  v9[0] = a1;
  v10 = v1;
  v11 = v2;
  v12 = v3;
  v4 = sub_7D5DD0(v9, 16);
  v5 = 0;
  if ( v4 )
  {
    v6 = *(_BYTE *)(v4 + 80);
    v5 = 1;
    if ( v6 != 19 )
    {
      v5 = 0;
      if ( v6 == 3 )
      {
        if ( *(_BYTE *)(v4 + 104) )
        {
          v8 = *(_QWORD *)(v4 + 88);
          if ( (*(_BYTE *)(v8 + 177) & 0x10) != 0 )
            return *(_QWORD *)(*(_QWORD *)(v8 + 168) + 168LL) != 0;
        }
      }
    }
  }
  return v5;
}
