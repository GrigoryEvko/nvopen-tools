// Function: sub_7D22B0
// Address: 0x7d22b0
//
__int64 __fastcall sub_7D22B0(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // rax
  __int64 v4; // rdi
  __int64 v5; // r12
  __int64 v6; // rsi
  char v7; // dl
  _OWORD v8[5]; // [rsp-A8h] [rbp-A8h] BYREF
  __m128i si128; // [rsp-58h] [rbp-58h]
  __m128i v10; // [rsp-48h] [rbp-48h]
  __m128i v11; // [rsp-38h] [rbp-38h]
  __m128i v12; // [rsp-28h] [rbp-28h]

  result = 0;
  if ( (*(_BYTE *)(a1 + 17) & 0x20) == 0 )
  {
    v3 = sub_7CFB70((_QWORD *)a1, 0);
    v4 = 0;
    v5 = v3;
    if ( dword_4F04C5C != -1 )
      v4 = qword_4F04C68[0] + 776LL * (int)dword_4F04C5C;
    v6 = 0;
    if ( v3 && *(_BYTE *)(v3 + 80) == 23 )
      v6 = *(_QWORD *)(a1 + 24);
    v8[0] = _mm_load_si128((const __m128i *)&xmmword_4F18000);
    v8[2] = _mm_load_si128((const __m128i *)&xmmword_4F18020);
    si128 = _mm_load_si128((const __m128i *)&xmmword_4F18050);
    DWORD2(v8[0]) = 1;
    si128.m128i_i32[2] = 1;
    v8[1] = _mm_load_si128((const __m128i *)&xmmword_4F18010);
    v8[3] = _mm_load_si128((const __m128i *)&xmmword_4F18030);
    v8[4] = _mm_load_si128((const __m128i *)&xmmword_4F18040);
    v10 = _mm_load_si128((const __m128i *)&xmmword_4F18060);
    v11 = _mm_load_si128((const __m128i *)&xmmword_4F18070);
    v12 = _mm_load_si128((const __m128i *)&xmmword_4F18080);
    result = sub_7D1000(v4, v6, a1, (int *)v8);
    if ( result )
    {
      *(_QWORD *)(a1 + 24) = result;
      v7 = *(_BYTE *)(result + 80);
      if ( v7 == 16 )
      {
        result = **(_QWORD **)(result + 88);
        v7 = *(_BYTE *)(result + 80);
      }
      if ( v7 == 24 )
        return *(_QWORD *)(result + 88);
    }
    else
    {
      return v5;
    }
  }
  return result;
}
