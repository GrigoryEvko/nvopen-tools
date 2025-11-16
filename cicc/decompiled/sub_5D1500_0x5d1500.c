// Function: sub_5D1500
// Address: 0x5d1500
//
__int64 __fastcall sub_5D1500(__int64 a1)
{
  const void *v2; // r14
  size_t v3; // r13
  const void *v4; // r15
  __int64 v5; // rbx
  __int64 v6; // rsi
  __int64 v7; // rax
  __m128i si128; // xmm0

  sub_7C9660();
  if ( word_4F06418[0] == 1
    && (v2 = *(const void **)(unk_4D04A00 + 8LL),
        v3 = *(_QWORD *)(unk_4D04A00 + 16LL),
        sub_7B8B50(),
        word_4F06418[0] == 1) )
  {
    v4 = *(const void **)(unk_4D04A00 + 8LL);
    v5 = *(_QWORD *)(unk_4D04A00 + 16LL);
    sub_7B8B50();
    sub_7C96B0(0);
    v6 = sub_724840(unk_4F073B8, v4);
    sub_5C7230(0, v6, (__int64)v2, (_QWORD *)(a1 + 56));
    v7 = sub_7279A0(v3 + v5 + 19);
    si128 = _mm_load_si128((const __m128i *)&xmmword_39FBE40);
    *(_QWORD *)(a1 + 80) = v7;
    *(__m128i *)v7 = si128;
    *(_BYTE *)(v7 + 16) = 32;
    memcpy((void *)(*(_QWORD *)(a1 + 80) + 17LL), v2, v3);
    *(_BYTE *)(*(_QWORD *)(a1 + 80) + v3 + 17) = 32;
    memcpy((void *)(*(_QWORD *)(a1 + 80) + v3 + 18), v4, v5 + 1);
    return sub_8543B0(a1, 0, 0);
  }
  else
  {
    sub_6851C0(40, &unk_4F07508);
    return sub_7C96B0(1);
  }
}
