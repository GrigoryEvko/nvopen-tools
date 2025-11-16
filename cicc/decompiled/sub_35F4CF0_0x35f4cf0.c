// Function: sub_35F4CF0
// Address: 0x35f4cf0
//
__int64 __fastcall sub_35F4CF0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v4; // rax
  bool v5; // zf
  __m128i *v6; // rdx
  __m128i si128; // xmm0
  __m128i v9; // xmm0

  v4 = *(_QWORD *)(a4 + 24);
  v5 = (*(_BYTE *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8) & 1) == 0;
  v6 = *(__m128i **)(a4 + 32);
  if ( v5 )
  {
    if ( (unsigned __int64)(v4 - (_QWORD)v6) <= 0x19 )
    {
      return sub_CB6200(a4, ".fence::before_thread_sync", 0x1Au);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_44FE880);
      qmemcpy(&v6[1], "hread_sync", 10);
      *v6 = si128;
      *(_QWORD *)(a4 + 32) += 26LL;
      return 25454;
    }
  }
  else if ( (unsigned __int64)(v4 - (_QWORD)v6) <= 0x18 )
  {
    return sub_CB6200(a4, ".fence::after_thread_sync", 0x19u);
  }
  else
  {
    v9 = _mm_load_si128((const __m128i *)&xmmword_44FE890);
    v6[1].m128i_i8[8] = 99;
    v6[1].m128i_i64[0] = 0x6E79735F64616572LL;
    *v6 = v9;
    *(_QWORD *)(a4 + 32) += 25LL;
    return 0x6E79735F64616572LL;
  }
}
