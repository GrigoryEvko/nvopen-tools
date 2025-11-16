// Function: sub_255A950
// Address: 0x255a950
//
__int64 __fastcall sub_255A950(__int64 a1, __int64 a2)
{
  __m128i *v2; // r13
  char v4; // al
  __int64 v5[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = (__m128i *)(a1 + 72);
  if ( *(_BYTE *)(*(_QWORD *)(sub_250D070((_QWORD *)(a1 + 72)) + 8) + 8LL) != 14 )
    return 1;
  v5[0] = 0x5400000053LL;
  if ( (unsigned __int8)sub_2516400(a2, v2, (__int64)v5, 2, 0, 0) )
  {
    v4 = *(_BYTE *)(a1 + 96) & 0xFD | *(_BYTE *)(a1 + 97) & 0xFD;
    *(_BYTE *)(a1 + 96) &= ~2u;
    *(_BYTE *)(a1 + 97) = v4;
  }
  sub_2515E10(a2, v2->m128i_i64, (__int64)dword_438A680, 3);
  return sub_255A4D0(a1, a2);
}
