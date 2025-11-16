// Function: sub_3422C90
// Address: 0x3422c90
//
__int64 __fastcall sub_3422C90(__int64 a1, __int64 a2)
{
  return sub_3415C70(
           *(const __m128i **)(a1 + 64),
           a2,
           10,
           **(unsigned __int16 **)(a2 + 48),
           *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL));
}
