// Function: sub_3422DA0
// Address: 0x3422da0
//
__int64 __fastcall sub_3422DA0(__int64 a1, __int64 a2)
{
  return sub_3415C70(
           *(const __m128i **)(a1 + 64),
           a2,
           46,
           **(unsigned __int16 **)(a2 + 48),
           *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL));
}
