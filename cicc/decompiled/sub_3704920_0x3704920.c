// Function: sub_3704920
// Address: 0x3704920
//
__int64 __fastcall sub_3704920(__int64 a1, __int64 a2)
{
  __int64 v2; // rsi

  v2 = a2 + 16;
  *(_DWORD *)(v2 + 88) = *(_QWORD *)(*(_QWORD *)(v2 - 8) + 56LL);
  sub_370EDF0(a1, v2);
  return a1;
}
