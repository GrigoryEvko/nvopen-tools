// Function: sub_127F610
// Address: 0x127f610
//
__int64 __fastcall sub_127F610(__int64 a1, const __m128i *a2, __int64 a3)
{
  __int64 v3; // rax
  _QWORD v5[4]; // [rsp+0h] [rbp-20h] BYREF

  v3 = *(_QWORD *)(a1 + 360);
  v5[0] = a1;
  v5[1] = 0;
  v5[2] = 0;
  v5[3] = v3;
  return sub_127D8B0((__int64)v5, a2, a3);
}
