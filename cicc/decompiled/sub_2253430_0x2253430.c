// Function: sub_2253430
// Address: 0x2253430
//
__int64 __fastcall sub_2253430(__int64 a1, __int64 a2, __int64 a3)
{
  *(_DWORD *)(a1 - 128) = 0;
  *(_QWORD *)(a1 - 112) = a2;
  *(_QWORD *)(a1 - 104) = a3;
  *(_QWORD *)(a1 - 96) = sub_2207560();
  *(_QWORD *)(a1 - 88) = sub_2207520();
  *(_QWORD *)(a1 - 32) = 0x474E5543432B2B00LL;
  *(_QWORD *)(a1 - 24) = sub_22533F0;
  return a1 - 128;
}
