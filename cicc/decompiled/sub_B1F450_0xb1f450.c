// Function: sub_B1F450
// Address: 0xb1f450
//
__int64 __fastcall sub_B1F450(__int64 a1, __int64 a2, __int64 a3)
{
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x100000000LL;
  *(_QWORD *)(a1 + 24) = a1 + 40;
  *(_QWORD *)(a1 + 116) = 0;
  *(_QWORD *)(a1 + 32) = 0x600000000LL;
  *(_QWORD *)(a1 + 96) = 0;
  *(_BYTE *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 104) = a3;
  *(_DWORD *)(a1 + 120) = *(_DWORD *)(a3 + 92);
  sub_B1F440(a1);
  return a1;
}
