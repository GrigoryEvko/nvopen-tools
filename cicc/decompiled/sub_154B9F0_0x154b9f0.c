// Function: sub_154B9F0
// Address: 0x154b9f0
//
__int64 __fastcall sub_154B9F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  *(_QWORD *)a1 = 0;
  *(_WORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = a3;
  *(_QWORD *)(a1 + 24) = a4;
  *(_QWORD *)(a1 + 32) = a2;
  return 0;
}
