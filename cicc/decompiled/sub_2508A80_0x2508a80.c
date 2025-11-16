// Function: sub_2508A80
// Address: 0x2508a80
//
__int64 __fastcall sub_2508A80(__int64 a1)
{
  *(_BYTE *)(a1 + 22) = 110;
  *(_QWORD *)a1 = a1 + 16;
  *(_DWORD *)(a1 + 16) = 1816215873;
  *(_WORD *)(a1 + 20) = 26473;
  *(_QWORD *)(a1 + 8) = 7;
  *(_BYTE *)(a1 + 23) = 0;
  return a1;
}
