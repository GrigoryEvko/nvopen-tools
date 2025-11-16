// Function: sub_2FCE3D0
// Address: 0x2fce3d0
//
__int64 __fastcall sub_2FCE3D0(__int64 a1, __int64 a2, __int64 a3)
{
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_DWORD *)(a1 + 32) = 8;
  *(_WORD *)(a1 + 36) = 0;
  *(_BYTE *)(a1 + 38) = 0;
  *(_BYTE *)(a1 + 36) = sub_2FCD340(a3, a1);
  *(_DWORD *)(a1 + 32) = sub_B2D810(a3, "stack-protector-buffer-size", 0x1Bu, 8);
  return a1;
}
