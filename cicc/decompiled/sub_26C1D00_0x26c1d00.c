// Function: sub_26C1D00
// Address: 0x26c1d00
//
__int64 __fastcall sub_26C1D00(__int64 a1, __int64 a2, __int64 a3, int a4, __int64 *a5, char a6, unsigned int a7)
{
  __int64 v12; // rdx

  *(_QWORD *)a1 = a1 + 16;
  sub_26BA410((__int64 *)a1, *(_BYTE **)a2, *(_QWORD *)a2 + *(_QWORD *)(a2 + 8));
  *(_QWORD *)(a1 + 32) = a1 + 48;
  sub_26BA410((__int64 *)(a1 + 32), *(_BYTE **)a3, *(_QWORD *)a3 + *(_QWORD *)(a3 + 8));
  *(_DWORD *)(a1 + 64) = a4;
  v12 = *a5;
  *a5 = 0;
  *(_BYTE *)(a1 + 80) = a6;
  *(_QWORD *)(a1 + 72) = v12;
  *(_BYTE *)(a1 + 81) = a7;
  return a7;
}
