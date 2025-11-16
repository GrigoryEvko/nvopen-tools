// Function: sub_1648CB0
// Address: 0x1648cb0
//
__int64 __fastcall sub_1648CB0(__int64 a1, __int64 a2, char a3)
{
  *(_DWORD *)(a1 + 20) &= 0xC0000000;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = a3;
  *(_BYTE *)(a1 + 17) = 0;
  *(_WORD *)(a1 + 18) = 0;
  return 0;
}
