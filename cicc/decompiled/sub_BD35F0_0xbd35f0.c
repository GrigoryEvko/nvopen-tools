// Function: sub_BD35F0
// Address: 0xbd35f0
//
__int64 __fastcall sub_BD35F0(__int64 a1, __int64 a2, char a3)
{
  *(_DWORD *)(a1 + 4) &= 0xC0000000;
  *(_BYTE *)a1 = a3;
  *(_BYTE *)(a1 + 1) = 0;
  *(_WORD *)(a1 + 2) = 0;
  *(_QWORD *)(a1 + 8) = a2;
  *(_QWORD *)(a1 + 16) = 0;
  return 0;
}
