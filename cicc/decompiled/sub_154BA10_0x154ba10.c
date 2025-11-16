// Function: sub_154BA10
// Address: 0x154ba10
//
void __fastcall sub_154BA10(__int64 a1, __int64 a2, char a3)
{
  *(_QWORD *)a1 = 0;
  *(_BYTE *)(a1 + 9) = a3;
  *(_QWORD *)(a1 + 16) = a2;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_BYTE *)(a1 + 8) = a2 != 0;
}
