// Function: sub_27380F0
// Address: 0x27380f0
//
__int64 __fastcall sub_27380F0(__int64 a1, __int64 a2, char a3)
{
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = a1 + 24;
  *(_QWORD *)(a1 + 32) = a2;
  *(_BYTE *)(a1 + 40) = a3;
  *(_QWORD *)(a1 + 16) = 0x300000001LL;
  *(_QWORD *)(a1 + 24) = 1;
  return 0x300000001LL;
}
