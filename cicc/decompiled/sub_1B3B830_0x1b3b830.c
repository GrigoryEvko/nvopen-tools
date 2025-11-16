// Function: sub_1B3B830
// Address: 0x1b3b830
//
__int64 __fastcall sub_1B3B830(__int64 a1, __int64 a2)
{
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = a1 + 32;
  *(_QWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 48) = a2;
  return a1 + 32;
}
