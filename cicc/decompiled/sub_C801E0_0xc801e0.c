// Function: sub_C801E0
// Address: 0xc801e0
//
__int64 __fastcall sub_C801E0(__int64 a1, __int64 a2, __int64 a3)
{
  *(_QWORD *)a1 = a2;
  *(_OWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 8) = a3;
  *(_QWORD *)(a1 + 32) = a3;
  *(_OWORD *)(a1 + 16) = 0;
  return a1;
}
