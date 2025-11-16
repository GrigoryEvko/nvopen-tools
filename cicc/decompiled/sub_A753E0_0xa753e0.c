// Function: sub_A753E0
// Address: 0xa753e0
//
__int64 __fastcall sub_A753E0(__int64 a1)
{
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = a1 + 24;
  *(_QWORD *)(a1 + 48) = a1 + 24;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)a1 = 0x10000000000LL;
  *(_QWORD *)(a1 + 8) = 201326592;
  return a1;
}
