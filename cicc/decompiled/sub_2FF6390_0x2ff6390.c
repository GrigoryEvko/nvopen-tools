// Function: sub_2FF6390
// Address: 0x2ff6390
//
__int64 __fastcall sub_2FF6390(__int64 a1, int a2, __int64 a3)
{
  *(_DWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = sub_2FF5460;
  *(_QWORD *)(a1 + 8) = a3;
  *(_QWORD *)(a1 + 24) = sub_2FF59B0;
  return a1;
}
