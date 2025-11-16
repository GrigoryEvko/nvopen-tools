// Function: sub_1F4AA70
// Address: 0x1f4aa70
//
__int64 __fastcall sub_1F4AA70(__int64 a1, int a2, __int64 a3)
{
  *(_DWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = sub_1F49D90;
  *(_QWORD *)(a1 + 8) = a3;
  *(_QWORD *)(a1 + 24) = sub_1F4A2E0;
  return a1;
}
