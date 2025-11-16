// Function: sub_2E79E20
// Address: 0x2e79e20
//
__int64 __fastcall sub_2E79E20(__int64 a1, int a2)
{
  *(_DWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = sub_2E78010;
  *(_QWORD *)(a1 + 24) = sub_2E785F0;
  return a1;
}
