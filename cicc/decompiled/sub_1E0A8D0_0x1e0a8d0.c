// Function: sub_1E0A8D0
// Address: 0x1e0a8d0
//
__int64 __fastcall sub_1E0A8D0(__int64 a1, int a2)
{
  *(_DWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = sub_1E094F0;
  *(_QWORD *)(a1 + 24) = sub_1E099C0;
  return a1;
}
