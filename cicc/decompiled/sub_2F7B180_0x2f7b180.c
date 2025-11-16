// Function: sub_2F7B180
// Address: 0x2f7b180
//
__int64 __fastcall sub_2F7B180(__int64 a1, __int64 a2, __int64 a3)
{
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  sub_2F7B040(a1, a3);
  return a1;
}
