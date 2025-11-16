// Function: sub_167FAB0
// Address: 0x167fab0
//
__int64 __fastcall sub_167FAB0(__int64 a1, int a2, int a3)
{
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_DWORD *)(a1 + 40) = a2;
  *(_DWORD *)(a1 + 44) = a3;
  *(_BYTE *)(a1 + 48) = 0;
  return sub_167FA60(a1);
}
