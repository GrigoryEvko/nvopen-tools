// Function: sub_AA87D0
// Address: 0xaa87d0
//
__int64 __fastcall sub_AA87D0(__int64 a1)
{
  *(_QWORD *)a1 = 0;
  *(_DWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = a1 + 48;
  *(_QWORD *)(a1 + 32) = 2;
  *(_DWORD *)(a1 + 40) = 0;
  *(_BYTE *)(a1 + 44) = 1;
  return a1 + 48;
}
