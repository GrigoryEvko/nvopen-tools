// Function: sub_AA87A0
// Address: 0xaa87a0
//
__int64 __fastcall sub_AA87A0(__int64 a1, __int64 a2)
{
  int v2; // eax

  *(_QWORD *)a1 = *(_QWORD *)a2;
  v2 = *(_DWORD *)(a2 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 8) = v2;
  *(_QWORD *)(a1 + 24) = a1 + 48;
  *(_QWORD *)(a1 + 32) = 2;
  *(_DWORD *)(a1 + 40) = 0;
  *(_BYTE *)(a1 + 44) = 1;
  return a1 + 48;
}
