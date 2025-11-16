// Function: sub_C0BFB0
// Address: 0xc0bfb0
//
__int64 __fastcall sub_C0BFB0(__int64 a1, int a2, char a3)
{
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_DWORD *)(a1 + 40) = a2;
  *(_BYTE *)(a1 + 44) = a3;
  *(_BYTE *)(a1 + 45) = 0;
  return sub_C0BF50(a1);
}
