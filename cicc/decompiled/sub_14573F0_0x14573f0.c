// Function: sub_14573F0
// Address: 0x14573f0
//
__int64 __fastcall sub_14573F0(__int64 a1, __int64 a2)
{
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = a2;
  *(_BYTE *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = a1 + 64;
  *(_QWORD *)(a1 + 40) = a1 + 64;
  *(_QWORD *)(a1 + 48) = 4;
  *(_DWORD *)(a1 + 56) = 0;
  return a1 + 64;
}
