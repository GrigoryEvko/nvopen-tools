// Function: sub_140B840
// Address: 0x140b840
//
__int64 __fastcall sub_140B840(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = a3;
  *(_BYTE *)(a1 + 18) = BYTE2(a5);
  *(_WORD *)(a1 + 16) = a5;
  *(_DWORD *)(a1 + 32) = 1;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = a1 + 80;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 64) = 8;
  *(_DWORD *)(a1 + 72) = 0;
  return a1 + 80;
}
