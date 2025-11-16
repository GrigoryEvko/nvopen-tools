// Function: sub_11F3840
// Address: 0x11f3840
//
void __fastcall sub_11F3840(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  *(_QWORD *)a1 = a2;
  *(_BYTE *)(a1 + 41) = a4 != 0;
  *(_QWORD *)(a1 + 8) = a3;
  *(_QWORD *)(a1 + 16) = a4;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = a5;
  *(_BYTE *)(a1 + 40) = 0;
  *(_BYTE *)(a1 + 42) = a3 != 0;
}
