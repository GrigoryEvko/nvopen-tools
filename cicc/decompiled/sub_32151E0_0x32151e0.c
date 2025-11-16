// Function: sub_32151E0
// Address: 0x32151e0
//
__int64 __fastcall sub_32151E0(__int64 a1, __int16 a2)
{
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_DWORD *)(a1 + 32) = -1;
  *(_QWORD *)a1 = &unk_4A355F0;
  *(_WORD *)(a1 + 36) = a2;
  *(_QWORD *)(a1 + 8) = (a1 + 8) | 4;
  *(_BYTE *)(a1 + 38) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 48) = a1 | 4;
  return a1 | 4;
}
