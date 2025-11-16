// Function: sub_12F0CE0
// Address: 0x12f0ce0
//
__int64 __fastcall sub_12F0CE0(__int64 a1, char a2, char a3)
{
  int v5; // ecx
  __int16 v6; // dx

  *(_QWORD *)a1 = &unk_49EED30;
  v5 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  v6 = *(_WORD *)(a1 + 12) & 0xF000;
  *(_DWORD *)(a1 + 8) = v5;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 72) = qword_4FA01C0;
  *(_WORD *)(a1 + 12) = v6 | a2 & 7 | (32 * (a3 & 3));
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = a1 + 120;
  *(_QWORD *)(a1 + 96) = a1 + 120;
  *(_QWORD *)(a1 + 104) = 4;
  *(_DWORD *)(a1 + 112) = 0;
  *(_BYTE *)(a1 + 152) = 0;
  return a1 + 120;
}
