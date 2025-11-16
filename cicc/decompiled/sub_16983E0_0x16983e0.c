// Function: sub_16983E0
// Address: 0x16983e0
//
__int64 __fastcall sub_16983E0(__int64 a1, __int64 a2)
{
  char v2; // al

  sub_16983A0(a1);
  *(_QWORD *)a1 = *(_QWORD *)a2;
  *(_QWORD *)(a1 + 8) = *(_QWORD *)(a2 + 8);
  *(_WORD *)(a1 + 16) = *(_WORD *)(a2 + 16);
  v2 = *(_BYTE *)(a2 + 18) & 7 | *(_BYTE *)(a1 + 18) & 0xF8;
  *(_BYTE *)(a1 + 18) = v2;
  *(_BYTE *)(a1 + 18) = *(_BYTE *)(a2 + 18) & 8 | v2 & 0xF7;
  *(_QWORD *)a2 = &unk_42AE9A0;
  return a1;
}
