// Function: sub_335EC50
// Address: 0x335ec50
//
unsigned __int64 __fastcall sub_335EC50(unsigned __int64 *a1, __int64 a2)
{
  unsigned __int64 v2; // r8
  char v3; // al
  char v4; // al
  char v5; // al
  char v6; // al
  char v7; // al
  char v8; // al
  char v9; // al

  v2 = sub_335EAA0(a1, *(_QWORD *)a2);
  *(_QWORD *)(v2 + 8) = *(_QWORD *)(a2 + 8);
  *(_WORD *)(v2 + 252) = *(_WORD *)(a2 + 252);
  v3 = *(_BYTE *)(a2 + 248) & 1 | *(_BYTE *)(v2 + 248) & 0xFE;
  *(_BYTE *)(v2 + 248) = v3;
  v4 = *(_BYTE *)(a2 + 248) & 2 | v3 & 0xFD;
  *(_BYTE *)(v2 + 248) = v4;
  v5 = *(_BYTE *)(a2 + 248) & 4 | v4 & 0xFB;
  *(_BYTE *)(v2 + 248) = v5;
  v6 = *(_BYTE *)(a2 + 248) & 8 | v5 & 0xF7;
  *(_BYTE *)(v2 + 248) = v6;
  v7 = *(_BYTE *)(a2 + 248) & 0x10 | v6 & 0xEF;
  *(_BYTE *)(v2 + 248) = v7;
  v8 = *(_BYTE *)(a2 + 248) & 0x40 | v7 & 0xBF;
  *(_BYTE *)(v2 + 248) = v8;
  *(_BYTE *)(v2 + 248) = *(_BYTE *)(a2 + 248) & 0x80 | v8 & 0x7F;
  v9 = *(_BYTE *)(a2 + 249) & 8 | *(_BYTE *)(v2 + 249) & 0xF7;
  *(_BYTE *)(v2 + 249) = v9;
  *(_BYTE *)(v2 + 249) = *(_BYTE *)(a2 + 249) & 0x10 | v9 & 0xEF;
  *(_BYTE *)(v2 + 254) = *(_BYTE *)(a2 + 254) & 0xF0 | *(_BYTE *)(v2 + 254) & 0xF;
  *(_BYTE *)(a2 + 249) |= 0x20u;
  return v2;
}
