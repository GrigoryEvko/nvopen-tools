// Function: sub_1D0E8B0
// Address: 0x1d0e8b0
//
__int64 __fastcall sub_1D0E8B0(__int64 *a1, __int64 a2)
{
  __int64 v2; // r8
  char v3; // al
  char v4; // al
  char v5; // al
  char v6; // al
  char v7; // al
  char v8; // al
  char v9; // al

  v2 = sub_1D0E6F0(a1, *(_QWORD *)a2);
  *(_QWORD *)(v2 + 16) = *(_QWORD *)(a2 + 16);
  *(_WORD *)(v2 + 226) = *(_WORD *)(a2 + 226);
  v3 = *(_BYTE *)(a2 + 228) & 1 | *(_BYTE *)(v2 + 228) & 0xFE;
  *(_BYTE *)(v2 + 228) = v3;
  v4 = *(_BYTE *)(a2 + 228) & 2 | v3 & 0xFD;
  *(_BYTE *)(v2 + 228) = v4;
  v5 = *(_BYTE *)(a2 + 228) & 4 | v4 & 0xFB;
  *(_BYTE *)(v2 + 228) = v5;
  v6 = *(_BYTE *)(a2 + 228) & 8 | v5 & 0xF7;
  *(_BYTE *)(v2 + 228) = v6;
  v7 = *(_BYTE *)(a2 + 228) & 0x10 | v6 & 0xEF;
  *(_BYTE *)(v2 + 228) = v7;
  v8 = *(_BYTE *)(a2 + 228) & 0x40 | v7 & 0xBF;
  *(_BYTE *)(v2 + 228) = v8;
  *(_BYTE *)(v2 + 228) = *(_BYTE *)(a2 + 228) & 0x80 | v8 & 0x7F;
  v9 = *(_BYTE *)(a2 + 229) & 8 | *(_BYTE *)(v2 + 229) & 0xF7;
  *(_BYTE *)(v2 + 229) = v9;
  *(_BYTE *)(v2 + 229) = *(_BYTE *)(a2 + 229) & 0x10 | v9 & 0xEF;
  *(_DWORD *)(v2 + 232) = *(_DWORD *)(a2 + 232);
  *(_BYTE *)(a2 + 229) |= 0x20u;
  return v2;
}
