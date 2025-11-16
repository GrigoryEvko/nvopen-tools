// Function: sub_C33870
// Address: 0xc33870
//
__int64 __fastcall sub_C33870(__int64 a1, __int64 a2)
{
  char v2; // al

  sub_C33830(a1);
  *(_QWORD *)a1 = *(_QWORD *)a2;
  *(_QWORD *)(a1 + 8) = *(_QWORD *)(a2 + 8);
  *(_DWORD *)(a1 + 16) = *(_DWORD *)(a2 + 16);
  v2 = *(_BYTE *)(a2 + 20) & 7 | *(_BYTE *)(a1 + 20) & 0xF8;
  *(_BYTE *)(a1 + 20) = v2;
  *(_BYTE *)(a1 + 20) = *(_BYTE *)(a2 + 20) & 8 | v2 & 0xF7;
  *(_QWORD *)a2 = &unk_3F655C0;
  return a1;
}
