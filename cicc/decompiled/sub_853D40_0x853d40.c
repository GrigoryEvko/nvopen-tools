// Function: sub_853D40
// Address: 0x853d40
//
__int64 __fastcall sub_853D40(unsigned __int8 a1, char a2, char a3, __int16 a4, char a5, __int16 a6, __int16 a7)
{
  __int16 v7; // bx
  char v8; // r14
  __int64 result; // rax
  __int16 v10; // dx
  __int16 v11; // bx

  v7 = a6 | 0x300 | (4 * a4);
  v8 = (8 * a5) | 0x80 | (16 * a3);
  result = sub_823970(24);
  v10 = *(_WORD *)(result + 18);
  *(_BYTE *)(result + 8) = a1;
  *(_DWORD *)(result + 12) = 5;
  v11 = (8 * a7) | v7;
  *(_BYTE *)(result + 16) = a2;
  LOBYTE(v11) = v11 & 0x1F;
  *(_BYTE *)(result + 17) = v8;
  *(_WORD *)(result + 18) = v10 & 0xE0 | v11;
  *(_QWORD *)result = unk_4D03E90;
  unk_4D03E90 = result;
  qword_4D03D40[a1] = result;
  return result;
}
