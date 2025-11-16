// Function: sub_885A40
// Address: 0x885a40
//
_QWORD *__fastcall sub_885A40(__int64 *a1, char a2, __int64 a3, int a4, int a5)
{
  _QWORD *v8; // r12
  char v9; // r13

  v8 = sub_87ECE0(a1, (_QWORD *)(a3 + 8), a4);
  v9 = (4 * (a2 & 1)) | v8[12] & 0xFB;
  *((_BYTE *)v8 + 81) = *(_BYTE *)(a3 + 17) & 0x20 | *((_BYTE *)v8 + 81) & 0xDF;
  *((_BYTE *)v8 + 96) = v9;
  *(_BYTE *)(a3 + 16) &= ~1u;
  *(_QWORD *)(a3 + 24) = v8;
  sub_885A00((__int64)v8, a4, a5);
  return v8;
}
