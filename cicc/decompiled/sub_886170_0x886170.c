// Function: sub_886170
// Address: 0x886170
//
_QWORD *__fastcall sub_886170(__int64 a1, const char *a2, int a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  char v6; // al
  _QWORD *v7; // rax
  _QWORD *v8; // r12

  if ( (*(_BYTE *)(a1 + 17) & 0x40) == 0 )
  {
    *(_BYTE *)(a1 + 16) &= ~0x80u;
    *(_QWORD *)(a1 + 24) = 0;
  }
  v4 = sub_7D2AC0((_QWORD *)a1, a2, 0x1000u);
  v5 = v4;
  if ( v4 )
  {
    v6 = *(_BYTE *)(v4 + 80);
    if ( (unsigned __int8)(v6 - 4) > 2u && (v6 != 3 || !*(_BYTE *)(v5 + 104)) )
      sub_6854C0(0xAD6u, (FILE *)(a1 + 8), v5);
  }
  v7 = sub_87EF90(2u, a1);
  *((_BYTE *)v7 + 81) |= 0x10u;
  v8 = v7;
  *((_DWORD *)v7 + 10) = a3;
  v7[8] = a2;
  sub_886160((__int64)v7);
  return v8;
}
