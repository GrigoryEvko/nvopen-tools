// Function: sub_B31710
// Address: 0xb31710
//
_BYTE *__fastcall sub_B31710(__int64 a1, _BYTE *a2)
{
  char v3; // al
  char v4; // al
  char v5; // al
  char v6; // al
  __int64 v7; // rdx
  const char *v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rdx
  _QWORD *v11; // rax
  __int64 v12; // rdx

  v3 = a2[32] & 0x30 | *(_BYTE *)(a1 + 32) & 0xCF;
  *(_BYTE *)(a1 + 32) = v3;
  if ( (v3 & 0xFu) - 7 <= 1 || (v3 & 0x30) != 0 && (v3 & 0xF) != 9 )
  {
    v4 = *(_BYTE *)(a1 + 33) | 0x40;
    *(_BYTE *)(a1 + 33) = v4;
  }
  else
  {
    v4 = *(_BYTE *)(a1 + 33);
  }
  *(_BYTE *)(a1 + 32) = a2[32] & 0xC0 | *(_BYTE *)(a1 + 32) & 0x3F;
  v5 = a2[33] & 0x1C | v4 & 0xE3;
  *(_BYTE *)(a1 + 33) = v5;
  v6 = a2[33] & 3 | v5 & 0xFC;
  *(_BYTE *)(a1 + 33) = v6;
  v7 = a2[33] & 0x40;
  *(_BYTE *)(a1 + 33) = v7 | v6 & 0xBF;
  v8 = sub_B30A70((__int64)a2, (__int64)a2, v7);
  sub_B30D10(a1, (__int64)v8, v9);
  if ( (a2[34] & 1) == 0 )
    return (_BYTE *)sub_B2FA10(a1, (__int64)v8, v10);
  v11 = sub_B31490((__int64)a2, (__int64)v8, v10);
  return sub_B311F0(a1, *(unsigned int *)v11, v12);
}
