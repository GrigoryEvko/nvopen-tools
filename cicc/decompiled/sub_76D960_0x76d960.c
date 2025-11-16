// Function: sub_76D960
// Address: 0x76d960
//
_BYTE *__fastcall sub_76D960(__int64 a1, int a2, int a3)
{
  __int64 v4; // r14
  char v5; // al
  _BYTE *result; // rax
  char v7; // bl
  char v8; // dl
  char v9; // r12
  char v10; // si
  char v11; // [rsp+Fh] [rbp-41h]
  _QWORD v12[7]; // [rsp+18h] [rbp-38h] BYREF

  v4 = *(_QWORD *)(a1 + 8);
  v12[0] = 0;
  v11 = *(_BYTE *)(v4 + 169) & 8;
  if ( (*(_QWORD *)(v4 + 168) & 0x80000004000LL) != 0 )
    goto LABEL_5;
  v5 = *(_BYTE *)(v4 + 177);
  if ( v5 == 2 )
  {
    if ( *(char *)(*(_QWORD *)(v4 + 184) + 50LL) < 0 || !v11 )
      goto LABEL_5;
  }
  else if ( v5 == 3 || !v11 )
  {
    goto LABEL_5;
  }
  *(_BYTE *)(a1 + 58) = 1;
  while ( 1 )
  {
    result = (_BYTE *)sub_7E2100(*(_QWORD *)(v4 + 120), v12);
    if ( !result )
      break;
    if ( ((result[173] & 0x20) != 0) == a2 && ((result[173] & 0x10) != 0) == a3 )
    {
      *(_BYTE *)(v12[0] + 16LL) = 1;
      *(_BYTE *)(a1 + 59) = 1;
      goto LABEL_6;
    }
  }
LABEL_5:
  result = (_BYTE *)sub_7E20D0(*(_QWORD *)(v4 + 120), 0);
LABEL_6:
  *(_DWORD *)(a1 + 16) = 1;
  *(_QWORD *)(a1 + 24) = result;
  if ( (*(_BYTE *)(v4 + 169) & 0x40) != 0 )
    result[169] |= 0x40u;
  if ( (*(_BYTE *)(v4 + 173) & 8) != 0 )
    result[173] |= 8u;
  if ( *(_BYTE *)(v4 + 177) == 3 )
    result[177] = 3;
  v7 = (16 * a3) | (32 * a2);
  v8 = *(_BYTE *)(v4 + 156) & 1 | result[156] & 0xFE;
  v9 = result[173] & 0xCF;
  result[156] = v8;
  v10 = *(_BYTE *)(v4 + 156);
  result[173] = v9 | v7 & 0x30;
  result[156] = v10 & 2 | v8 & 0xFD;
  if ( !v11 )
  {
    result[169] &= ~8u;
    result[144] = *(_BYTE *)(v4 + 144);
  }
  return result;
}
