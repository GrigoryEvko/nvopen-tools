// Function: sub_899CC0
// Address: 0x899cc0
//
__int64 __fastcall sub_899CC0(__int64 a1, char a2, char a3)
{
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rax
  __int64 v7; // r15
  _BYTE *v8; // r12
  char v9; // r12
  __int64 v10; // rax
  __int64 v11; // r9
  bool v12; // r12
  char v13; // cl
  char v14; // r12
  unsigned __int8 v15; // al
  unsigned int v16; // r8d
  char v17; // dl
  char v19; // cl
  char v20; // cl
  __int64 v21; // [rsp+8h] [rbp-38h]
  __int64 v22; // [rsp+8h] [rbp-38h]

  v4 = *(_QWORD *)(a1 + 32);
  switch ( *(_BYTE *)(v4 + 80) )
  {
    case 4:
    case 5:
      v5 = *(_QWORD *)(*(_QWORD *)(v4 + 96) + 80LL);
      break;
    case 6:
      v5 = *(_QWORD *)(*(_QWORD *)(v4 + 96) + 32LL);
      break;
    case 9:
    case 0xA:
      v5 = *(_QWORD *)(*(_QWORD *)(v4 + 96) + 56LL);
      break;
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      v5 = *(_QWORD *)(v4 + 88);
      break;
    default:
      v5 = 0;
      break;
  }
  v6 = *(_QWORD *)(a1 + 24);
  v7 = *(_QWORD *)(a1 + 16);
  v8 = *(_BYTE **)(v6 + 88);
  if ( ((*(_BYTE *)(v6 + 80) - 7) & 0xFD) == 0 )
  {
    v19 = v8[170];
    if ( (v19 & 0x40) != 0 )
      goto LABEL_10;
    if ( (v8[172] & 0x20) != 0 )
    {
      if ( (v8[176] & 1) != 0 || v19 < 0 )
        goto LABEL_10;
    }
    else
    {
      v20 = (unsigned __int8)v19 >> 7;
      if ( (*(_BYTE *)(v4 + 81) & 2) != 0 )
      {
        v14 = v20 ^ 1;
LABEL_7:
        if ( v14 )
          goto LABEL_8;
LABEL_10:
        v16 = 0;
        v17 = 0;
        goto LABEL_11;
      }
      if ( v20 )
        goto LABEL_10;
      if ( dword_4D04278 && (unsigned int)sub_890400(a1) )
        goto LABEL_8;
    }
    if ( (*(_BYTE *)(a1 + 80) & 0x22) != 0 )
      goto LABEL_10;
    if ( (*(_BYTE *)(v7 + 28) & 1) != 0 )
      goto LABEL_10;
    if ( !unk_4D0472C )
      goto LABEL_10;
    if ( (a2 & 1) == 0 )
      goto LABEL_10;
    sub_899B10(v4);
    if ( (v8[172] & 0x20) != 0 || (*(_BYTE *)(*(_QWORD *)(a1 + 32) + 81LL) & 2) == 0 )
      goto LABEL_10;
    goto LABEL_8;
  }
  v9 = v8[195];
  v21 = v5;
  v10 = sub_892400(v5);
  v11 = v21;
  v12 = (v9 & 2) != 0;
  if ( *(_QWORD *)(v10 + 8) )
  {
LABEL_5:
    v13 = 1;
LABEL_6:
    v14 = v13 & !v12;
    goto LABEL_7;
  }
  if ( *(_QWORD *)(v21 + 104) )
  {
    if ( (unsigned int)sub_825090() )
      goto LABEL_5;
    v11 = v21;
  }
  if ( v12 )
  {
    v13 = 0;
    goto LABEL_6;
  }
  if ( !dword_4D04278 || !(unsigned int)sub_890400(a1) )
  {
    if ( (*(_BYTE *)(a1 + 80) & 2) != 0 )
      goto LABEL_10;
    if ( (*(_BYTE *)(v7 + 28) & 1) != 0 )
      goto LABEL_10;
    if ( !unk_4D0472C )
      goto LABEL_10;
    if ( (a2 & 1) == 0 )
      goto LABEL_10;
    v22 = v11;
    sub_899B10(*(_QWORD *)(a1 + 32));
    if ( !*(_QWORD *)(sub_892400(v22) + 8) )
      goto LABEL_10;
  }
LABEL_8:
  if ( (*(_BYTE *)(v7 + 28) & 1) != 0 )
    goto LABEL_10;
  v15 = *(_BYTE *)(a1 + 80);
  if ( (v15 & 2) != 0 )
    goto LABEL_10;
  v16 = 1;
  v17 = (a3 | (v15 >> 5) ^ 1) & 1;
  if ( !v17 )
    goto LABEL_10;
LABEL_11:
  *(_BYTE *)(a1 + 80) = (v17 << 7) | *(_BYTE *)(a1 + 80) & 0x7F;
  return v16;
}
