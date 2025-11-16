// Function: sub_73D8E0
// Address: 0x73d8e0
//
__int64 __fastcall sub_73D8E0(__int64 a1, unsigned __int8 a2, __int64 a3, int a4, __int64 a5)
{
  char v6; // al
  unsigned int v7; // eax
  bool v8; // dl
  __int64 result; // rax
  char i; // dl
  __int64 v11; // r9
  char j; // cl
  char v13; // di
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // [rsp+0h] [rbp-40h]
  __int64 v17; // [rsp+0h] [rbp-40h]
  int v18; // [rsp+Ch] [rbp-34h]
  int v19; // [rsp+Ch] [rbp-34h]
  int v20; // [rsp+18h] [rbp-28h] BYREF
  _DWORD v21[9]; // [rsp+1Ch] [rbp-24h] BYREF

  v6 = *(_BYTE *)(a1 + 25);
  *(_QWORD *)a1 = a3;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 56) = a2;
  *(_QWORD *)(a1 + 72) = a5;
  *(_BYTE *)(a1 + 25) = a4 & 1 | v6 & 0xFC;
  v7 = dword_4D03F94;
  if ( !dword_4D03F94 )
    goto LABEL_12;
  if ( unk_4D04810 )
  {
    if ( a2 != 105 )
    {
      v16 = a5;
      v18 = a4;
      sub_737810(a2, &v20, v21);
      a5 = v16;
      *(_BYTE *)(a1 + 60) = *(_BYTE *)(a1 + 60) & 0xFC | v20 & 1 | (2 * (v21[0] & 1));
      v7 = dword_4D03F94;
      if ( !v18 )
        goto LABEL_12;
      goto LABEL_8;
    }
    if ( a5 && *(_BYTE *)(a5 + 24) == 20 && (v15 = *(_QWORD *)(a5 + 56), *(_BYTE *)(v15 + 174) == 5) )
    {
      v17 = a5;
      v19 = a4;
      sub_7377C0(*(_BYTE *)(v15 + 176), &v20, v21);
      a4 = v19;
      a5 = v17;
      *(_BYTE *)(a1 + 60) = *(_BYTE *)(a1 + 60) & 0xFC | v20 & 1 | (2 * (v21[0] & 1));
      v7 = dword_4D03F94;
    }
    else
    {
      *(_BYTE *)(a1 + 60) |= 1u;
    }
  }
  if ( !a4 )
  {
LABEL_12:
    v8 = 0;
    goto LABEL_13;
  }
LABEL_8:
  if ( !v7 )
    goto LABEL_12;
  v8 = a2 == 91 || a2 == 73;
  if ( !v8 )
  {
    if ( (unsigned __int8)(a2 - 103) <= 1u )
      *(_BYTE *)(a1 + 58) |= 1u;
    else
      *(_BYTE *)(a1 + 58) &= ~1u;
    goto LABEL_15;
  }
LABEL_13:
  *(_BYTE *)(a1 + 58) = v8 | *(_BYTE *)(a1 + 58) & 0xFE;
  if ( a2 == 91 )
    sub_7304E0(a5);
LABEL_15:
  if ( (*(_BYTE *)(a1 + 25) & 4) != 0 )
    sub_7304E0(a1);
  result = *(_QWORD *)a1;
  for ( i = *(_BYTE *)(*(_QWORD *)a1 + 140LL); i == 12; i = *(_BYTE *)(result + 140) )
    result = *(_QWORD *)(result + 160);
  v11 = *(_QWORD *)(a1 + 72);
  j = -1;
  if ( v11 )
  {
    result = *(_QWORD *)v11;
    for ( j = *(_BYTE *)(*(_QWORD *)v11 + 140LL); j == 12; j = *(_BYTE *)(result + 140) )
      result = *(_QWORD *)(result + 160);
  }
  v13 = *(_BYTE *)(a1 + 56);
  switch ( v13 )
  {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
    case 10:
    case 11:
    case 12:
    case 13:
    case 18:
    case 50:
    case 51:
    case 52:
    case 84:
    case 85:
    case 92:
      i = 6;
      goto LABEL_26;
    case 5:
    case 6:
    case 7:
    case 8:
      if ( i != j )
        goto LABEL_28;
      goto LABEL_24;
    case 9:
      i = 10;
      goto LABEL_26;
    case 14:
    case 15:
    case 19:
    case 25:
    case 26:
    case 27:
    case 28:
    case 32:
    case 33:
    case 34:
    case 35:
    case 36:
    case 37:
    case 38:
    case 39:
    case 40:
    case 41:
    case 42:
    case 43:
    case 55:
    case 56:
    case 57:
    case 71:
    case 72:
    case 116:
    case 117:
    case 118:
      goto LABEL_24;
    case 16:
    case 17:
    case 20:
    case 29:
    case 53:
    case 54:
    case 73:
      i = j;
      goto LABEL_24;
    case 21:
      i = 8;
      goto LABEL_26;
    case 22:
    case 23:
    case 86:
    case 91:
    case 94:
    case 95:
    case 96:
    case 97:
    case 98:
    case 99:
    case 100:
    case 101:
    case 102:
    case 103:
    case 105:
    case 106:
    case 107:
    case 108:
    case 109:
    case 110:
    case 111:
    case 112:
    case 113:
    case 114:
    case 115:
LABEL_28:
      *(_BYTE *)(a1 + 57) = 21;
      return result;
    case 24:
    case 87:
    case 88:
      i = 2;
      goto LABEL_26;
    case 30:
    case 31:
    case 65:
    case 66:
    case 67:
    case 68:
    case 69:
    case 70:
    case 89:
    case 90:
    case 93:
    case 104:
      *(_BYTE *)(a1 + 57) = 15;
      return result;
    case 44:
    case 45:
      i = 4;
      goto LABEL_26;
    case 46:
    case 47:
    case 48:
    case 49:
      i = 5;
      goto LABEL_26;
    case 58:
    case 59:
    case 60:
    case 61:
    case 62:
    case 63:
    case 64:
      if ( i == 14 )
        goto LABEL_26;
      i = sub_730040(v13, *(_QWORD *)v11, **(_QWORD **)(v11 + 16));
LABEL_24:
      result = 10;
      if ( (i & 0xFD) == 9 )
        i = 10;
LABEL_26:
      *(_BYTE *)(a1 + 57) = i;
      return result;
    case 74:
    case 75:
    case 76:
    case 77:
    case 78:
    case 79:
    case 80:
    case 81:
    case 82:
    case 83:
      v14 = sub_73D850(a1);
      for ( i = *(_BYTE *)(v14 + 140); i == 12; i = *(_BYTE *)(v14 + 140) )
        v14 = *(_QWORD *)(v14 + 160);
      goto LABEL_24;
    case 119:
      i = 0;
      goto LABEL_26;
    default:
      sub_721090();
  }
}
