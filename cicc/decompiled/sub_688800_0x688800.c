// Function: sub_688800
// Address: 0x688800
//
_BOOL8 __fastcall sub_688800(__int16 a1, int a2, __int16 a3)
{
  int v3; // eax
  int v4; // r13d
  char v5; // r12
  char v6; // r12

  switch ( a1 )
  {
    case 25:
    case 27:
    case 29:
    case 30:
    case 31:
    case 32:
    case 72:
      v6 = 1;
      v4 = 19;
      return (unsigned __int8)v6 & (v4 == a2);
    case 26:
    case 28:
    case 37:
    case 38:
    case 55:
    case 68:
    case 69:
    case 73:
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
    case 84:
    case 85:
    case 86:
    case 87:
    case 88:
    case 89:
    case 90:
    case 91:
    case 92:
    case 93:
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
    case 104:
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
    case 116:
    case 117:
    case 118:
    case 119:
    case 120:
    case 121:
    case 122:
    case 123:
    case 124:
    case 125:
    case 126:
    case 127:
    case 128:
    case 129:
    case 130:
    case 131:
    case 132:
    case 133:
    case 134:
    case 135:
    case 136:
    case 137:
    case 138:
    case 139:
    case 140:
    case 141:
    case 142:
    case 143:
    case 144:
    case 145:
    case 146:
      return 1;
    case 33:
      v4 = 8;
      v5 = 1;
      if ( (a3 & 0x400) == 0 )
        goto LABEL_5;
      goto LABEL_4;
    case 34:
    case 39:
    case 40:
      v4 = 15;
      v5 = 1;
      if ( (a3 & 0x400) == 0 )
        goto LABEL_5;
      goto LABEL_4;
    case 35:
    case 36:
      v4 = 14;
      goto LABEL_14;
    case 41:
      goto LABEL_24;
    case 42:
      if ( unk_4F07770 && (*(_BYTE *)(unk_4D03C50 + 18LL) & 1) != 0 && !*(_QWORD *)(unk_4D03C50 + 40LL) )
        return 1;
LABEL_24:
      v4 = 13;
      v5 = 1;
      if ( (a3 & 0x400) == 0 )
        goto LABEL_5;
      goto LABEL_4;
    case 43:
    case 45:
    case 46:
      goto LABEL_9;
    case 44:
      if ( (*(_BYTE *)(unk_4D03C50 + 18LL) & 1) != 0 )
      {
        v4 = 11;
        if ( !*(_QWORD *)(unk_4D03C50 + 40LL) )
          return 1;
LABEL_14:
        v5 = 1;
        if ( (a3 & 0x400) == 0 )
          goto LABEL_5;
      }
      else
      {
LABEL_9:
        v4 = 11;
        v5 = 1;
        if ( (a3 & 0x400) == 0 )
          goto LABEL_5;
      }
      goto LABEL_4;
    case 47:
    case 48:
      v4 = 10;
      v5 = 1;
      if ( (a3 & 0x400) == 0 )
        goto LABEL_5;
      goto LABEL_4;
    case 49:
      v4 = 12;
      v5 = 1;
      if ( (a3 & 0x400) == 0 )
        goto LABEL_5;
      goto LABEL_4;
    case 50:
      v4 = 7;
      v5 = 1;
      if ( (a3 & 0x400) == 0 )
        goto LABEL_5;
      goto LABEL_4;
    case 51:
      v4 = 6;
      v5 = 1;
      if ( (a3 & 0x400) == 0 )
        goto LABEL_5;
      goto LABEL_4;
    case 52:
      v4 = 5;
      v5 = 1;
      if ( (a3 & 0x400) == 0 )
        goto LABEL_5;
      goto LABEL_4;
    case 53:
      v4 = 4;
      v5 = 1;
      if ( (a3 & 0x400) == 0 )
        goto LABEL_5;
      goto LABEL_4;
    case 54:
      v3 = 3;
      if ( (a3 & 0x400) == 0 )
        return a2 > v3;
      v4 = 3;
      v5 = 0;
      goto LABEL_4;
    case 56:
    case 57:
    case 58:
    case 59:
    case 60:
    case 61:
    case 62:
    case 63:
    case 64:
    case 65:
    case 66:
      v3 = 2;
      if ( (a3 & 0x400) == 0 )
        return a2 > v3;
      v4 = 2;
      v5 = 0;
LABEL_4:
      if ( (unsigned __int16)sub_7BE840(0, 0) == 76 )
        return 1;
LABEL_5:
      v6 = v5 & 1;
      if ( a2 > v4 )
        return 1;
      return (unsigned __int8)v6 & (v4 == a2);
    case 67:
      if ( (a3 & 1) != 0 )
        return 1;
      v4 = 1;
      v5 = 1;
      if ( (a3 & 0x400) == 0 )
        goto LABEL_5;
      goto LABEL_4;
    case 70:
    case 71:
      v4 = 9;
      v5 = 1;
      if ( (a3 & 0x400) == 0 )
        goto LABEL_5;
      goto LABEL_4;
    case 147:
    case 148:
      v4 = 16;
      goto LABEL_14;
    default:
      return 1;
  }
}
