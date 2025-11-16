// Function: sub_20A1760
// Address: 0x20a1760
//
__int64 __fastcall sub_20A1760(__int64 a1, unsigned int a2)
{
  __int64 v2; // rsi
  char v3; // al
  __int64 v4; // rbx
  __int64 v6; // rdx
  char v7[8]; // [rsp+0h] [rbp-30h] BYREF
  __int64 v8; // [rsp+8h] [rbp-28h]
  char v9[8]; // [rsp+10h] [rbp-20h] BYREF
  __int64 v10; // [rsp+18h] [rbp-18h]

  v2 = *(_QWORD *)(a1 + 40) + 16LL * a2;
  v3 = *(_BYTE *)v2;
  v4 = *(_QWORD *)(v2 + 8);
  v7[0] = v3;
  v8 = v4;
  if ( v3 )
  {
    if ( (unsigned __int8)(v3 - 14) <= 0x5Fu )
    {
      switch ( v3 )
      {
        case 24:
        case 25:
        case 26:
        case 27:
        case 28:
        case 29:
        case 30:
        case 31:
        case 32:
        case 62:
        case 63:
        case 64:
        case 65:
        case 66:
        case 67:
          v9[0] = 3;
          v10 = 0;
          break;
        case 33:
        case 34:
        case 35:
        case 36:
        case 37:
        case 38:
        case 39:
        case 40:
        case 68:
        case 69:
        case 70:
        case 71:
        case 72:
        case 73:
          v9[0] = 4;
          v10 = 0;
          break;
        case 41:
        case 42:
        case 43:
        case 44:
        case 45:
        case 46:
        case 47:
        case 48:
        case 74:
        case 75:
        case 76:
        case 77:
        case 78:
        case 79:
          v9[0] = 5;
          v10 = 0;
          break;
        case 49:
        case 50:
        case 51:
        case 52:
        case 53:
        case 54:
        case 80:
        case 81:
        case 82:
        case 83:
        case 84:
        case 85:
          v9[0] = 6;
          v10 = 0;
          break;
        case 55:
          v9[0] = 7;
          v10 = 0;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v9[0] = 8;
          v10 = 0;
          break;
        case 89:
        case 90:
        case 91:
        case 92:
        case 93:
        case 101:
        case 102:
        case 103:
        case 104:
        case 105:
          v9[0] = 9;
          v10 = 0;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v9[0] = 10;
          v10 = 0;
          break;
        default:
          v9[0] = 2;
          v10 = 0;
          break;
      }
    }
    else
    {
      v9[0] = v3;
      v10 = v4;
    }
    return sub_1F3E310(v9);
  }
  if ( sub_1F58D20((__int64)v7) )
  {
    v9[0] = sub_1F596B0((__int64)v7);
    v10 = v6;
    if ( v9[0] )
      return sub_1F3E310(v9);
  }
  else
  {
    v9[0] = 0;
    v10 = v4;
  }
  return sub_1F58D40((__int64)v9);
}
