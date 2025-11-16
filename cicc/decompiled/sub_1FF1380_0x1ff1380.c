// Function: sub_1FF1380
// Address: 0x1ff1380
//
__int64 __fastcall sub_1FF1380(char *a1)
{
  __int64 v1; // rbx
  char v2; // r8
  unsigned int v3; // eax
  unsigned __int8 v4; // dl
  __int64 v6; // rax
  char v7; // di
  unsigned int v8; // eax
  unsigned __int8 v9; // cl
  char v10; // r8
  char v11; // di
  char v12; // r9
  int v13; // esi

  v2 = *a1;
  if ( !*a1 )
  {
    if ( !sub_1F58D20((__int64)a1) )
      return sub_1F58D80((__int64)a1);
    v6 = sub_1F5A910((__int64)a1);
    v1 = v6;
    goto LABEL_16;
  }
  if ( (unsigned __int8)(v2 - 14) <= 0x5Fu )
  {
    switch ( v2 )
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
        v7 = 3;
        goto LABEL_19;
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
        v7 = 4;
        goto LABEL_19;
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
        v7 = 5;
        goto LABEL_19;
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
        v7 = 6;
        goto LABEL_19;
      case 55:
        v8 = sub_1FEB8F0(7);
        if ( v8 != 32 )
          goto LABEL_20;
        v12 = 5;
        v13 = 1;
        goto LABEL_26;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        v7 = 8;
        goto LABEL_19;
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
        v7 = 9;
        goto LABEL_19;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        v7 = 10;
        goto LABEL_19;
      default:
        v7 = 2;
LABEL_19:
        v8 = sub_1FEB8F0(v7);
        if ( v8 == 32 )
        {
          v11 = 5;
        }
        else
        {
LABEL_20:
          if ( v8 > 0x20 )
          {
            v11 = 6;
            if ( v8 != 64 )
            {
              v11 = 0;
              if ( v8 == 128 )
                v11 = 7;
            }
          }
          else
          {
            v11 = 3;
            if ( v8 != 8 )
            {
              v11 = 4;
              if ( v8 != 16 )
                v11 = 2 * (v8 == 1);
            }
          }
        }
        v12 = v11;
        v13 = word_42FEB00[v9];
        if ( (unsigned __int8)(v10 - 56) <= 0x1Du || (unsigned __int8)(v10 - 98) <= 0xBu )
          LOBYTE(v6) = sub_1D154A0(v11, v13);
        else
LABEL_26:
          LOBYTE(v6) = sub_1D15020(v12, v13);
        break;
    }
LABEL_16:
    LOBYTE(v1) = v6;
    return v1;
  }
  v3 = sub_1FEB8F0(*a1);
  if ( v3 == 32 )
  {
    return 5;
  }
  else if ( v3 > 0x20 )
  {
    v4 = 6;
    if ( v3 != 64 )
    {
      v4 = 0;
      if ( v3 == 128 )
        return 7;
    }
  }
  else
  {
    v4 = 3;
    if ( v3 != 8 )
    {
      v4 = 4;
      if ( v3 != 16 )
        return (unsigned __int8)(2 * (v3 == 1));
    }
  }
  return v4;
}
