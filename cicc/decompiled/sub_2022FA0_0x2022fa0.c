// Function: sub_2022FA0
// Address: 0x2022fa0
//
_QWORD *__fastcall sub_2022FA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char *v6; // rdx
  _QWORD *v7; // r12
  char v8; // al
  __int64 v9; // rdx
  unsigned __int8 v10; // al
  __int64 v11; // rdx
  __int64 v12; // r8
  _QWORD *v13; // r12
  _BYTE v15[8]; // [rsp+0h] [rbp-40h] BYREF
  __int64 v16; // [rsp+8h] [rbp-38h]
  __int64 v17; // [rsp+10h] [rbp-30h] BYREF
  int v18; // [rsp+18h] [rbp-28h]

  v6 = *(char **)(a2 + 40);
  v7 = *(_QWORD **)(a1 + 8);
  v8 = *v6;
  v9 = *((_QWORD *)v6 + 1);
  v15[0] = v8;
  v16 = v9;
  if ( v8 )
  {
    switch ( v8 )
    {
      case 14:
      case 15:
      case 16:
      case 17:
      case 18:
      case 19:
      case 20:
      case 21:
      case 22:
      case 23:
      case 56:
      case 57:
      case 58:
      case 59:
      case 60:
      case 61:
        v10 = 2;
        break;
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
        v10 = 3;
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
        v10 = 4;
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
        v10 = 5;
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
        v10 = 6;
        break;
      case 55:
        v10 = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        v10 = 8;
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
        v10 = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        v10 = 10;
        break;
    }
    v12 = 0;
  }
  else
  {
    v10 = sub_1F596B0((__int64)v15);
    v12 = v11;
  }
  v17 = 0;
  v18 = 0;
  v13 = sub_1D2B300(v7, 0x30u, (__int64)&v17, v10, v12, a6);
  if ( v17 )
    sub_161E7C0((__int64)&v17, v17);
  return v13;
}
