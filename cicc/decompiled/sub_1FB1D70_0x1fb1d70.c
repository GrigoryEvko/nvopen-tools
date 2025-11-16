// Function: sub_1FB1D70
// Address: 0x1fb1d70
//
__int64 __fastcall sub_1FB1D70(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v5; // rax
  char v6; // di
  __int64 v7; // r15
  unsigned int v8; // eax
  __int64 result; // rax
  __int64 v10; // rdx
  unsigned __int8 v11; // [rsp+Fh] [rbp-51h]
  char v12[8]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v13; // [rsp+18h] [rbp-48h]
  unsigned __int64 v14; // [rsp+20h] [rbp-40h] BYREF
  __int64 v15; // [rsp+28h] [rbp-38h]

  v5 = *(_QWORD *)(a2 + 40) + 16LL * a3;
  v6 = *(_BYTE *)v5;
  v7 = *(_QWORD *)(v5 + 8);
  v12[0] = v6;
  v13 = v7;
  if ( v6 )
  {
    if ( (unsigned __int8)(v6 - 14) <= 0x5Fu )
    {
      switch ( v6 )
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
          v6 = 3;
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
          v6 = 4;
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
          v6 = 5;
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
          v6 = 6;
          break;
        case 55:
          v6 = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v6 = 8;
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
          v6 = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v6 = 10;
          break;
        default:
          v6 = 2;
          break;
      }
    }
    goto LABEL_3;
  }
  if ( sub_1F58D20((__int64)v12) )
  {
    LOBYTE(v14) = sub_1F596B0((__int64)v12);
    v6 = v14;
    v15 = v10;
    if ( (_BYTE)v14 )
    {
LABEL_3:
      v8 = sub_1F6C8D0(v6);
      goto LABEL_4;
    }
  }
  else
  {
    LOBYTE(v14) = 0;
    v15 = v7;
  }
  v8 = sub_1F58D40((__int64)&v14);
LABEL_4:
  LODWORD(v15) = v8;
  if ( v8 > 0x40 )
    sub_16A4EF0((__int64)&v14, -1, 1);
  else
    v14 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v8;
  result = sub_1FB1C90(a1, a2, a3, (int)&v14);
  if ( (unsigned int)v15 > 0x40 )
  {
    if ( v14 )
    {
      v11 = result;
      j_j___libc_free_0_0(v14);
      return v11;
    }
  }
  return result;
}
