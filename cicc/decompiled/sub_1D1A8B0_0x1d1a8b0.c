// Function: sub_1D1A8B0
// Address: 0x1d1a8b0
//
char __fastcall sub_1D1A8B0(__int64 a1, __int64 a2)
{
  char result; // al
  char *v4; // rdx
  char v5; // al
  __int64 v6; // rdx
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  char v11; // di
  unsigned int v12; // ebx
  char v13; // [rsp-69h] [rbp-69h]
  unsigned __int64 v14; // [rsp-68h] [rbp-68h] BYREF
  unsigned int v15; // [rsp-60h] [rbp-60h]
  char v16; // [rsp-58h] [rbp-58h] BYREF
  __int64 v17; // [rsp-50h] [rbp-50h]
  unsigned int v18; // [rsp-48h] [rbp-48h] BYREF
  __int64 v19; // [rsp-40h] [rbp-40h]

  result = 0;
  if ( *(_WORD *)(a1 + 24) != 104 )
    return result;
  v4 = *(char **)(a1 + 40);
  v15 = 1;
  v14 = 0;
  v5 = *v4;
  v6 = *((_QWORD *)v4 + 1);
  v16 = v5;
  v17 = v6;
  if ( v5 )
  {
    switch ( v5 )
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
        v11 = 2;
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
        v11 = 3;
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
        v11 = 4;
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
        v11 = 5;
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
        v11 = 6;
        break;
      case 55:
        v11 = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        v11 = 8;
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
        v11 = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        v11 = 10;
        break;
    }
    goto LABEL_14;
  }
  LOBYTE(v18) = sub_1F596B0(&v16);
  v11 = v18;
  v19 = v7;
  if ( (_BYTE)v18 )
  {
LABEL_14:
    v12 = sub_1D13440(v11);
    goto LABEL_5;
  }
  v12 = sub_1F58D40(&v18, a2, v7, v8, v9, v10);
LABEL_5:
  result = sub_1D19C30(a1, a2, &v14, &v18, (bool *)&v16, v12, 0);
  if ( result )
    result = v18 == v12;
  if ( v15 > 0x40 )
  {
    if ( v14 )
    {
      v13 = result;
      j_j___libc_free_0_0(v14);
      return v13;
    }
  }
  return result;
}
