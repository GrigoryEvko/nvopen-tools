// Function: sub_1D1F9F0
// Address: 0x1d1f9f0
//
__int64 __fastcall sub_1D1F9F0(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v7; // rax
  char v8; // di
  __int64 v9; // r15
  unsigned int v10; // eax
  unsigned int v11; // r15d
  __int64 v12; // rcx
  __int64 result; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // [rsp+0h] [rbp-60h]
  unsigned __int8 v18; // [rsp+8h] [rbp-58h]
  char v19[8]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v20; // [rsp+18h] [rbp-48h]
  __int64 v21; // [rsp+20h] [rbp-40h] BYREF
  __int64 v22; // [rsp+28h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 40) + 16LL * (unsigned int)a3;
  v8 = *(_BYTE *)v7;
  v9 = *(_QWORD *)(v7 + 8);
  v19[0] = v8;
  v20 = v9;
  if ( v8 )
  {
    if ( (unsigned __int8)(v8 - 14) <= 0x5Fu )
    {
      switch ( v8 )
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
          v8 = 3;
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
          v8 = 4;
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
          v8 = 5;
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
          v8 = 6;
          break;
        case 55:
          v8 = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v8 = 8;
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
          v8 = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v8 = 10;
          break;
        default:
          v8 = 2;
          break;
      }
    }
    goto LABEL_3;
  }
  if ( (unsigned __int8)sub_1F58D20(v19) )
  {
    LOBYTE(v21) = sub_1F596B0(v19);
    v8 = v21;
    v22 = v14;
    if ( (_BYTE)v21 )
    {
LABEL_3:
      v10 = sub_1D13440(v8);
      goto LABEL_4;
    }
  }
  else
  {
    LOBYTE(v21) = 0;
    v22 = v9;
  }
  v10 = sub_1F58D40(&v21, a2, v14, v15, v16, &v21);
LABEL_4:
  v11 = v10 - 1;
  LODWORD(v22) = v10;
  v12 = 1LL << ((unsigned __int8)v10 - 1);
  if ( v10 <= 0x40 )
  {
    v21 = 0;
LABEL_6:
    v21 |= v12;
    goto LABEL_7;
  }
  v17 = 1LL << ((unsigned __int8)v10 - 1);
  sub_16A4EF0((__int64)&v21, 0, 0);
  v12 = v17;
  if ( (unsigned int)v22 <= 0x40 )
    goto LABEL_6;
  *(_QWORD *)(v21 + 8LL * (v11 >> 6)) |= v17;
LABEL_7:
  result = sub_1D1F940(a1, a2, a3, (__int64)&v21, a4);
  if ( (unsigned int)v22 > 0x40 )
  {
    if ( v21 )
    {
      v18 = result;
      j_j___libc_free_0_0(v21);
      return v18;
    }
  }
  return result;
}
