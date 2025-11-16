// Function: sub_1D389D0
// Address: 0x1d389d0
//
__int64 __fastcall sub_1D389D0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        const void **a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        double a8,
        __m128i a9)
{
  unsigned __int8 v9; // r15
  char v10; // r12
  unsigned __int8 v11; // bl
  const void **v12; // rdx
  unsigned int v13; // eax
  __int64 result; // rax
  __int64 v15; // [rsp+0h] [rbp-60h]
  __int64 v16; // [rsp+10h] [rbp-50h] BYREF
  const void **v17; // [rsp+18h] [rbp-48h]
  unsigned __int64 v18; // [rsp+20h] [rbp-40h] BYREF
  const void **v19; // [rsp+28h] [rbp-38h]

  v9 = a6;
  v10 = a3;
  v11 = a5;
  v16 = a3;
  v17 = a4;
  if ( (_BYTE)a3 )
  {
    if ( (unsigned __int8)(a3 - 14) <= 0x5Fu )
    {
      switch ( (char)a3 )
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
        default:
          v10 = 2;
          break;
      }
      goto LABEL_14;
    }
    goto LABEL_3;
  }
  if ( !(unsigned __int8)sub_1F58D20(&v16) )
  {
LABEL_3:
    v12 = v17;
    goto LABEL_4;
  }
  v10 = sub_1F596B0(&v16);
LABEL_4:
  LOBYTE(v18) = v10;
  v19 = v12;
  if ( !v10 )
  {
    v13 = sub_1F58D40(&v18, a2, v12, a4, a5, a6);
    goto LABEL_6;
  }
LABEL_14:
  v13 = sub_1D13440(v10);
LABEL_6:
  LODWORD(v19) = v13;
  if ( v13 > 0x40 )
    sub_16A4EF0((__int64)&v18, -1, 1);
  else
    v18 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v13;
  result = sub_1D38970(a1, (__int64)&v18, a2, v16, v17, v11, a7, a8, a9, v9);
  if ( (unsigned int)v19 > 0x40 )
  {
    if ( v18 )
    {
      v15 = result;
      j_j___libc_free_0_0(v18);
      return v15;
    }
  }
  return result;
}
