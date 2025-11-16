// Function: sub_1D3C080
// Address: 0x1d3c080
//
__int64 *__fastcall sub_1D3C080(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        const void **a6,
        __m128i a7,
        double a8,
        __m128i a9)
{
  char v11; // r12
  const void **v12; // rdx
  unsigned int v13; // eax
  __int128 v14; // rax
  __int128 v16; // [rsp+0h] [rbp-70h]
  __int64 v17; // [rsp+10h] [rbp-60h] BYREF
  const void **v18; // [rsp+18h] [rbp-58h]
  char v19[8]; // [rsp+20h] [rbp-50h] BYREF
  const void **v20; // [rsp+28h] [rbp-48h]
  unsigned __int64 v21; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v22; // [rsp+38h] [rbp-38h]

  v11 = a5;
  v17 = a5;
  v18 = a6;
  if ( (_BYTE)a5 )
  {
    if ( (unsigned __int8)(a5 - 14) <= 0x5Fu )
    {
      switch ( (char)a5 )
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
          v19[0] = 3;
          v11 = 3;
          v20 = 0;
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
          v19[0] = 4;
          v11 = 4;
          v20 = 0;
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
          v19[0] = 5;
          v11 = 5;
          v20 = 0;
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
          v19[0] = 6;
          v11 = 6;
          v20 = 0;
          break;
        case 55:
          v19[0] = 7;
          v11 = 7;
          v20 = 0;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v19[0] = 8;
          v11 = 8;
          v20 = 0;
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
          v19[0] = 9;
          v11 = 9;
          v20 = 0;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v19[0] = 10;
          v11 = 10;
          v20 = 0;
          break;
        default:
          v19[0] = 2;
          v11 = 2;
          v20 = 0;
          break;
      }
      goto LABEL_14;
    }
    goto LABEL_3;
  }
  if ( !(unsigned __int8)sub_1F58D20(&v17) )
  {
LABEL_3:
    v12 = v18;
    goto LABEL_4;
  }
  v11 = sub_1F596B0(&v17);
LABEL_4:
  v19[0] = v11;
  v20 = v12;
  if ( !v11 )
  {
    v13 = sub_1F58D40(v19, a2, v12, a4, a5, a6);
    goto LABEL_6;
  }
LABEL_14:
  v13 = sub_1D13440(v11);
LABEL_6:
  v22 = v13;
  if ( v13 > 0x40 )
    sub_16A4EF0((__int64)&v21, -1, 1);
  else
    v21 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v13;
  *(_QWORD *)&v14 = sub_1D38970((__int64)a1, (__int64)&v21, a2, v17, v18, 0, a7, a8, a9, 0);
  if ( v22 > 0x40 && v21 )
  {
    v16 = v14;
    j_j___libc_free_0_0(v21);
    v14 = v16;
  }
  return sub_1D332F0(a1, 120, a2, (unsigned int)v17, v18, 0, *(double *)a7.m128i_i64, a8, a9, a3, a4, v14);
}
