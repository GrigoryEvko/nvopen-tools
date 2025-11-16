// Function: sub_1D38BB0
// Address: 0x1d38bb0
//
__int64 __fastcall sub_1D38BB0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        const void **a5,
        __int64 a6,
        __m128i a7,
        double a8,
        __m128i a9,
        unsigned __int8 a10)
{
  char v11; // r12
  const void **v12; // rdx
  unsigned __int8 v13; // bl
  unsigned int v14; // eax
  __int64 result; // rax
  char v16; // al
  char v17; // al
  unsigned int v18; // [rsp+Ch] [rbp-74h]
  unsigned __int8 v19; // [rsp+10h] [rbp-70h]
  __int64 v20; // [rsp+10h] [rbp-70h]
  __int64 v21; // [rsp+20h] [rbp-60h] BYREF
  const void **v22; // [rsp+28h] [rbp-58h]
  char v23[8]; // [rsp+30h] [rbp-50h] BYREF
  const void **v24; // [rsp+38h] [rbp-48h]
  unsigned __int64 v25; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v26; // [rsp+48h] [rbp-38h]

  v11 = a4;
  v21 = a4;
  v22 = a5;
  if ( (_BYTE)a4 )
  {
    if ( (unsigned __int8)(a4 - 14) <= 0x5Fu )
    {
      switch ( (char)a4 )
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
          v23[0] = 3;
          v13 = a6;
          v11 = 3;
          v24 = 0;
          v19 = a10;
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
          v23[0] = 4;
          v13 = a6;
          v11 = 4;
          v24 = 0;
          v19 = a10;
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
          v23[0] = 5;
          v13 = a6;
          v11 = 5;
          v24 = 0;
          v19 = a10;
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
          v23[0] = 6;
          v13 = a6;
          v11 = 6;
          v24 = 0;
          v19 = a10;
          break;
        case 55:
          v23[0] = 7;
          v13 = a6;
          v11 = 7;
          v24 = 0;
          v19 = a10;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v23[0] = 8;
          v13 = a6;
          v11 = 8;
          v24 = 0;
          v19 = a10;
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
          v23[0] = 9;
          v13 = a6;
          v11 = 9;
          v24 = 0;
          v19 = a10;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v23[0] = 10;
          v13 = a6;
          v11 = 10;
          v24 = 0;
          v19 = a10;
          break;
        default:
          v23[0] = 2;
          v13 = a6;
          v11 = 2;
          v24 = 0;
          v19 = a10;
          break;
      }
      goto LABEL_14;
    }
    goto LABEL_3;
  }
  v18 = a6;
  v16 = sub_1F58D20(&v21);
  a6 = v18;
  if ( !v16 )
  {
LABEL_3:
    v12 = v22;
    goto LABEL_4;
  }
  v17 = sub_1F596B0(&v21);
  a6 = v18;
  v11 = v17;
LABEL_4:
  v23[0] = v11;
  v13 = a6;
  v24 = v12;
  v19 = a10;
  if ( !v11 )
  {
    v14 = sub_1F58D40(v23, a2, v12, a4, a5, a6);
    goto LABEL_6;
  }
LABEL_14:
  v14 = sub_1D13440(v11);
LABEL_6:
  v26 = v14;
  if ( v14 > 0x40 )
    sub_16A4EF0((__int64)&v25, a2, 0);
  else
    v25 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v14) & a2;
  result = sub_1D38970(a1, (__int64)&v25, a3, v21, v22, v13, a7, a8, a9, v19);
  if ( v26 > 0x40 )
  {
    if ( v25 )
    {
      v20 = result;
      j_j___libc_free_0_0(v25);
      return v20;
    }
  }
  return result;
}
