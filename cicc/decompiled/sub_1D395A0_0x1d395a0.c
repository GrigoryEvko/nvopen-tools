// Function: sub_1D395A0
// Address: 0x1d395a0
//
__int64 __fastcall sub_1D395A0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        const void **a5,
        __int64 a6,
        double a7,
        __m128i a8,
        __m128i a9,
        __int128 a10)
{
  __int64 v10; // r13
  __int64 v11; // rbx
  __m128i v12; // xmm0
  _DWORD *v13; // r14
  char v14; // al
  unsigned int v15; // eax
  __int64 v16; // rsi
  __m128i v17; // rax
  const void **v18; // rdx
  unsigned int v19; // eax
  const void **v21; // [rsp+8h] [rbp-68h]
  __m128i v22; // [rsp+10h] [rbp-60h] BYREF
  __int64 v23; // [rsp+20h] [rbp-50h] BYREF
  const void **v24; // [rsp+28h] [rbp-48h]
  __m128i v25; // [rsp+30h] [rbp-40h] BYREF

  v10 = a3;
  v11 = a4;
  v12 = _mm_loadu_si128((const __m128i *)&a10);
  v22 = v12;
  if ( !(_BYTE)a2 )
  {
    LODWORD(a10) = 0;
    v16 = 0;
    goto LABEL_8;
  }
  a8 = _mm_load_si128(&v22);
  v13 = *(_DWORD **)(a1 + 16);
  v25 = a8;
  if ( v22.m128i_i8[0] )
  {
    if ( (unsigned __int8)(v22.m128i_i8[0] - 14) > 0x5Fu )
    {
      if ( (unsigned __int8)(v22.m128i_i8[0] - 86) <= 0x17u || (unsigned __int8)(v22.m128i_i8[0] - 8) <= 5u )
        goto LABEL_5;
      goto LABEL_11;
    }
  }
  else
  {
    v21 = a5;
    v22.m128i_i8[0] = sub_1F58CD0(&v25);
    v14 = sub_1F58D20(&v25);
    a5 = v21;
    if ( !v14 )
    {
      if ( v22.m128i_i8[0] )
      {
LABEL_5:
        v15 = v13[16];
        goto LABEL_6;
      }
LABEL_11:
      if ( v13[15] <= 1u )
        goto LABEL_7;
      goto LABEL_12;
    }
  }
  v15 = v13[17];
LABEL_6:
  if ( v15 <= 1 )
  {
LABEL_7:
    LODWORD(a10) = 0;
    a4 = (unsigned int)v11;
    a3 = v10;
    v16 = 1;
LABEL_8:
    v17.m128i_i64[0] = sub_1D38BB0(a1, v16, a3, a4, a5, 0, v12, *(double *)a8.m128i_i64, a9, a10);
    return v17.m128i_i64[0];
  }
LABEL_12:
  v23 = v11;
  v24 = a5;
  if ( (_BYTE)v11 )
  {
    if ( (unsigned __int8)(v11 - 14) <= 0x5Fu )
    {
      switch ( (char)v11 )
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
          LOBYTE(v11) = 3;
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
          LOBYTE(v11) = 4;
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
          LOBYTE(v11) = 5;
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
          LOBYTE(v11) = 6;
          break;
        case 55:
          LOBYTE(v11) = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          LOBYTE(v11) = 8;
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
          LOBYTE(v11) = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          LOBYTE(v11) = 10;
          break;
        default:
          LOBYTE(v11) = 2;
          break;
      }
      goto LABEL_29;
    }
    goto LABEL_14;
  }
  if ( !(unsigned __int8)sub_1F58D20(&v23) )
  {
LABEL_14:
    v18 = v24;
    goto LABEL_15;
  }
  LOBYTE(v11) = sub_1F596B0(&v23);
LABEL_15:
  v25.m128i_i8[0] = v11;
  v25.m128i_i64[1] = (__int64)v18;
  if ( (_BYTE)v11 )
  {
LABEL_29:
    v19 = sub_1D13440(v11);
    goto LABEL_17;
  }
  v19 = sub_1F58D40(&v25, a2, v18, a4, a5, a6);
LABEL_17:
  v25.m128i_i32[2] = v19;
  if ( v19 > 0x40 )
    sub_16A4EF0((__int64)&v25, -1, 1);
  else
    v25.m128i_i64[0] = 0xFFFFFFFFFFFFFFFFLL >> -(char)v19;
  v17.m128i_i64[0] = sub_1D38970(a1, (__int64)&v25, v10, v23, v24, 0, v12, *(double *)a8.m128i_i64, a9, 0);
  if ( v25.m128i_i32[2] > 0x40u && v25.m128i_i64[0] )
  {
    v22 = v17;
    j_j___libc_free_0_0(v25.m128i_i64[0]);
    v17.m128i_i64[0] = v22.m128i_i64[0];
  }
  return v17.m128i_i64[0];
}
