// Function: sub_20219E0
// Address: 0x20219e0
//
__int64 __fastcall sub_20219E0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, int a6, int a7, int a8)
{
  __int8 v8; // r15
  char v9; // r12
  __int64 v10; // rdx
  int i; // r12d
  __int64 v12; // rcx
  unsigned int v13; // r15d
  char v14; // r12
  bool v15; // al
  int v16; // r15d
  _QWORD *v17; // r14
  _BYTE *v18; // rbx
  char v20; // al
  unsigned int v21; // r9d
  unsigned int v22; // eax
  unsigned int v23; // r9d
  unsigned int v24; // [rsp+Ch] [rbp-A4h]
  char v25; // [rsp+13h] [rbp-9Dh]
  char v26; // [rsp+13h] [rbp-9Dh]
  unsigned int v27; // [rsp+14h] [rbp-9Ch]
  char v28; // [rsp+18h] [rbp-98h]
  unsigned int v30; // [rsp+24h] [rbp-8Ch]
  unsigned int v32; // [rsp+2Ch] [rbp-84h]
  _QWORD v33[2]; // [rsp+30h] [rbp-80h] BYREF
  __m128i v34; // [rsp+40h] [rbp-70h] BYREF
  __m128i v35; // [rsp+50h] [rbp-60h] BYREF
  char v36[80]; // [rsp+60h] [rbp-50h] BYREF

  v33[0] = a4;
  v33[1] = a5;
  if ( (_BYTE)a4 )
  {
    switch ( (char)a4 )
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
        v34.m128i_i8[0] = 2;
        v9 = 2;
        v34.m128i_i64[1] = 0;
        v30 = sub_2021900(a4);
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
        v34.m128i_i8[0] = 3;
        v9 = 3;
        v34.m128i_i64[1] = 0;
        v30 = sub_2021900(a4);
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
        v34.m128i_i8[0] = 4;
        v9 = 4;
        v34.m128i_i64[1] = 0;
        v30 = sub_2021900(a4);
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
        v34.m128i_i8[0] = 5;
        v9 = 5;
        v34.m128i_i64[1] = 0;
        v30 = sub_2021900(a4);
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
        v34.m128i_i8[0] = 6;
        v9 = 6;
        v34.m128i_i64[1] = 0;
        v30 = sub_2021900(a4);
        break;
      case 55:
        v34.m128i_i8[0] = 7;
        v9 = 7;
        v34.m128i_i64[1] = 0;
        v30 = sub_2021900(55);
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        v34.m128i_i8[0] = 8;
        v9 = 8;
        v34.m128i_i64[1] = 0;
        v30 = sub_2021900(a4);
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
        v34.m128i_i8[0] = 9;
        v9 = 9;
        v34.m128i_i64[1] = 0;
        v30 = sub_2021900(a4);
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        v34.m128i_i8[0] = 10;
        v9 = 10;
        v34.m128i_i64[1] = 0;
        v30 = sub_2021900(a4);
        break;
    }
  }
  else
  {
    v34.m128i_i8[0] = sub_1F596B0((__int64)v33);
    v8 = v34.m128i_i8[0];
    v9 = v34.m128i_i8[0];
    v34.m128i_i64[1] = v10;
    if ( LOBYTE(v33[0]) )
      v30 = sub_2021900(v33[0]);
    else
      v30 = sub_1F58D40((__int64)v33);
    if ( !v8 )
    {
      v32 = sub_1F58D40((__int64)&v34);
      goto LABEL_7;
    }
  }
  v32 = sub_2021900(v9);
LABEL_7:
  v35 = _mm_loadu_si128(&v34);
  v27 = 8 * a7;
  if ( a3 == v32 )
    return v35.m128i_i64[0];
  v25 = v9;
  for ( i = 7; i != 1; --i )
  {
    v28 = i;
    v13 = sub_2021900(i);
    if ( v32 >= v13 )
      break;
    LOBYTE(v12) = i;
    sub_1F40D10((__int64)v36, a2, *(_QWORD *)(a1 + 48), v12, 0);
    if ( v36[0] <= 1u
      && !(v30 % v13)
      && v30 >= v13
      && ((v30 / v13) & (v30 / v13 - 1)) == 0
      && (a3 >= v13 || a7 && v27 >= v13 && a8 + a3 >= v13) )
    {
      v35.m128i_i64[1] = 0;
      v14 = v25;
      v35.m128i_i8[0] = v28;
      goto LABEL_28;
    }
  }
  v14 = v25;
  v28 = v25;
LABEL_28:
  v16 = 109;
  v18 = (_BYTE *)(a2 + 30838);
  v17 = (_QWORD *)(a2 + 992);
  v26 = v33[0];
  while ( 1 )
  {
    if ( *v17 || (a6 == 2 ? (v15 = v18[1] == 4) : (v15 = *v18 == 4), v15) )
    {
      switch ( (char)v16 )
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
          v20 = 3;
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
          v20 = 4;
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
          v20 = 5;
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
          v20 = 6;
          break;
        case 55:
          v20 = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v20 = 8;
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
          v20 = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v20 = 10;
          break;
        default:
          v20 = 2;
          break;
      }
      if ( v14 == v20 )
      {
        v21 = sub_2021900(v16);
        if ( !(v30 % v21)
          && v30 >= v21
          && ((v30 / v21) & (v30 / v21 - 1)) == 0
          && (a3 >= v21 || a7 && v27 >= v21 && a8 + a3 >= v21) )
        {
          if ( v28 )
          {
            v22 = sub_2021900(v28);
          }
          else
          {
            v24 = v21;
            v22 = sub_1F58D40((__int64)&v35);
            v23 = v24;
          }
          if ( v23 > v22 || v26 == (_BYTE)v16 )
            break;
        }
      }
    }
    --v16;
    --v17;
    v18 -= 259;
    if ( v16 == 13 )
      return v35.m128i_i64[0];
  }
  return (unsigned __int8)v16;
}
