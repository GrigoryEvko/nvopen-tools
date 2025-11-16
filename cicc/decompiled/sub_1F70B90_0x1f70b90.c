// Function: sub_1F70B90
// Address: 0x1f70b90
//
__int64 *__fastcall sub_1F70B90(__int64 **a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5, double a6, __m128i a7)
{
  __int64 v10; // rax
  char v11; // di
  const void **v12; // r15
  int v13; // r15d
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r13
  __int64 v17; // r12
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  const void **v21; // rdx
  __int128 v22; // [rsp-10h] [rbp-80h]
  __int128 v23; // [rsp+0h] [rbp-70h]
  unsigned int v24; // [rsp+20h] [rbp-50h] BYREF
  const void **v25; // [rsp+28h] [rbp-48h]
  char v26[8]; // [rsp+30h] [rbp-40h] BYREF
  const void **v27; // [rsp+38h] [rbp-38h]

  v10 = *(_QWORD *)(a2 + 40) + 16LL * (unsigned int)a3;
  v11 = *(_BYTE *)v10;
  v12 = *(const void ***)(v10 + 8);
  LOBYTE(v24) = v11;
  v25 = v12;
  if ( v11 )
  {
    if ( (unsigned __int8)(v11 - 14) <= 0x5Fu )
    {
      switch ( v11 )
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
        default:
          v11 = 2;
          break;
      }
    }
    goto LABEL_3;
  }
  if ( !sub_1F58D20((__int64)&v24) )
  {
    v26[0] = 0;
    v27 = v12;
LABEL_7:
    v13 = sub_1F58D40((__int64)v26);
    goto LABEL_4;
  }
  v26[0] = sub_1F596B0((__int64)&v24);
  v11 = v26[0];
  v27 = v21;
  if ( !v26[0] )
    goto LABEL_7;
LABEL_3:
  v13 = sub_1F6C8D0(v11);
LABEL_4:
  *((_QWORD *)&v23 + 1) = a3;
  *(_QWORD *)&v23 = a2;
  v14 = sub_1D309E0(*a1, 129, a4, v24, v25, 0, *(double *)a5.m128i_i64, a6, *(double *)a7.m128i_i64, v23);
  v16 = v15;
  v17 = v14;
  v18 = sub_1D38BB0((__int64)*a1, (unsigned int)(v13 - 1), a4, v24, v25, 0, a5, a6, a7, 0);
  *((_QWORD *)&v22 + 1) = v16;
  *(_QWORD *)&v22 = v17;
  return sub_1D332F0(*a1, 53, a4, v24, v25, 0, *(double *)a5.m128i_i64, a6, a7, v18, v19, v22);
}
