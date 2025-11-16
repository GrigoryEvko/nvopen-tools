// Function: sub_213ECF0
// Address: 0x213ecf0
//
__int64 *__fastcall sub_213ECF0(__int64 a1, __int64 a2, __m128 a3, double a4, __m128i a5)
{
  unsigned int v5; // r14d
  unsigned int v7; // eax
  const void **v8; // rdx
  const void **v9; // r8
  __int64 v10; // rsi
  __int16 *v11; // rdx
  __int128 v12; // rax
  __int64 *v13; // r14
  const void **v15; // [rsp+8h] [rbp-78h]
  const void **v16; // [rsp+10h] [rbp-70h]
  unsigned __int64 v17; // [rsp+10h] [rbp-70h]
  __int16 *v18; // [rsp+18h] [rbp-68h]
  unsigned int v19; // [rsp+20h] [rbp-60h] BYREF
  const void **v20; // [rsp+28h] [rbp-58h]
  __int64 v21; // [rsp+30h] [rbp-50h] BYREF
  int v22; // [rsp+38h] [rbp-48h]
  const void **v23; // [rsp+40h] [rbp-40h]

  sub_1F40D10(
    (__int64)&v21,
    *(_QWORD *)a1,
    *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  LOBYTE(v19) = v22;
  v20 = v23;
  if ( (_BYTE)v22 )
  {
    switch ( (char)v22 )
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
        LOBYTE(v7) = 2;
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
        LOBYTE(v7) = 3;
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
        LOBYTE(v7) = 4;
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
        LOBYTE(v7) = 5;
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
        LOBYTE(v7) = 6;
        break;
      case 55:
        LOBYTE(v7) = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        LOBYTE(v7) = 8;
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
        LOBYTE(v7) = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        LOBYTE(v7) = 10;
        break;
    }
    v9 = 0;
  }
  else
  {
    LOBYTE(v7) = sub_1F596B0((__int64)&v19);
    v5 = v7;
    v9 = v8;
  }
  v10 = *(_QWORD *)(a2 + 72);
  LOBYTE(v5) = v7;
  v21 = v10;
  if ( v10 )
  {
    v16 = v9;
    sub_1623A60((__int64)&v21, v10, 2);
    v9 = v16;
  }
  v15 = v9;
  v22 = *(_DWORD *)(a2 + 64);
  v17 = sub_2138AD0(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v18 = v11;
  *(_QWORD *)&v12 = sub_1D309E0(
                      *(__int64 **)(a1 + 8),
                      144,
                      (__int64)&v21,
                      v5,
                      v15,
                      0,
                      *(double *)a3.m128_u64,
                      a4,
                      *(double *)a5.m128i_i64,
                      *(_OWORD *)(*(_QWORD *)(a2 + 32) + 40LL));
  v13 = sub_1D3A900(
          *(__int64 **)(a1 + 8),
          0x69u,
          (__int64)&v21,
          v19,
          v20,
          0,
          a3,
          a4,
          a5,
          v17,
          v18,
          v12,
          *(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL),
          *(_QWORD *)(*(_QWORD *)(a2 + 32) + 88LL));
  if ( v21 )
    sub_161E7C0((__int64)&v21, v21);
  return v13;
}
