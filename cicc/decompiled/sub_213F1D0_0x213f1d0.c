// Function: sub_213F1D0
// Address: 0x213f1d0
//
__int64 __fastcall sub_213F1D0(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v7; // rsi
  __int64 v8; // rax
  char *v9; // rcx
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // r13
  __int64 v12; // r12
  char v13; // dl
  char v14; // al
  __int64 v15; // rcx
  int v16; // esi
  int v17; // eax
  char v18; // di
  unsigned __int8 v19; // al
  __int128 v20; // rax
  __int64 v21; // r12
  char v23; // [rsp+Fh] [rbp-51h]
  __int64 v24; // [rsp+10h] [rbp-50h] BYREF
  int v25; // [rsp+18h] [rbp-48h]
  _BYTE v26[8]; // [rsp+20h] [rbp-40h] BYREF
  __int64 v27; // [rsp+28h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 72);
  v24 = v7;
  if ( v7 )
    sub_1623A60((__int64)&v24, v7, 2);
  v25 = *(_DWORD *)(a2 + 64);
  v8 = sub_2138AD0(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v9 = *(char **)(a2 + 40);
  v11 = v10;
  v12 = v8;
  v13 = *(_BYTE *)(*(_QWORD *)(v8 + 40) + 16LL * (unsigned int)v10);
  v14 = *v9;
  v15 = *((_QWORD *)v9 + 1);
  v26[0] = v14;
  v27 = v15;
  if ( v14 )
  {
    v16 = word_4310720[(unsigned __int8)(v14 - 14)];
  }
  else
  {
    v23 = v13;
    v17 = sub_1F58D30((__int64)v26);
    v13 = v23;
    v16 = v17;
  }
  switch ( v13 )
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
      v18 = 2;
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
      v18 = 3;
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
      v18 = 4;
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
      v18 = 5;
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
      v18 = 6;
      break;
    case 55:
      v18 = 7;
      break;
    case 86:
    case 87:
    case 88:
    case 98:
    case 99:
    case 100:
      v18 = 8;
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
      v18 = 9;
      break;
    case 94:
    case 95:
    case 96:
    case 97:
    case 106:
    case 107:
    case 108:
    case 109:
      v18 = 10;
      break;
  }
  v19 = sub_1D15020(v18, v16);
  *(_QWORD *)&v20 = sub_1D332F0(
                      *(__int64 **)(a1 + 8),
                      109,
                      (__int64)&v24,
                      v19,
                      0,
                      0,
                      a3,
                      a4,
                      a5,
                      v12,
                      v11,
                      *(_OWORD *)(*(_QWORD *)(a2 + 32) + 40LL));
  v21 = sub_1D309E0(
          *(__int64 **)(a1 + 8),
          145,
          (__int64)&v24,
          **(unsigned __int8 **)(a2 + 40),
          *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL),
          0,
          a3,
          a4,
          *(double *)a5.m128i_i64,
          v20);
  if ( v24 )
    sub_161E7C0((__int64)&v24, v24);
  return v21;
}
