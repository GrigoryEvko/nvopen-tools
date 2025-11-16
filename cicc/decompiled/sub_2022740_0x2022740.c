// Function: sub_2022740
// Address: 0x2022740
//
__int64 *__fastcall sub_2022740(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  unsigned int v5; // r12d
  char *v7; // rdx
  __int64 *v8; // r13
  __int64 v9; // rcx
  char v10; // al
  __int64 v11; // rdx
  unsigned int v12; // eax
  const void **v13; // rdx
  const void **v14; // r15
  __int64 v15; // rsi
  __int64 *v16; // r12
  __int64 v18; // [rsp+8h] [rbp-58h]
  __int64 v19; // [rsp+8h] [rbp-58h]
  __int64 v20; // [rsp+10h] [rbp-50h] BYREF
  int v21; // [rsp+18h] [rbp-48h]
  _BYTE v22[8]; // [rsp+20h] [rbp-40h] BYREF
  __int64 v23; // [rsp+28h] [rbp-38h]

  v7 = *(char **)(a2 + 40);
  v8 = *(__int64 **)(a1 + 8);
  v9 = *(_QWORD *)(a2 + 32);
  v10 = *v7;
  v11 = *((_QWORD *)v7 + 1);
  v22[0] = v10;
  v23 = v11;
  if ( v10 )
  {
    switch ( v10 )
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
        LOBYTE(v12) = 2;
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
        LOBYTE(v12) = 3;
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
        LOBYTE(v12) = 4;
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
        LOBYTE(v12) = 5;
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
        LOBYTE(v12) = 6;
        break;
      case 55:
        LOBYTE(v12) = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        LOBYTE(v12) = 8;
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
        LOBYTE(v12) = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        LOBYTE(v12) = 10;
        break;
    }
    v14 = 0;
  }
  else
  {
    v18 = v9;
    LOBYTE(v12) = sub_1F596B0((__int64)v22);
    v9 = v18;
    v5 = v12;
    v14 = v13;
  }
  v15 = *(_QWORD *)(a2 + 72);
  LOBYTE(v5) = v12;
  v20 = v15;
  if ( v15 )
  {
    v19 = v9;
    sub_1623A60((__int64)&v20, v15, 2);
    v9 = v19;
  }
  v21 = *(_DWORD *)(a2 + 64);
  v16 = sub_1D332F0(
          v8,
          106,
          (__int64)&v20,
          v5,
          v14,
          0,
          a3,
          a4,
          a5,
          *(_QWORD *)v9,
          *(_QWORD *)(v9 + 8),
          *(_OWORD *)(v9 + 40));
  if ( v20 )
    sub_161E7C0((__int64)&v20, v20);
  return v16;
}
