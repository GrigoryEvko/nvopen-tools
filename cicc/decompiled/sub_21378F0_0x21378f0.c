// Function: sub_21378F0
// Address: 0x21378f0
//
__int64 __fastcall sub_21378F0(__int64 a1, __int64 a2, double a3, double a4, double a5)
{
  unsigned int v5; // r14d
  __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rsi
  char v10; // al
  unsigned int v11; // eax
  const void **v12; // rdx
  const void **v13; // r8
  __int128 v14; // rax
  __int64 v15; // r14
  __int64 v17; // [rsp+0h] [rbp-60h] BYREF
  int v18; // [rsp+8h] [rbp-58h]
  unsigned int v19; // [rsp+10h] [rbp-50h] BYREF
  const void **v20; // [rsp+18h] [rbp-48h]
  const void **v21; // [rsp+20h] [rbp-40h]

  v7 = *(_QWORD *)(a2 + 72);
  v17 = v7;
  if ( v7 )
    sub_1623A60((__int64)&v17, v7, 2);
  v8 = *(_QWORD *)(a1 + 8);
  v9 = *(_QWORD *)a1;
  v18 = *(_DWORD *)(a2 + 64);
  sub_1F40D10(
    (__int64)&v19,
    v9,
    *(_QWORD *)(v8 + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v10 = (char)v20;
  LOBYTE(v19) = (_BYTE)v20;
  v20 = v21;
  if ( (_BYTE)v19 )
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
        LOBYTE(v11) = 2;
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
    }
    v13 = 0;
  }
  else
  {
    LOBYTE(v11) = sub_1F596B0((__int64)&v19);
    v5 = v11;
    v13 = v12;
  }
  LOBYTE(v5) = v11;
  *(_QWORD *)&v14 = sub_1D309E0(
                      *(__int64 **)(a1 + 8),
                      144,
                      (__int64)&v17,
                      v5,
                      v13,
                      0,
                      a3,
                      a4,
                      a5,
                      *(_OWORD *)*(_QWORD *)(a2 + 32));
  v15 = sub_1D309E0(*(__int64 **)(a1 + 8), 111, (__int64)&v17, v19, v20, 0, a3, a4, a5, v14);
  if ( v17 )
    sub_161E7C0((__int64)&v17, v17);
  return v15;
}
