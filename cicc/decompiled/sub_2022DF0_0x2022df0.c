// Function: sub_2022DF0
// Address: 0x2022df0
//
__int64 __fastcall sub_2022DF0(__int64 a1, __int64 a2, double a3, double a4, double a5)
{
  unsigned int v5; // r14d
  __int64 v6; // rcx
  char *v8; // rdx
  char v9; // al
  __int64 v10; // rdx
  unsigned int v11; // eax
  char v12; // si
  const void **v13; // rdx
  const void **v14; // r15
  __int64 *v15; // rax
  __int64 v16; // r12
  __int64 v17; // r13
  __int64 v18; // rdx
  __int64 v20; // rsi
  __int64 *v21; // r11
  __int128 v22; // [rsp-10h] [rbp-60h]
  __int64 *v23; // [rsp+0h] [rbp-50h]
  __int64 v24; // [rsp+10h] [rbp-40h] BYREF
  __int64 v25; // [rsp+18h] [rbp-38h]

  v6 = a1;
  v8 = *(char **)(a2 + 40);
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  LOBYTE(v24) = v9;
  v25 = v10;
  if ( v9 )
  {
    switch ( v9 )
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
        v12 = 2;
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
        v12 = 3;
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
        v12 = 4;
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
        v12 = 5;
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
        v12 = 6;
        break;
      case 55:
        v12 = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        v12 = 8;
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
        v12 = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        v12 = 10;
        break;
    }
    v14 = 0;
  }
  else
  {
    LOBYTE(v11) = sub_1F596B0((__int64)&v24);
    v6 = a1;
    v5 = v11;
    v12 = v11;
    v14 = v13;
  }
  v15 = *(__int64 **)(a2 + 32);
  LOBYTE(v5) = v12;
  v16 = *v15;
  v17 = v15[1];
  v18 = *(_QWORD *)(*v15 + 40) + 16LL * *((unsigned int *)v15 + 2);
  if ( *(_BYTE *)v18 != v12 || *(const void ***)(v18 + 8) != v14 && !v12 )
  {
    v20 = *(_QWORD *)(a2 + 72);
    v21 = *(__int64 **)(v6 + 8);
    v24 = v20;
    if ( v20 )
    {
      v23 = v21;
      sub_1623A60((__int64)&v24, v20, 2);
      v21 = v23;
    }
    *((_QWORD *)&v22 + 1) = v17;
    *(_QWORD *)&v22 = v16;
    LODWORD(v25) = *(_DWORD *)(a2 + 64);
    v16 = sub_1D309E0(v21, 145, (__int64)&v24, v5, v14, 0, a3, a4, a5, v22);
    if ( v24 )
      sub_161E7C0((__int64)&v24, v24);
  }
  return v16;
}
