// Function: sub_20228B0
// Address: 0x20228b0
//
__int64 __fastcall sub_20228B0(__int64 a1, __int64 a2, double a3, double a4, double a5, __int64 a6, __int64 a7)
{
  __int64 v7; // r9
  __int64 v9; // rax
  char *v10; // rdx
  __int64 v11; // r14
  __int64 v12; // r15
  __int64 v13; // r13
  __int64 v14; // rbx
  char v15; // al
  __int64 v16; // rdx
  __int64 v17; // rax
  char v18; // si
  const void **v19; // rdx
  const void **v20; // r8
  __int64 v21; // rax
  __int64 v23; // rsi
  __int64 *v24; // r13
  __int128 v25; // [rsp-10h] [rbp-70h]
  __int64 v26; // [rsp+8h] [rbp-58h]
  const void **v27; // [rsp+10h] [rbp-50h]
  __int64 v28; // [rsp+20h] [rbp-40h] BYREF
  __int64 v29; // [rsp+28h] [rbp-38h]

  v7 = a1;
  v9 = *(_QWORD *)(a2 + 32);
  v10 = *(char **)(a2 + 40);
  v11 = *(_QWORD *)(v9 + 40);
  v12 = *(_QWORD *)(v9 + 48);
  v13 = v11;
  v14 = *(unsigned int *)(v9 + 48);
  v15 = *v10;
  v16 = *((_QWORD *)v10 + 1);
  LOBYTE(v28) = v15;
  v29 = v16;
  if ( v15 )
  {
    switch ( v15 )
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
    v20 = 0;
  }
  else
  {
    LOBYTE(v17) = sub_1F596B0((__int64)&v28);
    v7 = a1;
    a7 = v17;
    v18 = v17;
    v20 = v19;
  }
  LOBYTE(a7) = v18;
  v21 = *(_QWORD *)(v11 + 40) + 16 * v14;
  if ( *(_BYTE *)v21 != v18 || *(const void ***)(v21 + 8) != v20 && !v18 )
  {
    v23 = *(_QWORD *)(a2 + 72);
    v24 = *(__int64 **)(v7 + 8);
    v28 = v23;
    if ( v23 )
    {
      v26 = a7;
      v27 = v20;
      sub_1623A60((__int64)&v28, v23, 2);
      a7 = v26;
      v20 = v27;
    }
    *((_QWORD *)&v25 + 1) = v12;
    *(_QWORD *)&v25 = v11;
    LODWORD(v29) = *(_DWORD *)(a2 + 64);
    v13 = sub_1D309E0(v24, 145, (__int64)&v28, a7, v20, 0, a3, a4, a5, v25);
    if ( v28 )
      sub_161E7C0((__int64)&v28, v28);
  }
  return v13;
}
