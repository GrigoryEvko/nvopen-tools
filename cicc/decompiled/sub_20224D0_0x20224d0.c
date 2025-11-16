// Function: sub_20224D0
// Address: 0x20224d0
//
__int64 __fastcall sub_20224D0(__int64 a1, __int64 a2, double a3, double a4, double a5)
{
  char *v6; // rdx
  char v7; // al
  __int64 v8; // rdx
  char v9; // al
  const void **v10; // rdx
  __int64 *v11; // rdx
  char v12; // cl
  __int64 v13; // r14
  __int64 v14; // r15
  bool v15; // dl
  __int64 v16; // r8
  __int64 v17; // rsi
  __int64 *v18; // r12
  __int64 *v20; // rax
  __int64 *v21; // rax
  __int64 *v22; // rax
  __int64 *v23; // rax
  __int64 *v24; // rax
  __int64 *v25; // rax
  __int64 *v26; // rax
  __int64 *v27; // rax
  __int64 *v28; // rax
  __int128 v29; // [rsp-10h] [rbp-70h]
  __int64 v30; // [rsp+0h] [rbp-60h]
  unsigned int v31; // [rsp+10h] [rbp-50h] BYREF
  const void **v32; // [rsp+18h] [rbp-48h]
  __int64 v33; // [rsp+20h] [rbp-40h] BYREF
  __int64 v34; // [rsp+28h] [rbp-38h]

  v6 = *(char **)(a2 + 40);
  v7 = *v6;
  v8 = *((_QWORD *)v6 + 1);
  LOBYTE(v33) = v7;
  v34 = v8;
  if ( v7 )
  {
    switch ( v7 )
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
        v21 = *(__int64 **)(a2 + 32);
        LOBYTE(v31) = 2;
        v12 = 2;
        v32 = 0;
        v13 = *v21;
        v14 = v21[1];
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
        v20 = *(__int64 **)(a2 + 32);
        LOBYTE(v31) = 3;
        v12 = 3;
        v32 = 0;
        v13 = *v20;
        v14 = v20[1];
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
        v22 = *(__int64 **)(a2 + 32);
        LOBYTE(v31) = 4;
        v12 = 4;
        v32 = 0;
        v13 = *v22;
        v14 = v22[1];
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
        v23 = *(__int64 **)(a2 + 32);
        LOBYTE(v31) = 5;
        v12 = 5;
        v32 = 0;
        v13 = *v23;
        v14 = v23[1];
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
        v24 = *(__int64 **)(a2 + 32);
        LOBYTE(v31) = 6;
        v12 = 6;
        v32 = 0;
        v13 = *v24;
        v14 = v24[1];
        break;
      case 55:
        v28 = *(__int64 **)(a2 + 32);
        LOBYTE(v31) = 7;
        v12 = 7;
        v32 = 0;
        v13 = *v28;
        v14 = v28[1];
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        v27 = *(__int64 **)(a2 + 32);
        LOBYTE(v31) = 8;
        v12 = 8;
        v32 = 0;
        v13 = *v27;
        v14 = v27[1];
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
        v25 = *(__int64 **)(a2 + 32);
        LOBYTE(v31) = 9;
        v12 = 9;
        v32 = 0;
        v13 = *v25;
        v14 = v25[1];
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        v26 = *(__int64 **)(a2 + 32);
        LOBYTE(v31) = 10;
        v12 = 10;
        v32 = 0;
        v13 = *v26;
        v14 = v26[1];
        break;
    }
  }
  else
  {
    v9 = sub_1F596B0((__int64)&v33);
    v32 = v10;
    v11 = *(__int64 **)(a2 + 32);
    v12 = v9;
    LOBYTE(v31) = v9;
    v13 = *v11;
    v14 = v11[1];
    if ( !v9 )
    {
      v15 = sub_1F58CF0((__int64)&v31);
      goto LABEL_5;
    }
  }
  v15 = (unsigned __int8)(v12 - 14) <= 0x47u || (unsigned __int8)(v12 - 2) <= 5u;
LABEL_5:
  v16 = v13;
  if ( v15 )
  {
    v17 = *(_QWORD *)(a2 + 72);
    v18 = *(__int64 **)(a1 + 8);
    v33 = v17;
    if ( v17 )
      sub_1623A60((__int64)&v33, v17, 2);
    *((_QWORD *)&v29 + 1) = v14;
    *(_QWORD *)&v29 = v13;
    LODWORD(v34) = *(_DWORD *)(a2 + 64);
    v16 = sub_1D309E0(v18, 145, (__int64)&v33, v31, v32, 0, a3, a4, a5, v29);
    if ( v33 )
    {
      v30 = v16;
      sub_161E7C0((__int64)&v33, v33);
      return v30;
    }
  }
  return v16;
}
