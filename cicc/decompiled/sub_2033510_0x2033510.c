// Function: sub_2033510
// Address: 0x2033510
//
__int64 *__fastcall sub_2033510(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  unsigned int v5; // r13d
  unsigned int v6; // r15d
  char *v8; // rdx
  char v9; // al
  __int64 v10; // rdx
  unsigned int v11; // eax
  const void **v12; // rdx
  unsigned __int64 *v13; // rdx
  unsigned __int64 v14; // rsi
  char v15; // al
  __int64 v16; // rsi
  unsigned int v17; // eax
  unsigned __int64 v18; // rdx
  unsigned __int64 v19; // r8
  __int64 v20; // rax
  __int64 *v21; // rbx
  unsigned __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r9
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rsi
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rsi
  __int64 *v31; // r12
  __int128 v33; // [rsp-10h] [rbp-90h]
  unsigned __int64 v34; // [rsp+0h] [rbp-80h]
  __int64 v35; // [rsp+0h] [rbp-80h]
  __int64 v36; // [rsp+8h] [rbp-78h]
  __int64 v37; // [rsp+10h] [rbp-70h]
  unsigned __int64 v38; // [rsp+18h] [rbp-68h]
  const void **v39; // [rsp+28h] [rbp-58h]
  char v40[8]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v41; // [rsp+38h] [rbp-48h]
  __int64 v42; // [rsp+40h] [rbp-40h] BYREF
  __int64 v43; // [rsp+48h] [rbp-38h]

  v8 = *(char **)(a2 + 40);
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  v40[0] = v9;
  v41 = v10;
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
    v39 = 0;
  }
  else
  {
    LOBYTE(v11) = sub_1F596B0((__int64)v40);
    v39 = v12;
    v6 = v11;
  }
  v13 = *(unsigned __int64 **)(a2 + 32);
  LOBYTE(v6) = v11;
  v14 = v13[5];
  v15 = *(_BYTE *)(v14 + 88);
  v16 = *(_QWORD *)(v14 + 96);
  LOBYTE(v42) = v15;
  v43 = v16;
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
        LOBYTE(v17) = 2;
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
        LOBYTE(v17) = 3;
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
        LOBYTE(v17) = 4;
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
        LOBYTE(v17) = 5;
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
        LOBYTE(v17) = 6;
        break;
      case 55:
        LOBYTE(v17) = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        LOBYTE(v17) = 8;
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
        LOBYTE(v17) = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        LOBYTE(v17) = 10;
        break;
      default:
        *(_QWORD *)(a1 + 144) = 0;
        BUG();
    }
    v19 = 0;
  }
  else
  {
    LOBYTE(v17) = sub_1F596B0((__int64)&v42);
    v19 = v18;
    v13 = *(unsigned __int64 **)(a2 + 32);
    v5 = v17;
  }
  v34 = v19;
  LOBYTE(v5) = v17;
  v20 = sub_2032580(a1, *v13, v13[1]);
  v21 = *(__int64 **)(a1 + 8);
  v38 = v22;
  v37 = v20;
  v25 = sub_1D2EF30(v21, v5, v34, v23, v34, v24);
  v27 = *(_QWORD *)(a2 + 72);
  v28 = v25;
  v29 = v26;
  v42 = v27;
  if ( v27 )
  {
    v36 = v26;
    v35 = v25;
    sub_1623A60((__int64)&v42, v27, 2);
    v28 = v35;
    v29 = v36;
  }
  v30 = *(unsigned __int16 *)(a2 + 24);
  *((_QWORD *)&v33 + 1) = v29;
  *(_QWORD *)&v33 = v28;
  LODWORD(v43) = *(_DWORD *)(a2 + 64);
  v31 = sub_1D332F0(v21, v30, (__int64)&v42, v6, v39, 0, a3, a4, a5, v37, v38, v33);
  if ( v42 )
    sub_161E7C0((__int64)&v42, v42);
  return v31;
}
