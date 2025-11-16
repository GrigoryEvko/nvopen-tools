// Function: sub_2032ED0
// Address: 0x2032ed0
//
__int64 *__fastcall sub_2032ED0(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  unsigned int v5; // r12d
  __int64 v6; // rcx
  char *v7; // rdx
  char v8; // al
  __int64 v9; // rdx
  unsigned int v10; // eax
  const void **v11; // rdx
  const void **v12; // r8
  __int64 v13; // rax
  const void **v14; // r8
  __int64 v15; // r14
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // r15
  __int64 *v18; // r10
  __int64 v19; // rcx
  __int64 *v20; // r14
  __int64 v22; // [rsp+8h] [rbp-58h]
  const void **v23; // [rsp+10h] [rbp-50h]
  __int64 v24; // [rsp+18h] [rbp-48h]
  __int64 *v25; // [rsp+18h] [rbp-48h]
  __int64 v26; // [rsp+20h] [rbp-40h] BYREF
  __int64 v27; // [rsp+28h] [rbp-38h]

  v6 = a1;
  v7 = *(char **)(a2 + 40);
  v8 = *v7;
  v9 = *((_QWORD *)v7 + 1);
  LOBYTE(v26) = v8;
  v27 = v9;
  if ( v8 )
  {
    switch ( v8 )
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
        LOBYTE(v10) = 2;
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
        LOBYTE(v10) = 3;
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
        LOBYTE(v10) = 4;
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
        LOBYTE(v10) = 5;
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
        LOBYTE(v10) = 6;
        break;
      case 55:
        LOBYTE(v10) = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        LOBYTE(v10) = 8;
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
        LOBYTE(v10) = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        LOBYTE(v10) = 10;
        break;
    }
    v12 = 0;
  }
  else
  {
    LOBYTE(v10) = sub_1F596B0((__int64)&v26);
    v6 = a1;
    v5 = v10;
    v12 = v11;
  }
  LOBYTE(v5) = v10;
  v23 = v12;
  v24 = v6;
  v13 = sub_2032580(v6, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v14 = v23;
  v15 = v13;
  v17 = v16;
  v18 = *(__int64 **)(v24 + 8);
  v19 = *(_QWORD *)(a2 + 32);
  v26 = *(_QWORD *)(a2 + 72);
  if ( v26 )
  {
    v22 = v19;
    v25 = v18;
    sub_1623A60((__int64)&v26, v26, 2);
    v19 = v22;
    v14 = v23;
    v18 = v25;
  }
  LODWORD(v27) = *(_DWORD *)(a2 + 64);
  v20 = sub_1D332F0(v18, 154, (__int64)&v26, v5, v14, 0, a3, a4, a5, v15, v17, *(_OWORD *)(v19 + 40));
  if ( v26 )
    sub_161E7C0((__int64)&v26, v26);
  return v20;
}
