// Function: sub_2032C80
// Address: 0x2032c80
//
__int64 __fastcall sub_2032C80(_QWORD *a1, __int64 a2, double a3, double a4, double a5, __int64 a6, __int64 a7)
{
  unsigned __int64 *v8; // rax
  unsigned __int64 v9; // r12
  __int64 v10; // r13
  unsigned __int8 *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r14
  char *v14; // rdx
  char v15; // al
  __int64 v16; // rdx
  __int64 v17; // rax
  const void **v18; // rdx
  const void **v19; // r8
  __int64 v20; // rsi
  __int64 *v21; // rbx
  __int64 v22; // r12
  __int64 v24; // rax
  unsigned int v25; // edx
  bool v26; // al
  int v27; // eax
  __int128 v28; // [rsp-10h] [rbp-80h]
  __int64 v29; // [rsp+0h] [rbp-70h]
  const void **v30; // [rsp+8h] [rbp-68h]
  __int64 v31; // [rsp+8h] [rbp-68h]
  __int64 v32; // [rsp+8h] [rbp-68h]
  __int64 v33; // [rsp+10h] [rbp-60h]
  unsigned __int64 v34; // [rsp+18h] [rbp-58h]
  _BYTE v35[8]; // [rsp+20h] [rbp-50h] BYREF
  __int64 v36; // [rsp+28h] [rbp-48h]
  __int64 v37; // [rsp+30h] [rbp-40h] BYREF
  __int64 v38; // [rsp+38h] [rbp-38h]

  v8 = *(unsigned __int64 **)(a2 + 32);
  v9 = *v8;
  v10 = v8[1];
  v34 = *v8;
  v33 = *((unsigned int *)v8 + 2);
  v11 = (unsigned __int8 *)(*(_QWORD *)(*v8 + 40) + 16 * v33);
  v12 = *v11;
  v13 = *((_QWORD *)v11 + 1);
  v35[0] = v12;
  v36 = v13;
  if ( (_BYTE)v12 )
  {
    if ( (unsigned __int8)(v12 - 14) <= 0x5Fu
      && word_4305480[(unsigned __int8)(v12 - 14)] == 1
      && !*(_QWORD *)(*a1 + 8 * v12 + 120) )
    {
LABEL_13:
      v31 = a7;
      v24 = sub_2032580((__int64)a1, v9, v10);
      a7 = v31;
      v34 = v24;
      v33 = v25;
    }
  }
  else
  {
    v32 = a7;
    v26 = sub_1F58D20((__int64)v35);
    a7 = v32;
    if ( v26 )
    {
      v38 = v13;
      LOBYTE(v37) = 0;
      v27 = sub_1F58D30((__int64)&v37);
      a7 = v32;
      if ( v27 == 1 )
        goto LABEL_13;
    }
  }
  v14 = *(char **)(a2 + 40);
  v15 = *v14;
  v16 = *((_QWORD *)v14 + 1);
  LOBYTE(v37) = v15;
  v38 = v16;
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
    }
    v19 = 0;
  }
  else
  {
    LOBYTE(v17) = sub_1F596B0((__int64)&v37);
    a7 = v17;
    v19 = v18;
  }
  v20 = *(_QWORD *)(a2 + 72);
  v21 = (__int64 *)a1[1];
  LOBYTE(a7) = v17;
  v37 = v20;
  if ( v20 )
  {
    v29 = a7;
    v30 = v19;
    sub_1623A60((__int64)&v37, v20, 2);
    a7 = v29;
    v19 = v30;
  }
  LODWORD(v38) = *(_DWORD *)(a2 + 64);
  *((_QWORD *)&v28 + 1) = v10 & 0xFFFFFFFF00000000LL | v33;
  *(_QWORD *)&v28 = v34;
  v22 = sub_1D309E0(v21, 158, (__int64)&v37, a7, v19, 0, a3, a4, a5, v28);
  if ( v37 )
    sub_161E7C0((__int64)&v37, v37);
  return v22;
}
