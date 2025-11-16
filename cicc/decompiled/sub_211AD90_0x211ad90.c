// Function: sub_211AD90
// Address: 0x211ad90
//
__int64 *__fastcall sub_211AD90(__int64 *a1, __int64 a2, unsigned int a3, double a4, double a5, __m128i a6)
{
  __int64 *v6; // rbx
  unsigned __int8 *v7; // rax
  unsigned __int8 v8; // r15
  __int64 v9; // rax
  __int64 *v10; // r13
  __int64 v11; // r12
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // r11
  __int64 v14; // r10
  __int64 v15; // rax
  char v16; // dl
  __int64 v17; // rax
  unsigned int v18; // eax
  const void **v19; // rdx
  const void **v20; // r8
  unsigned int v21; // ecx
  __int64 v22; // rsi
  __int64 v24; // [rsp+0h] [rbp-90h]
  unsigned __int64 v25; // [rsp+8h] [rbp-88h]
  unsigned int v26; // [rsp+18h] [rbp-78h]
  unsigned int v27; // [rsp+18h] [rbp-78h]
  __int64 v28; // [rsp+20h] [rbp-70h]
  __int64 v29; // [rsp+20h] [rbp-70h]
  const void **v30; // [rsp+20h] [rbp-70h]
  unsigned __int64 v31; // [rsp+28h] [rbp-68h]
  _BYTE v32[8]; // [rsp+30h] [rbp-60h] BYREF
  __int64 v33; // [rsp+38h] [rbp-58h]
  __int64 v34; // [rsp+40h] [rbp-50h] BYREF
  int v35; // [rsp+48h] [rbp-48h]
  __int64 v36; // [rsp+50h] [rbp-40h]

  v6 = (__int64 *)a2;
  v7 = (unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL * a3);
  v8 = *v7;
  v28 = *((_QWORD *)v7 + 1);
  sub_1F40D10((__int64)&v34, *a1, *(_QWORD *)(a1[1] + 48), *v7, v28);
  if ( v8 != (_BYTE)v35 )
    goto LABEL_2;
  if ( v28 == v36 )
  {
    if ( v8 )
    {
LABEL_11:
      if ( *(_QWORD *)(*a1 + 8LL * v8 + 120) )
        return v6;
    }
  }
  else if ( v8 )
  {
    goto LABEL_11;
  }
LABEL_2:
  v9 = sub_200D430(
         (__int64)a1,
         **(_QWORD **)(a2 + 32),
         *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
         a4,
         a5,
         *(double *)a6.m128i_i64);
  v10 = (__int64 *)a1[1];
  v11 = *(_QWORD *)(a2 + 32);
  v13 = v12;
  v14 = v9;
  v15 = *(_QWORD *)(v9 + 40) + 16LL * (unsigned int)v12;
  v16 = *(_BYTE *)v15;
  v17 = *(_QWORD *)(v15 + 8);
  v32[0] = v16;
  v33 = v17;
  if ( v16 )
  {
    switch ( v16 )
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
        LOBYTE(v18) = 2;
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
        LOBYTE(v18) = 3;
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
        LOBYTE(v18) = 4;
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
        LOBYTE(v18) = 5;
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
        LOBYTE(v18) = 6;
        break;
      case 55:
        LOBYTE(v18) = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        LOBYTE(v18) = 8;
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
        LOBYTE(v18) = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        LOBYTE(v18) = 10;
        break;
    }
    v20 = 0;
  }
  else
  {
    v29 = v14;
    v31 = v13;
    LOBYTE(v18) = sub_1F596B0((__int64)v32);
    v14 = v29;
    v13 = v31;
    v26 = v18;
    v20 = v19;
  }
  v21 = v26;
  v22 = *(_QWORD *)(a2 + 72);
  LOBYTE(v21) = v18;
  v34 = v22;
  v27 = v21;
  if ( v22 )
  {
    v30 = v20;
    v24 = v14;
    v25 = v13;
    sub_1623A60((__int64)&v34, v22, 2);
    v14 = v24;
    v13 = v25;
    v20 = v30;
  }
  v35 = *((_DWORD *)v6 + 16);
  v6 = sub_1D332F0(v10, 106, (__int64)&v34, v27, v20, 0, a4, a5, a6, v14, v13, *(_OWORD *)(v11 + 40));
  if ( v34 )
    sub_161E7C0((__int64)&v34, v34);
  return v6;
}
