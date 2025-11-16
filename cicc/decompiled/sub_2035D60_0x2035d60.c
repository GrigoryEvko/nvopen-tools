// Function: sub_2035D60
// Address: 0x2035d60
//
__int64 __fastcall sub_2035D60(__int64 a1, __int64 a2, double a3, double a4, __m128i a5, __int64 a6, __int64 a7)
{
  __int64 v8; // rax
  __int64 *v9; // r11
  __int64 v10; // r12
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // r13
  char *v13; // rdx
  char v14; // al
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rax
  const void **v18; // rdx
  const void **v19; // r8
  __int64 v20; // rsi
  __int64 *v21; // r12
  __int64 v22; // rdx
  __int64 v23; // r13
  __int64 v24; // rsi
  __int64 *v25; // r10
  const void **v26; // r8
  unsigned int v27; // ebx
  __int64 v28; // r12
  __int128 v30; // [rsp-10h] [rbp-90h]
  __int64 v31; // [rsp+8h] [rbp-78h]
  const void **v32; // [rsp+10h] [rbp-70h]
  __int64 *v33; // [rsp+18h] [rbp-68h]
  __int64 *v35; // [rsp+20h] [rbp-60h]
  __int64 *v36; // [rsp+20h] [rbp-60h]
  __int64 v37; // [rsp+28h] [rbp-58h]
  const void **v38; // [rsp+28h] [rbp-58h]
  __int64 v39; // [rsp+30h] [rbp-50h] BYREF
  int v40; // [rsp+38h] [rbp-48h]
  __int64 v41; // [rsp+40h] [rbp-40h] BYREF
  __int64 v42; // [rsp+48h] [rbp-38h]

  v8 = sub_2032580(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v9 = *(__int64 **)(a1 + 8);
  v10 = v8;
  v12 = v11;
  v13 = *(char **)(a2 + 40);
  v37 = *(_QWORD *)(a2 + 32);
  v14 = *v13;
  v15 = *((_QWORD *)v13 + 1);
  LOBYTE(v41) = v14;
  v42 = v15;
  if ( v14 )
  {
    v16 = a7;
    switch ( v14 )
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
    v35 = v9;
    LOBYTE(v17) = sub_1F596B0((__int64)&v41);
    v9 = v35;
    v16 = v17;
    v19 = v18;
  }
  v20 = *(_QWORD *)(a2 + 72);
  LOBYTE(v16) = v17;
  v39 = v20;
  if ( v20 )
  {
    v31 = v16;
    v32 = v19;
    v33 = v9;
    sub_1623A60((__int64)&v39, v20, 2);
    v16 = v31;
    v19 = v32;
    v9 = v33;
  }
  v40 = *(_DWORD *)(a2 + 64);
  v21 = sub_1D332F0(v9, 154, (__int64)&v39, v16, v19, 0, a3, a4, a5, v10, v12, *(_OWORD *)(v37 + 40));
  v23 = v22;
  if ( v39 )
    sub_161E7C0((__int64)&v39, v39);
  v24 = *(_QWORD *)(a2 + 72);
  v25 = *(__int64 **)(a1 + 8);
  v26 = *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL);
  v27 = **(unsigned __int8 **)(a2 + 40);
  v41 = v24;
  if ( v24 )
  {
    v36 = v25;
    v38 = v26;
    sub_1623A60((__int64)&v41, v24, 2);
    v25 = v36;
    v26 = v38;
  }
  *((_QWORD *)&v30 + 1) = v23;
  *(_QWORD *)&v30 = v21;
  LODWORD(v42) = *(_DWORD *)(a2 + 64);
  v28 = sub_1D309E0(v25, 111, (__int64)&v41, v27, v26, 0, a3, a4, *(double *)a5.m128i_i64, v30);
  if ( v41 )
    sub_161E7C0((__int64)&v41, v41);
  return v28;
}
