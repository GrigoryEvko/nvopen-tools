// Function: sub_2145520
// Address: 0x2145520
//
__int64 __fastcall sub_2145520(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v5; // r13
  __int64 v7; // rsi
  __int64 *v8; // r15
  char *v9; // rax
  unsigned __int8 v10; // r14
  const void **v11; // r8
  __int64 v12; // rax
  unsigned __int8 v13; // dl
  __int64 v14; // rsi
  bool v15; // al
  _QWORD *v16; // r9
  bool v17; // al
  __int64 v18; // r12
  unsigned __int8 v20; // al
  const void **v21; // rdx
  const void **v22; // r8
  __int64 v23; // rsi
  char v24; // al
  unsigned int v25; // ecx
  char *v26; // r14
  char v27; // al
  __int64 v28; // rdx
  const void **v29; // rdx
  __int64 v30; // rax
  const void **v31; // rdx
  const void **v32; // rsi
  __int128 v33; // rax
  __int128 v34; // [rsp-20h] [rbp-120h]
  const void **v35; // [rsp+18h] [rbp-E8h]
  const void **v36; // [rsp+18h] [rbp-E8h]
  unsigned int v37; // [rsp+18h] [rbp-E8h]
  unsigned int v38; // [rsp+18h] [rbp-E8h]
  __int64 v39; // [rsp+20h] [rbp-E0h] BYREF
  int v40; // [rsp+28h] [rbp-D8h]
  __int64 v41; // [rsp+30h] [rbp-D0h] BYREF
  const void **v42; // [rsp+38h] [rbp-C8h]
  __int64 *v43; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v44; // [rsp+48h] [rbp-B8h]
  __int64 v45[22]; // [rsp+50h] [rbp-B0h] BYREF

  v7 = *(_QWORD *)(a2 + 72);
  v39 = v7;
  if ( v7 )
    sub_1623A60((__int64)&v39, v7, 2);
  v8 = *(__int64 **)(a2 + 32);
  v40 = *(_DWORD *)(a2 + 64);
  v9 = *(char **)(a2 + 40);
  v10 = *v9;
  v11 = (const void **)*((_QWORD *)v9 + 1);
  LOBYTE(v41) = v10;
  v42 = v11;
  if ( v10 )
  {
    if ( (unsigned __int8)(v10 - 14) <= 0x5Fu )
      goto LABEL_5;
LABEL_10:
    v18 = sub_200D7B0(a1, *v8, v8[1], v10, (__int64)v11);
    goto LABEL_11;
  }
  v36 = v11;
  v17 = sub_1F58D20((__int64)&v41);
  v11 = v36;
  if ( !v17 )
    goto LABEL_10;
LABEL_5:
  v12 = *(_QWORD *)(*v8 + 40) + 16LL * *((unsigned int *)v8 + 2);
  v13 = *(_BYTE *)v12;
  v14 = *(_QWORD *)(v12 + 8);
  LOBYTE(v43) = v13;
  v44 = v14;
  if ( v13 )
  {
    v15 = (unsigned __int8)(v13 - 14) <= 0x47u || (unsigned __int8)(v13 - 2) <= 5u;
  }
  else
  {
    v35 = v11;
    v15 = sub_1F58CF0((__int64)&v43);
    v13 = 0;
    v11 = v35;
  }
  if ( !v15 )
    goto LABEL_10;
  sub_1F40D10((__int64)&v43, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), v13, v14);
  v16 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 48LL);
  switch ( (char)v44 )
  {
    case 2:
      v20 = 15;
      goto LABEL_28;
    case 3:
      v20 = 25;
      goto LABEL_28;
    case 4:
      v20 = 34;
      goto LABEL_28;
    case 5:
      v20 = 42;
      goto LABEL_28;
    case 6:
      v20 = 50;
      goto LABEL_28;
    case 8:
      v20 = 86;
      goto LABEL_28;
    case 9:
      v20 = 90;
      goto LABEL_28;
    case 10:
      v20 = 95;
LABEL_28:
      v22 = 0;
      break;
    default:
      v20 = sub_1F593D0(v16, (unsigned __int8)v44, v45[0], 2u);
      v22 = v21;
      v16 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 48LL);
      break;
  }
  v23 = *(_QWORD *)a1;
  LOBYTE(v41) = v20;
  v42 = v22;
  sub_1F40D10((__int64)&v43, v23, (__int64)v16, v20, (__int64)v22);
  if ( (_BYTE)v43 )
  {
    v26 = *(char **)(a2 + 40);
    v27 = *v26;
    v28 = *((_QWORD *)v26 + 1);
    LOBYTE(v43) = v27;
    v44 = v28;
    if ( v27 )
      v25 = word_4310E40[(unsigned __int8)(v27 - 14)];
    else
      v25 = sub_1F58D30((__int64)&v43);
    v24 = *v26;
    v29 = (const void **)*((_QWORD *)v26 + 1);
    LOBYTE(v41) = *v26;
    v42 = v29;
  }
  else
  {
    v24 = v41;
    v25 = 2;
  }
  v43 = v45;
  v44 = 0x800000000LL;
  if ( v24 )
  {
    switch ( v24 )
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
        LOBYTE(v30) = 2;
        v32 = 0;
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
        LOBYTE(v30) = 3;
        v32 = 0;
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
        LOBYTE(v30) = 4;
        v32 = 0;
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
        LOBYTE(v30) = 5;
        v32 = 0;
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
        LOBYTE(v30) = 6;
        v32 = 0;
        break;
      case 55:
        LOBYTE(v30) = 7;
        v32 = 0;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        LOBYTE(v30) = 8;
        v32 = 0;
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
        LOBYTE(v30) = 9;
        v32 = 0;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        LOBYTE(v30) = 10;
        v32 = 0;
        break;
    }
  }
  else
  {
    v37 = v25;
    LOBYTE(v30) = sub_1F596B0((__int64)&v41);
    v25 = v37;
    v5 = v30;
    v32 = v31;
  }
  LOBYTE(v5) = v30;
  v38 = v25;
  sub_21453A0(
    a1,
    **(_QWORD **)(a2 + 32),
    *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
    v25,
    (__int64)&v43,
    a3,
    a4,
    a5,
    *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
    v5,
    v32);
  *((_QWORD *)&v34 + 1) = v38;
  *(_QWORD *)&v34 = v43;
  *(_QWORD *)&v33 = sub_1D359D0(*(__int64 **)(a1 + 8), 104, (__int64)&v39, v41, v42, 0, a3, a4, a5, v34);
  v18 = sub_1D309E0(
          *(__int64 **)(a1 + 8),
          158,
          (__int64)&v39,
          **(unsigned __int8 **)(a2 + 40),
          *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL),
          0,
          a3,
          a4,
          *(double *)a5.m128i_i64,
          v33);
  if ( v43 != v45 )
    _libc_free((unsigned __int64)v43);
LABEL_11:
  if ( v39 )
    sub_161E7C0((__int64)&v39, v39);
  return v18;
}
