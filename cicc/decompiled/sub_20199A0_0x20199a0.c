// Function: sub_20199A0
// Address: 0x20199a0
//
__int64 *__fastcall sub_20199A0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __m128i a5,
        double a6,
        __m128i a7,
        __int64 a8,
        __int64 a9)
{
  unsigned __int8 *v10; // rax
  __int64 v11; // r8
  const void **v12; // rax
  __int64 v14; // rax
  __int64 *result; // rax
  __int64 v16; // rsi
  __int64 v17; // rax
  char v18; // r9
  const void **v19; // rax
  bool v20; // al
  const void **v21; // rdx
  int v22; // eax
  char v23; // r9
  const void **v24; // rdx
  int v25; // eax
  __int64 v26; // r10
  __int64 v27; // rax
  __int64 v28; // rdx
  unsigned __int64 v29; // r13
  __int64 *v30; // rax
  unsigned int v31; // edx
  __int64 v32; // rax
  const void **v33; // rax
  bool v34; // al
  char v35; // al
  __int128 v36; // [rsp-20h] [rbp-B0h]
  char v37; // [rsp+7h] [rbp-89h]
  char v38; // [rsp+8h] [rbp-88h]
  char v39; // [rsp+8h] [rbp-88h]
  int v40; // [rsp+10h] [rbp-80h]
  __int128 v41; // [rsp+10h] [rbp-80h]
  __int64 *v42; // [rsp+10h] [rbp-80h]
  unsigned int v43; // [rsp+20h] [rbp-70h] BYREF
  const void **v44; // [rsp+28h] [rbp-68h]
  __int64 v45; // [rsp+30h] [rbp-60h] BYREF
  int v46; // [rsp+38h] [rbp-58h]
  char v47[8]; // [rsp+40h] [rbp-50h] BYREF
  const void **v48; // [rsp+48h] [rbp-48h]
  char v49[8]; // [rsp+50h] [rbp-40h] BYREF
  const void **v50; // [rsp+58h] [rbp-38h]

  v10 = (unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL * (unsigned int)a3);
  v11 = *v10;
  v12 = (const void **)*((_QWORD *)v10 + 1);
  LOBYTE(v43) = v11;
  v44 = v12;
  if ( !(_BYTE)v11 )
    return sub_1D40890(*(__int64 **)a1, a2, 0, a4, v11, a9, a5, a6, a7);
  v14 = *(_QWORD *)(a1 + 8) + 259LL * (unsigned __int8)v11;
  if ( *(_BYTE *)(v14 + 2545) == 2 || *(_BYTE *)(v14 + 2544) == 2 )
    return sub_1D40890(*(__int64 **)a1, a2, 0, a4, v11, a9, a5, a6, a7);
  v16 = *(_QWORD *)(a2 + 72);
  v45 = v16;
  if ( !v16 )
  {
    v46 = *(_DWORD *)(a2 + 64);
    v32 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL);
    v18 = *(_BYTE *)(v32 + 88);
    v33 = *(const void ***)(v32 + 96);
    v47[0] = v18;
    v48 = v33;
LABEL_19:
    if ( (unsigned __int8)(v11 - 14) <= 0x5Fu )
    {
      switch ( (char)v11 )
      {
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
        default:
          LOBYTE(v11) = 2;
          break;
      }
      goto LABEL_24;
    }
    goto LABEL_9;
  }
  sub_1623A60((__int64)&v45, v16, 2);
  LOBYTE(v11) = v43;
  v46 = *(_DWORD *)(a2 + 64);
  v17 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL);
  v18 = *(_BYTE *)(v17 + 88);
  v19 = *(const void ***)(v17 + 96);
  v47[0] = v18;
  v48 = v19;
  if ( (_BYTE)v43 )
    goto LABEL_19;
  v38 = v18;
  v20 = sub_1F58D20((__int64)&v43);
  v18 = v38;
  LOBYTE(v11) = 0;
  if ( !v20 )
  {
LABEL_9:
    v21 = v44;
    goto LABEL_10;
  }
  v35 = sub_1F596B0((__int64)&v43);
  v18 = v47[0];
  LOBYTE(v11) = v35;
LABEL_10:
  v49[0] = v11;
  v50 = v21;
  if ( !(_BYTE)v11 )
  {
    v39 = v18;
    v22 = sub_1F58D40((__int64)v49);
    v23 = v39;
    v40 = v22;
    if ( v39 )
      goto LABEL_12;
    goto LABEL_25;
  }
LABEL_24:
  v40 = sub_2018C90(v11);
  if ( v23 )
  {
LABEL_12:
    if ( (unsigned __int8)(v23 - 14) <= 0x5Fu )
    {
      switch ( v23 )
      {
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
          v23 = 3;
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
          v23 = 4;
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
          v23 = 5;
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
          v23 = 6;
          break;
        case 55:
          v23 = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v23 = 8;
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
          v23 = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v23 = 10;
          break;
        default:
          v23 = 2;
          break;
      }
      goto LABEL_22;
    }
    goto LABEL_13;
  }
LABEL_25:
  v37 = v23;
  v34 = sub_1F58D20((__int64)v47);
  v23 = v37;
  if ( !v34 )
  {
LABEL_13:
    v24 = v48;
    goto LABEL_14;
  }
  v23 = sub_1F596B0((__int64)v47);
LABEL_14:
  v49[0] = v23;
  v50 = v24;
  if ( !v23 )
  {
    v25 = sub_1F58D40((__int64)v49);
    goto LABEL_16;
  }
LABEL_22:
  v25 = sub_2018C90(v23);
LABEL_16:
  v26 = sub_1D38BB0(*(_QWORD *)a1, (unsigned int)(v40 - v25), (__int64)&v45, v43, v44, 0, a5, a6, a7, 0);
  v27 = *(_QWORD *)(a2 + 32);
  *(_QWORD *)&v41 = v26;
  *((_QWORD *)&v36 + 1) = v28;
  *(_QWORD *)&v36 = v26;
  *((_QWORD *)&v41 + 1) = v28;
  v29 = *(unsigned int *)(v27 + 8) | a3 & 0xFFFFFFFF00000000LL;
  v30 = sub_1D332F0(
          *(__int64 **)a1,
          122,
          (__int64)&v45,
          v43,
          v44,
          0,
          *(double *)a5.m128i_i64,
          a6,
          a7,
          *(_QWORD *)v27,
          v29,
          v36);
  result = sub_1D332F0(
             *(__int64 **)a1,
             123,
             (__int64)&v45,
             v43,
             v44,
             0,
             *(double *)a5.m128i_i64,
             a6,
             a7,
             (__int64)v30,
             v31 | v29 & 0xFFFFFFFF00000000LL,
             v41);
  if ( v45 )
  {
    v42 = result;
    sub_161E7C0((__int64)&v45, v45);
    return v42;
  }
  return result;
}
