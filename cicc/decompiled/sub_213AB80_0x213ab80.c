// Function: sub_213AB80
// Address: 0x213ab80
//
__int64 *__fastcall sub_213AB80(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // r12
  __int64 v8; // rdx
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rax
  char v13; // r15
  const void **v14; // rax
  const void **v15; // rdx
  int v16; // r15d
  char v17; // r8
  const void **v18; // rdx
  int v19; // eax
  __int64 *v20; // r15
  __int64 v21; // rax
  __int64 v22; // rax
  const void **v23; // rdx
  __int128 v24; // rax
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  __int64 *v27; // r12
  bool v29; // al
  __int128 v30; // [rsp-20h] [rbp-B0h]
  __int64 v31; // [rsp+8h] [rbp-88h]
  unsigned int v33; // [rsp+10h] [rbp-80h]
  __int128 v34; // [rsp+10h] [rbp-80h]
  char v35[8]; // [rsp+20h] [rbp-70h] BYREF
  const void **v36; // [rsp+28h] [rbp-68h]
  unsigned int v37; // [rsp+30h] [rbp-60h] BYREF
  const void **v38; // [rsp+38h] [rbp-58h]
  __int64 v39; // [rsp+40h] [rbp-50h] BYREF
  int v40; // [rsp+48h] [rbp-48h]
  char v41[8]; // [rsp+50h] [rbp-40h] BYREF
  const void **v42; // [rsp+58h] [rbp-38h]

  v5 = sub_2138AD0(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v6 = a2;
  v7 = v5;
  v9 = v8;
  v10 = *(_QWORD *)(a2 + 40);
  v11 = *(_QWORD *)(a2 + 72);
  LOBYTE(v8) = *(_BYTE *)v10;
  v36 = *(const void ***)(v10 + 8);
  v12 = *(_QWORD *)(v7 + 40) + 16LL * (unsigned int)v9;
  v35[0] = v8;
  v13 = *(_BYTE *)v12;
  v14 = *(const void ***)(v12 + 8);
  v39 = v11;
  LOBYTE(v37) = v13;
  v38 = v14;
  if ( v11 )
  {
    sub_1623A60((__int64)&v39, v11, 2);
    v13 = v37;
    v6 = a2;
  }
  v40 = *(_DWORD *)(v6 + 64);
  if ( v13 )
  {
    if ( (unsigned __int8)(v13 - 14) <= 0x5Fu )
    {
      switch ( v13 )
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
          v13 = 3;
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
          v13 = 4;
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
          v13 = 5;
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
          v13 = 6;
          break;
        case 55:
          v13 = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v13 = 8;
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
          v13 = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v13 = 10;
          break;
        default:
          v13 = 2;
          break;
      }
      goto LABEL_21;
    }
    goto LABEL_5;
  }
  if ( !sub_1F58D20((__int64)&v37) )
  {
LABEL_5:
    v15 = v38;
    goto LABEL_6;
  }
  v13 = sub_1F596B0((__int64)&v37);
LABEL_6:
  v41[0] = v13;
  v42 = v15;
  if ( !v13 )
  {
    v16 = sub_1F58D40((__int64)v41);
    goto LABEL_8;
  }
LABEL_21:
  v16 = sub_2127930(v13);
LABEL_8:
  v17 = v35[0];
  if ( v35[0] )
  {
    if ( (unsigned __int8)(v35[0] - 14) <= 0x5Fu )
    {
      switch ( v35[0] )
      {
        case 0x18:
        case 0x19:
        case 0x1A:
        case 0x1B:
        case 0x1C:
        case 0x1D:
        case 0x1E:
        case 0x1F:
        case 0x20:
        case 0x3E:
        case 0x3F:
        case 0x40:
        case 0x41:
        case 0x42:
        case 0x43:
          v17 = 3;
          break;
        case 0x21:
        case 0x22:
        case 0x23:
        case 0x24:
        case 0x25:
        case 0x26:
        case 0x27:
        case 0x28:
        case 0x44:
        case 0x45:
        case 0x46:
        case 0x47:
        case 0x48:
        case 0x49:
          v17 = 4;
          break;
        case 0x29:
        case 0x2A:
        case 0x2B:
        case 0x2C:
        case 0x2D:
        case 0x2E:
        case 0x2F:
        case 0x30:
        case 0x4A:
        case 0x4B:
        case 0x4C:
        case 0x4D:
        case 0x4E:
        case 0x4F:
          v17 = 5;
          break;
        case 0x31:
        case 0x32:
        case 0x33:
        case 0x34:
        case 0x35:
        case 0x36:
        case 0x50:
        case 0x51:
        case 0x52:
        case 0x53:
        case 0x54:
        case 0x55:
          v17 = 6;
          break;
        case 0x37:
          v17 = 7;
          break;
        case 0x56:
        case 0x57:
        case 0x58:
        case 0x62:
        case 0x63:
        case 0x64:
          v17 = 8;
          break;
        case 0x59:
        case 0x5A:
        case 0x5B:
        case 0x5C:
        case 0x5D:
        case 0x65:
        case 0x66:
        case 0x67:
        case 0x68:
        case 0x69:
          v17 = 9;
          break;
        case 0x5E:
        case 0x5F:
        case 0x60:
        case 0x61:
        case 0x6A:
        case 0x6B:
        case 0x6C:
        case 0x6D:
          v17 = 10;
          break;
        default:
          v17 = 2;
          break;
      }
      goto LABEL_17;
    }
    goto LABEL_10;
  }
  v29 = sub_1F58D20((__int64)v35);
  v17 = 0;
  if ( !v29 )
  {
LABEL_10:
    v18 = v36;
    goto LABEL_11;
  }
  v17 = sub_1F596B0((__int64)v35);
LABEL_11:
  v41[0] = v17;
  v42 = v18;
  if ( !v17 )
  {
    v19 = sub_1F58D40((__int64)v41);
    goto LABEL_13;
  }
LABEL_17:
  v19 = sub_2127930(v17);
LABEL_13:
  v33 = v16 - v19;
  v20 = *(__int64 **)(a1 + 8);
  v31 = *(_QWORD *)a1;
  v21 = sub_1E0A0C0(v20[4]);
  v22 = sub_1F40B60(v31, v37, (__int64)v38, v21, 1);
  *(_QWORD *)&v24 = sub_1D38BB0((__int64)v20, v33, (__int64)&v39, v22, v23, 0, a3, a4, a5, 0);
  *((_QWORD *)&v30 + 1) = v9;
  *(_QWORD *)&v30 = v7;
  v34 = v24;
  v25 = sub_1D309E0(
          *(__int64 **)(a1 + 8),
          131,
          (__int64)&v39,
          v37,
          v38,
          0,
          *(double *)a3.m128i_i64,
          a4,
          *(double *)a5.m128i_i64,
          v30);
  v27 = sub_1D332F0(v20, 124, (__int64)&v39, v37, v38, 0, *(double *)a3.m128i_i64, a4, a5, v25, v26, v34);
  if ( v39 )
    sub_161E7C0((__int64)&v39, v39);
  return v27;
}
