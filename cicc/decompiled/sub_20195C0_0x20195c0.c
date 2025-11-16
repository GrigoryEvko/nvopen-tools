// Function: sub_20195C0
// Address: 0x20195c0
//
__int64 *__fastcall sub_20195C0(__int64 *a1, __int64 a2, __int64 a3, __m128i a4, double a5, __m128i a6)
{
  __int64 v6; // rcx
  unsigned int v7; // r15d
  __int64 v10; // rsi
  __int64 *v11; // rdi
  __int64 v12; // rax
  char v13; // dl
  const void **v14; // r9
  __int64 *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rsi
  __int64 v18; // rax
  char v19; // cl
  const void **v20; // rax
  __int64 v21; // rax
  char v22; // r15
  __int64 v23; // r12
  unsigned int v24; // edx
  unsigned __int64 v25; // r13
  const void **v26; // rdx
  int v27; // r15d
  char v28; // r8
  const void **v29; // rdx
  int v30; // eax
  __int128 v31; // rax
  __int64 *v32; // r15
  __int64 *v33; // rax
  unsigned __int64 v34; // rdx
  __int64 *v35; // r12
  bool v37; // al
  __int64 v38; // [rsp+10h] [rbp-90h]
  __int128 v39; // [rsp+10h] [rbp-90h]
  __int64 v40; // [rsp+30h] [rbp-70h] BYREF
  int v41; // [rsp+38h] [rbp-68h]
  unsigned int v42; // [rsp+40h] [rbp-60h] BYREF
  const void **v43; // [rsp+48h] [rbp-58h]
  char v44[8]; // [rsp+50h] [rbp-50h] BYREF
  const void **v45; // [rsp+58h] [rbp-48h]
  char v46[8]; // [rsp+60h] [rbp-40h] BYREF
  const void **v47; // [rsp+68h] [rbp-38h]

  v6 = a2;
  v7 = a3;
  v10 = *(_QWORD *)(a2 + 72);
  v40 = v10;
  if ( v10 )
  {
    v38 = v6;
    sub_1623A60((__int64)&v40, v10, 2);
    v6 = v38;
  }
  v11 = (__int64 *)*a1;
  v41 = *(_DWORD *)(v6 + 64);
  v12 = *(_QWORD *)(v6 + 40) + 16LL * v7;
  v13 = *(_BYTE *)v12;
  v14 = *(const void ***)(v12 + 8);
  v15 = *(__int64 **)(v6 + 32);
  v43 = v14;
  LOBYTE(v42) = v13;
  v16 = v15[1];
  v17 = *v15;
  v18 = *(_QWORD *)(*v15 + 40) + 16LL * *((unsigned int *)v15 + 2);
  v19 = *(_BYTE *)v18;
  v20 = *(const void ***)(v18 + 8);
  v44[0] = v19;
  v45 = v20;
  v21 = sub_1D327B0(v11, v17, v16, (__int64)&v40, v42, v14, *(double *)a4.m128i_i64, a5, *(double *)a6.m128i_i64);
  v22 = v42;
  v23 = v21;
  v25 = v24 | a3 & 0xFFFFFFFF00000000LL;
  if ( (_BYTE)v42 )
  {
    if ( (unsigned __int8)(v42 - 14) <= 0x5Fu )
    {
      switch ( (char)v42 )
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
          v22 = 3;
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
          v22 = 4;
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
          v22 = 5;
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
          v22 = 6;
          break;
        case 55:
          v22 = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v22 = 8;
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
          v22 = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v22 = 10;
          break;
        default:
          v22 = 2;
          break;
      }
      goto LABEL_21;
    }
    goto LABEL_5;
  }
  if ( !sub_1F58D20((__int64)&v42) )
  {
LABEL_5:
    v26 = v43;
    goto LABEL_6;
  }
  v22 = sub_1F596B0((__int64)&v42);
LABEL_6:
  v46[0] = v22;
  v47 = v26;
  if ( !v22 )
  {
    v27 = sub_1F58D40((__int64)v46);
    goto LABEL_8;
  }
LABEL_21:
  v27 = sub_2018C90(v22);
LABEL_8:
  v28 = v44[0];
  if ( v44[0] )
  {
    if ( (unsigned __int8)(v44[0] - 14) <= 0x5Fu )
    {
      switch ( v44[0] )
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
          v28 = 3;
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
          v28 = 4;
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
          v28 = 5;
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
          v28 = 6;
          break;
        case 0x37:
          v28 = 7;
          break;
        case 0x56:
        case 0x57:
        case 0x58:
        case 0x62:
        case 0x63:
        case 0x64:
          v28 = 8;
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
          v28 = 9;
          break;
        case 0x5E:
        case 0x5F:
        case 0x60:
        case 0x61:
        case 0x6A:
        case 0x6B:
        case 0x6C:
        case 0x6D:
          v28 = 10;
          break;
        default:
          v28 = 2;
          break;
      }
      goto LABEL_17;
    }
    goto LABEL_10;
  }
  v37 = sub_1F58D20((__int64)v44);
  v28 = 0;
  if ( !v37 )
  {
LABEL_10:
    v29 = v45;
    goto LABEL_11;
  }
  v28 = sub_1F596B0((__int64)v44);
LABEL_11:
  v46[0] = v28;
  v47 = v29;
  if ( !v28 )
  {
    v30 = sub_1F58D40((__int64)v46);
    goto LABEL_13;
  }
LABEL_17:
  v30 = sub_2018C90(v28);
LABEL_13:
  *(_QWORD *)&v31 = sub_1D38BB0(*a1, (unsigned int)(v27 - v30), (__int64)&v40, v42, v43, 0, a4, a5, a6, 0);
  v32 = (__int64 *)*a1;
  v39 = v31;
  v33 = sub_1D332F0((__int64 *)*a1, 122, (__int64)&v40, v42, v43, 0, *(double *)a4.m128i_i64, a5, a6, v23, v25, v31);
  v35 = sub_1D332F0(v32, 123, (__int64)&v40, v42, v43, 0, *(double *)a4.m128i_i64, a5, a6, (__int64)v33, v34, v39);
  if ( v40 )
    sub_161E7C0((__int64)&v40, v40);
  return v35;
}
