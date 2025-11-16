// Function: sub_21394A0
// Address: 0x21394a0
//
__int64 *__fastcall sub_21394A0(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  __int64 *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rsi
  __int64 *v10; // r12
  __int64 v11; // r13
  __int64 *v12; // r8
  unsigned int v13; // ecx
  __int64 *v14; // rdi
  __int64 v15; // rsi
  __int64 v16; // rax
  char v17; // dl
  __int64 v18; // rax
  const void **v19; // r8
  __int64 v20; // rax
  char v21; // bl
  __int64 v22; // r12
  unsigned int v23; // edx
  __int64 *v24; // r15
  unsigned __int64 v25; // r13
  const void **v26; // rdx
  int v27; // ebx
  char v28; // r8
  const void **v29; // rdx
  int v30; // eax
  __int128 v31; // rax
  __int64 *v32; // r12
  bool v34; // al
  __int128 v35; // [rsp-10h] [rbp-A0h]
  __int64 *v36; // [rsp+0h] [rbp-90h]
  unsigned int v37; // [rsp+8h] [rbp-88h]
  __int64 v38; // [rsp+20h] [rbp-70h] BYREF
  int v39; // [rsp+28h] [rbp-68h]
  char v40[8]; // [rsp+30h] [rbp-60h] BYREF
  const void **v41; // [rsp+38h] [rbp-58h]
  unsigned int v42; // [rsp+40h] [rbp-50h] BYREF
  const void **v43; // [rsp+48h] [rbp-48h]
  char v44[8]; // [rsp+50h] [rbp-40h] BYREF
  const void **v45; // [rsp+58h] [rbp-38h]

  v7 = sub_2139210(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL), a3, a4, a5);
  v9 = *(_QWORD *)(a2 + 72);
  v10 = v7;
  v11 = v8;
  v12 = v7;
  v13 = v8;
  v38 = v9;
  if ( v9 )
  {
    v37 = v8;
    v36 = v7;
    sub_1623A60((__int64)&v38, v9, 2);
    v12 = v36;
    v13 = v37;
  }
  v14 = *(__int64 **)(a1 + 8);
  v15 = *(unsigned __int16 *)(a2 + 24);
  v39 = *(_DWORD *)(a2 + 64);
  v16 = *(_QWORD *)(a2 + 40);
  v17 = *(_BYTE *)v16;
  v41 = *(const void ***)(v16 + 8);
  v18 = v12[5] + 16LL * v13;
  v40[0] = v17;
  v19 = *(const void ***)(v18 + 8);
  *((_QWORD *)&v35 + 1) = v11;
  *(_QWORD *)&v35 = v10;
  LOBYTE(v42) = *(_BYTE *)v18;
  v43 = v19;
  v20 = sub_1D309E0(v14, v15, (__int64)&v38, v42, v19, 0, *(double *)a3.m128i_i64, a4, *(double *)a5.m128i_i64, v35);
  v21 = v42;
  v22 = v20;
  v24 = *(__int64 **)(a1 + 8);
  v25 = v23 | v11 & 0xFFFFFFFF00000000LL;
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
          v21 = 3;
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
          v21 = 4;
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
          v21 = 5;
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
          v21 = 6;
          break;
        case 55:
          v21 = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v21 = 8;
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
          v21 = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v21 = 10;
          break;
        default:
          v21 = 2;
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
  v21 = sub_1F596B0((__int64)&v42);
LABEL_6:
  v44[0] = v21;
  v45 = v26;
  if ( !v21 )
  {
    v27 = sub_1F58D40((__int64)v44);
    goto LABEL_8;
  }
LABEL_21:
  v27 = sub_2127930(v21);
LABEL_8:
  v28 = v40[0];
  if ( v40[0] )
  {
    if ( (unsigned __int8)(v40[0] - 14) <= 0x5Fu )
    {
      switch ( v40[0] )
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
  v34 = sub_1F58D20((__int64)v40);
  v28 = 0;
  if ( !v34 )
  {
LABEL_10:
    v29 = v41;
    goto LABEL_11;
  }
  v28 = sub_1F596B0((__int64)v40);
LABEL_11:
  v44[0] = v28;
  v45 = v29;
  if ( !v28 )
  {
    v30 = sub_1F58D40((__int64)v44);
    goto LABEL_13;
  }
LABEL_17:
  v30 = sub_2127930(v28);
LABEL_13:
  *(_QWORD *)&v31 = sub_1D38BB0((__int64)v24, (unsigned int)(v27 - v30), (__int64)&v38, v42, v43, 0, a3, a4, a5, 0);
  v32 = sub_1D332F0(v24, 53, (__int64)&v38, v42, v43, 0, *(double *)a3.m128i_i64, a4, a5, v22, v25, v31);
  if ( v38 )
    sub_161E7C0((__int64)&v38, v38);
  return v32;
}
