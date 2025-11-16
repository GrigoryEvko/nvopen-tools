// Function: sub_1D364E0
// Address: 0x1d364e0
//
_QWORD *__fastcall sub_1D364E0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        const void **a4,
        unsigned __int8 a5,
        double a6,
        double a7,
        __m128i a8)
{
  unsigned __int8 v8; // r12
  unsigned __int8 v9; // bl
  const void **v10; // r15
  unsigned int v11; // r14d
  __m128i v12; // xmm0
  _QWORD *v13; // r12
  __int16 *v15; // r8
  const void **v16; // rdx
  __int16 *v17; // rax
  __int64 v18; // r14
  __int64 v19; // rsi
  __int64 v20; // rbx
  __int64 v21; // rsi
  __int64 v22; // rbx
  __int64 v23; // rsi
  __int64 v24; // rbx
  __int16 *v25; // [rsp+8h] [rbp-A8h]
  __int16 *v26; // [rsp+10h] [rbp-A0h]
  __int16 *v27; // [rsp+10h] [rbp-A0h]
  void *v28; // [rsp+18h] [rbp-98h]
  __int64 v30; // [rsp+30h] [rbp-80h] BYREF
  const void **v31; // [rsp+38h] [rbp-78h]
  __int64 v32[4]; // [rsp+40h] [rbp-70h] BYREF
  _BYTE v33[8]; // [rsp+60h] [rbp-50h] BYREF
  void *v34; // [rsp+68h] [rbp-48h] BYREF
  __int64 v35; // [rsp+70h] [rbp-40h]

  v8 = a3;
  v9 = a5;
  v30 = a3;
  v31 = a4;
  if ( (_BYTE)a3 )
  {
    if ( (unsigned __int8)(a3 - 14) <= 0x5Fu )
    {
      switch ( (char)a3 )
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
          v9 = a5;
          v11 = 3;
          v10 = 0;
          v28 = sub_16982C0();
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
          v9 = a5;
          v11 = 4;
          v10 = 0;
          v28 = sub_16982C0();
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
          v9 = a5;
          v11 = 5;
          v10 = 0;
          v28 = sub_16982C0();
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
          v9 = a5;
          v11 = 6;
          v10 = 0;
          v28 = sub_16982C0();
          break;
        case 55:
          v9 = a5;
          v11 = 7;
          v10 = 0;
          v28 = sub_16982C0();
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v9 = a5;
          v11 = 8;
          v10 = 0;
          v28 = sub_16982C0();
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
          v9 = a5;
          v28 = sub_16982C0();
          goto LABEL_5;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v9 = a5;
          v28 = sub_16982C0();
          v15 = (__int16 *)sub_1698280();
          goto LABEL_9;
        default:
          v11 = 2;
          v10 = 0;
          v28 = sub_16982C0();
          break;
      }
      v15 = (__int16 *)sub_1698280();
      goto LABEL_15;
    }
    goto LABEL_3;
  }
  if ( !(unsigned __int8)sub_1F58D20(&v30) )
  {
LABEL_3:
    v10 = v31;
    goto LABEL_4;
  }
  v8 = sub_1F596B0(&v30);
  v10 = v16;
LABEL_4:
  v28 = sub_16982C0();
  v11 = v8;
  if ( v8 != 9 )
  {
    v15 = (__int16 *)sub_1698280();
    if ( v8 == 10 )
    {
LABEL_9:
      v27 = v15;
      sub_169D3F0((__int64)v32, a6);
      sub_169E320(&v34, v32, v27);
      sub_1698460((__int64)v32);
      v13 = sub_1D36490(a1, (__int64)v33, a2, v30, v31, v9, a6, a7, a8);
      if ( v34 != v28 )
        goto LABEL_6;
      v18 = v35;
      if ( !v35 )
        return v13;
      v21 = 32LL * *(_QWORD *)(v35 - 8);
      v22 = v35 + v21;
      if ( v35 != v35 + v21 )
      {
        do
        {
          v22 -= 32;
          sub_127D120((_QWORD *)(v22 + 8));
        }
        while ( v18 != v22 );
      }
LABEL_21:
      j_j_j___libc_free_0_0(v18 - 8);
      return v13;
    }
LABEL_15:
    v25 = v15;
    sub_169D3F0((__int64)v32, a6);
    sub_169E320(&v34, v32, v25);
    sub_1698460((__int64)v32);
    v17 = (__int16 *)sub_1D15FA0(v11, (__int64)v10);
    sub_16A3360((__int64)v33, v17, 0, (bool *)v32);
    v13 = sub_1D36490(a1, (__int64)v33, a2, v30, v31, v9, a6, a7, a8);
    if ( v34 != v28 )
      goto LABEL_6;
    v18 = v35;
    if ( !v35 )
      return v13;
    v23 = 32LL * *(_QWORD *)(v35 - 8);
    v24 = v35 + v23;
    if ( v35 != v35 + v23 )
    {
      do
      {
        v24 -= 32;
        sub_127D120((_QWORD *)(v24 + 8));
      }
      while ( v18 != v24 );
    }
    goto LABEL_21;
  }
LABEL_5:
  v12 = 0;
  v26 = (__int16 *)sub_1698270();
  *(float *)v12.m128i_i32 = a6;
  sub_169D3B0((__int64)v32, v12);
  sub_169E320(&v34, v32, v26);
  sub_1698460((__int64)v32);
  v13 = sub_1D36490(a1, (__int64)v33, a2, v30, v31, v9, *(double *)v12.m128i_i64, a7, a8);
  if ( v34 == v28 )
  {
    v18 = v35;
    if ( !v35 )
      return v13;
    v19 = 32LL * *(_QWORD *)(v35 - 8);
    v20 = v35 + v19;
    if ( v35 != v35 + v19 )
    {
      do
      {
        v20 -= 32;
        sub_127D120((_QWORD *)(v20 + 8));
      }
      while ( v18 != v20 );
    }
    goto LABEL_21;
  }
LABEL_6:
  sub_1698460((__int64)&v34);
  return v13;
}
