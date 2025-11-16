// Function: sub_1B45E10
// Address: 0x1b45e10
//
__int64 __fastcall sub_1B45E10(_BYTE *a1)
{
  __m128i *v2; // rdi
  _BYTE *v3; // rsi
  __int64 v4; // rdx
  __int64 v5; // r14
  int v6; // r12d
  __int64 v7; // rax
  __int64 i; // r15
  _QWORD *v9; // rax
  __m128i v11; // [rsp+0h] [rbp-F0h]
  _BYTE v12[16]; // [rsp+10h] [rbp-E0h] BYREF
  void (__fastcall *v13)(_BYTE *, _BYTE *, __int64); // [rsp+20h] [rbp-D0h]
  unsigned __int8 (__fastcall *v14)(_BYTE *, __int64); // [rsp+28h] [rbp-C8h]
  __int64 v15; // [rsp+30h] [rbp-C0h]
  __int64 v16; // [rsp+38h] [rbp-B8h]
  _BYTE v17[16]; // [rsp+40h] [rbp-B0h] BYREF
  void (__fastcall *v18)(_BYTE *, _BYTE *, __int64); // [rsp+50h] [rbp-A0h]
  __int64 v19; // [rsp+58h] [rbp-98h]
  __m128i v20; // [rsp+60h] [rbp-90h] BYREF
  _BYTE v21[16]; // [rsp+70h] [rbp-80h] BYREF
  void (__fastcall *v22)(_BYTE *, _BYTE *, __int64); // [rsp+80h] [rbp-70h]
  unsigned __int8 (__fastcall *v23)(_BYTE *, __int64); // [rsp+88h] [rbp-68h]
  __int64 v24; // [rsp+90h] [rbp-60h]
  __int64 v25; // [rsp+98h] [rbp-58h]
  _BYTE v26[16]; // [rsp+A0h] [rbp-50h] BYREF
  void (__fastcall *v27)(_BYTE *, _BYTE *, __int64); // [rsp+B0h] [rbp-40h]
  __int64 v28; // [rsp+B8h] [rbp-38h]

  v2 = &v20;
  v3 = a1;
  sub_1580910(&v20);
  v13 = 0;
  v11 = v20;
  if ( v22 )
  {
    v3 = v21;
    v2 = (__m128i *)v12;
    v22(v12, v21, 2);
    v14 = v23;
    v13 = v22;
  }
  v18 = 0;
  v15 = v24;
  v16 = v25;
  if ( v27 )
  {
    v3 = v26;
    v2 = (__m128i *)v17;
    v27(v17, v26, 2);
    v19 = v28;
    v18 = v27;
  }
  v5 = v11.m128i_i64[0];
  v6 = 12;
  if ( v11.m128i_i64[0] == v15 )
  {
LABEL_32:
    sub_A17130((__int64)v17);
    sub_A17130((__int64)v12);
    sub_A17130((__int64)v26);
    sub_A17130((__int64)v21);
    return 1;
  }
  else
  {
    while ( 1 )
    {
      v7 = v5 - 24;
      if ( !v5 )
        v7 = 0;
      if ( !--v6 )
        break;
      for ( i = *(_QWORD *)(v7 + 8); i; i = *(_QWORD *)(i + 8) )
      {
        v2 = (__m128i *)i;
        v9 = sub_1648700(i);
        if ( a1 != (_BYTE *)v9[5] || *((_BYTE *)v9 + 16) == 77 )
          goto LABEL_14;
      }
      v5 = *(_QWORD *)(v5 + 8);
      v11.m128i_i64[0] = v5;
      if ( v5 != v11.m128i_i64[1] )
      {
        while ( 1 )
        {
          if ( v5 )
            v5 -= 24;
          if ( !v13 )
            sub_4263D6(v2, v3, v4);
          v3 = (_BYTE *)v5;
          v2 = (__m128i *)v12;
          if ( v14(v12, v5) )
            break;
          v5 = *(_QWORD *)(v11.m128i_i64[0] + 8);
          v11.m128i_i64[0] = v5;
          if ( v11.m128i_i64[1] == v5 )
            goto LABEL_31;
        }
        v5 = v11.m128i_i64[0];
      }
LABEL_31:
      if ( v15 == v5 )
        goto LABEL_32;
    }
LABEL_14:
    if ( v18 )
      v18(v17, v17, 3);
    if ( v13 )
      v13(v12, v12, 3);
    if ( v27 )
      v27(v26, v26, 3);
    if ( v22 )
      v22(v21, v21, 3);
    return 0;
  }
}
