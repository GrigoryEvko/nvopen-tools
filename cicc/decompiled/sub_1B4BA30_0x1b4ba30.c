// Function: sub_1B4BA30
// Address: 0x1b4ba30
//
__int64 sub_1B4BA30()
{
  unsigned int v0; // ebx
  unsigned int v1; // ecx
  __int64 v2; // rsi
  unsigned __int64 v3; // rdi
  unsigned __int8 v4; // al
  __int64 v5; // rdx
  unsigned __int8 v7; // [rsp+Fh] [rbp-F1h]
  unsigned __int8 v8; // [rsp+Fh] [rbp-F1h]
  unsigned __int8 v9; // [rsp+Fh] [rbp-F1h]
  __m128i v10; // [rsp+10h] [rbp-F0h]
  _BYTE v11[16]; // [rsp+20h] [rbp-E0h] BYREF
  void (__fastcall *v12)(_BYTE *, _BYTE *, __int64); // [rsp+30h] [rbp-D0h]
  unsigned __int8 (__fastcall *v13)(_BYTE *, __int64); // [rsp+38h] [rbp-C8h]
  __int64 v14; // [rsp+40h] [rbp-C0h]
  __int64 v15; // [rsp+48h] [rbp-B8h]
  _BYTE v16[16]; // [rsp+50h] [rbp-B0h] BYREF
  void (__fastcall *v17)(_BYTE *, _BYTE *, __int64); // [rsp+60h] [rbp-A0h]
  __m128i v18; // [rsp+70h] [rbp-90h] BYREF
  _BYTE v19[16]; // [rsp+80h] [rbp-80h] BYREF
  void (__fastcall *v20)(_BYTE *, _BYTE *, __int64); // [rsp+90h] [rbp-70h]
  __int64 v21; // [rsp+A0h] [rbp-60h]
  __int64 v22; // [rsp+A8h] [rbp-58h]
  _BYTE v23[16]; // [rsp+B0h] [rbp-50h] BYREF
  void (__fastcall *v24)(_BYTE *, _BYTE *, __int64); // [rsp+C0h] [rbp-40h]

  v0 = 0;
  sub_1580910(&v18);
  v10 = v18;
  sub_1974F30((__int64)v11, (__int64)v19);
  v14 = v21;
  v15 = v22;
  sub_1974F30((__int64)v16, (__int64)v23);
  v2 = v10.m128i_i64[0];
  if ( v10.m128i_i64[0] == v14 )
  {
LABEL_14:
    if ( v17 )
      v17(v16, v16, 3);
    if ( v12 )
      v12(v11, v11, 3);
    if ( v24 )
      v24(v23, v23, 3);
    if ( v20 )
      v20(v19, v19, 3);
    LOBYTE(v1) = dword_4FB7680 + 1 >= v0;
    return v1;
  }
  while ( 1 )
  {
    if ( !v2 )
      BUG();
    v3 = *(unsigned __int8 *)(v2 - 8);
    v4 = *(_BYTE *)(v2 - 8);
    v5 = (unsigned int)(v3 - 55);
    LOBYTE(v5) = (unsigned __int8)(v3 - 55) <= 1u;
    LOBYTE(v1) = v5 | ((unsigned int)(v3 - 35) <= 0x11);
    if ( (_BYTE)v1 )
    {
      ++v0;
      goto LABEL_5;
    }
    v5 = (unsigned int)(v3 - 25);
    if ( (unsigned int)v5 > 9 )
    {
      v3 = (unsigned int)(v3 - 24);
      if ( v4 <= 0x17u )
      {
        if ( v4 != 5 )
          break;
        v3 = *(unsigned __int16 *)(v2 - 6);
      }
      if ( (_DWORD)v3 != 47 || *(_BYTE *)(*(_QWORD *)(v2 - 24) + 8LL) != 15 )
        break;
    }
LABEL_5:
    v2 = *(_QWORD *)(v2 + 8);
    v10.m128i_i64[0] = v2;
    if ( v2 != v10.m128i_i64[1] )
    {
      while ( 1 )
      {
        if ( v2 )
          v2 -= 24;
        if ( !v12 )
          sub_4263D6(v3, v2, v5);
        v3 = (unsigned __int64)v11;
        if ( v13(v11, v2) )
          break;
        v2 = *(_QWORD *)(v10.m128i_i64[0] + 8);
        v10.m128i_i64[0] = v2;
        if ( v10.m128i_i64[1] == v2 )
          goto LABEL_13;
      }
      v2 = v10.m128i_i64[0];
    }
LABEL_13:
    if ( v14 == v2 )
      goto LABEL_14;
  }
  if ( v17 )
  {
    v17(v16, v16, 3);
    v1 = 0;
  }
  if ( v12 )
  {
    v7 = v1;
    v12(v11, v11, 3);
    v1 = v7;
  }
  if ( v24 )
  {
    v8 = v1;
    v24(v23, v23, 3);
    v1 = v8;
  }
  if ( v20 )
  {
    v9 = v1;
    v20(v19, v19, 3);
    return v9;
  }
  return v1;
}
