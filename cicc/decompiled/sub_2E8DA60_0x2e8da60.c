// Function: sub_2E8DA60
// Address: 0x2e8da60
//
__int64 __fastcall sub_2E8DA60(__int64 a1, unsigned int a2, int a3)
{
  __int64 v6; // r14
  bool v7; // zf
  _BYTE *v8; // rsi
  _BYTE *v9; // rdx
  __int64 (__fastcall **v10)(__m128i *, __m128i *, int); // r8
  __int64 (__fastcall **v11)(__m128i *, __m128i *, int); // rdi
  unsigned __int64 v12; // rsi
  __int64 v13; // rdx
  __int64 result; // rax
  __m128i v15; // [rsp+0h] [rbp-F0h] BYREF
  __int64 (__fastcall *v16)(__m128i *, __m128i *, int); // [rsp+10h] [rbp-E0h] BYREF
  bool (__fastcall *v17)(_DWORD *, __int64); // [rsp+18h] [rbp-D8h]
  void (__fastcall *v18)(__int64 (__fastcall **)(__m128i *, __m128i *, int), __int64 (__fastcall **)(__m128i *, __m128i *, int), __int64); // [rsp+20h] [rbp-D0h]
  unsigned __int8 (__fastcall *v19)(__int64 (__fastcall **)(__m128i *, __m128i *, int), __int64 (__fastcall **)(__m128i *, __m128i *, int)); // [rsp+28h] [rbp-C8h]
  _QWORD v20[2]; // [rsp+30h] [rbp-C0h] BYREF
  _QWORD v21[2]; // [rsp+40h] [rbp-B0h] BYREF
  void (__fastcall *v22)(_QWORD *, _QWORD *, __int64); // [rsp+50h] [rbp-A0h]
  __int64 v23; // [rsp+58h] [rbp-98h]
  __m128i v24; // [rsp+60h] [rbp-90h] BYREF
  __int64 (__fastcall *v25[2])(__m128i *, __m128i *, int); // [rsp+70h] [rbp-80h] BYREF
  void (__fastcall *v26)(__int64 (__fastcall **)(__m128i *, __m128i *, int), __int64 (__fastcall **)(__m128i *, __m128i *, int), __int64); // [rsp+80h] [rbp-70h]
  unsigned __int8 (__fastcall *v27)(__int64 (__fastcall **)(__m128i *, __m128i *, int), __int64 (__fastcall **)(__m128i *, __m128i *, int)); // [rsp+88h] [rbp-68h]
  __int64 v28; // [rsp+90h] [rbp-60h]
  __int64 v29; // [rsp+98h] [rbp-58h]
  _BYTE v30[16]; // [rsp+A0h] [rbp-50h] BYREF
  void (__fastcall *v31)(_QWORD *, _BYTE *, __int64); // [rsp+B0h] [rbp-40h]
  __int64 v32; // [rsp+B8h] [rbp-38h]

  v6 = sub_2E894A0(a1, a3);
  if ( *(_WORD *)(a1 + 68) == 14 )
    sub_2EAB3B0(*(_QWORD *)(a1 + 32) + 40LL, 0, 0);
  v15.m128i_i32[0] = a3;
  v21[0] = 0;
  v17 = sub_2E85490;
  v16 = (__int64 (__fastcall *)(__m128i *, __m128i *, int))sub_2E854D0;
  sub_2E854D0(v20, &v15, 2);
  v7 = *(_WORD *)(a1 + 68) == 14;
  v8 = *(_BYTE **)(a1 + 32);
  v21[1] = v17;
  v21[0] = v16;
  if ( v7 )
  {
    v9 = v8 + 40;
  }
  else
  {
    v9 = &v8[40 * (*(_DWORD *)(a1 + 40) & 0xFFFFFF)];
    v8 += 80;
  }
  sub_2E85EC0(&v24, v8, v9, (__int64)v20);
  if ( v21[0] )
    ((void (__fastcall *)(_QWORD *, _QWORD *, __int64))v21[0])(v20, v20, 3);
  if ( v16 )
    v16(&v15, &v15, 3);
  v18 = 0;
  v15 = v24;
  if ( v26 )
  {
    v26(&v16, (__int64 (__fastcall **)(__m128i *, __m128i *, int))v25, 2);
    v19 = v27;
    v18 = v26;
  }
  v22 = 0;
  v20[0] = v28;
  v20[1] = v29;
  if ( v31 )
  {
    v31(v21, v30, 2);
    v23 = v32;
    v22 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v31;
  }
LABEL_13:
  v10 = (__int64 (__fastcall **)(__m128i *, __m128i *, int))v15.m128i_i64[0];
LABEL_14:
  while ( (__int64 (__fastcall **)(__m128i *, __m128i *, int))v20[0] != v10 )
  {
    v11 = v10;
    v12 = a2;
    sub_2EAB4C0(v10, a2, 0);
    v10 = (__int64 (__fastcall **)(__m128i *, __m128i *, int))(v15.m128i_i64[0] + 40);
    v15.m128i_i64[0] = (__int64)v10;
    if ( v10 != (__int64 (__fastcall **)(__m128i *, __m128i *, int))v15.m128i_i64[1] )
    {
      while ( 1 )
      {
        if ( !v18 )
          sub_4263D6(v11, v12, v13);
        v12 = (unsigned __int64)v10;
        v11 = &v16;
        if ( v19(&v16, v10) )
          goto LABEL_13;
        v10 = (__int64 (__fastcall **)(__m128i *, __m128i *, int))(v15.m128i_i64[0] + 40);
        v15.m128i_i64[0] = (__int64)v10;
        if ( (__int64 (__fastcall **)(__m128i *, __m128i *, int))v15.m128i_i64[1] == v10 )
          goto LABEL_14;
      }
    }
  }
  if ( v22 )
    v22(v21, v21, 3);
  if ( v18 )
    v18(&v16, &v16, 3);
  if ( v31 )
    v31(v30, v30, 3);
  if ( v26 )
    v26(
      (__int64 (__fastcall **)(__m128i *, __m128i *, int))v25,
      (__int64 (__fastcall **)(__m128i *, __m128i *, int))v25,
      3);
  result = sub_2E891A0(a1);
  *(_QWORD *)(result + 24) = v6;
  return result;
}
