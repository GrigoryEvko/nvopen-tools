// Function: sub_F94F50
// Address: 0xf94f50
//
__int64 __fastcall sub_F94F50(__int64 a1, __int64 a2)
{
  __m128i *v3; // rdi
  __m128i v5; // xmm2
  __m128i v6; // xmm3
  unsigned __int64 v7; // rsi
  void (__fastcall *v8)(_BYTE *, _BYTE *, __int64); // rax
  int v9; // r12d
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  unsigned __int64 v13; // [rsp+0h] [rbp-120h]
  unsigned __int64 v14; // [rsp+0h] [rbp-120h]
  unsigned __int64 v15; // [rsp+10h] [rbp-110h]
  _BYTE v16[16]; // [rsp+20h] [rbp-100h] BYREF
  void (__fastcall *v17)(_BYTE *, _BYTE *, __int64); // [rsp+30h] [rbp-F0h]
  unsigned __int8 (__fastcall *v18)(_BYTE *, unsigned __int64); // [rsp+38h] [rbp-E8h]
  __m128i v19; // [rsp+40h] [rbp-E0h]
  __m128i v20; // [rsp+50h] [rbp-D0h]
  _BYTE v21[16]; // [rsp+60h] [rbp-C0h] BYREF
  void (__fastcall *v22)(_BYTE *, _BYTE *, __int64); // [rsp+70h] [rbp-B0h]
  __int64 v23; // [rsp+78h] [rbp-A8h]
  __m128i v24; // [rsp+80h] [rbp-A0h] BYREF
  __m128i v25; // [rsp+90h] [rbp-90h] BYREF
  _BYTE v26[16]; // [rsp+A0h] [rbp-80h] BYREF
  void (__fastcall *v27)(_BYTE *, _BYTE *, __int64); // [rsp+B0h] [rbp-70h]
  unsigned __int8 (__fastcall *v28)(_BYTE *, unsigned __int64); // [rsp+B8h] [rbp-68h]
  __m128i v29; // [rsp+C0h] [rbp-60h] BYREF
  __m128i v30; // [rsp+D0h] [rbp-50h] BYREF
  _BYTE v31[16]; // [rsp+E0h] [rbp-40h] BYREF
  void (__fastcall *v32)(_BYTE *, _BYTE *, __int64); // [rsp+F0h] [rbp-30h]
  __int64 v33; // [rsp+F8h] [rbp-28h]

  v3 = &v24;
  sub_AA72C0(&v24, a1, 0);
  v17 = 0;
  v13 = _mm_loadu_si128(&v24).m128i_u64[0];
  v15 = _mm_loadu_si128(&v25).m128i_u64[0];
  if ( v27 )
  {
    v3 = (__m128i *)v16;
    v27(v16, v26, 2);
    v18 = v28;
    v17 = v27;
  }
  v5 = _mm_loadu_si128(&v29);
  v6 = _mm_loadu_si128(&v30);
  v22 = 0;
  v19 = v5;
  v20 = v6;
  if ( !v32 )
  {
    v7 = v13;
    if ( v19.m128i_i64[0] != v13 )
      goto LABEL_5;
LABEL_23:
    if ( v17 )
      v17(v16, v16, 3);
    if ( v32 )
      v32(v31, v31, 3);
    if ( v27 )
      v27(v26, v26, 3);
    return 1;
  }
  v3 = (__m128i *)v21;
  v32(v21, v31, 2);
  v7 = v13;
  v23 = v33;
  v8 = v32;
  v22 = v32;
  if ( v13 == v19.m128i_i64[0] )
  {
LABEL_21:
    if ( v8 )
      v8(v21, v21, 3);
    goto LABEL_23;
  }
LABEL_5:
  v9 = 0;
  while ( 1 )
  {
    v10 = v7 - 24;
    if ( !v7 )
      v10 = 0;
    if ( a2 == v10 )
      goto LABEL_11;
    if ( v9 == 1 )
      break;
    v9 = 1;
LABEL_11:
    v7 = *(_QWORD *)(v7 + 8);
    v11 = 0;
    v14 = v7;
    if ( v7 != v15 )
    {
      while ( 1 )
      {
        if ( v7 )
          v7 -= 24LL;
        if ( !v17 )
          sub_4263D6(v3, v7, v11);
        v3 = (__m128i *)v16;
        if ( v18(v16, v7) )
          break;
        v7 = *(_QWORD *)(v14 + 8);
        v14 = v7;
        if ( v15 == v7 )
          goto LABEL_19;
      }
      v7 = v14;
    }
LABEL_19:
    if ( v19.m128i_i64[0] == v7 )
    {
      v8 = v22;
      goto LABEL_21;
    }
  }
  if ( v22 )
    v22(v21, v21, 3);
  if ( v17 )
    v17(v16, v16, 3);
  if ( v32 )
    v32(v31, v31, 3);
  if ( v27 )
    v27(v26, v26, 3);
  return 0;
}
