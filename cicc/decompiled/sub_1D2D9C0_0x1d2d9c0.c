// Function: sub_1D2D9C0
// Address: 0x1d2d9c0
//
unsigned __int64 __fastcall sub_1D2D9C0(const __m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  __int64 v8; // rax
  int v9; // r8d
  int v10; // r9d
  __m128i v11; // xmm0
  __int64 v12; // rax
  __int64 v13; // rbx
  const __m128i *v14; // r13
  _BYTE **v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rbx
  __int64 v18; // r13
  __m128i v20; // [rsp+0h] [rbp-500h] BYREF
  __int64 v21; // [rsp+10h] [rbp-4F0h]
  __int64 v22; // [rsp+18h] [rbp-4E8h]
  __int64 v23; // [rsp+20h] [rbp-4E0h]
  __int64 v24; // [rsp+28h] [rbp-4D8h]
  __m128i v25; // [rsp+30h] [rbp-4D0h]
  _QWORD v26[7]; // [rsp+40h] [rbp-4C0h] BYREF
  __int64 v27; // [rsp+78h] [rbp-488h]
  int v28; // [rsp+80h] [rbp-480h]
  __int64 v29; // [rsp+88h] [rbp-478h]
  int v30; // [rsp+90h] [rbp-470h]
  __int64 v31; // [rsp+98h] [rbp-468h] BYREF
  __int64 v32; // [rsp+A0h] [rbp-460h]
  _QWORD *v33; // [rsp+A8h] [rbp-458h]
  __int64 v34; // [rsp+B0h] [rbp-450h]
  __int64 v35; // [rsp+B8h] [rbp-448h] BYREF
  _BYTE *v36; // [rsp+C0h] [rbp-440h] BYREF
  __int64 i; // [rsp+C8h] [rbp-438h]
  _BYTE v38[1072]; // [rsp+D0h] [rbp-430h] BYREF

  v7 = a1[11].m128i_i64[0];
  v20 = _mm_loadu_si128(a1 + 11);
  v8 = sub_1D274F0(1u, a3, a4, a5, a6);
  v33 = v26;
  v11 = _mm_load_si128(&v20);
  v26[5] = v8;
  v27 = 0x100000000LL;
  v25 = v11;
  v35 = 0;
  v26[6] = 0;
  v28 = 0;
  v34 = 0;
  v29 = 0;
  v30 = -65536;
  LODWORD(v32) = v11.m128i_i32[2];
  v31 = v11.m128i_i64[0];
  v12 = *(_QWORD *)(v7 + 48);
  memset(v26, 0, 24);
  v26[3] = -4294967084LL;
  v35 = v12;
  if ( v12 )
    *(_QWORD *)(v12 + 24) = &v35;
  v34 = v7 + 48;
  *(_QWORD *)(v7 + 48) = &v31;
  v13 = a1[12].m128i_i64[1];
  v14 = a1 + 12;
  v15 = &v36;
  v26[4] = &v31;
  LODWORD(v27) = 1;
  v36 = v38;
  for ( i = 0x8000000000LL; v14 != (const __m128i *)v13; v13 = *(_QWORD *)(v13 + 8) )
  {
    while ( 1 )
    {
      if ( !v13 )
        BUG();
      if ( !*(_QWORD *)(v13 + 40) )
        break;
      v13 = *(_QWORD *)(v13 + 8);
      if ( v14 == (const __m128i *)v13 )
        goto LABEL_11;
    }
    v16 = (unsigned int)i;
    if ( (unsigned int)i >= HIDWORD(i) )
    {
      v20.m128i_i64[0] = (__int64)v15;
      sub_16CD150((__int64)v15, v38, 0, 8, v9, v10);
      v16 = (unsigned int)i;
      v15 = (_BYTE **)v20.m128i_i64[0];
    }
    *(_QWORD *)&v36[8 * v16] = v13 - 8;
    LODWORD(i) = i + 1;
  }
LABEL_11:
  sub_1D2D860((__int64)a1, (__int64)&v36);
  v17 = v31;
  v18 = v32;
  if ( v31 )
  {
    nullsub_686();
    a1[11].m128i_i64[0] = v17;
    a1[11].m128i_i32[2] = v18;
    v23 = v17;
    v24 = v18;
    sub_1D23870();
  }
  else
  {
    v21 = 0;
    v22 = v32;
    a1[11].m128i_i64[0] = 0;
    a1[11].m128i_i32[2] = v18;
  }
  if ( v36 != v38 )
    _libc_free((unsigned __int64)v36);
  return sub_1D189A0((__int64)v26);
}
