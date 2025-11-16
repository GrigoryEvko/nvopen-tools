// Function: sub_23B2900
// Address: 0x23b2900
//
void __fastcall sub_23B2900(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __m128i *v5; // rax
  size_t v6; // rcx
  __m128i *v7; // rsi
  size_t v8; // rdx
  int v9; // eax
  size_t v10; // r15
  unsigned int v11; // eax
  __m128i *v12; // r11
  __int64 *v13; // r9
  __int64 v14; // rax
  __int64 *v15; // [rsp+8h] [rbp-F8h]
  size_t v16; // [rsp+10h] [rbp-F0h]
  __m128i *v17; // [rsp+18h] [rbp-E8h]
  unsigned int v18; // [rsp+18h] [rbp-E8h]
  _QWORD v19[2]; // [rsp+20h] [rbp-E0h] BYREF
  _QWORD v20[2]; // [rsp+30h] [rbp-D0h] BYREF
  __m128i *v21; // [rsp+40h] [rbp-C0h] BYREF
  size_t v22; // [rsp+48h] [rbp-B8h]
  __m128i v23; // [rsp+50h] [rbp-B0h] BYREF
  __m128i *v24; // [rsp+60h] [rbp-A0h] BYREF
  size_t v25; // [rsp+68h] [rbp-98h]
  __m128i v26; // [rsp+70h] [rbp-90h] BYREF
  __m128i v27; // [rsp+80h] [rbp-80h] BYREF
  __m128i *v28; // [rsp+90h] [rbp-70h]
  size_t v29; // [rsp+98h] [rbp-68h]
  __m128i v30; // [rsp+A0h] [rbp-60h] BYREF
  __m128i *v31; // [rsp+B0h] [rbp-50h]
  size_t v32; // [rsp+B8h] [rbp-48h]
  _OWORD v33[4]; // [rsp+C0h] [rbp-40h] BYREF

  v20[0] = a2;
  v20[1] = a3;
  v19[0] = a4;
  v19[1] = a5;
  sub_95CA80((__int64 *)&v21, (__int64)v20);
  sub_95CA80((__int64 *)&v24, (__int64)v19);
  v5 = v21;
  v28 = &v30;
  if ( v21 == &v23 )
  {
    v5 = &v30;
    v30 = _mm_load_si128(&v23);
  }
  else
  {
    v28 = v21;
    v30.m128i_i64[0] = v23.m128i_i64[0];
  }
  v6 = v22;
  v7 = v24;
  v21 = &v23;
  v29 = v22;
  v22 = 0;
  v23.m128i_i8[0] = 0;
  v31 = (__m128i *)v33;
  if ( v24 == &v26 )
  {
    v7 = (__m128i *)v33;
    v33[0] = _mm_load_si128(&v26);
  }
  else
  {
    v31 = v24;
    *(_QWORD *)&v33[0] = v26.m128i_i64[0];
  }
  v8 = v25;
  v25 = v6;
  v24 = v5;
  v32 = v8;
  v26.m128i_i64[0] = (__int64)&v27;
  sub_23AEDD0(v26.m128i_i64, v7, (__int64)v7->m128i_i64 + v8);
  v9 = sub_C92610();
  v10 = v25;
  v17 = v24;
  v16 = v25;
  v11 = sub_C92740(a1, v24, v25, v9);
  v12 = v17;
  v13 = (__int64 *)(*(_QWORD *)a1 + 8LL * v11);
  if ( *v13 )
  {
    if ( *v13 != -8 )
      goto LABEL_7;
    --*(_DWORD *)(a1 + 16);
  }
  v15 = v13;
  v18 = v11;
  v14 = sub_23AE710(40, 8, v12, v10);
  if ( v14 )
  {
    *(_QWORD *)(v14 + 8) = v14 + 24;
    *(_QWORD *)v14 = v16;
    if ( (__m128i *)v26.m128i_i64[0] == &v27 )
    {
      *(__m128i *)(v14 + 24) = _mm_load_si128(&v27);
    }
    else
    {
      *(_QWORD *)(v14 + 8) = v26.m128i_i64[0];
      *(_QWORD *)(v14 + 24) = v27.m128i_i64[0];
    }
    *(_QWORD *)(v14 + 16) = v26.m128i_i64[1];
    v26 = (__m128i)(unsigned __int64)&v27;
    v27.m128i_i8[0] = 0;
  }
  *v15 = v14;
  ++*(_DWORD *)(a1 + 12);
  sub_C929D0((__int64 *)a1, v18);
LABEL_7:
  if ( (__m128i *)v26.m128i_i64[0] != &v27 )
    j_j___libc_free_0(v26.m128i_u64[0]);
  if ( v31 != (__m128i *)v33 )
    j_j___libc_free_0((unsigned __int64)v31);
  if ( v28 != &v30 )
    j_j___libc_free_0((unsigned __int64)v28);
}
