// Function: sub_2F18B70
// Address: 0x2f18b70
//
void __fastcall sub_2F18B70(__int64 a1, unsigned __int64 *a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v5; // rbx
  __int64 v6; // rdx
  __int64 v7; // rcx
  __m128i *v8; // rsi
  __m128i *v9; // rdi
  __m128i v10; // xmm0
  __m128i *v11; // rax
  __m128i v12; // xmm1
  unsigned __int64 v15; // [rsp+38h] [rbp-118h]
  unsigned __int64 v16; // [rsp+50h] [rbp-100h] BYREF
  unsigned __int64 v17; // [rsp+58h] [rbp-F8h]
  __int64 v18; // [rsp+60h] [rbp-F0h]
  __m128i *v19; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v20; // [rsp+78h] [rbp-D8h]
  __m128i v21; // [rsp+80h] [rbp-D0h] BYREF
  __m128i *v22; // [rsp+90h] [rbp-C0h]
  __int64 v23; // [rsp+98h] [rbp-B8h]
  __m128i v24; // [rsp+A0h] [rbp-B0h] BYREF
  __m128i v25; // [rsp+B0h] [rbp-A0h] BYREF
  __m128i v26; // [rsp+C0h] [rbp-90h] BYREF
  __m128i v27; // [rsp+D0h] [rbp-80h] BYREF
  _QWORD v28[14]; // [rsp+E0h] [rbp-70h] BYREF

  v16 = 0;
  v17 = 0;
  v18 = 0;
  sub_3531F30(a4, &v16);
  v5 = v16;
  v15 = v17;
  if ( v16 == v17 )
    goto LABEL_24;
  do
  {
    v28[5] = 0x100000000LL;
    v19 = &v21;
    v28[0] = &unk_49DD210;
    v20 = 0;
    v28[6] = &v19;
    v21.m128i_i8[0] = 0;
    memset(&v28[1], 0, 32);
    sub_CB5980((__int64)v28, 0, 0, 0);
    sub_A62C00(*(const char **)(v5 + 8), (__int64)v28, a4, *(_QWORD *)(*(_QWORD *)a3 + 40LL));
    v11 = v19;
    if ( v19 == &v21 )
    {
      v12 = _mm_load_si128(&v21);
      v7 = v20;
      v21.m128i_i8[0] = 0;
      v20 = 0;
      v25.m128i_i64[0] = (__int64)&v26;
      v24 = v12;
    }
    else
    {
      v6 = v21.m128i_i64[0];
      v7 = v20;
      v19 = &v21;
      v20 = 0;
      v24.m128i_i64[0] = v21.m128i_i64[0];
      v21.m128i_i8[0] = 0;
      v25.m128i_i64[0] = (__int64)&v26;
      if ( v11 != &v24 )
      {
        v25.m128i_i64[0] = (__int64)v11;
        v26.m128i_i64[0] = v6;
        goto LABEL_5;
      }
    }
    v26 = _mm_load_si128(&v24);
LABEL_5:
    v25.m128i_i64[1] = v7;
    v24.m128i_i8[0] = 0;
    v8 = (__m128i *)a2[71];
    v22 = &v24;
    v23 = 0;
    v27 = 0u;
    if ( v8 == (__m128i *)a2[72] )
    {
      sub_2F188A0(a2 + 70, v8, &v25);
      v9 = (__m128i *)v25.m128i_i64[0];
    }
    else
    {
      v9 = (__m128i *)v25.m128i_i64[0];
      if ( v8 )
      {
        v8->m128i_i64[0] = (__int64)v8[1].m128i_i64;
        if ( (__m128i *)v25.m128i_i64[0] == &v26 )
        {
          v8[1] = _mm_load_si128(&v26);
        }
        else
        {
          v8->m128i_i64[0] = v25.m128i_i64[0];
          v8[1].m128i_i64[0] = v26.m128i_i64[0];
        }
        v25.m128i_i64[0] = (__int64)&v26;
        v9 = &v26;
        v8->m128i_i64[1] = v25.m128i_i64[1];
        v10 = _mm_load_si128(&v27);
        v25.m128i_i64[1] = 0;
        v26.m128i_i8[0] = 0;
        v8[2] = v10;
        v8 = (__m128i *)a2[71];
      }
      a2[71] = (unsigned __int64)&v8[3];
    }
    if ( v9 != &v26 )
      j_j___libc_free_0((unsigned __int64)v9);
    if ( v22 != &v24 )
      j_j___libc_free_0((unsigned __int64)v22);
    v28[0] = &unk_49DD210;
    sub_CB5840((__int64)v28);
    if ( v19 != &v21 )
      j_j___libc_free_0((unsigned __int64)v19);
    v5 += 16LL;
  }
  while ( v15 != v5 );
  v15 = v16;
LABEL_24:
  if ( v15 )
    j_j___libc_free_0(v15);
}
