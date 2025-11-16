// Function: sub_BB92D0
// Address: 0xbb92d0
//
__int64 __fastcall sub_BB92D0(__int64 a1, __int64 *a2)
{
  __int64 v4; // r12
  unsigned int v5; // r15d
  _BYTE *v7; // rsi
  __int64 v8; // rdx
  __m128i *v9; // rax
  __int64 v10; // rsi
  __m128i *v11; // rax
  __int64 v12; // rsi
  _OWORD *v13; // rdi
  __int64 v14; // rdi
  const char *(__fastcall *v15)(__int64, __int64); // rax
  pthread_rwlock_t *v16; // rax
  __int64 v17; // rax
  __m128i *v18; // rcx
  __int64 v19; // rdx
  const char *v20; // rsi
  __int64 v21; // rax
  unsigned __int8 (__fastcall *v22)(__int64, const char *, __int64, __m128i *, __int64); // [rsp+8h] [rbp-98h]
  __m128i *v23; // [rsp+10h] [rbp-90h]
  __int64 v24; // [rsp+18h] [rbp-88h]
  __m128i v25; // [rsp+20h] [rbp-80h] BYREF
  __int64 v26[2]; // [rsp+30h] [rbp-70h] BYREF
  _QWORD v27[2]; // [rsp+40h] [rbp-60h] BYREF
  _OWORD *v28; // [rsp+50h] [rbp-50h] BYREF
  __int64 v29; // [rsp+58h] [rbp-48h]
  _OWORD v30[4]; // [rsp+60h] [rbp-40h] BYREF

  v4 = sub_B6F960(*a2);
  v5 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v4 + 24LL))(v4);
  if ( (_BYTE)v5 )
  {
    v7 = (_BYTE *)a2[21];
    v22 = *(unsigned __int8 (__fastcall **)(__int64, const char *, __int64, __m128i *, __int64))(*(_QWORD *)v4 + 16LL);
    if ( v7 )
    {
      v8 = (__int64)&v7[a2[22]];
      v26[0] = (__int64)v27;
      sub_BB88C0(v26, v7, v8);
    }
    else
    {
      v26[1] = 0;
      v26[0] = (__int64)v27;
      LOBYTE(v27[0]) = 0;
    }
    v9 = (__m128i *)sub_2241130(v26, 0, 0, "module (", 8);
    v28 = v30;
    if ( (__m128i *)v9->m128i_i64[0] == &v9[1] )
    {
      v30[0] = _mm_loadu_si128(v9 + 1);
    }
    else
    {
      v28 = (_OWORD *)v9->m128i_i64[0];
      *(_QWORD *)&v30[0] = v9[1].m128i_i64[0];
    }
    v10 = v9->m128i_i64[1];
    v9[1].m128i_i8[0] = 0;
    v29 = v10;
    v9->m128i_i64[0] = (__int64)v9[1].m128i_i64;
    v9->m128i_i64[1] = 0;
    if ( v29 == 0x3FFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"basic_string::append");
    v11 = (__m128i *)sub_2241490(&v28, ")", 1, v30);
    v23 = &v25;
    if ( (__m128i *)v11->m128i_i64[0] == &v11[1] )
    {
      v25 = _mm_loadu_si128(v11 + 1);
    }
    else
    {
      v23 = (__m128i *)v11->m128i_i64[0];
      v25.m128i_i64[0] = v11[1].m128i_i64[0];
    }
    v12 = v11->m128i_i64[1];
    v11[1].m128i_i8[0] = 0;
    v24 = v12;
    v11->m128i_i64[0] = (__int64)v11[1].m128i_i64;
    v13 = v28;
    v11->m128i_i64[1] = 0;
    if ( v13 != v30 )
    {
      v12 = *(_QWORD *)&v30[0] + 1LL;
      j_j___libc_free_0(v13, *(_QWORD *)&v30[0] + 1LL);
    }
    v14 = v26[0];
    if ( (_QWORD *)v26[0] != v27 )
    {
      v12 = v27[0] + 1LL;
      j_j___libc_free_0(v26[0], v27[0] + 1LL);
    }
    v15 = *(const char *(__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 16LL);
    if ( v15 == sub_BB8680 )
    {
      v16 = (pthread_rwlock_t *)sub_BC2B00(v14, v12);
      v17 = sub_BC2C30(v16);
      v18 = v23;
      v19 = 43;
      v20 = "Unnamed pass: implement Pass::getPassName()";
      if ( v17 )
      {
        v20 = *(const char **)v17;
        v19 = *(_QWORD *)(v17 + 8);
      }
    }
    else
    {
      v21 = ((__int64 (__fastcall *)(__int64))v15)(a1);
      v18 = v23;
      v20 = (const char *)v21;
    }
    if ( v22(v4, v20, v19, v18, v24) )
      v5 = 0;
    if ( v23 != &v25 )
      j_j___libc_free_0(v23, v25.m128i_i64[0] + 1);
  }
  return v5;
}
