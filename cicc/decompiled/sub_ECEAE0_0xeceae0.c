// Function: sub_ECEAE0
// Address: 0xeceae0
//
__int64 __fastcall sub_ECEAE0(__int64 a1, const char *a2)
{
  __int64 v2; // rax
  __int64 v3; // r12
  size_t v4; // rdx
  __int64 v5; // rcx
  __m128i *v6; // rax
  __int64 v7; // rcx
  __m128i *v8; // rax
  __int64 v9; // rcx
  __m128i *v10; // rcx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rbx
  __int64 v14; // rax
  unsigned int v15; // r12d
  _QWORD v17[2]; // [rsp+0h] [rbp-C0h] BYREF
  _QWORD v18[2]; // [rsp+10h] [rbp-B0h] BYREF
  __m128i *v19; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v20; // [rsp+28h] [rbp-98h]
  __m128i v21; // [rsp+30h] [rbp-90h] BYREF
  __m128i *v22; // [rsp+40h] [rbp-80h]
  __int64 v23; // [rsp+48h] [rbp-78h]
  __m128i v24; // [rsp+50h] [rbp-70h] BYREF
  _QWORD v25[4]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v26; // [rsp+80h] [rbp-40h]

  v2 = *(_QWORD *)(a1 + 32);
  v17[0] = v18;
  v3 = *(_QWORD *)(v2 + 8);
  strcpy((char *)v18, "Expected ");
  v17[1] = 9;
  v4 = strlen(a2);
  if ( v4 > 0x3FFFFFFFFFFFFFF6LL )
    goto LABEL_16;
  v6 = (__m128i *)sub_2241490(v17, a2, v4, v5);
  v19 = &v21;
  if ( (__m128i *)v6->m128i_i64[0] == &v6[1] )
  {
    v21 = _mm_loadu_si128(v6 + 1);
  }
  else
  {
    v19 = (__m128i *)v6->m128i_i64[0];
    v21.m128i_i64[0] = v6[1].m128i_i64[0];
  }
  v7 = v6->m128i_i64[1];
  v6[1].m128i_i8[0] = 0;
  v20 = v7;
  v6->m128i_i64[0] = (__int64)v6[1].m128i_i64;
  v6->m128i_i64[1] = 0;
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v20) <= 0xE )
LABEL_16:
    sub_4262D8((__int64)"basic_string::append");
  v8 = (__m128i *)sub_2241490(&v19, ", instead got: ", 15, v7);
  v22 = &v24;
  if ( (__m128i *)v8->m128i_i64[0] == &v8[1] )
  {
    v24 = _mm_loadu_si128(v8 + 1);
  }
  else
  {
    v22 = (__m128i *)v8->m128i_i64[0];
    v24.m128i_i64[0] = v8[1].m128i_i64[0];
  }
  v9 = v8->m128i_i64[1];
  v8[1].m128i_i8[0] = 0;
  v23 = v9;
  v8->m128i_i64[0] = (__int64)v8[1].m128i_i64;
  v10 = v22;
  v8->m128i_i64[1] = 0;
  v11 = *(_QWORD *)(v3 + 16);
  v12 = *(_QWORD *)(v3 + 8);
  v13 = *(_QWORD *)(a1 + 24);
  v25[0] = v10;
  v25[3] = v11;
  v25[2] = v12;
  v25[1] = v23;
  v26 = 1285;
  v14 = sub_ECD6A0(v3);
  v15 = sub_ECDA70(v13, v14, (__int64)v25, 0, 0);
  if ( v22 != &v24 )
    j_j___libc_free_0(v22, v24.m128i_i64[0] + 1);
  if ( v19 != &v21 )
    j_j___libc_free_0(v19, v21.m128i_i64[0] + 1);
  if ( (_QWORD *)v17[0] != v18 )
    j_j___libc_free_0(v17[0], v18[0] + 1LL);
  return v15;
}
