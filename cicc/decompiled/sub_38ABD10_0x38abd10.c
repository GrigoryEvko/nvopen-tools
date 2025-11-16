// Function: sub_38ABD10
// Address: 0x38abd10
//
__int64 __fastcall sub_38ABD10(__int64 a1, __int64 *a2, __int64 *a3, int a4, double a5, double a6, double a7)
{
  unsigned __int64 v9; // r12
  unsigned int v10; // r15d
  __m128i *v12; // rax
  __int64 v13; // rcx
  __m128i *v14; // rax
  size_t v15; // rcx
  __m128i *v16; // r10
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rdi
  __m128i *v19; // rax
  __m128i *v20; // rdx
  unsigned __int64 v21; // rcx
  __m128i *v22; // rax
  __int64 *v23; // [rsp+20h] [rbp-120h] BYREF
  __int64 v24; // [rsp+28h] [rbp-118h] BYREF
  unsigned __int64 *v25; // [rsp+30h] [rbp-110h] BYREF
  __int16 v26; // [rsp+40h] [rbp-100h]
  __int64 v27[2]; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v28; // [rsp+60h] [rbp-E0h] BYREF
  __m128i *v29; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v30; // [rsp+78h] [rbp-C8h]
  __m128i v31; // [rsp+80h] [rbp-C0h] BYREF
  __m128i *v32; // [rsp+90h] [rbp-B0h] BYREF
  size_t v33; // [rsp+98h] [rbp-A8h]
  __m128i v34; // [rsp+A0h] [rbp-A0h] BYREF
  char *v35; // [rsp+B0h] [rbp-90h] BYREF
  size_t v36; // [rsp+B8h] [rbp-88h]
  _QWORD v37[2]; // [rsp+C0h] [rbp-80h] BYREF
  __m128i *v38; // [rsp+D0h] [rbp-70h] BYREF
  __int64 v39; // [rsp+D8h] [rbp-68h]
  __m128i v40; // [rsp+E0h] [rbp-60h] BYREF
  unsigned __int64 v41[2]; // [rsp+F0h] [rbp-50h] BYREF
  _OWORD v42[4]; // [rsp+100h] [rbp-40h] BYREF

  v9 = *(_QWORD *)(a1 + 56);
  v24 = 0;
  if ( (unsigned __int8)sub_38AB270((__int64 **)a1, &v23, a3, a5, a6, a7) )
    return 1;
  if ( (unsigned __int8)sub_388AF10(a1, 53, "expected 'to' after cast value") )
    return 1;
  v41[0] = (unsigned __int64)"expected type";
  LOWORD(v42[0]) = 259;
  v10 = sub_3891B00(a1, &v24, (__int64)v41, 0);
  if ( (_BYTE)v10 )
    return 1;
  if ( (unsigned __int8)sub_15FC090(a4, v23, v24) )
  {
    LOWORD(v42[0]) = 257;
    *a2 = sub_15FDBD0(a4, (__int64)v23, v24, (__int64)v41, 0);
    return v10;
  }
  sub_15FC090(a4, v23, v24);
  sub_3888960((__int64 *)&v35, v24);
  sub_3888960(v27, *v23);
  v12 = (__m128i *)sub_2241130((unsigned __int64 *)v27, 0, 0, "invalid cast opcode for cast from '", 0x23u);
  v29 = &v31;
  if ( (__m128i *)v12->m128i_i64[0] == &v12[1] )
  {
    v31 = _mm_loadu_si128(v12 + 1);
  }
  else
  {
    v29 = (__m128i *)v12->m128i_i64[0];
    v31.m128i_i64[0] = v12[1].m128i_i64[0];
  }
  v13 = v12->m128i_i64[1];
  v12[1].m128i_i8[0] = 0;
  v30 = v13;
  v12->m128i_i64[0] = (__int64)v12[1].m128i_i64;
  v12->m128i_i64[1] = 0;
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v30) <= 5 )
LABEL_42:
    sub_4262D8((__int64)"basic_string::append");
  v14 = (__m128i *)sub_2241490((unsigned __int64 *)&v29, "' to '", 6u);
  v32 = &v34;
  if ( (__m128i *)v14->m128i_i64[0] == &v14[1] )
  {
    v34 = _mm_loadu_si128(v14 + 1);
  }
  else
  {
    v32 = (__m128i *)v14->m128i_i64[0];
    v34.m128i_i64[0] = v14[1].m128i_i64[0];
  }
  v15 = v14->m128i_u64[1];
  v14[1].m128i_i8[0] = 0;
  v33 = v15;
  v14->m128i_i64[0] = (__int64)v14[1].m128i_i64;
  v16 = v32;
  v14->m128i_i64[1] = 0;
  v17 = 15;
  v18 = 15;
  if ( v16 != &v34 )
    v18 = v34.m128i_i64[0];
  if ( v33 + v36 <= v18 )
    goto LABEL_18;
  if ( v35 != (char *)v37 )
    v17 = v37[0];
  if ( v33 + v36 <= v17 )
  {
    v19 = (__m128i *)sub_2241130((unsigned __int64 *)&v35, 0, 0, v16, v33);
    v20 = v19 + 1;
    v38 = &v40;
    v21 = v19->m128i_i64[0];
    if ( (__m128i *)v19->m128i_i64[0] != &v19[1] )
      goto LABEL_19;
  }
  else
  {
LABEL_18:
    v19 = (__m128i *)sub_2241490((unsigned __int64 *)&v32, v35, v36);
    v20 = v19 + 1;
    v38 = &v40;
    v21 = v19->m128i_i64[0];
    if ( (__m128i *)v19->m128i_i64[0] != &v19[1] )
    {
LABEL_19:
      v38 = (__m128i *)v21;
      v40.m128i_i64[0] = v19[1].m128i_i64[0];
      goto LABEL_20;
    }
  }
  v40 = _mm_loadu_si128(v19 + 1);
LABEL_20:
  v39 = v19->m128i_i64[1];
  v19->m128i_i64[0] = (__int64)v20;
  v19->m128i_i64[1] = 0;
  v19[1].m128i_i8[0] = 0;
  if ( v39 == 0x3FFFFFFFFFFFFFFFLL )
    goto LABEL_42;
  v22 = (__m128i *)sub_2241490((unsigned __int64 *)&v38, "'", 1u);
  v41[0] = (unsigned __int64)v42;
  if ( (__m128i *)v22->m128i_i64[0] == &v22[1] )
  {
    v42[0] = _mm_loadu_si128(v22 + 1);
  }
  else
  {
    v41[0] = v22->m128i_i64[0];
    *(_QWORD *)&v42[0] = v22[1].m128i_i64[0];
  }
  v41[1] = v22->m128i_u64[1];
  v22->m128i_i64[0] = (__int64)v22[1].m128i_i64;
  v22->m128i_i64[1] = 0;
  v22[1].m128i_i8[0] = 0;
  v26 = 260;
  v25 = v41;
  v10 = sub_38814C0(a1 + 8, v9, (__int64)&v25);
  if ( (_OWORD *)v41[0] != v42 )
    j_j___libc_free_0(v41[0]);
  if ( v38 != &v40 )
    j_j___libc_free_0((unsigned __int64)v38);
  if ( v32 != &v34 )
    j_j___libc_free_0((unsigned __int64)v32);
  if ( v29 != &v31 )
    j_j___libc_free_0((unsigned __int64)v29);
  if ( (__int64 *)v27[0] != &v28 )
    j_j___libc_free_0(v27[0]);
  if ( v35 != (char *)v37 )
    j_j___libc_free_0((unsigned __int64)v35);
  return v10;
}
