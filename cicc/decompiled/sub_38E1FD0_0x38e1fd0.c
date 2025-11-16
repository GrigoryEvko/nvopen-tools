// Function: sub_38E1FD0
// Address: 0x38e1fd0
//
__m128i *__fastcall sub_38E1FD0(
        __m128i *a1,
        __int64 a2,
        __int8 *a3,
        size_t a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __m128i a8,
        __int64 a9,
        __int64 a10,
        __int64 a11,
        __int64 a12,
        __int64 a13,
        __int64 a14,
        __int64 a15)
{
  __int64 *v19; // rdi
  _DWORD *m128i_i32; // r9
  unsigned __int64 v21; // rax
  size_t v22; // rdx
  __m128i v23; // xmm2
  __m128i v24; // xmm3
  __int64 v26; // rax
  __int64 v28; // [rsp+8h] [rbp-98h]
  __int64 v29; // [rsp+10h] [rbp-90h]
  __int64 v30; // [rsp+18h] [rbp-88h]
  __int64 v31; // [rsp+20h] [rbp-80h]
  __int64 v32; // [rsp+28h] [rbp-78h]
  __int64 v33; // [rsp+30h] [rbp-70h]
  __int64 v34; // [rsp+38h] [rbp-68h]
  __m128i v35; // [rsp+40h] [rbp-60h] BYREF
  __m128i v36; // [rsp+50h] [rbp-50h] BYREF
  size_t v37[7]; // [rsp+68h] [rbp-38h] BYREF

  v19 = &a1->m128i_i64[1];
  v34 = a9;
  v36 = _mm_loadu_si128(&a7);
  v33 = a10;
  v35 = _mm_loadu_si128(&a8);
  v32 = a11;
  v31 = a12;
  v30 = a13;
  v29 = a14;
  v28 = a15;
  *(v19 - 1) = (__int64)&unk_49EE580;
  a1->m128i_i64[1] = (__int64)&a1[1].m128i_i64[1];
  sub_38E1AC0(v19, *(_BYTE **)a2, *(_QWORD *)a2 + *(_QWORD *)(a2 + 8));
  m128i_i32 = a1[5].m128i_i32;
  a1[2].m128i_i64[1] = *(_QWORD *)(a2 + 32);
  a1[3].m128i_i64[0] = *(_QWORD *)(a2 + 40);
  a1[3].m128i_i64[1] = *(_QWORD *)(a2 + 48);
  if ( !a3 )
  {
    a1[4].m128i_i64[0] = (__int64)m128i_i32;
    v22 = 0;
    a1[4].m128i_i64[1] = 0;
    a1[5].m128i_i8[0] = 0;
    goto LABEL_7;
  }
  a1[4].m128i_i64[0] = (__int64)m128i_i32;
  v21 = a4;
  v37[0] = a4;
  if ( a4 > 0xF )
  {
    v26 = sub_22409D0((__int64)a1[4].m128i_i64, v37, 0);
    a1[4].m128i_i64[0] = v26;
    m128i_i32 = (_DWORD *)v26;
    a1[5].m128i_i64[0] = v37[0];
    goto LABEL_11;
  }
  if ( a4 != 1 )
  {
    if ( !a4 )
      goto LABEL_5;
LABEL_11:
    memcpy(m128i_i32, a3, a4);
    v21 = v37[0];
    m128i_i32 = (_DWORD *)a1[4].m128i_i64[0];
    goto LABEL_5;
  }
  a1[5].m128i_i8[0] = *a3;
LABEL_5:
  a1[4].m128i_i64[1] = v21;
  *((_BYTE *)m128i_i32 + v21) = 0;
  v22 = a1[4].m128i_u64[1];
  m128i_i32 = (_DWORD *)a1[4].m128i_i64[0];
LABEL_7:
  v23 = _mm_load_si128(&v36);
  v24 = _mm_load_si128(&v35);
  a1[12].m128i_i64[0] = 0;
  a1[8].m128i_i64[0] = v34;
  a1[6] = v23;
  a1[8].m128i_i64[1] = v33;
  a1[12].m128i_i64[1] = 0;
  a1[9].m128i_i64[0] = v32;
  a1[13].m128i_i64[0] = 0;
  a1[9].m128i_i64[1] = v31;
  a1[7] = v24;
  a1[10].m128i_i64[1] = v30;
  a1[11].m128i_i64[0] = v29;
  a1[11].m128i_i64[1] = v28;
  return sub_38E1E90(a1, m128i_i32, v22, a5, a6);
}
