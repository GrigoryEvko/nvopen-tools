// Function: sub_38ADC60
// Address: 0x38adc60
//
__int64 __fastcall sub_38ADC60(__int64 a1, __int64 *a2, __int64 *a3, double a4, double a5, double a6)
{
  unsigned __int64 v8; // rbx
  unsigned int v9; // r15d
  __int64 v11; // rax
  __int64 v12; // r14
  __m128i *v13; // rax
  __int64 v14; // rcx
  __m128i *v15; // rax
  size_t v16; // rcx
  __m128i *v17; // r10
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rdi
  __m128i *v20; // rax
  unsigned __int64 v21; // rcx
  __m128i *v22; // rdx
  __m128i *v23; // rax
  const char *v24; // rax
  __int64 v25; // rbx
  _QWORD *v26; // rax
  __int64 v27; // r15
  unsigned __int64 v28; // [rsp+8h] [rbp-168h]
  unsigned int *v29; // [rsp+8h] [rbp-168h]
  __int64 *v30; // [rsp+18h] [rbp-158h]
  char v31; // [rsp+2Fh] [rbp-141h] BYREF
  __int64 *v32; // [rsp+30h] [rbp-140h] BYREF
  __int64 *v33; // [rsp+38h] [rbp-138h] BYREF
  unsigned __int64 *v34; // [rsp+40h] [rbp-130h] BYREF
  __int16 v35; // [rsp+50h] [rbp-120h]
  unsigned int *v36; // [rsp+60h] [rbp-110h] BYREF
  __int64 v37; // [rsp+68h] [rbp-108h]
  _BYTE v38[16]; // [rsp+70h] [rbp-100h] BYREF
  __int64 v39[2]; // [rsp+80h] [rbp-F0h] BYREF
  __int64 v40; // [rsp+90h] [rbp-E0h] BYREF
  __m128i *v41; // [rsp+A0h] [rbp-D0h] BYREF
  __int64 v42; // [rsp+A8h] [rbp-C8h]
  __m128i v43; // [rsp+B0h] [rbp-C0h] BYREF
  __m128i *v44; // [rsp+C0h] [rbp-B0h] BYREF
  size_t v45; // [rsp+C8h] [rbp-A8h]
  __m128i v46; // [rsp+D0h] [rbp-A0h] BYREF
  char *v47; // [rsp+E0h] [rbp-90h] BYREF
  size_t v48; // [rsp+E8h] [rbp-88h]
  _QWORD v49[2]; // [rsp+F0h] [rbp-80h] BYREF
  __m128i *v50; // [rsp+100h] [rbp-70h] BYREF
  __int64 v51; // [rsp+108h] [rbp-68h]
  __m128i v52; // [rsp+110h] [rbp-60h] BYREF
  unsigned __int64 v53[2]; // [rsp+120h] [rbp-50h] BYREF
  __m128i v54; // [rsp+130h] [rbp-40h] BYREF

  v8 = *(_QWORD *)(a1 + 56);
  v36 = (unsigned int *)v38;
  v37 = 0x400000000LL;
  if ( (unsigned __int8)sub_38AB270((__int64 **)a1, &v32, a3, a4, a5, a6)
    || (unsigned __int8)sub_388AF10(a1, 4, "expected comma after insertvalue operand")
    || (v28 = *(_QWORD *)(a1 + 56), (unsigned __int8)sub_38AB270((__int64 **)a1, &v33, a3, a4, a5, a6))
    || (unsigned __int8)sub_388D130(a1, (__int64)&v36, &v31) )
  {
    v9 = 1;
    goto LABEL_3;
  }
  if ( (unsigned int)*(unsigned __int8 *)(*v32 + 8) - 13 > 1 )
  {
    v54.m128i_i8[1] = 1;
    v24 = "insertvalue operand must be aggregate type";
LABEL_41:
    v53[0] = (unsigned __int64)v24;
    v54.m128i_i8[0] = 3;
    v9 = (unsigned __int8)sub_38814C0(a1 + 8, v8, (__int64)v53);
    goto LABEL_3;
  }
  v11 = sub_15FB2A0(*v32, v36, (unsigned int)v37);
  if ( !v11 )
  {
    v54.m128i_i8[1] = 1;
    v24 = "invalid indices for insertvalue";
    goto LABEL_41;
  }
  v12 = (__int64)v33;
  if ( v11 != *v33 )
  {
    sub_3888960((__int64 *)&v47, v11);
    sub_3888960(v39, *v33);
    v13 = (__m128i *)sub_2241130(
                       (unsigned __int64 *)v39,
                       0,
                       0,
                       "insertvalue operand and field disagree in type: '",
                       0x31u);
    v41 = &v43;
    if ( (__m128i *)v13->m128i_i64[0] == &v13[1] )
    {
      v43 = _mm_loadu_si128(v13 + 1);
    }
    else
    {
      v41 = (__m128i *)v13->m128i_i64[0];
      v43.m128i_i64[0] = v13[1].m128i_i64[0];
    }
    v14 = v13->m128i_i64[1];
    v13[1].m128i_i8[0] = 0;
    v42 = v14;
    v13->m128i_i64[0] = (__int64)v13[1].m128i_i64;
    v13->m128i_i64[1] = 0;
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v42) <= 0xD )
      goto LABEL_52;
    v15 = (__m128i *)sub_2241490((unsigned __int64 *)&v41, "' instead of '", 0xEu);
    v44 = &v46;
    if ( (__m128i *)v15->m128i_i64[0] == &v15[1] )
    {
      v46 = _mm_loadu_si128(v15 + 1);
    }
    else
    {
      v44 = (__m128i *)v15->m128i_i64[0];
      v46.m128i_i64[0] = v15[1].m128i_i64[0];
    }
    v16 = v15->m128i_u64[1];
    v15[1].m128i_i8[0] = 0;
    v45 = v16;
    v15->m128i_i64[0] = (__int64)v15[1].m128i_i64;
    v17 = v44;
    v15->m128i_i64[1] = 0;
    v18 = 15;
    v19 = 15;
    if ( v17 != &v46 )
      v19 = v46.m128i_i64[0];
    if ( v45 + v48 <= v19 )
      goto LABEL_23;
    if ( v47 != (char *)v49 )
      v18 = v49[0];
    if ( v45 + v48 <= v18 )
    {
      v20 = (__m128i *)sub_2241130((unsigned __int64 *)&v47, 0, 0, v17, v45);
      v50 = &v52;
      v21 = v20->m128i_i64[0];
      v22 = v20 + 1;
      if ( (__m128i *)v20->m128i_i64[0] != &v20[1] )
        goto LABEL_24;
    }
    else
    {
LABEL_23:
      v20 = (__m128i *)sub_2241490((unsigned __int64 *)&v44, v47, v48);
      v50 = &v52;
      v21 = v20->m128i_i64[0];
      v22 = v20 + 1;
      if ( (__m128i *)v20->m128i_i64[0] != &v20[1] )
      {
LABEL_24:
        v50 = (__m128i *)v21;
        v52.m128i_i64[0] = v20[1].m128i_i64[0];
        goto LABEL_25;
      }
    }
    v52 = _mm_loadu_si128(v20 + 1);
LABEL_25:
    v51 = v20->m128i_i64[1];
    v20->m128i_i64[0] = (__int64)v22;
    v20->m128i_i64[1] = 0;
    v20[1].m128i_i8[0] = 0;
    if ( v51 != 0x3FFFFFFFFFFFFFFFLL )
    {
      v23 = (__m128i *)sub_2241490((unsigned __int64 *)&v50, "'", 1u);
      v53[0] = (unsigned __int64)&v54;
      if ( (__m128i *)v23->m128i_i64[0] == &v23[1] )
      {
        v54 = _mm_loadu_si128(v23 + 1);
      }
      else
      {
        v53[0] = v23->m128i_i64[0];
        v54.m128i_i64[0] = v23[1].m128i_i64[0];
      }
      v53[1] = v23->m128i_u64[1];
      v23->m128i_i64[0] = (__int64)v23[1].m128i_i64;
      v23->m128i_i64[1] = 0;
      v23[1].m128i_i8[0] = 0;
      v35 = 260;
      v34 = v53;
      v9 = (unsigned __int8)sub_38814C0(a1 + 8, v28, (__int64)&v34);
      if ( (__m128i *)v53[0] != &v54 )
        j_j___libc_free_0(v53[0]);
      if ( v50 != &v52 )
        j_j___libc_free_0((unsigned __int64)v50);
      if ( v44 != &v46 )
        j_j___libc_free_0((unsigned __int64)v44);
      if ( v41 != &v43 )
        j_j___libc_free_0((unsigned __int64)v41);
      if ( (__int64 *)v39[0] != &v40 )
        j_j___libc_free_0(v39[0]);
      if ( v47 != (char *)v49 )
        j_j___libc_free_0((unsigned __int64)v47);
      goto LABEL_3;
    }
LABEL_52:
    sub_4262D8((__int64)"basic_string::append");
  }
  v25 = (unsigned int)v37;
  v54.m128i_i16[0] = 257;
  v29 = v36;
  v30 = v32;
  v26 = sub_1648A60(88, 2u);
  v27 = (__int64)v26;
  if ( v26 )
  {
    sub_15F1EA0((__int64)v26, *v30, 63, (__int64)(v26 - 6), 2, 0);
    *(_QWORD *)(v27 + 64) = 0x400000000LL;
    *(_QWORD *)(v27 + 56) = v27 + 72;
    sub_15FAD90(v27, (__int64)v30, v12, v29, v25, (__int64)v53);
  }
  *a2 = v27;
  v9 = 2 * (v31 != 0);
LABEL_3:
  if ( v36 != (unsigned int *)v38 )
    _libc_free((unsigned __int64)v36);
  return v9;
}
