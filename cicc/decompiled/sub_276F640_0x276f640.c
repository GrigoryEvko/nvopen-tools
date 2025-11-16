// Function: sub_276F640
// Address: 0x276f640
//
_QWORD *__fastcall sub_276F640(unsigned __int64 *a1, __int64 *a2, __int64 *a3, __int64 *a4)
{
  __int64 v7; // r9
  __int64 v8; // r11
  __int64 v9; // r10
  __int64 v10; // rdi
  __int64 v11; // r15
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // r14
  unsigned __int64 v15; // r8
  __int64 v16; // r14
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  _QWORD *result; // rax
  __int64 v21; // rdx
  unsigned __int64 v22; // rax
  __int64 v23; // rax
  __m128i v24; // xmm1
  __int64 v25; // rdx
  unsigned __int64 v26; // rax
  __int64 v27; // rax
  __m128i v28; // xmm3
  __int64 v29; // [rsp+8h] [rbp-D8h]
  unsigned __int64 v30; // [rsp+8h] [rbp-D8h]
  unsigned __int64 v31; // [rsp+8h] [rbp-D8h]
  __m128i v32; // [rsp+10h] [rbp-D0h] BYREF
  __m128i v33; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v34; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v35; // [rsp+38h] [rbp-A8h]
  __int64 v36; // [rsp+40h] [rbp-A0h]
  __int64 v37; // [rsp+48h] [rbp-98h]
  __m128i v38; // [rsp+50h] [rbp-90h] BYREF
  __int64 v39; // [rsp+60h] [rbp-80h]
  __int64 v40; // [rsp+68h] [rbp-78h]
  __m128i v41; // [rsp+70h] [rbp-70h] BYREF
  __m128i v42; // [rsp+80h] [rbp-60h]
  __m128i v43; // [rsp+90h] [rbp-50h] BYREF
  __int64 v44; // [rsp+A0h] [rbp-40h]
  __int64 v45; // [rsp+A8h] [rbp-38h]

  v7 = a4[3];
  v8 = *a4;
  v9 = a4[1];
  v10 = *a3;
  v11 = a4[2];
  v29 = a3[1];
  v12 = a3[2];
  v13 = a3[3];
  v14 = a1[2];
  v15 = ((v8 - v9) >> 3) + ((((v7 - v13) >> 3) - 1) << 6) + ((v12 - v10) >> 3);
  if ( *a2 == v14 )
  {
    v25 = a1[3];
    v26 = (v14 - v25) >> 3;
    if ( v15 > v26 )
    {
      v31 = v15;
      sub_27697E0(a1, v15 - v26);
      v14 = a1[2];
      v25 = a1[3];
      v15 = v31;
    }
    v27 = a1[4];
    v32.m128i_i64[1] = v25;
    v32.m128i_i64[0] = v14;
    v33.m128i_i64[0] = v27;
    v33.m128i_i64[1] = a1[5];
    sub_2765500(v32.m128i_i64, -(__int64)v15);
    v41 = v32;
    v42 = v33;
    v38 = *(__m128i *)a4;
    v39 = a4[2];
    v40 = a4[3];
    v34 = *a3;
    v35 = a3[1];
    v36 = a3[2];
    v37 = a3[3];
    result = sub_2769FE0(&v43, &v34, v38.m128i_i64, &v41);
    v28 = _mm_loadu_si128(&v33);
    *((__m128i *)a1 + 1) = _mm_loadu_si128(&v32);
    *((__m128i *)a1 + 2) = v28;
  }
  else
  {
    v16 = a1[6];
    if ( *a2 == v16 )
    {
      v21 = a1[8];
      v22 = ((v21 - v16) >> 3) - 1;
      if ( v15 > v22 )
      {
        v30 = v15;
        sub_2769620(a1, v15 - v22);
        v16 = a1[6];
        v21 = a1[8];
        v15 = v30;
      }
      v23 = a1[7];
      v33.m128i_i64[0] = v21;
      v32.m128i_i64[0] = v16;
      v32.m128i_i64[1] = v23;
      v33.m128i_i64[1] = a1[9];
      sub_2765500(v32.m128i_i64, v15);
      v41 = *((__m128i *)a1 + 3);
      v42 = *((__m128i *)a1 + 4);
      v38 = *(__m128i *)a4;
      v39 = a4[2];
      v40 = a4[3];
      v34 = *a3;
      v35 = a3[1];
      v36 = a3[2];
      v37 = a3[3];
      result = sub_2769FE0(&v43, &v34, v38.m128i_i64, &v41);
      v24 = _mm_loadu_si128(&v33);
      *((__m128i *)a1 + 3) = _mm_loadu_si128(&v32);
      *((__m128i *)a1 + 4) = v24;
    }
    else
    {
      v38.m128i_i64[0] = *a2;
      v17 = a2[1];
      v41.m128i_i64[0] = v10;
      v38.m128i_i64[1] = v17;
      v18 = a2[2];
      v42.m128i_i64[0] = v12;
      v39 = v18;
      v19 = a2[3];
      v42.m128i_i64[1] = v13;
      v43.m128i_i64[0] = v8;
      v43.m128i_i64[1] = v9;
      v44 = v11;
      v45 = v7;
      v41.m128i_i64[1] = v29;
      v40 = v19;
      return sub_276E8A0(a1, &v38, &v41, &v43, v15);
    }
  }
  return result;
}
