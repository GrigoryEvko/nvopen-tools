// Function: sub_6D0040
// Address: 0x6d0040
//
__int64 __fastcall sub_6D0040(unsigned __int16 a1, _QWORD *a2, __int64 *a3, __m128i *a4, __m128i *a5)
{
  _QWORD *v7; // r15
  __int64 result; // rax
  __int64 v9; // rcx
  __int64 v10; // rax
  __m128i v11; // xmm1
  __m128i v12; // xmm2
  __m128i v13; // xmm3
  __m128i v14; // xmm4
  __m128i v15; // xmm5
  __m128i v16; // xmm6
  __int8 v17; // dl
  __m128i v18; // xmm7
  __m128i v19; // xmm0
  __m128i v20; // xmm2
  __m128i v21; // xmm3
  __m128i v22; // xmm4
  __m128i v23; // xmm5
  __m128i v24; // xmm6
  __m128i v25; // xmm7
  __m128i v26; // xmm1
  __m128i v27; // xmm2
  __m128i v28; // xmm3
  __m128i v29; // xmm4
  __m128i v30; // xmm5
  __m128i v31; // xmm6
  __int64 v32; // [rsp+0h] [rbp-1B0h]
  _QWORD *v34; // [rsp+10h] [rbp-1A0h] BYREF
  _QWORD *v35; // [rsp+18h] [rbp-198h]
  __m128i v36[9]; // [rsp+20h] [rbp-190h] BYREF
  __m128i v37; // [rsp+B0h] [rbp-100h]
  __m128i v38; // [rsp+C0h] [rbp-F0h]
  __m128i v39; // [rsp+D0h] [rbp-E0h]
  __m128i v40; // [rsp+E0h] [rbp-D0h]
  __m128i v41; // [rsp+F0h] [rbp-C0h]
  __m128i v42; // [rsp+100h] [rbp-B0h]
  __m128i v43; // [rsp+110h] [rbp-A0h]
  __m128i v44; // [rsp+120h] [rbp-90h]
  __m128i v45; // [rsp+130h] [rbp-80h]
  __m128i v46; // [rsp+140h] [rbp-70h]
  __m128i v47; // [rsp+150h] [rbp-60h]
  __m128i v48; // [rsp+160h] [rbp-50h]
  __m128i v49; // [rsp+170h] [rbp-40h]

  v7 = (_QWORD *)*a2;
  *a2 = 0;
  sub_6E6610(a2, a5, 1);
  result = sub_6E18E0(a5);
  if ( v7 )
  {
    v9 = *(_QWORD *)(qword_4D03C50 + 136LL);
    *(_QWORD *)(qword_4D03C50 + 136LL) = &v34;
    v32 = v9;
    do
    {
      v11 = _mm_loadu_si128(a5 + 1);
      v12 = _mm_loadu_si128(a5 + 2);
      v13 = _mm_loadu_si128(a5 + 3);
      v14 = _mm_loadu_si128(a5 + 4);
      v15 = _mm_loadu_si128(a5 + 5);
      v36[0] = _mm_loadu_si128(a5);
      v16 = _mm_loadu_si128(a5 + 6);
      v17 = a5[1].m128i_i8[0];
      v36[1] = v11;
      v18 = _mm_loadu_si128(a5 + 7);
      v36[2] = v12;
      v19 = _mm_loadu_si128(a5 + 8);
      v36[3] = v13;
      v36[4] = v14;
      v36[5] = v15;
      v36[6] = v16;
      v36[7] = v18;
      v36[8] = v19;
      if ( v17 == 2 )
      {
        v20 = _mm_loadu_si128(a5 + 10);
        v21 = _mm_loadu_si128(a5 + 11);
        v22 = _mm_loadu_si128(a5 + 12);
        v23 = _mm_loadu_si128(a5 + 13);
        v37 = _mm_loadu_si128(a5 + 9);
        v24 = _mm_loadu_si128(a5 + 14);
        v25 = _mm_loadu_si128(a5 + 15);
        v38 = v20;
        v26 = _mm_loadu_si128(a5 + 16);
        v27 = _mm_loadu_si128(a5 + 17);
        v39 = v21;
        v28 = _mm_loadu_si128(a5 + 18);
        v40 = v22;
        v29 = _mm_loadu_si128(a5 + 19);
        v41 = v23;
        v30 = _mm_loadu_si128(a5 + 20);
        v42 = v24;
        v31 = _mm_loadu_si128(a5 + 21);
        v43 = v25;
        v44 = v26;
        v45 = v27;
        v46 = v28;
        v47 = v29;
        v48 = v30;
        v49 = v31;
      }
      else if ( v17 == 5 || v17 == 1 )
      {
        v37.m128i_i64[0] = a5[9].m128i_i64[0];
      }
      v10 = v7[3];
      v34 = v7;
      v35 = v7;
      sub_6E18E0(v10 + 8);
      v7 = (_QWORD *)*v7;
      *v35 = 0;
      sub_6CFD10(a1, v36[0].m128i_i64, *a3, a4, a5->m128i_i64);
      a5[5].m128i_i64[1] = 0;
    }
    while ( v7 );
    result = qword_4D03C50;
    *(_QWORD *)(qword_4D03C50 + 136LL) = v32;
  }
  return result;
}
