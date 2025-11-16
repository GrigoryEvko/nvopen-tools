// Function: sub_68D1C0
// Address: 0x68d1c0
//
unsigned int *__fastcall sub_68D1C0(
        __int64 a1,
        unsigned __int8 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned int a7,
        unsigned int a8,
        const __m128i *a9)
{
  __int64 v9; // r14
  __int64 v10; // r13
  char v11; // al
  unsigned int *result; // rax
  __int64 v13; // rdx
  __int64 v14; // r9
  __int64 v15; // r8
  __int64 v16; // r14
  __int64 v17; // rax
  __int64 v18; // rdi
  __m128i v19; // xmm1
  __m128i v20; // xmm2
  __m128i v21; // xmm3
  __m128i v22; // xmm4
  __m128i v23; // xmm5
  __m128i v24; // xmm6
  __int8 v25; // al
  __m128i v26; // xmm7
  __m128i v27; // xmm0
  __int64 v28; // r13
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // rax
  __m128i v33; // xmm2
  __m128i v34; // xmm3
  __m128i v35; // xmm4
  __m128i v36; // xmm5
  __m128i v37; // xmm6
  __m128i v38; // xmm7
  __m128i v39; // xmm1
  __m128i v40; // xmm2
  __m128i v41; // xmm3
  __m128i v42; // xmm4
  __m128i v43; // xmm5
  __m128i v44; // xmm6
  __int64 v45; // [rsp-10h] [rbp-320h]
  __int64 v47; // [rsp+10h] [rbp-300h] BYREF
  __int64 v48; // [rsp+18h] [rbp-2F8h] BYREF
  _BYTE v49[352]; // [rsp+20h] [rbp-2F0h] BYREF
  _OWORD v50[9]; // [rsp+180h] [rbp-190h] BYREF
  __m128i v51; // [rsp+210h] [rbp-100h]
  __m128i v52; // [rsp+220h] [rbp-F0h]
  __m128i v53; // [rsp+230h] [rbp-E0h]
  __m128i v54; // [rsp+240h] [rbp-D0h]
  __m128i v55; // [rsp+250h] [rbp-C0h]
  __m128i v56; // [rsp+260h] [rbp-B0h]
  __m128i v57; // [rsp+270h] [rbp-A0h]
  __m128i v58; // [rsp+280h] [rbp-90h]
  __m128i v59; // [rsp+290h] [rbp-80h]
  __m128i v60; // [rsp+2A0h] [rbp-70h]
  __m128i v61; // [rsp+2B0h] [rbp-60h]
  __m128i v62; // [rsp+2C0h] [rbp-50h]
  __m128i v63; // [rsp+2D0h] [rbp-40h]

  v9 = a1;
  v10 = *(_QWORD *)(a3 + 24);
  v11 = *(_BYTE *)(v10 + 80);
  if ( v11 == 16 )
  {
    v10 = **(_QWORD **)(v10 + 88);
    v11 = *(_BYTE *)(v10 + 80);
  }
  if ( v11 == 24 )
    v10 = *(_QWORD *)(v10 + 88);
  sub_6EAEF0(a3, a4, a5, v49);
  result = (unsigned int *)sub_6F7B90(a1, v49, a2, a6, a7, a9);
  v15 = a8;
  if ( a8 )
  {
    *(_BYTE *)(a9[9].m128i_i64[0] + 27) |= 2u;
    result = (unsigned int *)a9[9].m128i_i64[0];
    a1 = result[7];
    if ( !(_DWORD)a1 )
    {
      v13 = *(_QWORD *)(v9 + 68);
      *(_QWORD *)(result + 7) = v13;
    }
  }
  if ( *(_QWORD *)(v10 + 96) )
  {
    v16 = a9[9].m128i_i64[0];
    sub_73E3D0(v16, v10, 0);
    a1 = v16;
    result = (unsigned int *)sub_6E7170(v16, a9);
  }
  if ( !a7 )
  {
    result = &word_4D04898;
    if ( word_4D04898 )
    {
      v17 = sub_724DC0(a1, a7, v13, word_4D04898, v15, v14);
      v18 = a9[9].m128i_i64[0];
      v47 = v17;
      if ( (unsigned int)sub_71ABE0(v18, v17) )
      {
        v19 = _mm_loadu_si128(a9 + 1);
        v20 = _mm_loadu_si128(a9 + 2);
        v21 = _mm_loadu_si128(a9 + 3);
        v22 = _mm_loadu_si128(a9 + 4);
        v23 = _mm_loadu_si128(a9 + 5);
        v50[0] = _mm_loadu_si128(a9);
        v24 = _mm_loadu_si128(a9 + 6);
        v25 = a9[1].m128i_i8[0];
        v50[1] = v19;
        v26 = _mm_loadu_si128(a9 + 7);
        v50[2] = v20;
        v27 = _mm_loadu_si128(a9 + 8);
        v50[3] = v21;
        v50[4] = v22;
        v50[5] = v23;
        v50[6] = v24;
        v50[7] = v26;
        v50[8] = v27;
        if ( v25 == 2 )
        {
          v33 = _mm_loadu_si128(a9 + 10);
          v34 = _mm_loadu_si128(a9 + 11);
          v35 = _mm_loadu_si128(a9 + 12);
          v36 = _mm_loadu_si128(a9 + 13);
          v51 = _mm_loadu_si128(a9 + 9);
          v37 = _mm_loadu_si128(a9 + 14);
          v38 = _mm_loadu_si128(a9 + 15);
          v52 = v33;
          v39 = _mm_loadu_si128(a9 + 16);
          v40 = _mm_loadu_si128(a9 + 17);
          v53 = v34;
          v41 = _mm_loadu_si128(a9 + 18);
          v54 = v35;
          v42 = _mm_loadu_si128(a9 + 19);
          v55 = v36;
          v43 = _mm_loadu_si128(a9 + 20);
          v56 = v37;
          v44 = _mm_loadu_si128(a9 + 21);
          v57 = v38;
          v58 = v39;
          v59 = v40;
          v60 = v41;
          v61 = v42;
          v62 = v43;
          v63 = v44;
        }
        else if ( v25 == 5 || v25 == 1 )
        {
          v51.m128i_i64[0] = a9[9].m128i_i64[0];
        }
        if ( *(_BYTE *)(v47 + 173) == 10 )
        {
          v28 = sub_6ECAE0(a9->m128i_i64[0], 0, 0, 1, 2, (int)a9 + 68, (__int64)&v48);
          v32 = sub_724E50(&v47, 0, v29, v30, v31);
          sub_72F900(v48, v32);
          sub_6E70E0(v28, a9);
          sub_6E4BC0(a9, v50);
          a9[18].m128i_i64[0] = v51.m128i_i64[0];
          return (unsigned int *)v45;
        }
        sub_6E6A50(v47, a9);
        sub_6E4BC0(a9, v50);
        a9[18].m128i_i64[0] = v51.m128i_i64[0];
      }
      return (unsigned int *)sub_724E30(&v47);
    }
  }
  return result;
}
