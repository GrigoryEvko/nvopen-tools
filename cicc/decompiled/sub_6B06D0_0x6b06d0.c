// Function: sub_6B06D0
// Address: 0x6b06d0
//
__int64 __fastcall sub_6B06D0(__int64 a1, __m128i *a2, _DWORD *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 *v10; // r14
  int v11; // r12d
  __int64 result; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rcx
  __int8 v16; // dl
  bool v17; // si
  char v18; // dl
  __int64 v19; // rax
  __m128i v20; // xmm1
  __m128i v21; // xmm2
  __m128i v22; // xmm3
  __m128i v23; // xmm4
  __m128i v24; // xmm5
  __m128i v25; // xmm6
  __m128i v26; // xmm7
  __m128i v27; // xmm0
  __m128i v28; // xmm1
  __m128i v29; // xmm2
  __m128i v30; // xmm3
  __m128i v31; // xmm4
  __m128i v32; // xmm5
  __m128i v33; // xmm6
  __m128i v34; // xmm7
  __m128i v35; // xmm1
  __m128i v36; // xmm2
  __m128i v37; // xmm3
  __m128i v38; // xmm4
  __m128i v39; // xmm5
  char v40; // [rsp+Fh] [rbp-191h]
  __m128i v41; // [rsp+10h] [rbp-190h] BYREF
  __m128i v42; // [rsp+20h] [rbp-180h] BYREF
  __m128i v43; // [rsp+30h] [rbp-170h] BYREF
  __m128i v44; // [rsp+40h] [rbp-160h] BYREF
  __m128i v45; // [rsp+50h] [rbp-150h] BYREF
  __m128i v46; // [rsp+60h] [rbp-140h] BYREF
  __m128i v47; // [rsp+70h] [rbp-130h] BYREF
  __m128i v48; // [rsp+80h] [rbp-120h] BYREF
  __m128i v49; // [rsp+90h] [rbp-110h] BYREF
  __m128i v50; // [rsp+A0h] [rbp-100h] BYREF
  __m128i v51; // [rsp+B0h] [rbp-F0h] BYREF
  __m128i v52; // [rsp+C0h] [rbp-E0h] BYREF
  __m128i v53; // [rsp+D0h] [rbp-D0h] BYREF
  __m128i v54; // [rsp+E0h] [rbp-C0h] BYREF
  __m128i v55; // [rsp+F0h] [rbp-B0h] BYREF
  __m128i v56; // [rsp+100h] [rbp-A0h] BYREF
  __m128i v57; // [rsp+110h] [rbp-90h] BYREF
  __m128i v58; // [rsp+120h] [rbp-80h] BYREF
  __m128i v59; // [rsp+130h] [rbp-70h] BYREF
  __m128i v60; // [rsp+140h] [rbp-60h] BYREF
  __m128i v61; // [rsp+150h] [rbp-50h] BYREF
  __m128i v62[4]; // [rsp+160h] [rbp-40h] BYREF

  v8 = *(_QWORD *)(a1 + 56);
  v9 = *(_QWORD *)(v8 + 16);
  if ( !v9 )
  {
    v10 = (__int64 *)(v8 + 16);
    v11 = *(_BYTE *)(a1 + 64) & 1;
    if ( word_4F06418[0] != 67 )
      goto LABEL_3;
LABEL_6:
    v13 = qword_4D03C50;
    v14 = *(_BYTE *)(qword_4D03C50 + 17LL) & 1;
    *(_BYTE *)(qword_4D03C50 + 17LL) = v11 | *(_BYTE *)(qword_4D03C50 + 17LL) & 0xFE;
    v40 = v14;
    sub_7B8B50(a1, a2, v13, v14);
    sub_69ED20((__int64)&v41, 0, 0, 1);
    if ( !v11 )
    {
LABEL_12:
      *(_BYTE *)(qword_4D03C50 + 17LL) = *(_BYTE *)(qword_4D03C50 + 17LL) & 0xFE | v40;
      result = sub_6F6F40(&v41, 0);
      *v10 = result;
      return result;
    }
    if ( *(_BYTE *)(qword_4D03C50 + 16LL) <= 3u )
    {
      sub_6F69D0(&v41, 0);
      sub_6E6B60(&v41, 0);
      if ( v42.m128i_i8[0] != 2 )
      {
        v15 = v41.m128i_i64[0];
        if ( v42.m128i_i8[0] )
        {
          v18 = *(_BYTE *)(v41.m128i_i64[0] + 140);
          if ( v18 == 12 )
          {
            v19 = v41.m128i_i64[0];
            do
            {
              v19 = *(_QWORD *)(v19 + 160);
              v18 = *(_BYTE *)(v19 + 140);
            }
            while ( v18 == 12 );
          }
          if ( v18 )
          {
            sub_6E68E0(28, &v41);
            v15 = v41.m128i_i64[0];
          }
        }
LABEL_9:
        v16 = v42.m128i_i8[1];
        *(_QWORD *)a1 = v15;
        if ( v16 == 1 )
        {
          if ( !(unsigned int)sub_6ED0A0(&v41) )
          {
            v15 = v41.m128i_i64[0];
            v16 = v42.m128i_i8[1];
            v17 = 1;
            goto LABEL_11;
          }
          v16 = v42.m128i_i8[1];
          v15 = v41.m128i_i64[0];
        }
        v17 = v16 == 3;
LABEL_11:
        *(_BYTE *)(a1 + 25) = v17 | *(_BYTE *)(a1 + 25) & 0xFE;
        a2->m128i_i64[0] = v15;
        a2[1].m128i_i8[1] = v16;
        goto LABEL_12;
      }
      v20 = _mm_loadu_si128(&v42);
      v21 = _mm_loadu_si128(&v43);
      v22 = _mm_loadu_si128(&v44);
      v23 = _mm_loadu_si128(&v45);
      v24 = _mm_loadu_si128(&v46);
      *a2 = _mm_loadu_si128(&v41);
      v25 = _mm_loadu_si128(&v47);
      v26 = _mm_loadu_si128(&v48);
      a2[1] = v20;
      v27 = _mm_loadu_si128(&v49);
      v28 = _mm_loadu_si128(&v50);
      a2[2] = v21;
      a2[3] = v22;
      v29 = _mm_loadu_si128(&v51);
      v30 = _mm_loadu_si128(&v52);
      a2[4] = v23;
      v31 = _mm_loadu_si128(&v53);
      a2[5] = v24;
      v32 = _mm_loadu_si128(&v54);
      a2[6] = v25;
      v33 = _mm_loadu_si128(&v55);
      a2[7] = v26;
      v34 = _mm_loadu_si128(&v56);
      a2[9] = v28;
      a2[10] = v29;
      a2[11] = v30;
      a2[12] = v31;
      a2[13] = v32;
      a2[8] = v27;
      a2[14] = v33;
      a2[15] = v34;
      v35 = _mm_loadu_si128(&v58);
      v36 = _mm_loadu_si128(&v59);
      v37 = _mm_loadu_si128(&v60);
      v38 = _mm_loadu_si128(&v61);
      v39 = _mm_loadu_si128(v62);
      a2[16] = _mm_loadu_si128(&v57);
      a2[18] = v36;
      a2[17] = v35;
      a2[18].m128i_i64[0] = a1;
      a2[19] = v37;
      a2[20] = v38;
      a2[21] = v39;
    }
    v15 = v41.m128i_i64[0];
    goto LABEL_9;
  }
  v10 = (__int64 *)(v9 + 16);
  v11 = (*(_BYTE *)(a1 + 64) & 1) == 0;
  if ( word_4F06418[0] == 67 )
    goto LABEL_6;
LABEL_3:
  if ( !*a3 )
  {
    if ( (unsigned int)sub_6E5430(a1, a2, a3, a4, a5, a6) )
      sub_6851C0(0xFDu, &dword_4F063F8);
  }
  sub_7BE180();
  *a3 = 1;
  result = sub_7305B0();
  *v10 = result;
  return result;
}
