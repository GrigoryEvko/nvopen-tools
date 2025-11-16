// Function: sub_16D6AB0
// Address: 0x16d6ab0
//
__int64 __fastcall sub_16D6AB0(__m128i *a1, __m128i *a2, __int64 a3)
{
  __int64 result; // rax
  __m128i *v4; // rbx
  double v5; // xmm1_8
  _BYTE *v6; // rsi
  double v7; // xmm2_8
  __m128i *v8; // r12
  __int64 v9; // rdx
  double v10; // xmm0_8
  __m128i v11; // xmm2
  _BYTE *v12; // rsi
  __int64 v13; // rdx
  __m128i *v14; // r15
  __m128i *v15; // r12
  __m128i v16; // xmm4
  __int64 v17; // rdi
  double v18; // xmm1_8
  __m128i *v19; // rbx
  __m128i *v20; // r12
  double *m128i_i64; // rax
  double v22; // xmm0_8
  _BYTE *v23; // rsi
  __m128i v24; // xmm4
  __int64 v25; // rdx
  _BYTE *v26; // rsi
  __int64 v27; // rdx
  __m128i v28; // xmm2
  __m128i *v29; // r12
  __m128i *v30; // r15
  __m128i v31; // xmm7
  _BYTE *v32; // rsi
  __int64 v33; // rdx
  __m128i v34; // xmm6
  __m128i v35; // xmm7
  __m128i v36; // xmm3
  __m128i v37; // xmm5
  __m128i *v38; // r9
  __int64 v39; // r12
  const __m128i *v40; // r13
  const __m128i *i; // rbx
  _BYTE *v42; // rsi
  __m128i v43; // xmm5
  __int64 v44; // rdx
  _BYTE *v45; // rsi
  __int64 v46; // rdx
  __m128i v47; // xmm7
  __m128i *v48; // r9
  __m128i *v49; // r15
  _BYTE *v50; // rsi
  __m128i v51; // xmm3
  __int64 v52; // r12
  __int64 v53; // rdx
  _BYTE *v54; // rsi
  __int64 v55; // rdx
  __m128i v56; // xmm3
  __m128i v57; // xmm2
  _BYTE *v58; // rsi
  __int64 v59; // rdx
  __m128i v60; // xmm7
  __m128i v61; // xmm3
  _BYTE *v62; // rsi
  __int64 v63; // rdx
  __m128i v64; // xmm6
  __m128i *v65; // [rsp+0h] [rbp-150h]
  __int64 v66; // [rsp+8h] [rbp-148h]
  __m128i *v67; // [rsp+8h] [rbp-148h]
  __m128i *v68; // [rsp+10h] [rbp-140h]
  __m128i *v69; // [rsp+18h] [rbp-138h]
  __m128i *v70; // [rsp+20h] [rbp-130h]
  __int64 v71; // [rsp+28h] [rbp-128h]
  __m128i *v72; // [rsp+30h] [rbp-120h]
  double *v73; // [rsp+38h] [rbp-118h]
  __m128i v74; // [rsp+60h] [rbp-F0h] BYREF
  __m128i v75; // [rsp+70h] [rbp-E0h] BYREF
  _BYTE *v76; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v77; // [rsp+88h] [rbp-C8h]
  _QWORD v78[2]; // [rsp+90h] [rbp-C0h] BYREF
  _BYTE *v79; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v80; // [rsp+A8h] [rbp-A8h]
  _QWORD v81[2]; // [rsp+B0h] [rbp-A0h] BYREF
  __m128i v82; // [rsp+C0h] [rbp-90h] BYREF
  __m128i v83; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v84[2]; // [rsp+E0h] [rbp-70h] BYREF
  _QWORD v85[2]; // [rsp+F0h] [rbp-60h] BYREF
  __int64 v86[2]; // [rsp+100h] [rbp-50h] BYREF
  _QWORD v87[8]; // [rsp+110h] [rbp-40h] BYREF

  result = (char *)a2 - (char *)a1;
  v71 = a3;
  v72 = a2;
  if ( (char *)a2 - (char *)a1 <= 1536 )
    return result;
  v4 = a1;
  if ( !a3 )
  {
    v38 = a2;
    v70 = a1 + 2;
    v69 = a1 + 4;
    goto LABEL_32;
  }
  v68 = a1 + 6;
  v70 = a1 + 2;
  v69 = a1 + 4;
  while ( 2 )
  {
    v5 = *(double *)v4[6].m128i_i64;
    v6 = (_BYTE *)v4[2].m128i_i64[0];
    --v71;
    v7 = *(double *)v72[-6].m128i_i64;
    v8 = &v4[2 * ((__int64)(0xAAAAAAAAAAAAAAABLL * (((char *)v72 - (char *)v4) >> 5)) / 2)
           + 2
           * ((0xAAAAAAAAAAAAAAABLL * (((char *)v72 - (char *)v4) >> 5)
             + ((0xAAAAAAAAAAAAAAABLL * (((char *)v72 - (char *)v4) >> 5)) >> 63))
            & 0xFFFFFFFFFFFFFFFELL)];
    v9 = (__int64)&v6[v4[2].m128i_i64[1]];
    v10 = *(double *)v8->m128i_i64;
    if ( *(double *)v8->m128i_i64 > v5 )
    {
      if ( v7 > v10 )
      {
        v82 = _mm_loadu_si128(v4);
        v57 = _mm_loadu_si128(v4 + 1);
        v84[0] = (__int64)v85;
        v83 = v57;
        sub_16D5EB0(v84, v6, v9);
        v58 = (_BYTE *)v4[4].m128i_i64[0];
        v59 = (__int64)&v58[v4[4].m128i_i64[1]];
        v86[0] = (__int64)v87;
        sub_16D5EB0(v86, v58, v59);
        *v4 = _mm_loadu_si128(v8);
        v4[1] = _mm_loadu_si128(v8 + 1);
        sub_2240AE0(v70, &v8[2]);
        sub_2240AE0(v69, &v8[4]);
        v60 = _mm_loadu_si128(&v83);
        *v8 = _mm_loadu_si128(&v82);
        v8[1] = v60;
        sub_2240AE0(&v8[2], v84);
        sub_2240AE0(&v8[4], v86);
        v17 = v86[0];
        if ( (_QWORD *)v86[0] == v87 )
          goto LABEL_10;
        goto LABEL_9;
      }
      if ( v7 > v5 )
      {
        v11 = _mm_loadu_si128(v4);
        v83 = _mm_loadu_si128(v4 + 1);
        v82 = v11;
        v84[0] = (__int64)v85;
        sub_16D5EB0(v84, v6, v9);
        v12 = (_BYTE *)v4[4].m128i_i64[0];
        v13 = (__int64)&v12[v4[4].m128i_i64[1]];
        v86[0] = (__int64)v87;
        sub_16D5EB0(v86, v12, v13);
        v14 = v72 - 4;
        *v4 = _mm_loadu_si128(v72 - 6);
        v4[1] = _mm_loadu_si128(v72 - 5);
        sub_2240AE0(v70, &v72[-4]);
        v15 = v72 - 2;
        sub_2240AE0(v69, &v72[-2]);
        v16 = _mm_loadu_si128(&v82);
        v72[-5] = _mm_loadu_si128(&v83);
        v72[-6] = v16;
        goto LABEL_8;
      }
      goto LABEL_30;
    }
    if ( v7 > v5 )
    {
LABEL_30:
      v14 = v4 + 8;
      v15 = v4 + 10;
      v82 = _mm_loadu_si128(v4);
      v35 = _mm_loadu_si128(v4 + 1);
      v84[0] = (__int64)v85;
      v83 = v35;
      sub_16D5EB0(v84, v6, v9);
      v86[0] = (__int64)v87;
      sub_16D5EB0(v86, (_BYTE *)v4[4].m128i_i64[0], v4[4].m128i_i64[0] + v4[4].m128i_i64[1]);
      v36 = _mm_loadu_si128(v4 + 7);
      *v4 = _mm_loadu_si128(v4 + 6);
      v4[1] = v36;
      sub_2240AE0(v70, &v4[8]);
      sub_2240AE0(v69, &v4[10]);
      v37 = _mm_loadu_si128(&v83);
      v4[6] = _mm_loadu_si128(&v82);
      v4[7] = v37;
      goto LABEL_8;
    }
    if ( v7 <= v10 )
    {
      v61 = _mm_loadu_si128(v4 + 1);
      v82 = _mm_loadu_si128(v4);
      v83 = v61;
      v84[0] = (__int64)v85;
      sub_16D5EB0(v84, v6, v9);
      v62 = (_BYTE *)v4[4].m128i_i64[0];
      v63 = (__int64)&v62[v4[4].m128i_i64[1]];
      v86[0] = (__int64)v87;
      sub_16D5EB0(v86, v62, v63);
      *v4 = _mm_loadu_si128(v8);
      v4[1] = _mm_loadu_si128(v8 + 1);
      sub_2240AE0(v70, &v8[2]);
      sub_2240AE0(v69, &v8[4]);
      v64 = _mm_loadu_si128(&v82);
      v8[1] = _mm_loadu_si128(&v83);
      *v8 = v64;
      sub_2240AE0(&v8[2], v84);
      sub_2240AE0(&v8[4], v86);
      v17 = v86[0];
      if ( (_QWORD *)v86[0] == v87 )
        goto LABEL_10;
      goto LABEL_9;
    }
    v31 = _mm_loadu_si128(v4 + 1);
    v82 = _mm_loadu_si128(v4);
    v83 = v31;
    v84[0] = (__int64)v85;
    sub_16D5EB0(v84, v6, v9);
    v32 = (_BYTE *)v4[4].m128i_i64[0];
    v33 = (__int64)&v32[v4[4].m128i_i64[1]];
    v86[0] = (__int64)v87;
    sub_16D5EB0(v86, v32, v33);
    v14 = v72 - 4;
    *v4 = _mm_loadu_si128(v72 - 6);
    v4[1] = _mm_loadu_si128(v72 - 5);
    sub_2240AE0(v70, &v72[-4]);
    v15 = v72 - 2;
    sub_2240AE0(v69, &v72[-2]);
    v34 = _mm_loadu_si128(&v82);
    v72[-5] = _mm_loadu_si128(&v83);
    v72[-6] = v34;
LABEL_8:
    sub_2240AE0(v14, v84);
    sub_2240AE0(v15, v86);
    v17 = v86[0];
    if ( (_QWORD *)v86[0] != v87 )
LABEL_9:
      j_j___libc_free_0(v17, v87[0] + 1LL);
LABEL_10:
    if ( (_QWORD *)v84[0] != v85 )
      j_j___libc_free_0(v84[0], v85[0] + 1LL);
    v18 = *(double *)v4->m128i_i64;
    v73 = (double *)v4;
    v19 = v68;
    v20 = v72;
    while ( 1 )
    {
      if ( v18 > *(double *)v19->m128i_i64 )
        goto LABEL_22;
      m128i_i64 = (double *)v20[-12].m128i_i64;
      if ( *(double *)v20[-6].m128i_i64 <= v18 )
      {
        v20 -= 6;
      }
      else
      {
        do
        {
          v20 = (__m128i *)m128i_i64;
          v22 = *m128i_i64;
          m128i_i64 -= 12;
        }
        while ( v22 > v18 );
      }
      if ( v19 >= v20 )
        break;
      v23 = (_BYTE *)v19[2].m128i_i64[0];
      v24 = _mm_loadu_si128(v19 + 1);
      v25 = (__int64)&v23[v19[2].m128i_i64[1]];
      v82 = _mm_loadu_si128(v19);
      v83 = v24;
      v84[0] = (__int64)v85;
      sub_16D5EB0(v84, v23, v25);
      v26 = (_BYTE *)v19[4].m128i_i64[0];
      v27 = (__int64)&v26[v19[4].m128i_i64[1]];
      v86[0] = (__int64)v87;
      sub_16D5EB0(v86, v26, v27);
      *v19 = _mm_loadu_si128(v20);
      v19[1] = _mm_loadu_si128(v20 + 1);
      sub_2240AE0(&v19[2], &v20[2]);
      sub_2240AE0(&v19[4], &v20[4]);
      v28 = _mm_loadu_si128(&v83);
      *v20 = _mm_loadu_si128(&v82);
      v20[1] = v28;
      sub_2240AE0(&v20[2], v84);
      sub_2240AE0(&v20[4], v86);
      if ( (_QWORD *)v86[0] != v87 )
        j_j___libc_free_0(v86[0], v87[0] + 1LL);
      if ( (_QWORD *)v84[0] != v85 )
        j_j___libc_free_0(v84[0], v85[0] + 1LL);
      v18 = *v73;
LABEL_22:
      v19 += 6;
    }
    v29 = v19;
    v30 = v19;
    v4 = (__m128i *)v73;
    sub_16D6AB0(v29, v72, v71);
    result = (char *)v29 - (char *)v73;
    if ( (char *)v29 - (char *)v73 > 1536 )
    {
      if ( v71 )
      {
        v72 = v29;
        continue;
      }
      v38 = v30;
LABEL_32:
      v65 = v38;
      v66 = 0xAAAAAAAAAAAAAAABLL * (result >> 5);
      v39 = (v66 - 2) >> 1;
      v40 = v4;
      for ( i = &v4[2 * v39 + 2 * ((v66 - 2) & 0xFFFFFFFFFFFFFFFELL)]; ; i -= 6 )
      {
        v42 = (_BYTE *)i[2].m128i_i64[0];
        v43 = _mm_loadu_si128(i + 1);
        v44 = (__int64)&v42[i[2].m128i_i64[1]];
        v74 = _mm_loadu_si128(i);
        v75 = v43;
        v76 = v78;
        sub_16D5EB0((__int64 *)&v76, v42, v44);
        v45 = (_BYTE *)i[4].m128i_i64[0];
        v46 = (__int64)&v45[i[4].m128i_i64[1]];
        v79 = v81;
        sub_16D5EB0((__int64 *)&v79, v45, v46);
        v47 = _mm_loadu_si128(&v75);
        v82 = _mm_loadu_si128(&v74);
        v83 = v47;
        v84[0] = (__int64)v85;
        sub_16D5EB0(v84, v76, (__int64)&v76[v77]);
        v86[0] = (__int64)v87;
        sub_16D5EB0(v86, v79, (__int64)&v79[v80]);
        sub_16D6720((__int64)v40, v39, v66, &v82);
        if ( (_QWORD *)v86[0] != v87 )
          j_j___libc_free_0(v86[0], v87[0] + 1LL);
        if ( (_QWORD *)v84[0] != v85 )
          j_j___libc_free_0(v84[0], v85[0] + 1LL);
        if ( !v39 )
          break;
        --v39;
        if ( v79 != (_BYTE *)v81 )
          j_j___libc_free_0(v79, v81[0] + 1LL);
        if ( v76 != (_BYTE *)v78 )
          j_j___libc_free_0(v76, v78[0] + 1LL);
      }
      v48 = v65;
      if ( v79 != (_BYTE *)v81 )
      {
        j_j___libc_free_0(v79, v81[0] + 1LL);
        v48 = v65;
      }
      if ( v76 != (_BYTE *)v78 )
      {
        v67 = v48;
        j_j___libc_free_0(v76, v78[0] + 1LL);
        v48 = v67;
      }
      v49 = v48 - 6;
      do
      {
        v50 = (_BYTE *)v49[2].m128i_i64[0];
        v51 = _mm_loadu_si128(v49);
        v52 = (char *)v49 - (char *)v40;
        v53 = (__int64)&v50[v49[2].m128i_i64[1]];
        v75 = _mm_loadu_si128(v49 + 1);
        v74 = v51;
        v76 = v78;
        sub_16D5EB0((__int64 *)&v76, v50, v53);
        v54 = (_BYTE *)v49[4].m128i_i64[0];
        v55 = (__int64)&v54[v49[4].m128i_i64[1]];
        v79 = v81;
        sub_16D5EB0((__int64 *)&v79, v54, v55);
        *v49 = _mm_loadu_si128(v40);
        v49[1] = _mm_loadu_si128(v40 + 1);
        sub_2240AE0(&v49[2], v70);
        sub_2240AE0(&v49[4], v69);
        v56 = _mm_loadu_si128(&v75);
        v82 = _mm_loadu_si128(&v74);
        v83 = v56;
        v84[0] = (__int64)v85;
        sub_16D5EB0(v84, v76, (__int64)&v76[v77]);
        v86[0] = (__int64)v87;
        sub_16D5EB0(v86, v79, (__int64)&v79[v80]);
        result = sub_16D6720((__int64)v40, 0, 0xAAAAAAAAAAAAAAABLL * (((char *)v49 - (char *)v40) >> 5), &v82);
        if ( (_QWORD *)v86[0] != v87 )
          result = j_j___libc_free_0(v86[0], v87[0] + 1LL);
        if ( (_QWORD *)v84[0] != v85 )
          result = j_j___libc_free_0(v84[0], v85[0] + 1LL);
        if ( v79 != (_BYTE *)v81 )
          result = j_j___libc_free_0(v79, v81[0] + 1LL);
        if ( v76 != (_BYTE *)v78 )
          result = j_j___libc_free_0(v76, v78[0] + 1LL);
        v49 -= 6;
      }
      while ( v52 > 96 );
    }
    return result;
  }
}
