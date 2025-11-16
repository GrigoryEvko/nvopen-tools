// Function: sub_C9D290
// Address: 0xc9d290
//
__int64 __fastcall sub_C9D290(__int64 a1, __m128i *a2, __int64 a3)
{
  __int64 result; // rax
  double *v4; // rbx
  double v5; // xmm1_8
  _BYTE *v6; // rsi
  double v7; // xmm2_8
  __m128i *v8; // r12
  double v9; // xmm0_8
  __int64 v10; // rdx
  __m128i v11; // xmm4
  __m128i v12; // xmm2
  _BYTE *v13; // rsi
  __int64 v14; // rdx
  double *m128i_i64; // r15
  double *v16; // r12
  __m128i *v17; // rcx
  __m128i v18; // xmm5
  __int64 v19; // rdi
  double v20; // xmm1_8
  __m128i *v21; // rbx
  __m128i *v22; // r12
  double *v23; // rax
  double v24; // xmm0_8
  _BYTE *v25; // rsi
  __int64 v26; // rdx
  __m128i v27; // xmm3
  __m128i v28; // xmm4
  _BYTE *v29; // rsi
  __int64 v30; // rdx
  __m128i v31; // xmm7
  __m128i v32; // xmm2
  __m128i *v33; // r12
  __m128i *v34; // r15
  __m128i v35; // xmm6
  __m128i v36; // xmm7
  _BYTE *v37; // rsi
  __int64 v38; // rdx
  __m128i v39; // xmm7
  __m128i v40; // xmm2
  __m128i v41; // xmm3
  __m128i v42; // xmm4
  __m128i v43; // xmm5
  __m128i *v44; // r9
  __int64 v45; // r12
  const __m128i *v46; // r13
  const __m128i *i; // rbx
  _BYTE *v48; // rsi
  __m128i v49; // xmm4
  __m128i v50; // xmm5
  __int64 v51; // rdx
  _BYTE *v52; // rsi
  __int64 v53; // rdx
  __m128i v54; // xmm6
  __m128i v55; // xmm7
  __m128i *v56; // r9
  __m128i *v57; // r15
  _BYTE *v58; // rsi
  __int64 v59; // rdx
  __m128i v60; // xmm4
  __int64 v61; // r12
  __m128i v62; // xmm3
  _BYTE *v63; // rsi
  __int64 v64; // rdx
  __m128i v65; // xmm7
  __m128i v66; // xmm3
  __int64 v67; // rcx
  __m128i v68; // xmm2
  _BYTE *v69; // rsi
  __int64 v70; // rdx
  __m128i v71; // xmm6
  __m128i v72; // xmm7
  __m128i v73; // xmm2
  __m128i v74; // xmm3
  _BYTE *v75; // rsi
  __int64 v76; // rdx
  __m128i v77; // xmm6
  __m128i v78; // xmm7
  __m128i *v79; // [rsp+0h] [rbp-170h]
  __int64 v80; // [rsp+8h] [rbp-168h]
  __m128i *v81; // [rsp+8h] [rbp-168h]
  __m128i *v82; // [rsp+10h] [rbp-160h]
  __int64 v83; // [rsp+18h] [rbp-158h]
  __int64 v84; // [rsp+20h] [rbp-150h]
  __int64 v85; // [rsp+28h] [rbp-148h]
  __m128i *v86; // [rsp+30h] [rbp-140h]
  double *v87; // [rsp+38h] [rbp-138h]
  __m128i v88; // [rsp+60h] [rbp-110h] BYREF
  __m128i v89; // [rsp+70h] [rbp-100h] BYREF
  __int64 v90; // [rsp+80h] [rbp-F0h]
  _BYTE *v91; // [rsp+88h] [rbp-E8h] BYREF
  __int64 v92; // [rsp+90h] [rbp-E0h]
  _QWORD v93[2]; // [rsp+98h] [rbp-D8h] BYREF
  _BYTE *v94; // [rsp+A8h] [rbp-C8h] BYREF
  __int64 v95; // [rsp+B0h] [rbp-C0h]
  _QWORD v96[3]; // [rsp+B8h] [rbp-B8h] BYREF
  __m128i v97; // [rsp+D0h] [rbp-A0h] BYREF
  __m128i v98; // [rsp+E0h] [rbp-90h] BYREF
  __int64 v99; // [rsp+F0h] [rbp-80h]
  __int64 v100[2]; // [rsp+F8h] [rbp-78h] BYREF
  _QWORD v101[2]; // [rsp+108h] [rbp-68h] BYREF
  __int64 v102[2]; // [rsp+118h] [rbp-58h] BYREF
  _QWORD v103[9]; // [rsp+128h] [rbp-48h] BYREF

  result = (__int64)a2->m128i_i64 - a1;
  v85 = a3;
  v86 = a2;
  if ( (__int64)a2->m128i_i64 - a1 <= 1664 )
    return result;
  v4 = (double *)a1;
  if ( !a3 )
  {
    v44 = a2;
    v84 = a1 + 40;
    v83 = a1 + 72;
    goto LABEL_33;
  }
  v82 = (__m128i *)(a1 + 104);
  v84 = a1 + 40;
  v83 = a1 + 72;
  while ( 2 )
  {
    v5 = v4[13];
    v6 = (_BYTE *)*((_QWORD *)v4 + 5);
    --v85;
    v7 = *(double *)&v86[-7].m128i_i64[1];
    v8 = (__m128i *)&v4[4 * (0x4EC4EC4EC4EC4EC5LL * (((char *)v86 - (char *)v4) >> 3) / 2)
                      + 4
                      * ((0x4EC4EC4EC4EC4EC5LL * (((char *)v86 - (char *)v4) >> 3)
                        + ((unsigned __int64)(0x4EC4EC4EC4EC4EC5LL * (((char *)v86 - (char *)v4) >> 3)) >> 63))
                       & 0xFFFFFFFFFFFFFFFELL)
                      + 0x4EC4EC4EC4EC4EC5LL * (((char *)v86 - (char *)v4) >> 3) / 2];
    v9 = *(double *)v8->m128i_i64;
    v10 = (__int64)&v6[*((_QWORD *)v4 + 6)];
    if ( *(double *)v8->m128i_i64 <= v5 )
    {
      if ( v7 <= v5 )
      {
        if ( v7 <= v9 )
        {
          v73 = _mm_loadu_si128((const __m128i *)v4);
          v74 = _mm_loadu_si128((const __m128i *)v4 + 1);
          v99 = *((_QWORD *)v4 + 4);
          v97 = v73;
          v98 = v74;
          v100[0] = (__int64)v101;
          sub_C9CAB0(v100, v6, v10);
          v75 = (_BYTE *)*((_QWORD *)v4 + 9);
          v76 = (__int64)&v75[*((_QWORD *)v4 + 10)];
          v102[0] = (__int64)v103;
          sub_C9CAB0(v102, v75, v76);
          *(__m128i *)v4 = _mm_loadu_si128(v8);
          *((__m128i *)v4 + 1) = _mm_loadu_si128(v8 + 1);
          v4[4] = *(double *)v8[2].m128i_i64;
          sub_2240AE0(v84, &v8[2].m128i_u64[1]);
          sub_2240AE0(v83, &v8[4].m128i_u64[1]);
          v77 = _mm_loadu_si128(&v97);
          v78 = _mm_loadu_si128(&v98);
          v8[2].m128i_i64[0] = v99;
          *v8 = v77;
          v8[1] = v78;
          sub_2240AE0(&v8[2].m128i_u64[1], v100);
          sub_2240AE0(&v8[4].m128i_u64[1], v102);
          v19 = v102[0];
          if ( (_QWORD *)v102[0] == v103 )
            goto LABEL_11;
          goto LABEL_10;
        }
        v35 = _mm_loadu_si128((const __m128i *)v4);
        v36 = _mm_loadu_si128((const __m128i *)v4 + 1);
        v99 = *((_QWORD *)v4 + 4);
        v97 = v35;
        v98 = v36;
        v100[0] = (__int64)v101;
        sub_C9CAB0(v100, v6, v10);
        v37 = (_BYTE *)*((_QWORD *)v4 + 9);
        v38 = (__int64)&v37[*((_QWORD *)v4 + 10)];
        v102[0] = (__int64)v103;
        sub_C9CAB0(v102, v37, v38);
        m128i_i64 = (double *)v86[-4].m128i_i64;
        *(__m128i *)v4 = _mm_loadu_si128((__m128i *)((char *)v86 - 104));
        *((__m128i *)v4 + 1) = _mm_loadu_si128((__m128i *)((char *)v86 - 88));
        v4[4] = *(double *)&v86[-5].m128i_i64[1];
        sub_2240AE0(v84, &v86[-4]);
        v16 = (double *)v86[-2].m128i_i64;
        sub_2240AE0(v83, &v86[-2]);
        v17 = v86;
        v39 = _mm_loadu_si128(&v98);
        *(__m128i *)((char *)v86 - 104) = _mm_loadu_si128(&v97);
        *(__m128i *)((char *)v86 - 88) = v39;
        goto LABEL_8;
      }
    }
    else
    {
      if ( v7 > v9 )
      {
        v67 = *((_QWORD *)v4 + 4);
        v97 = _mm_loadu_si128((const __m128i *)v4);
        v68 = _mm_loadu_si128((const __m128i *)v4 + 1);
        v99 = v67;
        v98 = v68;
        v100[0] = (__int64)v101;
        sub_C9CAB0(v100, v6, v10);
        v69 = (_BYTE *)*((_QWORD *)v4 + 9);
        v70 = (__int64)&v69[*((_QWORD *)v4 + 10)];
        v102[0] = (__int64)v103;
        sub_C9CAB0(v102, v69, v70);
        *(__m128i *)v4 = _mm_loadu_si128(v8);
        *((__m128i *)v4 + 1) = _mm_loadu_si128(v8 + 1);
        v4[4] = *(double *)v8[2].m128i_i64;
        sub_2240AE0(v84, &v8[2].m128i_u64[1]);
        sub_2240AE0(v83, &v8[4].m128i_u64[1]);
        v71 = _mm_loadu_si128(&v97);
        v72 = _mm_loadu_si128(&v98);
        v8[2].m128i_i64[0] = v99;
        *v8 = v71;
        v8[1] = v72;
        sub_2240AE0(&v8[2].m128i_u64[1], v100);
        sub_2240AE0(&v8[4].m128i_u64[1], v102);
        v19 = v102[0];
        if ( (_QWORD *)v102[0] == v103 )
          goto LABEL_11;
        goto LABEL_10;
      }
      if ( v7 > v5 )
      {
        v11 = _mm_loadu_si128((const __m128i *)v4 + 1);
        v12 = _mm_loadu_si128((const __m128i *)v4);
        v99 = *((_QWORD *)v4 + 4);
        v98 = v11;
        v97 = v12;
        v100[0] = (__int64)v101;
        sub_C9CAB0(v100, v6, v10);
        v13 = (_BYTE *)*((_QWORD *)v4 + 9);
        v14 = (__int64)&v13[*((_QWORD *)v4 + 10)];
        v102[0] = (__int64)v103;
        sub_C9CAB0(v102, v13, v14);
        m128i_i64 = (double *)v86[-4].m128i_i64;
        *(__m128i *)v4 = _mm_loadu_si128((__m128i *)((char *)v86 - 104));
        *((__m128i *)v4 + 1) = _mm_loadu_si128((__m128i *)((char *)v86 - 88));
        v4[4] = *(double *)&v86[-5].m128i_i64[1];
        sub_2240AE0(v84, &v86[-4]);
        v16 = (double *)v86[-2].m128i_i64;
        sub_2240AE0(v83, &v86[-2]);
        v17 = v86;
        v18 = _mm_loadu_si128(&v98);
        *(__m128i *)((char *)v86 - 104) = _mm_loadu_si128(&v97);
        *(__m128i *)((char *)v86 - 88) = v18;
LABEL_8:
        v17[-5].m128i_i64[1] = v99;
        goto LABEL_9;
      }
    }
    m128i_i64 = v4 + 18;
    v16 = v4 + 22;
    v97 = _mm_loadu_si128((const __m128i *)v4);
    v98 = _mm_loadu_si128((const __m128i *)v4 + 1);
    v99 = *((_QWORD *)v4 + 4);
    v100[0] = (__int64)v101;
    sub_C9CAB0(v100, v6, v10);
    v102[0] = (__int64)v103;
    sub_C9CAB0(v102, *((_BYTE **)v4 + 9), *((_QWORD *)v4 + 9) + *((_QWORD *)v4 + 10));
    v40 = _mm_loadu_si128((const __m128i *)(v4 + 13));
    v41 = _mm_loadu_si128((const __m128i *)(v4 + 15));
    v4[4] = v4[17];
    *(__m128i *)v4 = v40;
    *((__m128i *)v4 + 1) = v41;
    sub_2240AE0(v84, v4 + 18);
    sub_2240AE0(v83, v4 + 22);
    v42 = _mm_loadu_si128(&v97);
    v43 = _mm_loadu_si128(&v98);
    *((_QWORD *)v4 + 17) = v99;
    *(__m128i *)(v4 + 13) = v42;
    *(__m128i *)(v4 + 15) = v43;
LABEL_9:
    sub_2240AE0(m128i_i64, v100);
    sub_2240AE0(v16, v102);
    v19 = v102[0];
    if ( (_QWORD *)v102[0] != v103 )
LABEL_10:
      j_j___libc_free_0(v19, v103[0] + 1LL);
LABEL_11:
    if ( (_QWORD *)v100[0] != v101 )
      j_j___libc_free_0(v100[0], v101[0] + 1LL);
    v20 = *v4;
    v87 = v4;
    v21 = v82;
    v22 = v86;
    while ( 1 )
    {
      if ( v20 > *(double *)v21->m128i_i64 )
        goto LABEL_23;
      v23 = (double *)v22[-13].m128i_i64;
      if ( *(double *)&v22[-7].m128i_i64[1] <= v20 )
      {
        v22 = (__m128i *)((char *)v22 - 104);
      }
      else
      {
        do
        {
          v22 = (__m128i *)v23;
          v24 = *v23;
          v23 -= 13;
        }
        while ( v24 > v20 );
      }
      if ( v21 >= v22 )
        break;
      v25 = (_BYTE *)v21[2].m128i_i64[1];
      v26 = v21[3].m128i_i64[0];
      v27 = _mm_loadu_si128(v21);
      v28 = _mm_loadu_si128(v21 + 1);
      v99 = v21[2].m128i_i64[0];
      v97 = v27;
      v98 = v28;
      v100[0] = (__int64)v101;
      sub_C9CAB0(v100, v25, (__int64)&v25[v26]);
      v29 = (_BYTE *)v21[4].m128i_i64[1];
      v30 = (__int64)&v29[v21[5].m128i_i64[0]];
      v102[0] = (__int64)v103;
      sub_C9CAB0(v102, v29, v30);
      *v21 = _mm_loadu_si128(v22);
      v21[1] = _mm_loadu_si128(v22 + 1);
      v21[2].m128i_i64[0] = v22[2].m128i_i64[0];
      sub_2240AE0(&v21[2].m128i_u64[1], &v22[2].m128i_u64[1]);
      sub_2240AE0(&v21[4].m128i_u64[1], &v22[4].m128i_u64[1]);
      v31 = _mm_loadu_si128(&v97);
      v32 = _mm_loadu_si128(&v98);
      v22[2].m128i_i64[0] = v99;
      *v22 = v31;
      v22[1] = v32;
      sub_2240AE0(&v22[2].m128i_u64[1], v100);
      sub_2240AE0(&v22[4].m128i_u64[1], v102);
      if ( (_QWORD *)v102[0] != v103 )
        j_j___libc_free_0(v102[0], v103[0] + 1LL);
      if ( (_QWORD *)v100[0] != v101 )
        j_j___libc_free_0(v100[0], v101[0] + 1LL);
      v20 = *v87;
LABEL_23:
      v21 = (__m128i *)((char *)v21 + 104);
    }
    v33 = v21;
    v34 = v21;
    v4 = v87;
    sub_C9D290(v33, v86, v85);
    result = (char *)v33 - (char *)v87;
    if ( (char *)v33 - (char *)v87 > 1664 )
    {
      if ( v85 )
      {
        v86 = v33;
        continue;
      }
      v44 = v34;
LABEL_33:
      v79 = v44;
      v80 = 0x4EC4EC4EC4EC4EC5LL * (result >> 3);
      v45 = (v80 - 2) >> 1;
      v46 = (const __m128i *)v4;
      for ( i = (const __m128i *)&v4[4 * v45 + 4 * ((v80 - 2) & 0xFFFFFFFFFFFFFFFELL) + v45];
            ;
            i = (const __m128i *)((char *)i - 104) )
      {
        v48 = (_BYTE *)i[2].m128i_i64[1];
        v49 = _mm_loadu_si128(i);
        v50 = _mm_loadu_si128(i + 1);
        v90 = i[2].m128i_i64[0];
        v51 = i[3].m128i_i64[0];
        v88 = v49;
        v89 = v50;
        v91 = v93;
        sub_C9CAB0((__int64 *)&v91, v48, (__int64)&v48[v51]);
        v52 = (_BYTE *)i[4].m128i_i64[1];
        v53 = (__int64)&v52[i[5].m128i_i64[0]];
        v94 = v96;
        sub_C9CAB0((__int64 *)&v94, v52, v53);
        v54 = _mm_loadu_si128(&v88);
        v55 = _mm_loadu_si128(&v89);
        v99 = v90;
        v97 = v54;
        v98 = v55;
        v100[0] = (__int64)v101;
        sub_C9CAB0(v100, v91, (__int64)&v91[v92]);
        v102[0] = (__int64)v103;
        sub_C9CAB0(v102, v94, (__int64)&v94[v95]);
        sub_C9CED0((__int64)v46, v45, v80, &v97);
        if ( (_QWORD *)v102[0] != v103 )
          j_j___libc_free_0(v102[0], v103[0] + 1LL);
        if ( (_QWORD *)v100[0] != v101 )
          j_j___libc_free_0(v100[0], v101[0] + 1LL);
        if ( !v45 )
          break;
        --v45;
        if ( v94 != (_BYTE *)v96 )
          j_j___libc_free_0(v94, v96[0] + 1LL);
        if ( v91 != (_BYTE *)v93 )
          j_j___libc_free_0(v91, v93[0] + 1LL);
      }
      v56 = v79;
      if ( v94 != (_BYTE *)v96 )
      {
        j_j___libc_free_0(v94, v96[0] + 1LL);
        v56 = v79;
      }
      if ( v91 != (_BYTE *)v93 )
      {
        v81 = v56;
        j_j___libc_free_0(v91, v93[0] + 1LL);
        v56 = v81;
      }
      v57 = (__m128i *)((char *)v56 - 104);
      do
      {
        v58 = (_BYTE *)v57[2].m128i_i64[1];
        v59 = v57[3].m128i_i64[0];
        v60 = _mm_loadu_si128(v57 + 1);
        v61 = (char *)v57 - (char *)v46;
        v62 = _mm_loadu_si128(v57);
        v90 = v57[2].m128i_i64[0];
        v89 = v60;
        v88 = v62;
        v91 = v93;
        sub_C9CAB0((__int64 *)&v91, v58, (__int64)&v58[v59]);
        v63 = (_BYTE *)v57[4].m128i_i64[1];
        v64 = (__int64)&v63[v57[5].m128i_i64[0]];
        v94 = v96;
        sub_C9CAB0((__int64 *)&v94, v63, v64);
        *v57 = _mm_loadu_si128(v46);
        v57[1] = _mm_loadu_si128(v46 + 1);
        v57[2].m128i_i64[0] = v46[2].m128i_i64[0];
        sub_2240AE0(&v57[2].m128i_u64[1], v84);
        sub_2240AE0(&v57[4].m128i_u64[1], v83);
        v65 = _mm_loadu_si128(&v88);
        v66 = _mm_loadu_si128(&v89);
        v99 = v90;
        v97 = v65;
        v98 = v66;
        v100[0] = (__int64)v101;
        sub_C9CAB0(v100, v91, (__int64)&v91[v92]);
        v102[0] = (__int64)v103;
        sub_C9CAB0(v102, v94, (__int64)&v94[v95]);
        result = sub_C9CED0((__int64)v46, 0, 0x4EC4EC4EC4EC4EC5LL * (((char *)v57 - (char *)v46) >> 3), &v97);
        if ( (_QWORD *)v102[0] != v103 )
          result = j_j___libc_free_0(v102[0], v103[0] + 1LL);
        if ( (_QWORD *)v100[0] != v101 )
          result = j_j___libc_free_0(v100[0], v101[0] + 1LL);
        if ( v94 != (_BYTE *)v96 )
          result = j_j___libc_free_0(v94, v96[0] + 1LL);
        if ( v91 != (_BYTE *)v93 )
          result = j_j___libc_free_0(v91, v93[0] + 1LL);
        v57 = (__m128i *)((char *)v57 - 104);
      }
      while ( v61 > 104 );
    }
    return result;
  }
}
