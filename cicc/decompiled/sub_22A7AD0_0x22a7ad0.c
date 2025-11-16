// Function: sub_22A7AD0
// Address: 0x22a7ad0
//
__int64 __fastcall sub_22A7AD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v7; // rax
  __m128i *v8; // r14
  const __m128i *v9; // r13
  __int64 v10; // r12
  __int64 v12; // r15
  __int64 v13; // r12
  __int64 v14; // r11
  __int64 v15; // r8
  __int64 v16; // rax
  __m128i *v17; // r8
  __int64 v18; // rcx
  __int8 *v19; // r10
  int v20; // r11d
  __int64 v21; // r14
  char *v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rsi
  __int64 v27; // r9
  __int8 *v28; // rax
  __int64 v29; // rdx
  __int64 v30; // r9
  const __m128i *v31; // rax
  __int64 v32; // rdi
  __int8 *v33; // rsi
  __int64 v34; // rax
  __m128i *v35; // rdx
  __int64 v36; // rsi
  __int64 v37; // r8
  __int64 v38; // rax
  __int64 result; // rax
  __int64 v40; // r15
  __int64 v41; // rcx
  __int64 v42; // rdx
  const __m128i *v43; // rax
  __int64 v44; // rsi
  __int64 v45; // r15
  __int64 v46; // rdi
  __int64 v47; // rcx
  __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v50; // rsi
  const __m128i *v51; // r15
  const __m128i *v52; // r12
  const __m128i *v53; // r15
  __m128i *v54; // r14
  __int64 v55; // r8
  __int64 *v56; // rdx
  unsigned __int8 v57; // al
  unsigned __int32 v58; // ecx
  unsigned __int32 v59; // eax
  unsigned __int32 v60; // edi
  unsigned __int32 v61; // edi
  unsigned __int32 v62; // eax
  unsigned __int32 v63; // eax
  char v64; // al
  __int64 v65; // r15
  __int64 v66; // rdx
  __int64 v67; // rdi
  __int64 v68; // rdx
  __int64 v69; // rsi
  __int64 v70; // r9
  __m128i *v71; // rax
  __int64 v72; // rdx
  __int64 v73; // rsi
  __m128i *v74; // rax
  __int64 v75; // r9
  __int8 *v76; // rdx
  __int64 v77; // rax
  __int64 v78; // rdi
  __int64 v79; // r8
  __int64 v80; // rax
  const __m128i *v81; // r15
  __int64 v82; // rdx
  __int64 v83; // rcx
  int v84; // [rsp+8h] [rbp-58h]
  int v85; // [rsp+8h] [rbp-58h]
  __int64 v86; // [rsp+10h] [rbp-50h]
  __m128i *v87; // [rsp+10h] [rbp-50h]
  int v88; // [rsp+10h] [rbp-50h]
  __m128i *v90; // [rsp+20h] [rbp-40h]
  char *v91; // [rsp+20h] [rbp-40h]
  __int64 v92; // [rsp+20h] [rbp-40h]
  __int64 v93; // [rsp+20h] [rbp-40h]
  const __m128i *v94; // [rsp+20h] [rbp-40h]
  __int64 v95; // [rsp+20h] [rbp-40h]
  __int64 v96; // [rsp+20h] [rbp-40h]
  int v97; // [rsp+20h] [rbp-40h]
  char *v98; // [rsp+28h] [rbp-38h]
  __int64 *v99; // [rsp+28h] [rbp-38h]

  v7 = a5;
  v8 = (__m128i *)a3;
  v9 = (const __m128i *)a1;
  v10 = a2;
  if ( a7 <= a5 )
    v7 = a7;
  if ( a4 > v7 )
  {
    v12 = a5;
    if ( a7 >= a5 )
      goto LABEL_46;
    v13 = a4;
    v14 = a1;
    v15 = a2;
    if ( a4 <= v12 )
      goto LABEL_14;
LABEL_6:
    v84 = v14;
    v90 = (__m128i *)v15;
    v86 = v14 + 56 * (v13 / 2);
    v16 = sub_22A7890(v15, a3, v86);
    v17 = v90;
    v18 = v13 / 2;
    v19 = (__int8 *)v86;
    v20 = v84;
    v98 = (char *)v16;
    v13 -= v13 / 2;
    v21 = 0x6DB6DB6DB6DB6DB7LL * ((v16 - (__int64)v90) >> 3);
    if ( v13 <= v21 )
      goto LABEL_15;
LABEL_7:
    if ( v21 <= a7 )
    {
      v22 = v19;
      if ( !v21 )
        goto LABEL_9;
      v67 = v98 - (char *)v17;
      v96 = (char *)v17 - v19;
      v68 = 0x6DB6DB6DB6DB6DB7LL * (((char *)v17 - v19) >> 3);
      v69 = 0x6DB6DB6DB6DB6DB7LL * ((v98 - (char *)v17) >> 3);
      if ( v98 - (char *)v17 <= 0 )
      {
        if ( v96 <= 0 )
          goto LABEL_9;
        v73 = 0;
        v67 = 0;
      }
      else
      {
        v70 = a6;
        v71 = v17;
        do
        {
          v72 = v71[3].m128i_i64[0];
          v70 += 56;
          v71 = (__m128i *)((char *)v71 + 56);
          *(_QWORD *)(v70 - 8) = v72;
          *(__m128i *)(v70 - 40) = _mm_loadu_si128((__m128i *)((char *)v71 - 40));
          *(__m128i *)(v70 - 24) = _mm_loadu_si128((__m128i *)((char *)v71 - 24));
          *(__m128i *)(v70 - 56) = _mm_loadu_si128((__m128i *)((char *)v71 - 56));
          --v69;
        }
        while ( v69 );
        v68 = 0x6DB6DB6DB6DB6DB7LL * (((char *)v17 - v19) >> 3);
        if ( v67 <= 0 )
          v67 = 56;
        v73 = 0x6DB6DB6DB6DB6DB7LL * (v67 >> 3);
        if ( v96 <= 0 )
          goto LABEL_88;
      }
      v74 = (__m128i *)v98;
      do
      {
        v75 = v17[-1].m128i_i64[1];
        v17 = (__m128i *)((char *)v17 - 56);
        v74 = (__m128i *)((char *)v74 - 56);
        v74[3].m128i_i64[0] = v75;
        v74[1] = _mm_loadu_si128(v17 + 1);
        v74[2] = _mm_loadu_si128(v17 + 2);
        *v74 = _mm_loadu_si128(v17);
        --v68;
      }
      while ( v68 );
LABEL_88:
      if ( v67 <= 0 )
      {
        v22 = v19;
      }
      else
      {
        v76 = v19;
        v77 = a6;
        v78 = v73;
        do
        {
          v79 = *(_QWORD *)(v77 + 48);
          v76 += 56;
          v77 += 56;
          *((_QWORD *)v76 - 1) = v79;
          *(__m128i *)(v76 - 40) = _mm_loadu_si128((const __m128i *)(v77 - 40));
          *(__m128i *)(v76 - 24) = _mm_loadu_si128((const __m128i *)(v77 - 24));
          *(__m128i *)(v76 - 56) = _mm_loadu_si128((const __m128i *)(v77 - 56));
          --v78;
        }
        while ( v78 );
        v80 = 56 * v73;
        if ( v73 <= 0 )
          v80 = 56;
        v22 = &v19[v80];
      }
      goto LABEL_9;
    }
    while ( 1 )
    {
LABEL_15:
      if ( v13 > a7 )
      {
        v85 = v20;
        v88 = v18;
        v97 = (int)v19;
        v22 = sub_22A6590(v19, v17->m128i_i8, v98);
        LODWORD(v19) = v97;
        LODWORD(v18) = v88;
        v20 = v85;
        goto LABEL_9;
      }
      v22 = v98;
      if ( !v13 )
        goto LABEL_9;
      v93 = v98 - (char *)v17;
      v25 = 0x6DB6DB6DB6DB6DB7LL * ((v98 - (char *)v17) >> 3);
      v26 = 0x6DB6DB6DB6DB6DB7LL * (((char *)v17 - v19) >> 3);
      if ( (char *)v17 - v19 <= 0 )
      {
        if ( v93 <= 0 )
          goto LABEL_9;
        v31 = (const __m128i *)a6;
        v32 = 0;
        v30 = 0;
      }
      else
      {
        v27 = a6;
        v28 = v19;
        do
        {
          v29 = *((_QWORD *)v28 + 6);
          v27 += 56;
          v28 += 56;
          *(_QWORD *)(v27 - 8) = v29;
          *(__m128i *)(v27 - 40) = _mm_loadu_si128((const __m128i *)(v28 - 40));
          *(__m128i *)(v27 - 24) = _mm_loadu_si128((const __m128i *)(v28 - 24));
          *(__m128i *)(v27 - 56) = _mm_loadu_si128((const __m128i *)(v28 - 56));
          --v26;
        }
        while ( v26 );
        v30 = 56;
        v25 = 0x6DB6DB6DB6DB6DB7LL * ((v98 - (char *)v17) >> 3);
        if ( (char *)v17 - v19 > 0 )
          v30 = (char *)v17 - v19;
        v31 = (const __m128i *)(a6 + v30);
        v32 = 0x6DB6DB6DB6DB6DB7LL * (v30 >> 3);
        if ( v93 <= 0 )
          goto LABEL_26;
      }
      v94 = v31;
      v33 = v19;
      do
      {
        v34 = v17[3].m128i_i64[0];
        v33 += 56;
        v17 = (__m128i *)((char *)v17 + 56);
        *((_QWORD *)v33 - 1) = v34;
        *(__m128i *)(v33 - 40) = _mm_loadu_si128((__m128i *)((char *)v17 - 40));
        *(__m128i *)(v33 - 24) = _mm_loadu_si128((__m128i *)((char *)v17 - 24));
        *(__m128i *)(v33 - 56) = _mm_loadu_si128((__m128i *)((char *)v17 - 56));
        --v25;
      }
      while ( v25 );
      v31 = v94;
LABEL_26:
      if ( v30 <= 0 )
      {
        v22 = v98;
      }
      else
      {
        v35 = (__m128i *)v98;
        v36 = v32;
        do
        {
          v37 = v31[-1].m128i_i64[1];
          v31 = (const __m128i *)((char *)v31 - 56);
          v35 = (__m128i *)((char *)v35 - 56);
          v35[3].m128i_i64[0] = v37;
          v35[1] = _mm_loadu_si128(v31 + 1);
          v35[2] = _mm_loadu_si128(v31 + 2);
          *v35 = _mm_loadu_si128(v31);
          --v36;
        }
        while ( v36 );
        v38 = -56 * v32;
        if ( v32 <= 0 )
          v38 = -56;
        v22 = &v98[v38];
      }
LABEL_9:
      v12 -= v21;
      v91 = v22;
      sub_22A7AD0(v20, (_DWORD)v19, (_DWORD)v22, v18, v21, a6, a7);
      v23 = a7;
      if ( v12 <= a7 )
        v23 = v12;
      if ( v13 <= v23 )
      {
        v8 = (__m128i *)a3;
        v10 = (__int64)v98;
        v9 = (const __m128i *)v91;
        break;
      }
      if ( v12 <= a7 )
      {
        v8 = (__m128i *)a3;
        v10 = (__int64)v98;
        v9 = (const __m128i *)v91;
LABEL_46:
        result = 0x6DB6DB6DB6DB6DB7LL;
        v46 = (__int64)v8->m128i_i64 - v10;
        v47 = 0x6DB6DB6DB6DB6DB7LL * (((__int64)v8->m128i_i64 - v10) >> 3);
        if ( (__int64)v8->m128i_i64 - v10 <= 0 )
          return result;
        v48 = a6;
        v49 = v10;
        do
        {
          v50 = *(_QWORD *)(v49 + 48);
          v48 += 56;
          v49 += 56;
          *(_QWORD *)(v48 - 8) = v50;
          *(__m128i *)(v48 - 40) = _mm_loadu_si128((const __m128i *)(v49 - 40));
          *(__m128i *)(v48 - 24) = _mm_loadu_si128((const __m128i *)(v49 - 24));
          *(__m128i *)(v48 - 56) = _mm_loadu_si128((const __m128i *)(v49 - 56));
          --v47;
        }
        while ( v47 );
        result = 56;
        if ( v46 <= 0 )
          v46 = 56;
        v51 = (const __m128i *)(a6 + v46);
        if ( v9 == (const __m128i *)v10 )
        {
          result = 0x6DB6DB6DB6DB6DB7LL * (v46 >> 3);
          while ( 1 )
          {
            v51 = (const __m128i *)((char *)v51 - 56);
            v8[-1].m128i_i64[1] = v50;
            v8 = (__m128i *)((char *)v8 - 56);
            v8[1] = _mm_loadu_si128(v51 + 1);
            v8[2] = _mm_loadu_si128(v51 + 2);
            *v8 = _mm_loadu_si128(v51);
            if ( !--result )
              break;
            v50 = v51[-1].m128i_i64[1];
          }
          return result;
        }
        if ( (const __m128i *)a6 == v51 )
          return result;
        v52 = (const __m128i *)(v10 - 56);
        v53 = (const __m128i *)((char *)v51 - 56);
        v54 = (__m128i *)((char *)v8 - 56);
        v55 = (__int64)v53;
        v56 = (__int64 *)v52;
        while ( 2 )
        {
          v57 = v52->m128i_u8[10];
          if ( (unsigned int)v53->m128i_i8[10] >= v57 )
          {
            if ( v53->m128i_i8[10] == v57 )
            {
              v58 = v53[1].m128i_u32[0];
              v59 = v52[1].m128i_u32[0];
              if ( v58 < v59 )
                goto LABEL_59;
              if ( v58 == v59 )
              {
                v60 = v52[1].m128i_u32[1];
                if ( v53[1].m128i_i32[1] < v60 )
                  goto LABEL_59;
                if ( v53[1].m128i_i32[1] == v60 )
                {
                  v61 = v52[1].m128i_u32[2];
                  if ( v53[1].m128i_i32[2] < v61
                    || v53[1].m128i_i32[2] == v61 && v53[1].m128i_i32[3] < (unsigned __int32)v52[1].m128i_i32[3] )
                  {
                    goto LABEL_59;
                  }
                }
              }
              if ( v58 <= v59 )
              {
                v62 = v53[1].m128i_u32[1];
                if ( v52[1].m128i_i32[1] >= v62 )
                {
                  if ( v52[1].m128i_i32[1] != v62
                    || (v63 = v53[1].m128i_u32[2], v52[1].m128i_i32[2] >= v63)
                    && (v52[1].m128i_i32[2] != v63 || v52[1].m128i_i32[3] >= (unsigned __int32)v53[1].m128i_i32[3]) )
                  {
                    v95 = (__int64)v56;
                    v99 = (__int64 *)v55;
                    v64 = sub_22A6F20(v55, v56);
                    v55 = (__int64)v99;
                    if ( v64 )
                      goto LABEL_59;
                    sub_22A6F20(v95, v99);
                    v56 = (__int64 *)v95;
                  }
                }
              }
            }
            result = v53[3].m128i_i64[0];
            v54[3].m128i_i64[0] = result;
            v54[1] = _mm_loadu_si128(v53 + 1);
            v54[2] = _mm_loadu_si128(v53 + 2);
            *v54 = _mm_loadu_si128(v53);
            if ( (const __m128i *)a6 == v53 )
              return result;
            v53 = (const __m128i *)((char *)v53 - 56);
            v55 = (__int64)v53;
          }
          else
          {
LABEL_59:
            result = (__int64)v54;
            v54[3].m128i_i64[0] = v52[3].m128i_i64[0];
            v54[1] = _mm_loadu_si128(v52 + 1);
            v54[2] = _mm_loadu_si128(v52 + 2);
            *v54 = _mm_loadu_si128(v52);
            if ( v9 == v52 )
            {
              v81 = (const __m128i *)((char *)v53 + 56);
              v82 = 0x6DB6DB6DB6DB6DB7LL * (((__int64)v81->m128i_i64 - a6) >> 3);
              if ( (__int64)v81->m128i_i64 - a6 > 0 )
              {
                do
                {
                  v83 = v81[-1].m128i_i64[1];
                  v81 = (const __m128i *)((char *)v81 - 56);
                  result -= 56;
                  *(_QWORD *)(result + 48) = v83;
                  *(__m128i *)(result + 16) = _mm_loadu_si128(v81 + 1);
                  *(__m128i *)(result + 32) = _mm_loadu_si128(v81 + 2);
                  *(__m128i *)result = _mm_loadu_si128(v81);
                  --v82;
                }
                while ( v82 );
              }
              return result;
            }
            v52 = (const __m128i *)((char *)v52 - 56);
            v56 = (__int64 *)v52;
          }
          v54 = (__m128i *)((char *)v54 - 56);
          continue;
        }
      }
      v15 = (__int64)v98;
      v14 = (__int64)v91;
      if ( v13 > v12 )
        goto LABEL_6;
LABEL_14:
      v87 = (__m128i *)v15;
      v92 = v14;
      v21 = v12 / 2;
      v98 = (char *)(v15 + 56 * (v12 / 2));
      v24 = sub_22A79B0(v14, v15, (__int64)v98);
      v20 = v92;
      v17 = v87;
      v19 = (__int8 *)v24;
      v18 = 0x6DB6DB6DB6DB6DB7LL * ((v24 - v92) >> 3);
      v13 -= v18;
      if ( v13 > v12 / 2 )
        goto LABEL_7;
    }
  }
  result = 0x6DB6DB6DB6DB6DB7LL;
  v40 = v10 - (_QWORD)v9;
  v41 = 0x6DB6DB6DB6DB6DB7LL * ((v10 - (__int64)v9) >> 3);
  if ( v10 - (__int64)v9 > 0 )
  {
    v42 = a6;
    v43 = v9;
    do
    {
      v44 = v43[3].m128i_i64[0];
      v42 += 56;
      v43 = (const __m128i *)((char *)v43 + 56);
      *(_QWORD *)(v42 - 8) = v44;
      *(__m128i *)(v42 - 40) = _mm_loadu_si128((const __m128i *)((char *)v43 - 40));
      *(__m128i *)(v42 - 24) = _mm_loadu_si128((const __m128i *)((char *)v43 - 24));
      *(__m128i *)(v42 - 56) = _mm_loadu_si128((const __m128i *)((char *)v43 - 56));
      --v41;
    }
    while ( v41 );
    result = 56;
    if ( v40 <= 0 )
      v40 = 56;
    v45 = a6 + v40;
    if ( a6 != v45 )
    {
      while ( v8 != (__m128i *)v10 )
      {
        if ( (unsigned __int8)sub_22A71D0(v10, a6) )
        {
          result = *(_QWORD *)(v10 + 48);
          v9 = (const __m128i *)((char *)v9 + 56);
          v10 += 56;
          v9[-1].m128i_i64[1] = result;
          *(const __m128i *)((char *)&v9[-3] + 8) = _mm_loadu_si128((const __m128i *)(v10 - 40));
          *(const __m128i *)((char *)&v9[-2] + 8) = _mm_loadu_si128((const __m128i *)(v10 - 24));
          *(const __m128i *)((char *)&v9[-4] + 8) = _mm_loadu_si128((const __m128i *)(v10 - 56));
          if ( a6 == v45 )
            return result;
        }
        else
        {
          result = *(_QWORD *)(a6 + 48);
          a6 += 56;
          v9 = (const __m128i *)((char *)v9 + 56);
          v9[-1].m128i_i64[1] = result;
          *(const __m128i *)((char *)&v9[-3] + 8) = _mm_loadu_si128((const __m128i *)(a6 - 40));
          *(const __m128i *)((char *)&v9[-2] + 8) = _mm_loadu_si128((const __m128i *)(a6 - 24));
          *(const __m128i *)((char *)&v9[-4] + 8) = _mm_loadu_si128((const __m128i *)(a6 - 56));
          if ( a6 == v45 )
            return result;
        }
      }
      if ( a6 != v45 )
      {
        v65 = v45 - a6;
        result = 0x6DB6DB6DB6DB6DB7LL * (v65 >> 3);
        if ( v65 > 0 )
        {
          do
          {
            v66 = *(_QWORD *)(a6 + 48);
            v9 = (const __m128i *)((char *)v9 + 56);
            a6 += 56;
            v9[-1].m128i_i64[1] = v66;
            *(const __m128i *)((char *)&v9[-3] + 8) = _mm_loadu_si128((const __m128i *)(a6 - 40));
            *(const __m128i *)((char *)&v9[-2] + 8) = _mm_loadu_si128((const __m128i *)(a6 - 24));
            *(const __m128i *)((char *)&v9[-4] + 8) = _mm_loadu_si128((const __m128i *)(a6 - 56));
            --result;
          }
          while ( result );
        }
      }
    }
  }
  return result;
}
