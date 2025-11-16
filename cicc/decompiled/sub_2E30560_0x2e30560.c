// Function: sub_2E30560
// Address: 0x2e30560
//
__int64 __fastcall sub_2E30560(unsigned int *a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  unsigned int *v8; // r14
  __int64 v9; // r12
  unsigned int *v10; // r13
  unsigned int v11; // edi
  unsigned int v12; // edx
  unsigned int *v13; // rcx
  unsigned int v14; // eax
  unsigned int v15; // eax
  __int64 v16; // rdx
  __m128i v17; // xmm6
  __m128i v18; // xmm7
  __int64 v19; // rcx
  unsigned int *v20; // rbx
  unsigned int *v21; // rdx
  unsigned __int64 v22; // rax
  unsigned int *v23; // rax
  __m128i v24; // xmm0
  unsigned int v25; // eax
  unsigned int v26; // ecx
  __m128i v27; // xmm2
  __m128i v28; // xmm5
  __int64 v29; // rcx
  __m128i v30; // xmm6
  __m128i v31; // xmm7
  __int64 v32; // rbx
  __int64 i; // r12
  __int64 v34; // rax
  const __m128i *v35; // r14
  __int64 v36; // rax
  __m128i v37; // xmm3
  __int64 v38; // r12
  __m128i v39; // xmm6
  __int64 v40; // rdi
  unsigned int v41; // edx
  __m128i v42; // xmm7
  __m128i v43; // xmm6
  __m128i v44; // xmm3
  __int64 v45; // rdi
  unsigned int v46; // edx
  __m128i v47; // xmm4
  __m128i v48; // xmm5
  __int128 v49; // [rsp-58h] [rbp-58h] BYREF
  __int64 v50; // [rsp-48h] [rbp-48h]

  result = (char *)a2 - (char *)a1;
  if ( (char *)a2 - (char *)a1 <= 384 )
    return result;
  v8 = a2;
  v9 = a3;
  if ( !a3 )
    goto LABEL_26;
  v10 = a1 + 6;
  while ( 2 )
  {
    v11 = *(a2 - 6);
    --v9;
    v12 = a1[6];
    v13 = &a1[2 * ((__int64)(0xAAAAAAAAAAAAAAABLL * (((char *)a2 - (char *)a1) >> 3)) / 2)
            + 2
            * ((0xAAAAAAAAAAAAAAABLL * (((char *)a2 - (char *)a1) >> 3)
              + ((0xAAAAAAAAAAAAAAABLL * (((char *)a2 - (char *)a1) >> 3)) >> 63))
             & 0xFFFFFFFFFFFFFFFELL)];
    v14 = *v13;
    if ( v12 >= *v13 )
    {
      if ( v12 >= v11 )
      {
        if ( v14 >= v11 )
        {
          v44 = _mm_loadu_si128((const __m128i *)a1);
          v45 = *((_QWORD *)a1 + 2);
          v46 = *a1;
          *a1 = v14;
          v50 = v45;
          v47 = _mm_loadu_si128((const __m128i *)(v13 + 2));
          v49 = (__int128)v44;
          v48 = _mm_loadu_si128((const __m128i *)((char *)&v49 + 8));
          *(__m128i *)(a1 + 2) = v47;
          *v13 = v46;
          *(__m128i *)(v13 + 2) = v48;
          goto LABEL_9;
        }
        v15 = *a1;
        v49 = (__int128)_mm_loadu_si128((const __m128i *)a1);
        goto LABEL_8;
      }
      v15 = *a1;
LABEL_25:
      v28 = _mm_loadu_si128((const __m128i *)a1);
      v29 = *((_QWORD *)a1 + 2);
      *a1 = v12;
      v30 = _mm_loadu_si128((const __m128i *)a1 + 2);
      a1[6] = v15;
      v50 = v29;
      v49 = (__int128)v28;
      v31 = _mm_loadu_si128((const __m128i *)((char *)&v49 + 8));
      *(__m128i *)(a1 + 2) = v30;
      *((__m128i *)a1 + 2) = v31;
      goto LABEL_9;
    }
    if ( v14 < v11 )
    {
      v39 = _mm_loadu_si128((const __m128i *)a1);
      v40 = *((_QWORD *)a1 + 2);
      v41 = *a1;
      *a1 = v14;
      v50 = v40;
      v42 = _mm_loadu_si128((const __m128i *)(v13 + 2));
      v49 = (__int128)v39;
      v43 = _mm_loadu_si128((const __m128i *)((char *)&v49 + 8));
      *(__m128i *)(a1 + 2) = v42;
      *v13 = v41;
      *(__m128i *)(v13 + 2) = v43;
      goto LABEL_9;
    }
    v15 = *a1;
    if ( v12 >= v11 )
      goto LABEL_25;
    v49 = (__int128)_mm_loadu_si128((const __m128i *)a1);
LABEL_8:
    v16 = *((_QWORD *)a1 + 2);
    *a1 = v11;
    v17 = _mm_loadu_si128((const __m128i *)a2 - 1);
    v50 = v16;
    v18 = _mm_loadu_si128((const __m128i *)((char *)&v49 + 8));
    *(__m128i *)(a1 + 2) = v17;
    *(a2 - 6) = v15;
    *((__m128i *)a2 - 1) = v18;
LABEL_9:
    v19 = *a1;
    v20 = v10;
    v21 = a2;
    while ( 1 )
    {
      v8 = v20;
      if ( *v20 < (unsigned int)v19 )
        goto LABEL_16;
      v22 = (unsigned __int64)(v21 - 6);
      if ( *(v21 - 6) <= (unsigned int)v19 )
      {
        v21 -= 6;
        if ( (unsigned __int64)v20 >= v22 )
          break;
        goto LABEL_15;
      }
      v23 = v21 - 12;
      do
      {
        v21 = v23;
        v23 -= 6;
      }
      while ( v23[6] > (unsigned int)v19 );
      if ( v20 >= v21 )
        break;
LABEL_15:
      v24 = _mm_loadu_si128((const __m128i *)v20);
      v25 = *v20;
      v50 = *((_QWORD *)v20 + 2);
      v26 = *v21;
      v49 = (__int128)v24;
      v27 = _mm_loadu_si128((const __m128i *)((char *)&v49 + 8));
      *v20 = v26;
      *(__m128i *)(v20 + 2) = _mm_loadu_si128((const __m128i *)(v21 + 2));
      *v21 = v25;
      *(__m128i *)(v21 + 2) = v27;
      v19 = *a1;
LABEL_16:
      v20 += 6;
    }
    sub_2E30560(v20, a2, v9, v19, a5, a6, v49, *((_QWORD *)&v49 + 1), v50);
    result = (char *)v20 - (char *)a1;
    if ( (char *)v20 - (char *)a1 > 384 )
    {
      if ( v9 )
      {
        a2 = v20;
        continue;
      }
LABEL_26:
      v32 = 0xAAAAAAAAAAAAAAABLL * (result >> 3);
      for ( i = (v32 - 2) >> 1; ; --i )
      {
        v34 = *(_QWORD *)&a1[6 * i + 4];
        v49 = (__int128)_mm_loadu_si128((const __m128i *)&a1[6 * i]);
        v50 = v34;
        sub_2E2FA90((__int64)a1, i, v32, a4, a5, a6, v49, v34);
        if ( !i )
          break;
      }
      v35 = (const __m128i *)(v8 - 6);
      do
      {
        v36 = v35[1].m128i_i64[0];
        v37 = _mm_loadu_si128(v35);
        v38 = (char *)v35 - (char *)a1;
        v35 = (const __m128i *)((char *)v35 - 24);
        v50 = v36;
        LODWORD(v36) = *a1;
        v49 = (__int128)v37;
        v35[1].m128i_i32[2] = v36;
        v35[2] = _mm_loadu_si128((const __m128i *)(a1 + 2));
        result = sub_2E2FA90((__int64)a1, 0, 0xAAAAAAAAAAAAAAABLL * (v38 >> 3), a4, a5, a6, v49, v50);
      }
      while ( v38 > 24 );
    }
    return result;
  }
}
