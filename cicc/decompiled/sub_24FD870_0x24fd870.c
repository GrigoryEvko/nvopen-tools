// Function: sub_24FD870
// Address: 0x24fd870
//
__int64 *__fastcall sub_24FD870(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *result; // rax
  __int64 v6; // r9
  __int64 v8; // rbx
  __int64 v9; // r12
  __int64 v10; // rcx
  __int64 v11; // rsi
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rdx
  __m128i v15; // xmm2
  __int64 v16; // rdx
  __int64 v17; // rsi
  char v18; // cl
  _QWORD *v19; // rsi
  _QWORD *v20; // r13
  __int64 v21; // rdx
  _QWORD *v22; // r15
  unsigned __int64 v23; // rax
  __int64 i; // rdx
  __m128i v25; // xmm0
  char v26; // cl
  __int64 v27; // rdx
  __int64 v28; // rdi
  __m128i v29; // xmm3
  __int64 v30; // rax
  __int64 v31; // rsi
  char v32; // dl
  __int64 v33; // r10
  __m128i v34; // xmm4
  __int64 v35; // rcx
  char v36; // dl
  __int64 v37; // rax
  __m128i v38; // xmm6
  __int64 v39; // rbx
  __int64 v40; // r12
  __int64 v41; // r8
  __int64 v42; // r9
  __int128 v43; // rax
  __int64 v44; // rsi
  __int64 v45; // rdi
  __int64 v46; // rcx
  __m128i v47; // xmm7
  __m128i v48; // xmm5
  __int128 v49; // [rsp-78h] [rbp-78h]
  _QWORD *v50; // [rsp-60h] [rbp-60h]

  result = (_QWORD *)((char *)a2 - a1);
  if ( (__int64)a2 - a1 <= 512 )
    return result;
  v6 = (__int64)a2;
  v8 = a3;
  if ( !a3 )
  {
    v22 = a2;
    goto LABEL_27;
  }
  v9 = a1 + 32;
  v50 = (_QWORD *)(a1 + 64);
  while ( 2 )
  {
    v10 = *(_QWORD *)(a1 + 32);
    v11 = *(_QWORD *)(v6 - 32);
    --v8;
    v12 = *(_QWORD *)a1;
    v13 = a1 + 32 * ((__int64)result >> 6);
    v14 = *(_QWORD *)v13;
    if ( v10 >= *(_QWORD *)v13 )
    {
      if ( v10 >= v11 )
      {
        if ( v14 >= v11 )
        {
          *(_QWORD *)a1 = v14;
          v47 = _mm_loadu_si128((const __m128i *)(v13 + 8));
          *(_QWORD *)v13 = v12;
          v17 = *(_QWORD *)(a1 + 8);
          v18 = *(_BYTE *)(a1 + 16);
          v16 = *(_QWORD *)(a1 + 24);
          *(__m128i *)(a1 + 8) = v47;
          goto LABEL_7;
        }
        *(_QWORD *)a1 = v11;
        v38 = _mm_loadu_si128((const __m128i *)(v6 - 24));
        *(_QWORD *)(v6 - 32) = v12;
        v37 = *(_QWORD *)(a1 + 24);
        v35 = *(_QWORD *)(a1 + 8);
        v36 = *(_BYTE *)(a1 + 16);
        *(__m128i *)(a1 + 8) = v38;
LABEL_25:
        *(_QWORD *)(a1 + 24) = *(_QWORD *)(v6 - 8);
        *(_QWORD *)(v6 - 24) = v35;
        *(_BYTE *)(v6 - 16) = v36;
        *(_QWORD *)(v6 - 8) = v37;
        v12 = *(_QWORD *)(a1 + 32);
        v10 = *(_QWORD *)a1;
        goto LABEL_8;
      }
      v29 = _mm_loadu_si128((const __m128i *)(a1 + 40));
      v30 = *(_QWORD *)(a1 + 24);
      *(_QWORD *)a1 = v10;
      v31 = *(_QWORD *)(a1 + 8);
      v32 = *(_BYTE *)(a1 + 16);
      *(_QWORD *)(a1 + 32) = v12;
      *(__m128i *)(a1 + 8) = v29;
LABEL_20:
      v33 = *(_QWORD *)(a1 + 56);
      *(_QWORD *)(a1 + 40) = v31;
      *(_BYTE *)(a1 + 48) = v32;
      *(_QWORD *)(a1 + 24) = v33;
      *(_QWORD *)(a1 + 56) = v30;
      goto LABEL_8;
    }
    if ( v14 >= v11 )
    {
      if ( v10 < v11 )
      {
        *(_QWORD *)a1 = v11;
        v34 = _mm_loadu_si128((const __m128i *)(v6 - 24));
        *(_QWORD *)(v6 - 32) = v12;
        v35 = *(_QWORD *)(a1 + 8);
        v36 = *(_BYTE *)(a1 + 16);
        v37 = *(_QWORD *)(a1 + 24);
        *(__m128i *)(a1 + 8) = v34;
        goto LABEL_25;
      }
      v48 = _mm_loadu_si128((const __m128i *)(a1 + 40));
      v31 = *(_QWORD *)(a1 + 8);
      *(_QWORD *)a1 = v10;
      v32 = *(_BYTE *)(a1 + 16);
      v30 = *(_QWORD *)(a1 + 24);
      *(_QWORD *)(a1 + 32) = v12;
      *(__m128i *)(a1 + 8) = v48;
      goto LABEL_20;
    }
    *(_QWORD *)a1 = v14;
    v15 = _mm_loadu_si128((const __m128i *)(v13 + 8));
    *(_QWORD *)v13 = v12;
    v16 = *(_QWORD *)(a1 + 24);
    v17 = *(_QWORD *)(a1 + 8);
    v18 = *(_BYTE *)(a1 + 16);
    *(__m128i *)(a1 + 8) = v15;
LABEL_7:
    *(_QWORD *)(a1 + 24) = *(_QWORD *)(v13 + 24);
    *(_QWORD *)(v13 + 8) = v17;
    *(_BYTE *)(v13 + 16) = v18;
    *(_QWORD *)(v13 + 24) = v16;
    v12 = *(_QWORD *)(a1 + 32);
    v10 = *(_QWORD *)a1;
LABEL_8:
    v19 = v50;
    v20 = (_QWORD *)v9;
    v21 = v6;
    while ( 1 )
    {
      v22 = v20;
      if ( v12 < v10 )
        goto LABEL_14;
      v23 = v21 - 32;
      for ( i = *(_QWORD *)(v21 - 32); i > v10; v23 -= 32LL )
        i = *(_QWORD *)(v23 - 32);
      if ( v23 <= (unsigned __int64)v20 )
        break;
      *(v19 - 4) = i;
      v25 = _mm_loadu_si128((const __m128i *)(v23 + 8));
      *(_QWORD *)v23 = v12;
      v26 = *((_BYTE *)v19 - 16);
      v27 = *(v19 - 1);
      v28 = *(v19 - 3);
      *(__m128i *)(v19 - 3) = v25;
      *(v19 - 1) = *(_QWORD *)(v23 + 24);
      *(_QWORD *)(v23 + 24) = v27;
      v21 = v23;
      *(_QWORD *)(v23 + 8) = v28;
      *(_BYTE *)(v23 + 16) = v26;
      v10 = *(_QWORD *)a1;
LABEL_14:
      v12 = *v19;
      v20 += 4;
      v19 += 4;
    }
    sub_24FD870(v20, v6, v8);
    result = (_QWORD *)((char *)v20 - a1);
    if ( (__int64)v20 - a1 > 512 )
    {
      if ( v8 )
      {
        v6 = (__int64)v20;
        continue;
      }
LABEL_27:
      v39 = (__int64)result >> 5;
      v40 = (((__int64)result >> 5) - 2) >> 1;
      sub_24FD590(a1, (v39 - 2) >> 1, v39, a4, a5, v6, *(_OWORD *)(a1 + 32 * v40), *(_OWORD *)(a1 + 32 * v40 + 16));
      do
      {
        --v40;
        sub_24FD590(a1, v40, v39, 32 * v40, v41, v42, *(_OWORD *)(a1 + 32 * v40), *(_OWORD *)(a1 + 32 * v40 + 16));
      }
      while ( v40 );
      do
      {
        *(_QWORD *)&v43 = *(v22 - 4);
        v22 -= 4;
        *((_QWORD *)&v43 + 1) = v22[1];
        v44 = v22[2];
        *v22 = *(_QWORD *)a1;
        v45 = v22[3];
        *(__m128i *)(v22 + 1) = _mm_loadu_si128((const __m128i *)(a1 + 8));
        v46 = *(_QWORD *)(a1 + 24);
        v22[3] = v46;
        *((_QWORD *)&v49 + 1) = v45;
        *(_QWORD *)&v49 = v44;
        result = sub_24FD590(a1, 0, ((__int64)v22 - a1) >> 5, v46, ((__int64)v22 - a1) >> 5, v42, v43, v49);
      }
      while ( (__int64)v22 - a1 > 32 );
    }
    return result;
  }
}
