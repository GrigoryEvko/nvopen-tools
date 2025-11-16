// Function: sub_C9EA90
// Address: 0xc9ea90
//
__int64 __fastcall sub_C9EA90(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5)
{
  double *v7; // rax
  double *v8; // rsi
  double v9; // xmm0_8
  __int64 v10; // rdx
  double v11; // xmm2_8
  double v12; // xmm1_8
  __int64 v13; // rdx
  __int64 v14; // r14
  __int64 v15; // rax
  _DWORD *v16; // rdx
  unsigned __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rdi
  __int64 v21; // r8
  _BYTE *v22; // rax
  __int64 v23; // rdx
  __int64 v24; // r14
  __int64 v25; // rax
  _DWORD *v26; // rdx
  _BYTE *v27; // rax
  __int64 v28; // rax
  double v29; // xmm0_8
  __m128i v30; // xmm0
  char *v31; // rax
  __int64 v32; // rax
  __int64 v33; // rbx
  __int64 v34; // r12
  __int64 v35; // rdi
  _BYTE *v36; // rax
  __int64 v37; // rdx
  __int64 result; // rax
  _QWORD *v39; // r14
  _QWORD *v40; // r12
  _QWORD *v41; // rbx
  _QWORD *v42; // rdi
  _QWORD *v43; // rdi
  double *v44; // r12
  __int64 v45; // r14
  unsigned __int64 v46; // rax
  const __m128i *v47; // rbx
  const __m128i *v48; // rdi
  __m128i v49; // xmm0
  __m128i v50; // xmm0
  __m128i si128; // xmm0
  __int64 v52; // [rsp+8h] [rbp-88h]
  _QWORD *v53; // [rsp+10h] [rbp-80h] BYREF
  const char *v54; // [rsp+18h] [rbp-78h]
  _QWORD v55[2]; // [rsp+20h] [rbp-70h] BYREF
  double v56; // [rsp+30h] [rbp-60h] BYREF
  double v57; // [rsp+38h] [rbp-58h]
  double v58; // [rsp+40h] [rbp-50h]
  __int64 v59; // [rsp+48h] [rbp-48h]
  __int64 v60; // [rsp+50h] [rbp-40h]

  if ( qword_4F84F60 )
  {
    if ( !*(_BYTE *)(qword_4F84F60 + 600) )
      goto LABEL_3;
  }
  else
  {
    sub_C7D570(&qword_4F84F60, sub_CA0780, (__int64)sub_C9FD10);
    if ( !*(_BYTE *)(qword_4F84F60 + 600) )
      goto LABEL_3;
  }
  v44 = *(double **)(a1 + 80);
  v45 = *(_QWORD *)(a1 + 72);
  if ( v44 == (double *)v45 )
  {
    v56 = 0.0;
    v57 = 0.0;
    v58 = 0.0;
    v59 = 0;
    v60 = 0;
    goto LABEL_6;
  }
  _BitScanReverse64(&v46, 0x4EC4EC4EC4EC4EC5LL * (((__int64)v44 - v45) >> 3));
  sub_C9D290(*(_QWORD *)(a1 + 72), *(__m128i **)(a1 + 80), 2LL * (int)(63 - (v46 ^ 0x3F)));
  if ( (__int64)v44 - v45 <= 1664 )
  {
    sub_C9CCE0(v45, v44);
  }
  else
  {
    v47 = (const __m128i *)(v45 + 1664);
    sub_C9CCE0(v45, (double *)(v45 + 1664));
    if ( v44 != (double *)(v45 + 1664) )
    {
      do
      {
        v48 = v47;
        v47 = (const __m128i *)((char *)v47 + 104);
        sub_C9CB60(v48);
      }
      while ( v44 != (double *)v47 );
    }
  }
LABEL_3:
  v7 = *(double **)(a1 + 72);
  v8 = *(double **)(a1 + 80);
  v56 = 0.0;
  v57 = 0.0;
  v58 = 0.0;
  v59 = 0;
  v60 = 0;
  if ( v8 != v7 )
  {
    v9 = 0.0;
    a4 = 0;
    v10 = 0;
    v11 = 0.0;
    v12 = 0.0;
    do
    {
      v9 = v9 + *v7;
      v7 += 13;
      v56 = v9;
      v12 = v12 + *(v7 - 12);
      v57 = v12;
      v11 = v11 + *(v7 - 11);
      v58 = v11;
      v10 += *((_QWORD *)v7 - 10);
      v59 = v10;
      a4 += *((_QWORD *)v7 - 9);
      v60 = a4;
    }
    while ( v8 != v7 );
  }
LABEL_6:
  v13 = a2[4];
  if ( (unsigned __int64)(a2[3] - v13) <= 2 )
  {
    v14 = sub_CB6200(a2, "===", 3);
  }
  else
  {
    *(_BYTE *)(v13 + 2) = 61;
    v14 = (__int64)a2;
    *(_WORD *)v13 = 15677;
    a2[4] += 3LL;
  }
  v53 = v55;
  sub_2240A50(&v53, 73, 45, a4, a5);
  v15 = sub_CB6200(v14, v53, v54);
  v16 = *(_DWORD **)(v15 + 32);
  if ( *(_QWORD *)(v15 + 24) - (_QWORD)v16 <= 3u )
  {
    sub_CB6200(v15, "===\n", 4);
  }
  else
  {
    *v16 = 171785533;
    *(_QWORD *)(v15 + 32) += 4LL;
  }
  if ( v53 != v55 )
    j_j___libc_free_0(v53, v55[0] + 1LL);
  v17 = (unsigned __int64)(80LL - *(_QWORD *)(a1 + 40)) >> 1;
  if ( (unsigned int)v17 >= 0x51 )
    v17 = 0;
  v18 = sub_CB69B0(a2, v17);
  v20 = sub_CB6200(v18, *(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 40));
  v22 = *(_BYTE **)(v20 + 32);
  if ( (unsigned __int64)v22 >= *(_QWORD *)(v20 + 24) )
  {
    sub_CB5D20(v20, 10);
  }
  else
  {
    *(_QWORD *)(v20 + 32) = v22 + 1;
    *v22 = 10;
  }
  v23 = a2[4];
  if ( (unsigned __int64)(a2[3] - v23) <= 2 )
  {
    v24 = sub_CB6200(a2, "===", 3);
  }
  else
  {
    *(_BYTE *)(v23 + 2) = 61;
    v24 = (__int64)a2;
    *(_WORD *)v23 = 15677;
    a2[4] += 3LL;
  }
  v53 = v55;
  sub_2240A50(&v53, 73, 45, v19, v21);
  v25 = sub_CB6200(v24, v53, v54);
  v26 = *(_DWORD **)(v25 + 32);
  if ( *(_QWORD *)(v25 + 24) - (_QWORD)v26 <= 3u )
  {
    sub_CB6200(v25, "===\n", 4);
  }
  else
  {
    *v26 = 171785533;
    *(_QWORD *)(v25 + 32) += 4LL;
  }
  if ( v53 != v55 )
    j_j___libc_free_0(v53, v55[0] + 1LL);
  if ( !qword_4F84F60 )
    sub_C7D570(&qword_4F84F60, sub_CA0780, (__int64)sub_C9FD10);
  if ( a1 != qword_4F84F60 + 712 )
  {
    v54 = "  Total Execution Time: %5.4f seconds (%5.4f wall clock)\n";
    *(double *)v55 = v56;
    *(double *)&v55[1] = v57 + v58;
    v53 = &unk_49DCB78;
    sub_CB6620(a2, &v53);
  }
  v27 = (_BYTE *)a2[4];
  if ( (unsigned __int64)v27 >= a2[3] )
  {
    sub_CB5D20(a2, 10);
  }
  else
  {
    a2[4] = v27 + 1;
    *v27 = 10;
  }
  v28 = a2[4];
  if ( v57 != 0.0 )
  {
    if ( (unsigned __int64)(a2[3] - v28) <= 0x11 )
    {
      sub_CB6200(a2, "   ---User Time---", 18);
      v28 = a2[4];
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F67A40);
      *(_WORD *)(v28 + 16) = 11565;
      *(__m128i *)v28 = si128;
      v28 = a2[4] + 18LL;
      a2[4] = v28;
    }
  }
  v29 = v58;
  if ( v58 != 0.0 )
  {
    if ( (unsigned __int64)(a2[3] - v28) <= 0x11 )
    {
      sub_CB6200(a2, "   --System Time--", 18);
      v28 = a2[4];
      v29 = v58;
    }
    else
    {
      v50 = _mm_load_si128((const __m128i *)&xmmword_3F67A50);
      *(_WORD *)(v28 + 16) = 11565;
      *(__m128i *)v28 = v50;
      v29 = v58;
      v28 = a2[4] + 18LL;
      a2[4] = v28;
    }
  }
  if ( v29 + v57 != 0.0 )
  {
    if ( (unsigned __int64)(a2[3] - v28) <= 0x11 )
    {
      sub_CB6200(a2, "   --User+System--", 18);
      v28 = a2[4];
    }
    else
    {
      v49 = _mm_load_si128((const __m128i *)&xmmword_3F67A60);
      *(_WORD *)(v28 + 16) = 11565;
      *(__m128i *)v28 = v49;
      v28 = a2[4] + 18LL;
      a2[4] = v28;
    }
  }
  if ( (unsigned __int64)(a2[3] - v28) <= 0x11 )
  {
    sub_CB6200(a2, "   ---Wall Time---", 18);
    v31 = (char *)a2[4];
  }
  else
  {
    v30 = _mm_load_si128((const __m128i *)&xmmword_3F67A70);
    *(_WORD *)(v28 + 16) = 11565;
    *(__m128i *)v28 = v30;
    v31 = (char *)(a2[4] + 18LL);
    a2[4] = v31;
  }
  if ( v59 )
  {
    if ( a2[3] - (_QWORD)v31 <= 0xAu )
    {
      sub_CB6200(a2, "  ---Mem---", 11);
      v31 = (char *)a2[4];
    }
    else
    {
      qmemcpy(v31, "  ---Mem---", 11);
      v31 = (char *)(a2[4] + 11LL);
      a2[4] = v31;
    }
  }
  if ( v60 )
  {
    if ( a2[3] - (_QWORD)v31 <= 0xCu )
    {
      sub_CB6200(a2, "  ---Instr---", 13);
      v31 = (char *)a2[4];
    }
    else
    {
      qmemcpy(v31, "  ---Instr---", 13);
      v31 = (char *)(a2[4] + 13LL);
      a2[4] = v31;
    }
  }
  if ( a2[3] - (_QWORD)v31 <= 0xEu )
  {
    sub_CB6200(a2, "  --- Name ---\n", 15);
  }
  else
  {
    qmemcpy(v31, "  --- Name ---\n", 15);
    a2[4] += 15LL;
  }
  v52 = *(_QWORD *)(a1 + 72);
  v32 = *(_QWORD *)(a1 + 80);
  v33 = v32 - 104;
  if ( v52 != v32 )
  {
    do
    {
      while ( 1 )
      {
        v34 = v33;
        sub_C9E380((double *)v33, (__int64)&v56, (__int64)a2);
        v35 = sub_CB6200(a2, *(_QWORD *)(v33 + 72), *(_QWORD *)(v33 + 80));
        v36 = *(_BYTE **)(v35 + 32);
        if ( (unsigned __int64)v36 >= *(_QWORD *)(v35 + 24) )
          break;
        v33 -= 104;
        *(_QWORD *)(v35 + 32) = v36 + 1;
        *v36 = 10;
        if ( v34 == v52 )
          goto LABEL_42;
      }
      v33 -= 104;
      sub_CB5D20(v35, 10);
    }
    while ( v34 != v52 );
  }
LABEL_42:
  sub_C9E380(&v56, (__int64)&v56, (__int64)a2);
  v37 = a2[4];
  if ( (unsigned __int64)(a2[3] - v37) <= 6 )
  {
    sub_CB6200(a2, "Total\n\n", 7);
    result = a2[4];
  }
  else
  {
    *(_DWORD *)v37 = 1635020628;
    *(_WORD *)(v37 + 4) = 2668;
    *(_BYTE *)(v37 + 6) = 10;
    result = a2[4] + 7LL;
    a2[4] = result;
  }
  if ( a2[2] != result )
    result = sub_CB5AE0(a2);
  v39 = *(_QWORD **)(a1 + 72);
  v40 = *(_QWORD **)(a1 + 80);
  if ( v39 != v40 )
  {
    v41 = *(_QWORD **)(a1 + 72);
    do
    {
      v42 = (_QWORD *)v41[9];
      if ( v42 != v41 + 11 )
        j_j___libc_free_0(v42, v41[11] + 1LL);
      v43 = (_QWORD *)v41[5];
      result = (__int64)(v41 + 7);
      if ( v43 != v41 + 7 )
        result = j_j___libc_free_0(v43, v41[7] + 1LL);
      v41 += 13;
    }
    while ( v40 != v41 );
    *(_QWORD *)(a1 + 80) = v39;
  }
  return result;
}
