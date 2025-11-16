// Function: sub_16D80D0
// Address: 0x16d80d0
//
__int64 __fastcall sub_16D80D0(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __m128i *v4; // r14
  double *v5; // rdx
  unsigned __int64 v6; // rax
  __m128i *v7; // rbx
  const __m128i *v8; // rdi
  double v9; // xmm2_8
  double *v10; // rdi
  __int64 v11; // rax
  double v12; // xmm1_8
  double v13; // xmm0_8
  __int64 v14; // rax
  __int64 v15; // r15
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // rax
  _DWORD *v19; // rdx
  unsigned __int64 v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rdi
  __int64 v24; // r8
  _BYTE *v25; // rax
  _WORD *v26; // rdx
  void **v27; // r15
  __int64 v28; // rax
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  _DWORD *v32; // rdx
  _BYTE *v33; // rax
  __m128i *v34; // rax
  double v35; // xmm0_8
  __m128i v36; // xmm0
  char *v37; // rax
  __int64 v38; // rax
  __int64 v39; // r15
  __int64 v40; // rbx
  __int64 v41; // rdi
  _BYTE *v42; // rax
  char *v43; // rdx
  char *v44; // rax
  _QWORD *v45; // r14
  _QWORD *v46; // rbx
  _QWORD *v47; // r15
  _QWORD *v48; // rdi
  _QWORD *v49; // rdi
  __int64 result; // rax
  __m128i v51; // xmm0
  __m128i v52; // xmm0
  __m128i si128; // xmm0
  __int64 v55; // [rsp+18h] [rbp-C8h]
  _QWORD v56[2]; // [rsp+20h] [rbp-C0h] BYREF
  _QWORD v57[2]; // [rsp+30h] [rbp-B0h] BYREF
  double v58; // [rsp+40h] [rbp-A0h] BYREF
  double v59; // [rsp+48h] [rbp-98h]
  double v60; // [rsp+50h] [rbp-90h]
  __int64 v61; // [rsp+58h] [rbp-88h]
  const char *v62; // [rsp+60h] [rbp-80h] BYREF
  const char *v63; // [rsp+68h] [rbp-78h]
  _QWORD v64[2]; // [rsp+70h] [rbp-70h] BYREF
  void *v65; // [rsp+80h] [rbp-60h] BYREF
  char *v66; // [rsp+88h] [rbp-58h]
  unsigned __int64 v67; // [rsp+90h] [rbp-50h]
  char *v68; // [rsp+98h] [rbp-48h]
  int v69; // [rsp+A0h] [rbp-40h]
  const char **v70; // [rsp+A8h] [rbp-38h]

  v3 = *(_QWORD *)(a1 + 72);
  v4 = *(__m128i **)(a1 + 80);
  v5 = (double *)v3;
  if ( v4 == (__m128i *)v3 )
    goto LABEL_6;
  _BitScanReverse64(&v6, 0xAAAAAAAAAAAAAAABLL * (((__int64)v4->m128i_i64 - v3) >> 5));
  sub_16D6AB0((__m128i *)v3, v4, 2LL * (int)(63 - (v6 ^ 0x3F)));
  if ( (__int64)v4->m128i_i64 - v3 <= 1536 )
  {
    sub_16D63F0(v3, (double *)v4->m128i_i64);
    goto LABEL_60;
  }
  v7 = (__m128i *)(v3 + 1536);
  sub_16D63F0(v3, (double *)(v3 + 1536));
  if ( v4 == (__m128i *)(v3 + 1536) )
  {
LABEL_60:
    v3 = *(_QWORD *)(a1 + 72);
    v5 = *(double **)(a1 + 80);
    goto LABEL_6;
  }
  do
  {
    v8 = v7;
    v7 += 6;
    sub_16D62A0(v8);
  }
  while ( v4 != v7 );
  v3 = *(_QWORD *)(a1 + 72);
  v5 = *(double **)(a1 + 80);
LABEL_6:
  v56[1] = 0;
  v56[0] = v57;
  LOBYTE(v57[0]) = 0;
  v69 = 1;
  v68 = 0;
  v67 = 0;
  v66 = 0;
  v65 = &unk_49EFBE0;
  v70 = (const char **)v56;
  v58 = 0.0;
  v59 = 0.0;
  v60 = 0.0;
  v61 = 0;
  if ( (double *)v3 != v5 )
  {
    v9 = 0.0;
    v10 = (double *)v3;
    v11 = 0;
    v12 = 0.0;
    v13 = 0.0;
    do
    {
      v13 = v13 + *v10;
      v10 += 12;
      v58 = v13;
      v12 = v12 + *(v10 - 11);
      v59 = v12;
      v9 = v9 + *(v10 - 10);
      v60 = v9;
      v11 += *((_QWORD *)v10 - 9);
      v61 = v11;
    }
    while ( v5 != v10 );
  }
  v14 = sub_16E7EE0(&v65, "===", 3);
  v62 = (const char *)v64;
  v15 = v14;
  sub_2240A50(&v62, 73, 45, v16, v17);
  v18 = sub_16E7EE0(v15, v62, v63);
  v19 = *(_DWORD **)(v18 + 24);
  if ( *(_QWORD *)(v18 + 16) - (_QWORD)v19 <= 3u )
  {
    sub_16E7EE0(v18, "===\n", 4);
  }
  else
  {
    *v19 = 171785533;
    *(_QWORD *)(v18 + 24) += 4LL;
  }
  if ( v62 != (const char *)v64 )
    j_j___libc_free_0(v62, v64[0] + 1LL);
  v20 = (unsigned __int64)(80LL - *(_QWORD *)(a1 + 40)) >> 1;
  if ( (unsigned int)v20 >= 0x51 )
    v20 = 0;
  v21 = sub_16E8750(&v65, v20);
  v23 = sub_16E7EE0(v21, *(const char **)(a1 + 32), *(_QWORD *)(a1 + 40));
  v25 = *(_BYTE **)(v23 + 24);
  if ( (unsigned __int64)v25 >= *(_QWORD *)(v23 + 16) )
  {
    sub_16E7DE0(v23, 10);
  }
  else
  {
    *(_QWORD *)(v23 + 24) = v25 + 1;
    *v25 = 10;
  }
  v26 = v68;
  if ( v67 - (unsigned __int64)v68 <= 2 )
  {
    v27 = (void **)sub_16E7EE0(&v65, "===", 3);
  }
  else
  {
    v68[2] = 61;
    v27 = &v65;
    *v26 = 15677;
    v68 += 3;
  }
  v62 = (const char *)v64;
  sub_2240A50(&v62, 73, 45, v22, v24);
  v28 = sub_16E7EE0(v27, v62, v63);
  v32 = *(_DWORD **)(v28 + 24);
  if ( *(_QWORD *)(v28 + 16) - (_QWORD)v32 <= 3u )
  {
    sub_16E7EE0(v28, "===\n", 4);
  }
  else
  {
    *v32 = 171785533;
    *(_QWORD *)(v28 + 24) += 4LL;
  }
  if ( v62 != (const char *)v64 )
    j_j___libc_free_0(v62, v64[0] + 1LL);
  if ( !qword_4FA1420 )
    sub_16C1EA0((__int64)&qword_4FA1420, (__int64 (*)(void))sub_16D7F40, (__int64)sub_16D95A0, v29, v30, v31);
  if ( a1 != qword_4FA1420 )
  {
    v63 = "  Total Execution Time: %5.4f seconds (%5.4f wall clock)\n";
    *(double *)v64 = v58;
    v62 = (const char *)&unk_49EF688;
    *(double *)&v64[1] = v59 + v60;
    sub_16E8450(&v65, &v62);
  }
  v33 = v68;
  if ( (unsigned __int64)v68 >= v67 )
  {
    sub_16E7DE0(&v65, 10);
  }
  else
  {
    ++v68;
    *v33 = 10;
  }
  v34 = (__m128i *)v68;
  if ( v59 != 0.0 )
  {
    if ( v67 - (unsigned __int64)v68 <= 0x11 )
    {
      sub_16E7EE0(&v65, "   ---User Time---", 18);
      v34 = (__m128i *)v68;
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F67A40);
      *((_WORD *)v68 + 8) = 11565;
      *v34 = si128;
      v34 = (__m128i *)(v68 + 18);
      v68 += 18;
    }
  }
  v35 = v60;
  if ( v60 != 0.0 )
  {
    if ( v67 - (unsigned __int64)v34 <= 0x11 )
    {
      sub_16E7EE0(&v65, "   --System Time--", 18);
      v34 = (__m128i *)v68;
      v35 = v60;
    }
    else
    {
      v52 = _mm_load_si128((const __m128i *)&xmmword_3F67A50);
      v34[1].m128i_i16[0] = 11565;
      *v34 = v52;
      v35 = v60;
      v34 = (__m128i *)(v68 + 18);
      v68 += 18;
    }
  }
  if ( v35 + v59 != 0.0 )
  {
    if ( v67 - (unsigned __int64)v34 <= 0x11 )
    {
      sub_16E7EE0(&v65, "   --User+System--", 18);
      v34 = (__m128i *)v68;
    }
    else
    {
      v51 = _mm_load_si128((const __m128i *)&xmmword_3F67A60);
      v34[1].m128i_i16[0] = 11565;
      *v34 = v51;
      v34 = (__m128i *)(v68 + 18);
      v68 += 18;
    }
  }
  if ( v67 - (unsigned __int64)v34 <= 0x11 )
  {
    sub_16E7EE0(&v65, "   ---Wall Time---", 18);
    v37 = v68;
  }
  else
  {
    v36 = _mm_load_si128((const __m128i *)&xmmword_3F67A70);
    v34[1].m128i_i16[0] = 11565;
    *v34 = v36;
    v37 = v68 + 18;
    v68 += 18;
  }
  if ( v61 )
  {
    if ( v67 - (unsigned __int64)v37 <= 0xA )
    {
      sub_16E7EE0(&v65, "  ---Mem---", 11);
      v37 = v68;
    }
    else
    {
      qmemcpy(v37, "  ---Mem---", 11);
      v37 = v68 + 11;
      v68 += 11;
    }
  }
  if ( v67 - (unsigned __int64)v37 <= 0xE )
  {
    sub_16E7EE0(&v65, "  --- Name ---\n", 15);
  }
  else
  {
    qmemcpy(v37, "  --- Name ---\n", 15);
    v68 += 15;
  }
  v55 = *(_QWORD *)(a1 + 72);
  v38 = *(_QWORD *)(a1 + 80);
  v39 = v38 - 96;
  if ( v38 != v55 )
  {
    do
    {
      while ( 1 )
      {
        v40 = v39;
        sub_16D79D0((double *)v39, (__int64)&v58, (__int64)&v65);
        v41 = sub_16E7EE0(&v65, *(const char **)(v39 + 64), *(_QWORD *)(v39 + 72));
        v42 = *(_BYTE **)(v41 + 24);
        if ( (unsigned __int64)v42 >= *(_QWORD *)(v41 + 16) )
          break;
        v39 -= 96;
        *(_QWORD *)(v41 + 24) = v42 + 1;
        *v42 = 10;
        if ( v40 == v55 )
          goto LABEL_42;
      }
      v39 -= 96;
      sub_16E7DE0(v41, 10);
    }
    while ( v40 != v55 );
  }
LABEL_42:
  sub_16D79D0(&v58, (__int64)&v58, (__int64)&v65);
  v43 = v68;
  if ( v67 - (unsigned __int64)v68 <= 6 )
  {
    sub_16E7EE0(&v65, "Total\n\n", 7);
    v44 = v68;
  }
  else
  {
    *(_DWORD *)v68 = 1635020628;
    *((_WORD *)v43 + 2) = 2668;
    v43[6] = 10;
    v44 = v68 + 7;
    v68 += 7;
  }
  if ( v66 != v44 )
    sub_16E7BA0(&v65);
  sub_16E7EE0(a2, *v70, v70[1]);
  if ( *(_QWORD *)(a2 + 24) != *(_QWORD *)(a2 + 8) )
    sub_16E7BA0(a2);
  v45 = *(_QWORD **)(a1 + 72);
  v46 = *(_QWORD **)(a1 + 80);
  if ( v45 != v46 )
  {
    v47 = *(_QWORD **)(a1 + 72);
    do
    {
      v48 = (_QWORD *)v47[8];
      if ( v48 != v47 + 10 )
        j_j___libc_free_0(v48, v47[10] + 1LL);
      v49 = (_QWORD *)v47[4];
      if ( v49 != v47 + 6 )
        j_j___libc_free_0(v49, v47[6] + 1LL);
      v47 += 12;
    }
    while ( v46 != v47 );
    *(_QWORD *)(a1 + 80) = v45;
  }
  result = sub_16E7BC0(&v65);
  if ( (_QWORD *)v56[0] != v57 )
    return j_j___libc_free_0(v56[0], v57[0] + 1LL);
  return result;
}
