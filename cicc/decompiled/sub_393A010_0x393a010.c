// Function: sub_393A010
// Address: 0x393a010
//
_DWORD *__fastcall sub_393A010(__int64 a1, int a2, _DWORD *a3)
{
  unsigned int v4; // r12d
  __m128i **v5; // rdi
  _QWORD *v6; // rax
  _QWORD *v7; // r13
  unsigned int v8; // edx
  __int64 i; // rax
  __int64 v10; // r14
  unsigned __int64 v11; // rcx
  __m128i *v12; // rsi
  unsigned int v13; // r14d
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 *v17; // rax
  __int64 v18; // rdx
  unsigned __int64 v19; // rcx
  char *v20; // r14
  __int64 v21; // rax
  unsigned __int64 v22; // r15
  __m128i *v23; // rdx
  const __m128i *v24; // rax
  __int64 v25; // rax
  unsigned __int64 v26; // r14
  unsigned __int64 v27; // rdi
  _DWORD *v28; // r12
  __int64 v29; // r12
  char *v30; // rax
  char *v31; // r14
  void *v32; // rcx
  unsigned __int64 v33; // rax
  unsigned __int64 v34; // r12
  unsigned __int64 v35; // rdi
  unsigned __int64 v36; // r12
  unsigned __int64 v37; // rdi
  __int64 v39; // [rsp+8h] [rbp-118h]
  __int64 v40; // [rsp+10h] [rbp-110h]
  __int64 v41; // [rsp+18h] [rbp-108h]
  __int64 v42; // [rsp+20h] [rbp-100h]
  __int64 v43; // [rsp+28h] [rbp-F8h]
  __int64 v44; // [rsp+30h] [rbp-F0h]
  char *v45; // [rsp+38h] [rbp-E8h]
  unsigned __int64 v46; // [rsp+40h] [rbp-E0h]
  unsigned __int64 v48; // [rsp+58h] [rbp-C8h] BYREF
  __m128i *v49; // [rsp+60h] [rbp-C0h] BYREF
  __m128i *v50; // [rsp+68h] [rbp-B8h] BYREF
  unsigned __int64 v51; // [rsp+70h] [rbp-B0h]
  __m128i **v52; // [rsp+78h] [rbp-A8h]
  __m128i **v53; // [rsp+80h] [rbp-A0h]
  __int64 v54; // [rsp+88h] [rbp-98h]
  unsigned __int64 v55; // [rsp+90h] [rbp-90h]
  char *v56; // [rsp+98h] [rbp-88h]
  char *v57; // [rsp+A0h] [rbp-80h]
  unsigned __int64 v58; // [rsp+A8h] [rbp-78h]
  __int64 v59; // [rsp+B0h] [rbp-70h]
  __int64 v60; // [rsp+B8h] [rbp-68h]
  __int64 v61; // [rsp+C0h] [rbp-60h]
  __int64 v62; // [rsp+C8h] [rbp-58h]
  __int64 v63; // [rsp+D0h] [rbp-50h]
  __int64 v64; // [rsp+D8h] [rbp-48h]
  __int64 v65; // [rsp+E0h] [rbp-40h]

  if ( a2 <= 3 )
  {
    v29 = 4LL * unk_49D9478;
    if ( (unsigned __int64)(4LL * unk_49D9478) > 0x7FFFFFFFFFFFFFFCLL )
      sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
    if ( v29 )
    {
      v30 = (char *)sub_22077B0(4LL * unk_49D9478);
      v31 = &v30[v29];
      v32 = memcpy(v30, &unk_4530740, 4LL * unk_49D9478);
    }
    else
    {
      v31 = 0;
      v32 = 0;
    }
    v57 = v31;
    v52 = &v50;
    v53 = &v50;
    LODWORD(v50) = 0;
    v51 = 0;
    v54 = 0;
    v55 = (unsigned __int64)v32;
    v56 = v31;
    v58 = 0;
    v59 = 0;
    v60 = 0;
    v61 = 0;
    v62 = 0;
    v63 = 0;
    v64 = 0;
    v65 = 0;
    sub_393CBF0(&v48, &v49);
    v33 = v48;
    v48 = 0;
    v34 = *(_QWORD *)(a1 + 40);
    *(_QWORD *)(a1 + 40) = v33;
    if ( v34 )
    {
      v35 = *(_QWORD *)(v34 + 8);
      if ( v35 )
        j_j___libc_free_0(v35);
      j_j___libc_free_0(v34);
      v36 = v48;
      if ( v48 )
      {
        v37 = *(_QWORD *)(v48 + 8);
        if ( v37 )
          j_j___libc_free_0(v37);
        j_j___libc_free_0(v36);
      }
    }
    if ( v58 )
      j_j___libc_free_0(v58);
    if ( v55 )
      j_j___libc_free_0(v55);
    v28 = a3;
    sub_39380C0(v51);
    return v28;
  }
  v4 = 8 * (3 * a3[2] + *a3 + 2);
  v5 = (__m128i **)v4;
  v6 = (_QWORD *)sub_22077B0(v4);
  v7 = v6;
  if ( v6 )
  {
    v5 = (__m128i **)v6;
    memset(v6, 0, v4);
  }
  v8 = 0;
  for ( i = 0; v8 < (unsigned __int64)v4 >> 3; i = v8 )
  {
    v7[i] = *(_QWORD *)&a3[2 * i];
    ++v8;
  }
  v49 = 0;
  v10 = v7[1];
  v50 = 0;
  v51 = 0;
  if ( !v10 )
  {
    v22 = 0;
    v45 = 0;
    v46 = 0;
    v39 = v7[2];
    v40 = v7[3];
    v41 = v7[4];
    v42 = v7[6];
    v43 = v7[5];
    v44 = v7[7];
    goto LABEL_23;
  }
  v11 = 0;
  v12 = 0;
  v13 = 0;
  v14 = 0;
  while ( 1 )
  {
    v17 = &v7[3 * v14 + 2 + *v7];
    v18 = *v17;
    LODWORD(v48) = *v17;
    if ( (__m128i *)v11 != v12 )
      break;
    v5 = &v49;
    sub_3939E50((unsigned __int64 *)&v49, v12, &v48, v17 + 1, v17 + 2);
    v12 = v50;
    v14 = ++v13;
    if ( (unsigned __int64)v13 >= v7[1] )
      goto LABEL_14;
LABEL_11:
    v11 = v51;
  }
  if ( v12 )
  {
    v15 = v17[1];
    v16 = v17[2];
    v12->m128i_i32[0] = v18;
    v12->m128i_i64[1] = v15;
    v12[1].m128i_i64[0] = v16;
    v12 = v50;
  }
  v12 = (__m128i *)((char *)v12 + 24);
  v50 = v12;
  v14 = ++v13;
  if ( (unsigned __int64)v13 < v7[1] )
    goto LABEL_11;
LABEL_14:
  v19 = (unsigned __int64)v49;
  v39 = v7[2];
  v40 = v7[3];
  v41 = v7[4];
  v42 = v7[6];
  v43 = v7[5];
  v44 = v7[7];
  v20 = (char *)((char *)v12 - (char *)v49);
  if ( v12 == v49 )
  {
    v45 = 0;
    v22 = 0;
  }
  else
  {
    if ( (unsigned __int64)((char *)v12 - (char *)v49) > 0x7FFFFFFFFFFFFFF8LL )
      sub_4261EA(v5, v12, 0x7FFFFFFFFFFFFFF8LL);
    v21 = sub_22077B0((char *)v12 - (char *)v49);
    v12 = v50;
    v19 = (unsigned __int64)v49;
    v22 = v21;
    v45 = &v20[v21];
  }
  if ( (__m128i *)v19 == v12 )
  {
    v46 = v22;
  }
  else
  {
    v23 = (__m128i *)v22;
    v24 = (const __m128i *)v19;
    do
    {
      if ( v23 )
      {
        *v23 = _mm_loadu_si128(v24);
        v23[1].m128i_i64[0] = v24[1].m128i_i64[0];
      }
      v24 = (const __m128i *)((char *)v24 + 24);
      v23 = (__m128i *)((char *)v23 + 24);
    }
    while ( v24 != v12 );
    v46 = v22 + 8 * (((unsigned __int64)&v24[-2].m128i_u64[1] - v19) >> 3) + 24;
  }
LABEL_23:
  v25 = sub_22077B0(0x48u);
  if ( v25 )
  {
    *(_DWORD *)v25 = 0;
    *(_QWORD *)(v25 + 8) = v22;
    *(_QWORD *)(v25 + 16) = v46;
    *(_QWORD *)(v25 + 24) = v45;
    *(_QWORD *)(v25 + 32) = v44;
    *(_QWORD *)(v25 + 40) = v43;
    *(_QWORD *)(v25 + 48) = v42;
    *(_QWORD *)(v25 + 56) = v41;
    *(_DWORD *)(v25 + 64) = v40;
    *(_DWORD *)(v25 + 68) = v39;
  }
  else if ( v22 )
  {
    j_j___libc_free_0(v22);
    v25 = 0;
  }
  v26 = *(_QWORD *)(a1 + 40);
  *(_QWORD *)(a1 + 40) = v25;
  if ( v26 )
  {
    v27 = *(_QWORD *)(v26 + 8);
    if ( v27 )
      j_j___libc_free_0(v27);
    j_j___libc_free_0(v26);
  }
  v28 = &a3[v4 / 4];
  if ( v49 )
    j_j___libc_free_0((unsigned __int64)v49);
  j___libc_free_0((unsigned __int64)v7);
  return v28;
}
