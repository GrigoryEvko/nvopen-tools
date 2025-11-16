// Function: sub_31807D0
// Address: 0x31807d0
//
__int64 __fastcall sub_31807D0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // rbx
  size_t v8; // rdx
  size_t v9; // r15
  int *v10; // r13
  _QWORD *v11; // rdi
  __m128i *v12; // rax
  __m128i *v13; // r8
  __int64 v14; // rcx
  __int64 v15; // rdx
  unsigned __int64 *i; // rbx
  __int64 m128i_i64; // rsi
  __int64 v18; // rdx
  __int64 v19; // rdx
  const __m128i *v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  unsigned __int64 v27; // r15
  _QWORD *v28; // rdi
  __int64 *j; // rax
  __int64 v30; // r12
  __int64 v31; // rax
  __int64 v32; // rbx
  __int64 v33; // rax
  __int64 v34; // r15
  __int64 k; // rbx
  _QWORD *v36; // rax
  __int64 v38; // rdx
  __int64 v42; // [rsp+20h] [rbp-100h]
  __m128i *v43; // [rsp+28h] [rbp-F8h]
  __m128i *v44; // [rsp+28h] [rbp-F8h]
  size_t v45; // [rsp+38h] [rbp-E8h] BYREF
  unsigned __int64 v46[2]; // [rsp+40h] [rbp-E0h] BYREF
  unsigned __int64 *v47; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v48; // [rsp+58h] [rbp-C8h]
  unsigned __int64 v49; // [rsp+60h] [rbp-C0h]
  unsigned __int64 v50; // [rsp+68h] [rbp-B8h]
  __int64 v51; // [rsp+70h] [rbp-B0h]
  unsigned __int64 *v52; // [rsp+78h] [rbp-A8h]
  _QWORD *v53; // [rsp+80h] [rbp-A0h]
  __int64 v54; // [rsp+88h] [rbp-98h]
  __int64 v55; // [rsp+90h] [rbp-90h]
  __int64 v56; // [rsp+98h] [rbp-88h]

  v6 = sub_317E460(a4);
  v7 = *a3;
  v9 = v8;
  if ( v6 )
  {
    v10 = (int *)v6;
    sub_C7D030(&v47);
    sub_C7D280((int *)&v47, v10, v9);
    sub_C7D290(&v47, v46);
    v9 = v46[0];
  }
  v45 = v9 + 33 * v7;
  v11 = (_QWORD *)sub_317E450(a2);
  v12 = (__m128i *)v11[2];
  if ( !v12 )
  {
    v13 = (__m128i *)(v11 + 1);
LABEL_36:
    v47 = &v45;
    v13 = (__m128i *)sub_317F580(v11, (__int64)v13, &v47);
    goto LABEL_10;
  }
  v13 = (__m128i *)(v11 + 1);
  do
  {
    while ( 1 )
    {
      v14 = v12[1].m128i_i64[0];
      v15 = v12[1].m128i_i64[1];
      if ( v12[2].m128i_i64[0] >= v45 )
        break;
      v12 = (__m128i *)v12[1].m128i_i64[1];
      if ( !v15 )
        goto LABEL_8;
    }
    v13 = v12;
    v12 = (__m128i *)v12[1].m128i_i64[0];
  }
  while ( v14 );
LABEL_8:
  if ( v11 + 1 == (_QWORD *)v13 || v45 < v13[2].m128i_i64[0] )
    goto LABEL_36;
LABEL_10:
  v42 = (__int64)&v13[2].m128i_i64[1];
  if ( (unsigned __int64 *)a4 == &v13[2].m128i_u64[1] )
    goto LABEL_21;
  i = (unsigned __int64 *)v13[3].m128i_i64[1];
  m128i_i64 = (__int64)v13[3].m128i_i64;
  v47 = i;
  v18 = v13[4].m128i_i64[1];
  v49 = (unsigned __int64)&v13[2].m128i_u64[1];
  v48 = v18;
  if ( i )
  {
    i[1] = 0;
    v19 = *(_QWORD *)(v18 + 16);
    if ( v19 )
      v48 = v19;
    v13[3].m128i_i64[1] = 0;
    v13[4].m128i_i64[0] = m128i_i64;
    v13[4].m128i_i64[1] = m128i_i64;
    v13[5].m128i_i64[0] = 0;
    v20 = *(const __m128i **)(a4 + 16);
    if ( !v20 )
      goto LABEL_20;
  }
  else
  {
    v48 = 0;
    v13[3].m128i_i64[1] = 0;
    v13[4].m128i_i64[0] = m128i_i64;
    v13[4].m128i_i64[1] = m128i_i64;
    v13[5].m128i_i64[0] = 0;
    v20 = *(const __m128i **)(a4 + 16);
    if ( !v20 )
      goto LABEL_21;
  }
  v43 = v13;
  v21 = sub_31804D0(v20, m128i_i64, &v47);
  v13 = v43;
  v22 = v21;
  do
  {
    v23 = v21;
    v21 = *(_QWORD *)(v21 + 16);
  }
  while ( v21 );
  v43[4].m128i_i64[0] = v23;
  v24 = v22;
  do
  {
    v25 = v24;
    v24 = *(_QWORD *)(v24 + 24);
  }
  while ( v24 );
  v43[4].m128i_i64[1] = v25;
  v26 = *(_QWORD *)(a4 + 40);
  v43[3].m128i_i64[1] = v22;
  v43[5].m128i_i64[0] = v26;
  for ( i = v47; i; v13 = v44 )
  {
LABEL_20:
    v27 = (unsigned __int64)i;
    v44 = v13;
    sub_317D930((_QWORD *)i[3]);
    v28 = (_QWORD *)i[7];
    i = (unsigned __int64 *)i[2];
    sub_317D930(v28);
    j_j___libc_free_0(v27);
  }
LABEL_21:
  v13[5].m128i_i64[1] = *(_QWORD *)(a4 + 48);
  v13[6] = _mm_loadu_si128((const __m128i *)(a4 + 56));
  v13[7].m128i_i64[0] = *(_QWORD *)(a4 + 72);
  v13[7].m128i_i64[1] = *(_QWORD *)(a4 + 80);
  v13[8].m128i_i64[0] = *(_QWORD *)(a4 + 88);
  sub_317E670(v42, a3);
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  sub_26C4970((__int64 *)&v47, 0);
  sub_317E660(v42, a2);
  v46[0] = v42;
  sub_317FBD0((unsigned __int64 *)&v47, v46);
  for ( j = (__int64 *)v49; (_QWORD *)v49 != v53; j = (__int64 *)v49 )
  {
    v30 = *j;
    if ( j == (__int64 *)(v51 - 8) )
    {
      j_j___libc_free_0(v50);
      v38 = *++v52 + 512;
      v50 = *v52;
      v51 = v38;
      v49 = v50;
    }
    else
    {
      v49 = (unsigned __int64)(j + 1);
    }
    v31 = sub_317E470(v30);
    v32 = v31;
    if ( v31 )
    {
      v46[0] = v31;
      *(_QWORD *)sub_317EE30((unsigned __int64 *)(a1 + 56), v46) = v30;
      *(_DWORD *)(v32 + 48) |= 2u;
    }
    v33 = sub_317E450(v30);
    v34 = *(_QWORD *)(v33 + 24);
    for ( k = v33 + 8; k != v34; v34 = sub_220EEE0(v34) )
    {
      while ( 1 )
      {
        v46[0] = v34 + 40;
        sub_317E660(v34 + 40, v30);
        v36 = v53;
        if ( v53 != (_QWORD *)(v55 - 8) )
          break;
        sub_26C4A60((unsigned __int64 *)&v47, v46);
        v34 = sub_220EEE0(v34);
        if ( k == v34 )
          goto LABEL_33;
      }
      if ( v53 )
      {
        *v53 = v46[0];
        v36 = v53;
      }
      v53 = v36 + 1;
    }
LABEL_33:
    ;
  }
  sub_26C2C00((unsigned __int64 *)&v47);
  return v42;
}
