// Function: sub_EFC170
// Address: 0xefc170
//
__int64 *__fastcall sub_EFC170(__int64 *a1, _QWORD *a2, void **p_s)
{
  _QWORD *i; // rbx
  _QWORD *v5; // r15
  __int64 v6; // r12
  __int64 v7; // r13
  __int64 v8; // rbx
  __int64 v9; // r14
  _QWORD *v10; // rdi
  _QWORD *v12; // r13
  const __m128i *v13; // r15
  size_t v14; // rdx
  int *v15; // rsi
  size_t v16; // r14
  unsigned __int64 v17; // rcx
  __m128i **v18; // rax
  __int64 v19; // rbx
  __int64 v20; // r14
  __int64 v21; // r12
  __int64 v22; // r13
  _QWORD *v23; // rdi
  _QWORD *v24; // r14
  unsigned __int64 v25; // rsi
  _QWORD *v26; // rax
  _QWORD *v27; // rdi
  __m128i *v28; // rax
  __m128i *v29; // r12
  __m128i v30; // xmm0
  __m128i *v31; // rdx
  __m128i v32; // xmm1
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  __int32 v36; // ecx
  __int64 v37; // rax
  __m128i *v38; // rdx
  __int32 v39; // ecx
  unsigned __int64 v40; // r14
  unsigned __int64 v41; // rbx
  __m128i **v42; // rax
  __int64 v43; // rbx
  __int64 v44; // r13
  _QWORD *v45; // rdi
  __int64 v46; // rax
  _QWORD *v47; // rdi
  __int64 v48; // rcx
  __int64 v49; // rdx
  __int64 v52; // [rsp+40h] [rbp-1F0h]
  char v53; // [rsp+57h] [rbp-1D9h]
  _QWORD *v54; // [rsp+58h] [rbp-1D8h]
  __m128i *v55; // [rsp+58h] [rbp-1D8h]
  unsigned __int64 v56; // [rsp+60h] [rbp-1D0h] BYREF
  void *s; // [rsp+70h] [rbp-1C0h] BYREF
  unsigned __int64 v58; // [rsp+78h] [rbp-1B8h]
  _QWORD *v59; // [rsp+80h] [rbp-1B0h]
  __int64 v60; // [rsp+88h] [rbp-1A8h]
  int v61; // [rsp+90h] [rbp-1A0h]
  __int64 v62; // [rsp+98h] [rbp-198h]
  _QWORD v63[2]; // [rsp+A0h] [rbp-190h] BYREF
  _QWORD v64[48]; // [rsp+B0h] [rbp-180h] BYREF

  s = v63;
  v58 = 1;
  v59 = 0;
  v60 = 0;
  v61 = 1065353216;
  v62 = 0;
  v63[0] = 0;
  if ( LOBYTE(qword_4F8AE48[8]) )
    goto LABEL_18;
  if ( unk_4F838D3 )
  {
    v24 = sub_C52410();
    v12 = v24 + 1;
    v25 = sub_C959E0();
    v26 = (_QWORD *)v24[2];
    if ( v26 )
    {
      v27 = v24 + 1;
      do
      {
        if ( v25 > v26[4] )
        {
          v26 = (_QWORD *)v26[3];
        }
        else
        {
          v27 = v26;
          v26 = (_QWORD *)v26[2];
        }
      }
      while ( v26 );
      if ( v12 != v27 && v25 >= v27[4] )
        v12 = v27;
    }
    if ( v12 == (_QWORD *)((char *)sub_C52410() + 8) )
      goto LABEL_18;
    v46 = v12[7];
    if ( !v46 )
      goto LABEL_18;
    v47 = v12 + 6;
    do
    {
      while ( 1 )
      {
        v48 = *(_QWORD *)(v46 + 16);
        v49 = *(_QWORD *)(v46 + 24);
        if ( *(_DWORD *)(v46 + 32) >= unk_4F8AE08 )
          break;
        v46 = *(_QWORD *)(v46 + 24);
        if ( !v49 )
          goto LABEL_57;
      }
      v47 = (_QWORD *)v46;
      v46 = *(_QWORD *)(v46 + 16);
    }
    while ( v48 );
LABEL_57:
    if ( v12 + 6 == v47 || unk_4F8AE08 < *((_DWORD *)v47 + 8) || !*((_DWORD *)v47 + 9) )
    {
LABEL_18:
      v13 = (const __m128i *)p_s[2];
      if ( v13 )
      {
        while ( 1 )
        {
          v14 = v13[2].m128i_u64[1];
          v15 = (int *)v13[2].m128i_i64[0];
          v16 = v14;
          memset(&v64[20], 0, 0xB0u);
          v17 = v14;
          v52 = v14;
          v64[34] = 0;
          v64[32] = &v64[30];
          v64[33] = &v64[30];
          v64[37] = 0;
          v64[38] = &v64[36];
          v64[39] = &v64[36];
          v64[40] = 0;
          if ( v15 )
          {
            sub_C7D030(v64);
            sub_C7D280((int *)v64, v15, v16);
            sub_C7D290(v64, &v56);
            v17 = v56;
          }
          v64[0] = v17;
          v18 = (__m128i **)sub_C1DD00(&s, v17 % v58, v64, v17);
          if ( v18 )
          {
            v55 = *v18;
            if ( *v18 )
              break;
          }
          v28 = (__m128i *)sub_22077B0(200);
          v29 = v28;
          if ( v28 )
            v28->m128i_i64[0] = 0;
          v30 = _mm_loadu_si128((const __m128i *)&v64[22]);
          v31 = v28 + 6;
          v32 = _mm_loadu_si128((const __m128i *)&v64[24]);
          v28->m128i_i64[1] = v64[0];
          v33 = v64[20];
          v29[2] = v30;
          v29[1].m128i_i64[0] = v33;
          v34 = v64[21];
          v29[3] = v32;
          v29[1].m128i_i64[1] = v34;
          v29[4].m128i_i64[0] = v64[26];
          v29[4].m128i_i64[1] = v64[27];
          v29[5].m128i_i64[0] = v64[28];
          v35 = v64[31];
          if ( v64[31] )
          {
            v36 = v64[30];
            v29[6].m128i_i64[1] = v64[31];
            v29[6].m128i_i32[0] = v36;
            v29[7].m128i_i64[0] = v64[32];
            v29[7].m128i_i64[1] = v64[33];
            *(_QWORD *)(v35 + 8) = v31;
            v64[31] = 0;
            v29[8].m128i_i64[0] = v64[34];
            v64[34] = 0;
            v64[32] = &v64[30];
            v64[33] = &v64[30];
          }
          else
          {
            v29[6].m128i_i32[0] = 0;
            v29[6].m128i_i64[1] = 0;
            v29[7].m128i_i64[0] = (__int64)v31;
            v29[7].m128i_i64[1] = (__int64)v31;
            v29[8].m128i_i64[0] = 0;
          }
          v37 = v64[37];
          v38 = v29 + 9;
          if ( v64[37] )
          {
            v39 = v64[36];
            v29[9].m128i_i64[1] = v64[37];
            v29[9].m128i_i32[0] = v39;
            v29[10].m128i_i64[0] = v64[38];
            v29[10].m128i_i64[1] = v64[39];
            *(_QWORD *)(v37 + 8) = v38;
            v64[37] = 0;
            v29[11].m128i_i64[0] = v64[40];
            v64[40] = 0;
            v64[38] = &v64[36];
            v64[39] = &v64[36];
          }
          else
          {
            v29[9].m128i_i32[0] = 0;
            v29[9].m128i_i64[1] = 0;
            v29[10].m128i_i64[0] = (__int64)v38;
            v29[10].m128i_i64[1] = (__int64)v38;
            v29[11].m128i_i64[0] = 0;
          }
          v40 = v29->m128i_u64[1];
          v29[11].m128i_i64[1] = v64[41];
          v41 = v40 % v58;
          v42 = (__m128i **)sub_C1DD00(&s, v40 % v58, &v29->m128i_i64[1], v40);
          if ( v42 && (v55 = *v42) != 0 )
          {
            v43 = v29[9].m128i_i64[1];
            while ( v43 )
            {
              v44 = v43;
              sub_EF8F20(*(_QWORD **)(v43 + 24));
              v45 = *(_QWORD **)(v43 + 56);
              v43 = *(_QWORD *)(v43 + 16);
              sub_EF9170(v45);
              j_j___libc_free_0(v44, 88);
            }
            sub_EF88E0((_QWORD *)v29[6].m128i_i64[1]);
            j_j___libc_free_0(v29, 200);
          }
          else
          {
            v55 = (__m128i *)sub_EF8460(&s, v41, v40, v29, 1);
          }
          v19 = v64[37];
          v53 = 1;
          if ( v64[37] )
            goto LABEL_24;
          sub_EF88E0((_QWORD *)v64[31]);
LABEL_28:
          v55[2].m128i_i64[0] = (__int64)v15;
          v55[3].m128i_i64[0] = 0;
          v55[2].m128i_i64[1] = v52;
          v55[3].m128i_i64[1] = 0;
          v55[4].m128i_i64[0] = 0;
LABEL_29:
          sub_C1D5C0(v55 + 1, v13 + 1, 1u);
          v13 = (const __m128i *)v13->m128i_i64[0];
          if ( !v13 )
            goto LABEL_30;
        }
        v19 = v64[37];
        v53 = 0;
        if ( !v64[37] )
        {
          sub_EF88E0((_QWORD *)v64[31]);
          goto LABEL_29;
        }
        do
        {
LABEL_24:
          v20 = v19;
          sub_EF8F20(*(_QWORD **)(v19 + 24));
          v21 = *(_QWORD *)(v19 + 56);
          v19 = *(_QWORD *)(v19 + 16);
          while ( v21 )
          {
            v22 = v21;
            sub_EF9170(*(_QWORD **)(v21 + 24));
            v23 = *(_QWORD **)(v21 + 184);
            v21 = *(_QWORD *)(v21 + 16);
            sub_EF8F20(v23);
            sub_EF88E0(*(_QWORD **)(v22 + 136));
            j_j___libc_free_0(v22, 224);
          }
          j_j___libc_free_0(v20, 88);
        }
        while ( v19 );
        sub_EF88E0((_QWORD *)v64[31]);
        if ( !v53 )
          goto LABEL_29;
        goto LABEL_28;
      }
LABEL_30:
      p_s = &s;
    }
  }
  for ( i = p_s[2]; i; i = (_QWORD *)*i )
    sub_EFBF90((__int64)a2, (__int64)(i + 2), 0);
  sub_EFA1E0(a1, a2);
  v54 = v59;
  while ( v54 )
  {
    v5 = v54;
    v6 = v54[19];
    v54 = (_QWORD *)*v54;
    while ( v6 )
    {
      v7 = v6;
      sub_EF8F20(*(_QWORD **)(v6 + 24));
      v8 = *(_QWORD *)(v6 + 56);
      v6 = *(_QWORD *)(v6 + 16);
      while ( v8 )
      {
        v9 = v8;
        sub_EF9170(*(_QWORD **)(v8 + 24));
        v10 = *(_QWORD **)(v8 + 184);
        v8 = *(_QWORD *)(v8 + 16);
        sub_EF8F20(v10);
        sub_EF88E0(*(_QWORD **)(v9 + 136));
        j_j___libc_free_0(v9, 224);
      }
      j_j___libc_free_0(v7, 88);
    }
    sub_EF88E0((_QWORD *)v5[13]);
    j_j___libc_free_0(v5, 200);
  }
  memset(s, 0, 8 * v58);
  v60 = 0;
  v59 = 0;
  if ( s != v63 )
    j_j___libc_free_0(s, 8 * v58);
  return a1;
}
