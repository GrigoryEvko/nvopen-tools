// Function: sub_357EDB0
// Address: 0x357edb0
//
void __fastcall sub_357EDB0(__m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, char a7)
{
  size_t v7; // rbx
  const void *v8; // r9
  __m128i *v9; // r13
  size_t v10; // r12
  const void *v11; // r8
  size_t v12; // rdx
  signed __int64 v13; // rax
  size_t v14; // r15
  const void *v15; // rsi
  size_t v16; // rdx
  int v17; // eax
  __int64 v18; // r12
  size_t v19; // rdx
  int v20; // eax
  int v21; // r8d
  int v22; // r9d
  __int64 v23; // rdx
  __m128i *v24; // rbx
  const void *v25; // r13
  size_t v26; // r14
  unsigned __int64 v27; // r12
  size_t v28; // r15
  size_t v29; // rdx
  int v30; // eax
  size_t v31; // rbx
  unsigned __int64 i; // r15
  size_t v33; // r14
  size_t v34; // rdx
  signed __int64 v35; // rax
  unsigned int v36; // ecx
  __int64 v37; // rax
  __int64 v38; // rbx
  size_t v39; // r15
  const void *v40; // rsi
  size_t v41; // rdx
  int v42; // eax
  __int64 v43; // rbx
  size_t v44; // rdx
  int v45; // eax
  __int64 v46; // r12
  __int64 v47; // rax
  __int64 v48; // rax
  __m128i *v49; // r15
  __m128i *v50; // r13
  __int64 v51; // rax
  __int64 v52; // rax
  __m128i *v53; // rax
  __int64 v54; // rbx
  size_t v55; // rdx
  __m128i *v56; // [rsp+0h] [rbp-C0h]
  __int64 v57; // [rsp+8h] [rbp-B8h]
  __m128i *v58; // [rsp+10h] [rbp-B0h]
  const void *v59; // [rsp+20h] [rbp-A0h]
  const void *v60; // [rsp+20h] [rbp-A0h]
  __int64 v61; // [rsp+20h] [rbp-A0h]
  const void *v62; // [rsp+20h] [rbp-A0h]
  void *s2a; // [rsp+28h] [rbp-98h]
  __m128i *s2; // [rsp+28h] [rbp-98h]
  __m128i *v65; // [rsp+30h] [rbp-90h]
  __int64 v66; // [rsp+38h] [rbp-88h]
  __m128i v67; // [rsp+40h] [rbp-80h] BYREF
  __int64 v68; // [rsp+50h] [rbp-70h]
  __m128i v69; // [rsp+60h] [rbp-60h] BYREF
  __m128i v70; // [rsp+70h] [rbp-50h] BYREF
  __int64 v71; // [rsp+80h] [rbp-40h]

  v57 = a3;
  v58 = (__m128i *)a2;
  if ( a2 - (__int64)a1 <= 640 )
    return;
  if ( !a3 )
  {
    v61 = a2;
    goto LABEL_72;
  }
  v56 = (__m128i *)((char *)a1 + 40);
  while ( 2 )
  {
    --v57;
    v7 = a1[3].m128i_u64[0];
    v8 = (const void *)a1[2].m128i_i64[1];
    v9 = (__m128i *)((char *)a1 + 40 * ((__int64)(0xCCCCCCCCCCCCCCCDLL * (((char *)v58 - (char *)a1) >> 3)) / 2));
    v10 = v9->m128i_u64[1];
    v11 = (const void *)v9->m128i_i64[0];
    v12 = v10;
    if ( v7 <= v10 )
      v12 = a1[3].m128i_u64[0];
    if ( !v12
      || (v59 = (const void *)v9->m128i_i64[0],
          s2a = (void *)a1[2].m128i_i64[1],
          LODWORD(v13) = memcmp(s2a, (const void *)v9->m128i_i64[0], v12),
          v8 = s2a,
          v11 = v59,
          !(_DWORD)v13) )
    {
      v13 = v7 - v10;
      if ( (__int64)(v7 - v10) >= 0x80000000LL )
        goto LABEL_52;
      if ( v13 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
        goto LABEL_11;
    }
    if ( (int)v13 >= 0 )
    {
LABEL_52:
      v39 = v58[-2].m128i_u64[0];
      v40 = (const void *)v58[-3].m128i_i64[1];
      v41 = v39;
      if ( v7 <= v39 )
        v41 = v7;
      if ( !v41 || (v62 = v11, v42 = memcmp(v8, v40, v41), v11 = v62, !v42) )
      {
        v43 = v7 - v39;
        if ( v43 >= 0x80000000LL )
        {
LABEL_60:
          v44 = v39;
          if ( v10 <= v39 )
            v44 = v10;
          if ( !v44 || (v45 = memcmp(v11, v40, v44)) == 0 )
          {
            v46 = v10 - v39;
            if ( v46 >= 0x80000000LL )
              goto LABEL_69;
            if ( v46 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
              goto LABEL_68;
            v45 = v46;
          }
          if ( v45 < 0 )
            goto LABEL_68;
          goto LABEL_69;
        }
        if ( v43 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
          goto LABEL_24;
        v42 = v43;
      }
      if ( v42 < 0 )
        goto LABEL_24;
      goto LABEL_60;
    }
LABEL_11:
    v14 = v58[-2].m128i_u64[0];
    v15 = (const void *)v58[-3].m128i_i64[1];
    v16 = v14;
    if ( v10 <= v14 )
      v16 = v10;
    if ( !v16 || (v60 = v8, v17 = memcmp(v11, v15, v16), v8 = v60, !v17) )
    {
      v18 = v10 - v14;
      if ( v18 >= 0x80000000LL )
        goto LABEL_19;
      if ( v18 > (__int64)0xFFFFFFFF7FFFFFFFLL )
      {
        v17 = v18;
        goto LABEL_18;
      }
LABEL_69:
      sub_22415E0(a1, v9);
      v48 = a1[2].m128i_i64[0];
      a1[2].m128i_i64[0] = v9[2].m128i_i64[0];
      v9[2].m128i_i64[0] = v48;
      goto LABEL_25;
    }
LABEL_18:
    if ( v17 < 0 )
      goto LABEL_69;
LABEL_19:
    v19 = v14;
    if ( v7 <= v14 )
      v19 = v7;
    if ( !v19 || (v20 = memcmp(v8, v15, v19)) == 0 )
    {
      v38 = v7 - v14;
      if ( v38 >= 0x80000000LL )
        goto LABEL_24;
      if ( v38 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
      {
LABEL_68:
        sub_22415E0(a1, (__m128i *)((char *)v58 - 40));
        v47 = a1[2].m128i_i64[0];
        a1[2].m128i_i64[0] = v58[-1].m128i_i64[1];
        v58[-1].m128i_i64[1] = v47;
        goto LABEL_25;
      }
      v20 = v38;
    }
    if ( v20 < 0 )
      goto LABEL_68;
LABEL_24:
    sub_22415E0(a1, v56);
    v23 = a1[4].m128i_i64[1];
    a1[4].m128i_i64[1] = a1[2].m128i_i64[0];
    a1[2].m128i_i64[0] = v23;
LABEL_25:
    v24 = v56;
    v25 = (const void *)a1->m128i_i64[0];
    v26 = a1->m128i_u64[1];
    v27 = (unsigned __int64)v58;
    while ( 1 )
    {
      v28 = v24->m128i_u64[1];
      v29 = v26;
      v61 = (__int64)v24;
      if ( v28 <= v26 )
        v29 = v24->m128i_u64[1];
      if ( v29 )
      {
        v30 = memcmp((const void *)v24->m128i_i64[0], v25, v29);
        if ( v30 )
          break;
      }
      if ( (__int64)(v28 - v26) >= 0x80000000LL )
        goto LABEL_34;
      if ( (__int64)(v28 - v26) > (__int64)0xFFFFFFFF7FFFFFFFLL )
      {
        v30 = v28 - v26;
        break;
      }
LABEL_45:
      v24 = (__m128i *)((char *)v24 + 40);
    }
    if ( v30 < 0 )
      goto LABEL_45;
LABEL_34:
    s2 = v24;
    v31 = v26;
    for ( i = v27 - 40; ; i -= 40LL )
    {
      v33 = *(_QWORD *)(i + 8);
      v34 = v31;
      v27 = i;
      if ( v33 <= v31 )
        v34 = *(_QWORD *)(i + 8);
      if ( !v34 )
        break;
      LODWORD(v35) = memcmp(v25, *(const void **)i, v34);
      if ( !(_DWORD)v35 )
        break;
LABEL_36:
      if ( (int)v35 >= 0 )
        goto LABEL_43;
LABEL_37:
      ;
    }
    v36 = 0x80000000;
    v35 = v31 - v33;
    if ( (__int64)(v31 - v33) < 0x80000000LL )
    {
      if ( v35 <= (__int64)0xFFFFFFFF7FFFFFFFLL )
        goto LABEL_37;
      goto LABEL_36;
    }
LABEL_43:
    v24 = s2;
    if ( (unsigned __int64)s2 < i )
    {
      sub_22415E0(s2, (__m128i *)i);
      v37 = s2[2].m128i_i64[0];
      s2[2].m128i_i64[0] = *(_QWORD *)(i + 32);
      *(_QWORD *)(i + 32) = v37;
      v26 = a1->m128i_u64[1];
      v25 = (const void *)a1->m128i_i64[0];
      goto LABEL_45;
    }
    sub_357EDB0((_DWORD)s2, (_DWORD)v58, v57, v36, v21, v22, (char)v56);
    if ( (char *)s2 - (char *)a1 > 640 )
    {
      if ( v57 )
      {
        v58 = s2;
        continue;
      }
LABEL_72:
      v69.m128i_i8[0] = a7;
      v49 = (__m128i *)(v61 - 24);
      sub_357EC10((__int64)a1, v61);
      v50 = a1 + 1;
      do
      {
        v65 = &v67;
        if ( (__m128i *)v49[-1].m128i_i64[0] == v49 )
        {
          v67 = _mm_loadu_si128(v49);
        }
        else
        {
          v65 = (__m128i *)v49[-1].m128i_i64[0];
          v67.m128i_i64[0] = v49->m128i_i64[0];
        }
        v51 = v49[-1].m128i_i64[1];
        v49[-1].m128i_i64[0] = (__int64)v49;
        v49[-1].m128i_i64[1] = 0;
        v66 = v51;
        v52 = v49[1].m128i_i64[0];
        v49->m128i_i8[0] = 0;
        v68 = v52;
        if ( v50 == (__m128i *)a1->m128i_i64[0] )
        {
          v55 = a1->m128i_u64[1];
          if ( v55 )
          {
            if ( v55 == 1 )
              v49->m128i_i8[0] = a1[1].m128i_i8[0];
            else
              memcpy(v49, v50, v55);
            v55 = a1->m128i_u64[1];
          }
          v49[-1].m128i_i64[1] = v55;
          v49->m128i_i8[v55] = 0;
          v53 = (__m128i *)a1->m128i_i64[0];
        }
        else
        {
          v49[-1].m128i_i64[0] = a1->m128i_i64[0];
          v49[-1].m128i_i64[1] = a1->m128i_i64[1];
          v49->m128i_i64[0] = a1[1].m128i_i64[0];
          v53 = a1 + 1;
          a1->m128i_i64[0] = (__int64)v50;
        }
        a1->m128i_i64[1] = 0;
        v53->m128i_i8[0] = 0;
        v49[1].m128i_i64[0] = a1[2].m128i_i64[0];
        v69.m128i_i64[0] = (__int64)&v70;
        if ( v65 == &v67 )
        {
          v70 = _mm_load_si128(&v67);
        }
        else
        {
          v69.m128i_i64[0] = (__int64)v65;
          v70.m128i_i64[0] = v67.m128i_i64[0];
        }
        v54 = (char *)&v49[-1] - (char *)a1;
        v67.m128i_i8[0] = 0;
        v69.m128i_i64[1] = v66;
        v71 = v68;
        sub_357D140((__int64)a1, 0, 0xCCCCCCCCCCCCCCCDLL * (v54 >> 3), &v69);
        if ( (__m128i *)v69.m128i_i64[0] != &v70 )
          j_j___libc_free_0(v69.m128i_u64[0]);
        v49 = (__m128i *)((char *)v49 - 40);
      }
      while ( v54 > 40 );
    }
    break;
  }
}
