// Function: sub_2F0C170
// Address: 0x2f0c170
//
void __fastcall sub_2F0C170(unsigned __int64 a1, __int64 a2, __int64 a3, int *a4)
{
  __int64 v4; // rdx
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // rax
  const __m128i *v7; // r14
  const __m128i *v8; // r13
  __m128i *j; // r12
  const __m128i *v10; // r14
  const __m128i *v11; // r15
  __int64 v12; // rax
  __m128i *k; // r13
  bool v14; // r15
  unsigned __int64 *m; // r14
  unsigned __int64 *n; // r13
  __int64 v17; // rax
  _QWORD *v18; // rax
  unsigned __int64 v19; // r13
  unsigned __int64 *v20; // r12
  __int64 v21; // r14
  unsigned __int64 *ii; // rbx
  __int64 v23; // rbx
  unsigned int v24; // ecx
  const __m128i *v25; // rcx
  unsigned __int64 v26; // r13
  const __m128i *v27; // r14
  __m128i *v28; // r12
  __int64 v29; // rbx
  __int64 v30; // r8
  __int64 v31; // rax
  const __m128i *v32; // rbx
  __m128i *v33; // r14
  const __m128i *v34; // r15
  bool v35; // zf
  unsigned __int64 *jj; // r15
  unsigned __int64 *kk; // r14
  unsigned __int64 v38; // rax
  unsigned __int64 v39; // r14
  unsigned __int64 *v40; // r12
  __int64 v41; // rbx
  unsigned __int64 *v42; // r15
  unsigned __int64 v43; // r12
  unsigned __int64 *v44; // rbx
  unsigned __int64 *v45; // r14
  _QWORD *v46; // rdx
  _QWORD *v47; // rax
  unsigned __int64 v48; // r12
  unsigned __int64 *v49; // r15
  unsigned __int64 *v50; // r14
  __int64 v53; // [rsp+10h] [rbp-A0h]
  __int64 v54; // [rsp+10h] [rbp-A0h]
  __int64 v55; // [rsp+18h] [rbp-98h]
  unsigned int v56; // [rsp+20h] [rbp-90h]
  unsigned int v57; // [rsp+20h] [rbp-90h]
  unsigned int v58; // [rsp+24h] [rbp-8Ch]
  unsigned int v59; // [rsp+24h] [rbp-8Ch]
  __int64 v60; // [rsp+28h] [rbp-88h]
  unsigned int v61; // [rsp+28h] [rbp-88h]
  unsigned int v62; // [rsp+30h] [rbp-80h]
  const __m128i *v63; // [rsp+30h] [rbp-80h]
  unsigned int v64; // [rsp+38h] [rbp-78h]
  __int64 v65; // [rsp+38h] [rbp-78h]
  __int64 v66; // [rsp+40h] [rbp-70h]
  unsigned __int64 v67; // [rsp+48h] [rbp-68h]
  unsigned __int64 v68; // [rsp+48h] [rbp-68h]
  unsigned __int64 v69; // [rsp+50h] [rbp-60h]
  unsigned int v70; // [rsp+50h] [rbp-60h]
  bool v71; // [rsp+50h] [rbp-60h]
  __int64 i; // [rsp+58h] [rbp-58h]
  __int64 v73; // [rsp+58h] [rbp-58h]
  unsigned __int64 v74; // [rsp+60h] [rbp-50h]
  unsigned __int64 v75; // [rsp+68h] [rbp-48h]
  unsigned __int64 v76; // [rsp+68h] [rbp-48h]
  unsigned __int64 v77; // [rsp+70h] [rbp-40h]
  unsigned __int64 v78; // [rsp+70h] [rbp-40h]
  __int64 v79; // [rsp+78h] [rbp-38h]
  __int64 v80; // [rsp+78h] [rbp-38h]

  v4 = a3 - 1;
  v74 = a1;
  v55 = a2;
  v53 = v4 / 2;
  if ( a2 >= v4 / 2 )
  {
    v80 = a2;
  }
  else
  {
    for ( i = a2; ; i = v80 )
    {
      v79 = 2 * (i + 1);
      v5 = v74 + ((i + 1) << 6);
      v6 = v74 + 32 * (v79 - 1);
      v7 = *(const __m128i **)(v6 + 16);
      v69 = v6;
      v8 = *(const __m128i **)(v6 + 8);
      v62 = *(_DWORD *)v6;
      v58 = *(_DWORD *)(v6 + 4);
      v67 = (char *)v7 - (char *)v8;
      if ( v7 == v8 )
      {
        v77 = 0;
      }
      else
      {
        if ( (unsigned __int64)((char *)v7 - (char *)v8) > 0x7FFFFFFFFFFFFFF8LL )
          goto LABEL_105;
        a1 = (char *)v7 - (char *)v8;
        v77 = sub_22077B0(v67);
        v7 = *(const __m128i **)(v69 + 16);
        v8 = *(const __m128i **)(v69 + 8);
      }
      for ( j = (__m128i *)v77; v7 != v8; j = (__m128i *)((char *)j + 56) )
      {
        if ( j )
        {
          a1 = (unsigned __int64)j;
          j->m128i_i64[0] = (__int64)j[1].m128i_i64;
          a2 = v8->m128i_i64[0];
          sub_2F07250(j->m128i_i64, v8->m128i_i64[0], v8->m128i_i64[0] + v8->m128i_i64[1]);
          j[2] = _mm_loadu_si128(v8 + 2);
          j[3].m128i_i16[0] = v8[3].m128i_i16[0];
        }
        v8 = (const __m128i *)((char *)v8 + 56);
      }
      v10 = *(const __m128i **)(v5 + 16);
      v11 = *(const __m128i **)(v5 + 8);
      v64 = *(_DWORD *)v5;
      v56 = *(_DWORD *)(v5 + 4);
      v60 = (char *)v10 - (char *)v11;
      if ( v10 == v11 )
      {
        v75 = 0;
      }
      else
      {
        if ( (unsigned __int64)((char *)v10 - (char *)v11) > 0x7FFFFFFFFFFFFFF8LL )
LABEL_105:
          sub_4261EA(a1, a2, v4);
        a1 = (char *)v10 - (char *)v11;
        v12 = sub_22077B0((char *)v10 - (char *)v11);
        v10 = *(const __m128i **)(v5 + 16);
        v11 = *(const __m128i **)(v5 + 8);
        v75 = v12;
      }
      for ( k = (__m128i *)v75; v10 != v11; k = (__m128i *)((char *)k + 56) )
      {
        if ( k )
        {
          a1 = (unsigned __int64)k;
          k->m128i_i64[0] = (__int64)k[1].m128i_i64;
          a2 = v11->m128i_i64[0];
          sub_2F07250(k->m128i_i64, v11->m128i_i64[0], v11->m128i_i64[0] + v11->m128i_i64[1]);
          k[2] = _mm_loadu_si128(v11 + 2);
          k[3].m128i_i16[0] = v11[3].m128i_i16[0];
        }
        v11 = (const __m128i *)((char *)v11 + 56);
      }
      v14 = v64 < v62;
      if ( v64 == v62 )
        v14 = v58 > v56;
      for ( m = (unsigned __int64 *)v75; m != (unsigned __int64 *)k; m += 7 )
      {
        a1 = *m;
        if ( (unsigned __int64 *)*m != m + 2 )
        {
          a2 = m[2] + 1;
          j_j___libc_free_0(a1);
        }
      }
      if ( v75 )
      {
        a2 = v60;
        a1 = v75;
        j_j___libc_free_0(v75);
      }
      for ( n = (unsigned __int64 *)v77; n != (unsigned __int64 *)j; n += 7 )
      {
        a1 = *n;
        if ( (unsigned __int64 *)*n != n + 2 )
        {
          a2 = n[2] + 1;
          j_j___libc_free_0(a1);
        }
      }
      if ( v77 )
      {
        a2 = v67;
        a1 = v77;
        j_j___libc_free_0(v77);
      }
      v17 = v79 - 1;
      if ( v14 )
        v5 = v74 + 32 * (v79 - 1);
      else
        v17 = 2 * (i + 1);
      v80 = v17;
      v18 = (_QWORD *)(v74 + 32 * i);
      v19 = v18[1];
      v20 = (unsigned __int64 *)v18[2];
      *v18 = *(_QWORD *)v5;
      v21 = v18[3];
      v18[1] = *(_QWORD *)(v5 + 8);
      v18[2] = *(_QWORD *)(v5 + 16);
      v4 = *(_QWORD *)(v5 + 24);
      v18[3] = v4;
      *(_QWORD *)(v5 + 8) = 0;
      *(_QWORD *)(v5 + 16) = 0;
      *(_QWORD *)(v5 + 24) = 0;
      for ( ii = (unsigned __int64 *)v19; v20 != ii; ii += 7 )
      {
        a1 = *ii;
        if ( (unsigned __int64 *)*ii != ii + 2 )
        {
          a2 = ii[2] + 1;
          j_j___libc_free_0(a1);
        }
      }
      if ( v19 )
      {
        a1 = v19;
        a2 = v21 - v19;
        j_j___libc_free_0(v19);
      }
      if ( v80 >= v53 )
        break;
    }
  }
  if ( (a3 & 1) == 0 && (a3 - 2) / 2 == v80 )
  {
    v46 = (_QWORD *)(v74 + 32 * v80);
    v47 = (_QWORD *)(v74 + 32 * (2 * v80 + 1));
    v48 = v46[1];
    v49 = (unsigned __int64 *)v46[2];
    v50 = (unsigned __int64 *)v48;
    *v46 = *v47;
    v46[1] = v47[1];
    v46[2] = v47[2];
    v46[3] = v47[3];
    v47[1] = 0;
    v47[2] = 0;
    for ( v47[3] = 0; v49 != v50; v50 += 7 )
    {
      a1 = *v50;
      if ( (unsigned __int64 *)*v50 != v50 + 2 )
        j_j___libc_free_0(a1);
    }
    if ( v48 )
    {
      a1 = v48;
      j_j___libc_free_0(v48);
    }
    v80 = 2 * v80 + 1;
  }
  v23 = *((_QWORD *)a4 + 3);
  v24 = *a4;
  *((_QWORD *)a4 + 3) = 0;
  a2 = *((_QWORD *)a4 + 1);
  *((_QWORD *)a4 + 1) = 0;
  v54 = v23;
  v61 = v24;
  v4 = v80 - 1;
  v65 = a2;
  v57 = a4[1];
  v25 = (const __m128i *)*((_QWORD *)a4 + 2);
  *((_QWORD *)a4 + 2) = 0;
  v63 = v25;
  v73 = (v80 - 1) / 2;
  if ( v80 > v55 )
  {
    v68 = (unsigned __int64)v25 - a2;
    while ( 1 )
    {
      v26 = v74 + 32 * v73;
      if ( v68 )
      {
        if ( v68 > 0x7FFFFFFFFFFFFFF8LL )
          goto LABEL_105;
        a1 = v68;
        v78 = sub_22077B0(v68);
      }
      else
      {
        v78 = 0;
      }
      v27 = (const __m128i *)v65;
      v28 = (__m128i *)v78;
      if ( (const __m128i *)v65 != v63 )
      {
        do
        {
          if ( v28 )
          {
            a1 = (unsigned __int64)v28;
            v28->m128i_i64[0] = (__int64)v28[1].m128i_i64;
            a2 = v27->m128i_i64[0];
            sub_2F07250(v28->m128i_i64, v27->m128i_i64[0], v27->m128i_i64[0] + v27->m128i_i64[1]);
            v28[2] = _mm_loadu_si128(v27 + 2);
            v28[3].m128i_i16[0] = v27[3].m128i_i16[0];
          }
          v27 = (const __m128i *)((char *)v27 + 56);
          v28 = (__m128i *)((char *)v28 + 56);
        }
        while ( v63 != v27 );
      }
      v29 = *(_QWORD *)(v26 + 16);
      v30 = *(_QWORD *)(v26 + 8);
      v70 = *(_DWORD *)v26;
      v59 = *(_DWORD *)(v26 + 4);
      v66 = v29 - v30;
      if ( v29 == v30 )
        break;
      if ( (unsigned __int64)(v29 - v30) > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_105;
      a1 = v29 - v30;
      v31 = sub_22077B0(v29 - v30);
      v32 = *(const __m128i **)(v26 + 16);
      v76 = v31;
      if ( *(const __m128i **)(v26 + 8) == v32 )
        goto LABEL_86;
      v33 = (__m128i *)v31;
      v34 = *(const __m128i **)(v26 + 8);
      do
      {
        if ( v33 )
        {
          a1 = (unsigned __int64)v33;
          v33->m128i_i64[0] = (__int64)v33[1].m128i_i64;
          a2 = v34->m128i_i64[0];
          sub_2F07250(v33->m128i_i64, v34->m128i_i64[0], v34->m128i_i64[0] + v34->m128i_i64[1]);
          v33[2] = _mm_loadu_si128(v34 + 2);
          v33[3].m128i_i16[0] = v34[3].m128i_i16[0];
        }
        v34 = (const __m128i *)((char *)v34 + 56);
        v33 = (__m128i *)((char *)v33 + 56);
      }
      while ( v32 != v34 );
LABEL_62:
      v35 = v70 == v61;
      v71 = v70 < v61;
      if ( v35 )
        v71 = v57 > v59;
      for ( jj = (unsigned __int64 *)v76; v33 != (__m128i *)jj; jj += 7 )
      {
        a1 = *jj;
        if ( (unsigned __int64 *)*jj != jj + 2 )
        {
          a2 = jj[2] + 1;
          j_j___libc_free_0(a1);
        }
      }
      if ( v76 )
      {
        a2 = v66;
        a1 = v76;
        j_j___libc_free_0(v76);
      }
      for ( kk = (unsigned __int64 *)v78; kk != (unsigned __int64 *)v28; kk += 7 )
      {
        a1 = *kk;
        if ( (unsigned __int64 *)*kk != kk + 2 )
        {
          a2 = kk[2] + 1;
          j_j___libc_free_0(a1);
        }
      }
      if ( v78 )
      {
        a2 = v68;
        a1 = v78;
        j_j___libc_free_0(v78);
      }
      v38 = v74 + 32 * v80;
      v39 = *(_QWORD *)(v38 + 8);
      v40 = *(unsigned __int64 **)(v38 + 16);
      v41 = *(_QWORD *)(v38 + 24);
      if ( !v71 )
        goto LABEL_89;
      v42 = *(unsigned __int64 **)(v38 + 8);
      *(_QWORD *)v38 = *(_QWORD *)v26;
      *(_QWORD *)(v38 + 8) = *(_QWORD *)(v26 + 8);
      *(_QWORD *)(v38 + 16) = *(_QWORD *)(v26 + 16);
      *(_QWORD *)(v38 + 24) = *(_QWORD *)(v26 + 24);
      *(_QWORD *)(v26 + 8) = 0;
      *(_QWORD *)(v26 + 16) = 0;
      *(_QWORD *)(v26 + 24) = 0;
      if ( v40 != (unsigned __int64 *)v39 )
      {
        do
        {
          a1 = *v42;
          if ( (unsigned __int64 *)*v42 != v42 + 2 )
          {
            a2 = v42[2] + 1;
            j_j___libc_free_0(a1);
          }
          v42 += 7;
        }
        while ( v42 != v40 );
      }
      if ( v39 )
      {
        a1 = v39;
        a2 = v41 - v39;
        j_j___libc_free_0(v39);
      }
      v4 = v73 - 1;
      v80 = v73;
      if ( v55 >= v73 )
      {
        v38 = v74 + 32 * v73;
        goto LABEL_89;
      }
      v73 = (v73 - 1) / 2;
    }
    v76 = 0;
LABEL_86:
    v33 = (__m128i *)v76;
    goto LABEL_62;
  }
  v38 = v74 + 32 * v80;
LABEL_89:
  v43 = *(_QWORD *)(v38 + 8);
  v44 = *(unsigned __int64 **)(v38 + 16);
  *(_DWORD *)v38 = v61;
  v45 = (unsigned __int64 *)v43;
  *(_DWORD *)(v38 + 4) = v57;
  *(_QWORD *)(v38 + 8) = v65;
  *(_QWORD *)(v38 + 16) = v63;
  for ( *(_QWORD *)(v38 + 24) = v54; v44 != v45; v45 += 7 )
  {
    if ( (unsigned __int64 *)*v45 != v45 + 2 )
      j_j___libc_free_0(*v45);
  }
  if ( v43 )
    j_j___libc_free_0(v43);
}
