// Function: sub_2F22A40
// Address: 0x2f22a40
//
void __fastcall sub_2F22A40(unsigned __int64 a1, __m128i *a2, __int64 a3)
{
  __int64 v3; // r12
  int *m128i_i32; // r13
  __int64 v5; // rdx
  __m128i *v6; // r12
  __int64 v7; // rdx
  int *v8; // rsi
  __int64 v9; // rdx
  __int64 m128i_i64; // rsi
  unsigned __int64 v11; // rdi
  __int64 v12; // rdx
  const __m128i *v13; // r13
  const __m128i *v14; // rbx
  __m128i *i; // r15
  const __m128i *v16; // r14
  const __m128i *v17; // rbx
  __int64 v18; // rax
  unsigned __int64 v19; // r13
  __m128i *j; // r12
  bool v21; // bl
  unsigned __int64 *k; // r14
  unsigned __int64 *m; // r13
  unsigned __int64 v24; // r14
  const __m128i *v25; // r15
  const __m128i *v26; // r13
  __int64 v27; // rax
  __m128i *n; // rbx
  const __m128i *v29; // r15
  const __m128i *v30; // rax
  unsigned __int64 v31; // rcx
  __int64 v32; // rax
  unsigned __int64 v33; // r13
  __m128i *v34; // r12
  bool v35; // zf
  unsigned __int64 *ii; // r15
  unsigned __int64 *jj; // r13
  __int64 v38; // rcx
  __int64 v39; // rax
  __int64 v40; // rbx
  unsigned __int64 v41; // r15
  unsigned __int64 *v42; // rdx
  int v43; // edi
  unsigned __int64 *v44; // rcx
  int v45; // esi
  __int64 v46; // rax
  unsigned __int64 *v47; // r14
  unsigned __int64 *v48; // r12
  __int64 v49; // rdx
  __m128i *v50; // rbx
  unsigned __int64 *v51; // rcx
  unsigned __int64 *v52; // rdx
  __int64 v53; // rax
  int v54; // edi
  __int64 v55; // r9
  __int32 v56; // esi
  __int64 v57; // r9
  __int64 v58; // r9
  unsigned __int64 *v59; // r14
  unsigned __int64 *v60; // r12
  __int64 v61; // [rsp+8h] [rbp-B8h]
  __m128i *v62; // [rsp+10h] [rbp-B0h]
  __m128i *v63; // [rsp+18h] [rbp-A8h]
  __m128i *v64; // [rsp+20h] [rbp-A0h]
  unsigned int v65; // [rsp+2Ch] [rbp-94h]
  unsigned __int64 v66; // [rsp+30h] [rbp-90h]
  unsigned int v68; // [rsp+40h] [rbp-80h]
  __int64 v69; // [rsp+40h] [rbp-80h]
  __int64 v70; // [rsp+48h] [rbp-78h]
  __int64 v71; // [rsp+48h] [rbp-78h]
  unsigned __int32 v72; // [rsp+50h] [rbp-70h]
  const __m128i *v73; // [rsp+50h] [rbp-70h]
  unsigned int v74; // [rsp+58h] [rbp-68h]
  unsigned int v75; // [rsp+58h] [rbp-68h]
  unsigned __int32 v76; // [rsp+5Ch] [rbp-64h]
  unsigned int v77; // [rsp+5Ch] [rbp-64h]
  unsigned __int64 v78; // [rsp+60h] [rbp-60h]
  unsigned int v79; // [rsp+60h] [rbp-60h]
  bool v80; // [rsp+60h] [rbp-60h]
  unsigned __int64 v81; // [rsp+68h] [rbp-58h]
  unsigned __int64 v82; // [rsp+68h] [rbp-58h]
  __int64 v83; // [rsp+68h] [rbp-58h]
  __int64 v84; // [rsp+68h] [rbp-58h]
  int v85; // [rsp+70h] [rbp-50h] BYREF
  int v86; // [rsp+74h] [rbp-4Ch]
  unsigned __int64 *v87; // [rsp+78h] [rbp-48h]
  unsigned __int64 *v88; // [rsp+80h] [rbp-40h]
  __int64 v89; // [rsp+88h] [rbp-38h]

  v3 = (__int64)a2->m128i_i64 - a1;
  v61 = a3;
  v63 = a2;
  if ( (__int64)((__int64)a2->m128i_i64 - a1) <= 512 )
    return;
  if ( !a3 )
  {
LABEL_77:
    v83 = v3 >> 5;
    v40 = ((v3 >> 5) - 2) >> 1;
    v41 = a1 + 32 * v40;
    while ( 1 )
    {
      v42 = *(unsigned __int64 **)(v41 + 16);
      v43 = *(_DWORD *)v41;
      *(_QWORD *)(v41 + 16) = 0;
      v44 = *(unsigned __int64 **)(v41 + 8);
      v45 = *(_DWORD *)(v41 + 4);
      *(_QWORD *)(v41 + 8) = 0;
      v46 = *(_QWORD *)(v41 + 24);
      *(_QWORD *)(v41 + 24) = 0;
      v85 = v43;
      v88 = v42;
      v86 = v45;
      v87 = v44;
      v89 = v46;
      sub_2F0C170(a1, v40, v83, &v85);
      v47 = v88;
      v48 = v87;
      if ( v88 != v87 )
      {
        do
        {
          if ( (unsigned __int64 *)*v48 != v48 + 2 )
            j_j___libc_free_0(*v48);
          v48 += 7;
        }
        while ( v47 != v48 );
        v48 = v87;
      }
      if ( v48 )
        j_j___libc_free_0((unsigned __int64)v48);
      v41 -= 32LL;
      if ( !v40 )
        break;
      --v40;
    }
    v50 = v63 - 2;
    do
    {
      v51 = (unsigned __int64 *)v50->m128i_i64[1];
      v52 = (unsigned __int64 *)v50[1].m128i_i64[0];
      v50->m128i_i64[1] = 0;
      v53 = v50[1].m128i_i64[1];
      v54 = v50->m128i_i32[0];
      v50[1].m128i_i64[1] = 0;
      v50[1].m128i_i64[0] = 0;
      v55 = *(_QWORD *)a1;
      v56 = v50->m128i_i32[1];
      v89 = v53;
      v50->m128i_i64[0] = v55;
      v84 = (__int64)v50->m128i_i64 - a1;
      v50->m128i_i64[1] = *(_QWORD *)(a1 + 8);
      v57 = *(_QWORD *)(a1 + 16);
      v85 = v54;
      v50[1].m128i_i64[0] = v57;
      v58 = *(_QWORD *)(a1 + 24);
      v86 = v56;
      v50[1].m128i_i64[1] = v58;
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      v87 = v51;
      v88 = v52;
      sub_2F0C170(a1, 0, (__int64)((__int64)v50->m128i_i64 - a1) >> 5, &v85);
      v59 = v88;
      v60 = v87;
      if ( v88 != v87 )
      {
        do
        {
          if ( (unsigned __int64 *)*v60 != v60 + 2 )
            j_j___libc_free_0(*v60);
          v60 += 7;
        }
        while ( v59 != v60 );
        v60 = v87;
      }
      if ( v60 )
        j_j___libc_free_0((unsigned __int64)v60);
      v50 -= 2;
    }
    while ( v84 > 32 );
    return;
  }
  v62 = (__m128i *)(a1 + 32);
  while ( 2 )
  {
    --v61;
    m128i_i32 = v63[-2].m128i_i32;
    v5 = (__int64)((__int64)v63->m128i_i64 - a1) >> 5;
    v6 = (__m128i *)(a1 + 32 * ((__int64)(v5 + (((unsigned __int64)v63 - a1) >> 63)) >> 1));
    v8 = v63[-2].m128i_i32;
    if ( !(unsigned __int8)sub_2F0A430(v62, v6->m128i_i32, v5) )
    {
      if ( (unsigned __int8)sub_2F0A430(v62, v8, v7) )
      {
        v11 = a1;
        m128i_i64 = (__int64)v62;
        sub_2F229E0((__int64 *)a1, v62->m128i_i64);
        goto LABEL_7;
      }
      if ( !(unsigned __int8)sub_2F0A430(v6, m128i_i32, v49) )
      {
        v11 = a1;
        m128i_i64 = (__int64)v6;
        sub_2F229E0((__int64 *)a1, v6->m128i_i64);
        goto LABEL_7;
      }
LABEL_100:
      v11 = a1;
      m128i_i64 = (__int64)v63[-2].m128i_i64;
      sub_2F229E0((__int64 *)a1, (__int64 *)m128i_i32);
      goto LABEL_7;
    }
    if ( !(unsigned __int8)sub_2F0A430(v6, v8, v7) )
    {
      if ( !(unsigned __int8)sub_2F0A430(v62, m128i_i32, v9) )
      {
        m128i_i64 = (__int64)v62;
        v11 = a1;
        sub_2F229E0((__int64 *)a1, v62->m128i_i64);
        goto LABEL_7;
      }
      goto LABEL_100;
    }
    m128i_i64 = (__int64)v6;
    v11 = a1;
    sub_2F229E0((__int64 *)a1, v6->m128i_i64);
LABEL_7:
    v64 = v62;
    v66 = (unsigned __int64)v63;
    while ( 1 )
    {
      v13 = *(const __m128i **)(a1 + 16);
      v14 = *(const __m128i **)(a1 + 8);
      v72 = *(_DWORD *)a1;
      v74 = *(_DWORD *)(a1 + 4);
      v78 = (char *)v13 - (char *)v14;
      if ( v13 == v14 )
      {
        v81 = 0;
      }
      else
      {
        if ( (unsigned __int64)((char *)v13 - (char *)v14) > 0x7FFFFFFFFFFFFFF8LL )
          goto LABEL_104;
        v11 = (char *)v13 - (char *)v14;
        v81 = sub_22077B0(v78);
        v13 = *(const __m128i **)(a1 + 16);
        v14 = *(const __m128i **)(a1 + 8);
      }
      for ( i = (__m128i *)v81; v13 != v14; i = (__m128i *)((char *)i + 56) )
      {
        if ( i )
        {
          v11 = (unsigned __int64)i;
          i->m128i_i64[0] = (__int64)i[1].m128i_i64;
          m128i_i64 = v14->m128i_i64[0];
          sub_2F07250(i->m128i_i64, v14->m128i_i64[0], v14->m128i_i64[0] + v14->m128i_i64[1]);
          i[2] = _mm_loadu_si128(v14 + 2);
          i[3].m128i_i16[0] = v14[3].m128i_i16[0];
        }
        v14 = (const __m128i *)((char *)v14 + 56);
      }
      v16 = (const __m128i *)v64[1].m128i_i64[0];
      v17 = (const __m128i *)v64->m128i_i64[1];
      v76 = v64->m128i_i32[0];
      v68 = v64->m128i_u32[1];
      v70 = (char *)v16 - (char *)v17;
      if ( v16 == v17 )
      {
        v19 = 0;
      }
      else
      {
        if ( (unsigned __int64)((char *)v16 - (char *)v17) > 0x7FFFFFFFFFFFFFF8LL )
LABEL_104:
          sub_4261EA(v11, m128i_i64, v12);
        v11 = (char *)v16 - (char *)v17;
        v18 = sub_22077B0((char *)v16 - (char *)v17);
        v16 = (const __m128i *)v64[1].m128i_i64[0];
        v17 = (const __m128i *)v64->m128i_i64[1];
        v19 = v18;
      }
      for ( j = (__m128i *)v19; v16 != v17; j = (__m128i *)((char *)j + 56) )
      {
        if ( j )
        {
          v11 = (unsigned __int64)j;
          j->m128i_i64[0] = (__int64)j[1].m128i_i64;
          m128i_i64 = v17->m128i_i64[0];
          sub_2F07250(j->m128i_i64, v17->m128i_i64[0], v17->m128i_i64[0] + v17->m128i_i64[1]);
          j[2] = _mm_loadu_si128(v17 + 2);
          j[3].m128i_i16[0] = v17[3].m128i_i16[0];
        }
        v17 = (const __m128i *)((char *)v17 + 56);
      }
      v21 = v76 < v72;
      if ( v76 == v72 )
        v21 = v68 < v74;
      for ( k = (unsigned __int64 *)v19; k != (unsigned __int64 *)j; k += 7 )
      {
        v11 = *k;
        if ( (unsigned __int64 *)*k != k + 2 )
        {
          m128i_i64 = k[2] + 1;
          j_j___libc_free_0(v11);
        }
      }
      if ( v19 )
      {
        m128i_i64 = v70;
        v11 = v19;
        j_j___libc_free_0(v19);
      }
      for ( m = (unsigned __int64 *)v81; i != (__m128i *)m; m += 7 )
      {
        v11 = *m;
        if ( (unsigned __int64 *)*m != m + 2 )
        {
          m128i_i64 = m[2] + 1;
          j_j___libc_free_0(v11);
        }
      }
      if ( v81 )
      {
        m128i_i64 = v78;
        v11 = v81;
        j_j___libc_free_0(v81);
      }
      if ( !v21 )
        break;
LABEL_70:
      v64 += 2;
    }
    v24 = v66 - 32;
    do
    {
      v25 = *(const __m128i **)(v24 + 16);
      v66 = v24;
      v26 = *(const __m128i **)(v24 + 8);
      v77 = *(_DWORD *)v24;
      v75 = *(_DWORD *)(v24 + 4);
      v69 = (char *)v25 - (char *)v26;
      if ( v25 == v26 )
      {
        v82 = 0;
      }
      else
      {
        if ( (unsigned __int64)((char *)v25 - (char *)v26) > 0x7FFFFFFFFFFFFFF8LL )
          goto LABEL_104;
        v11 = (char *)v25 - (char *)v26;
        v27 = sub_22077B0((char *)v25 - (char *)v26);
        v25 = *(const __m128i **)(v24 + 16);
        v26 = *(const __m128i **)(v24 + 8);
        v82 = v27;
      }
      for ( n = (__m128i *)v82; v25 != v26; n = (__m128i *)((char *)n + 56) )
      {
        if ( n )
        {
          v11 = (unsigned __int64)n;
          n->m128i_i64[0] = (__int64)n[1].m128i_i64;
          m128i_i64 = v26->m128i_i64[0];
          sub_2F07250(n->m128i_i64, v26->m128i_i64[0], v26->m128i_i64[0] + v26->m128i_i64[1]);
          n[2] = _mm_loadu_si128(v26 + 2);
          n[3].m128i_i16[0] = v26[3].m128i_i16[0];
        }
        v26 = (const __m128i *)((char *)v26 + 56);
      }
      v29 = *(const __m128i **)(a1 + 8);
      v79 = *(_DWORD *)a1;
      v65 = *(_DWORD *)(a1 + 4);
      v30 = *(const __m128i **)(a1 + 16);
      v31 = (char *)v30 - (char *)v29;
      v71 = (char *)v30 - (char *)v29;
      if ( v30 == v29 )
      {
        v33 = 0;
      }
      else
      {
        if ( v31 > 0x7FFFFFFFFFFFFFF8LL )
          goto LABEL_104;
        v11 = *(_QWORD *)(a1 + 16) - (_QWORD)v29;
        v32 = sub_22077B0(v31);
        v29 = *(const __m128i **)(a1 + 8);
        v33 = v32;
        v30 = *(const __m128i **)(a1 + 16);
      }
      v34 = (__m128i *)v33;
      if ( v29 != v30 )
      {
        v73 = v30;
        do
        {
          if ( v34 )
          {
            v11 = (unsigned __int64)v34;
            v34->m128i_i64[0] = (__int64)v34[1].m128i_i64;
            m128i_i64 = v29->m128i_i64[0];
            sub_2F07250(v34->m128i_i64, v29->m128i_i64[0], v29->m128i_i64[0] + v29->m128i_i64[1]);
            v34[2] = _mm_loadu_si128(v29 + 2);
            v12 = v29[3].m128i_u16[0];
            v34[3].m128i_i16[0] = v12;
          }
          v34 = (__m128i *)((char *)v34 + 56);
          v29 = (const __m128i *)((char *)v29 + 56);
        }
        while ( v73 != v29 );
      }
      v35 = v79 == v77;
      v80 = v79 < v77;
      if ( v35 )
        v80 = v75 > v65;
      for ( ii = (unsigned __int64 *)v33; ii != (unsigned __int64 *)v34; ii += 7 )
      {
        v11 = *ii;
        if ( (unsigned __int64 *)*ii != ii + 2 )
        {
          m128i_i64 = ii[2] + 1;
          j_j___libc_free_0(v11);
        }
      }
      if ( v33 )
      {
        m128i_i64 = v71;
        v11 = v33;
        j_j___libc_free_0(v33);
      }
      for ( jj = (unsigned __int64 *)v82; n != (__m128i *)jj; jj += 7 )
      {
        v11 = *jj;
        if ( (unsigned __int64 *)*jj != jj + 2 )
        {
          m128i_i64 = jj[2] + 1;
          j_j___libc_free_0(v11);
        }
      }
      if ( v82 )
      {
        m128i_i64 = v69;
        v11 = v82;
        j_j___libc_free_0(v82);
      }
      v24 -= 32LL;
    }
    while ( v80 );
    if ( (unsigned __int64)v64 < v66 )
    {
      v38 = v64->m128i_i64[1];
      v12 = v64[1].m128i_i64[0];
      v64->m128i_i64[1] = 0;
      v39 = v64[1].m128i_i64[1];
      v11 = v64->m128i_u32[0];
      v64[1].m128i_i64[1] = 0;
      v64[1].m128i_i64[0] = 0;
      m128i_i64 = v64->m128i_u32[1];
      v64->m128i_i64[0] = *(_QWORD *)v66;
      v64->m128i_i64[1] = *(_QWORD *)(v66 + 8);
      v64[1].m128i_i64[0] = *(_QWORD *)(v66 + 16);
      v64[1].m128i_i64[1] = *(_QWORD *)(v66 + 24);
      *(_DWORD *)v66 = v11;
      *(_DWORD *)(v66 + 4) = m128i_i64;
      *(_QWORD *)(v66 + 8) = v38;
      *(_QWORD *)(v66 + 16) = v12;
      *(_QWORD *)(v66 + 24) = v39;
      goto LABEL_70;
    }
    sub_2F22A40(v64, v63, v61);
    v3 = (__int64)v64->m128i_i64 - a1;
    if ( (__int64)((__int64)v64->m128i_i64 - a1) > 512 )
    {
      v63 = v64;
      if ( !v61 )
        goto LABEL_77;
      continue;
    }
    break;
  }
}
