// Function: sub_26E2170
// Address: 0x26e2170
//
void __fastcall sub_26E2170(
        const __m128i **a1,
        const __m128i **a2,
        unsigned __int8 (__fastcall *a3)(unsigned __int64, __int64, unsigned __int64 *),
        unsigned __int64 a4,
        void (__fastcall *a5)(__int64, __int64, __int64),
        __int64 a6)
{
  __int64 v7; // rax
  int v8; // r12d
  unsigned __int64 v9; // r12
  char *v10; // rax
  char *v11; // rbx
  _DWORD *v12; // rcx
  unsigned __int64 *v13; // r12
  char *v14; // rbx
  char *v15; // r13
  unsigned __int64 v16; // r13
  void *v17; // rcx
  char *v18; // rax
  signed __int64 v19; // r13
  __int64 v20; // rsi
  unsigned __int64 v21; // rdi
  int v22; // ecx
  __int64 v23; // rax
  unsigned __int64 j; // rdx
  int i; // r15d
  int v26; // eax
  __int64 v27; // r13
  __int64 v28; // rbx
  int v29; // r12d
  const __m128i *v30; // rcx
  const __m128i *v31; // rax
  __int64 v32; // rax
  const __m128i *v33; // rcx
  const __m128i *v34; // rax
  unsigned __int64 v35; // r15
  __m128i *k; // rdx
  int v37; // r12d
  int v38; // eax
  unsigned __int64 v39; // rsi
  int v40; // edx
  int v41; // edi
  int v42; // eax
  __int64 *v43; // r14
  int v44; // r13d
  __int64 *v45; // rbx
  __int64 v46; // rdx
  __int64 v47; // rsi
  unsigned __int64 *v48; // rbx
  unsigned __int64 *v49; // r12
  unsigned __int64 *v50; // r13
  unsigned __int64 v52; // [rsp+10h] [rbp-D0h]
  int v53; // [rsp+18h] [rbp-C8h]
  int v54; // [rsp+1Ch] [rbp-C4h]
  unsigned __int64 v55; // [rsp+20h] [rbp-C0h]
  int v57; // [rsp+30h] [rbp-B0h]
  int v58; // [rsp+38h] [rbp-A8h]
  unsigned __int64 v59; // [rsp+38h] [rbp-A8h]
  int v60; // [rsp+44h] [rbp-9Ch]
  int v61; // [rsp+48h] [rbp-98h]
  unsigned __int64 v63; // [rsp+50h] [rbp-90h]
  unsigned __int64 *v65; // [rsp+58h] [rbp-88h]
  int v67; // [rsp+60h] [rbp-80h]
  int v68; // [rsp+68h] [rbp-78h]
  int v69; // [rsp+6Ch] [rbp-74h]
  int v70; // [rsp+6Ch] [rbp-74h]
  void *src; // [rsp+70h] [rbp-70h] BYREF
  char *v72; // [rsp+78h] [rbp-68h]
  char *v73; // [rsp+80h] [rbp-60h]
  unsigned __int64 *v74; // [rsp+90h] [rbp-50h] BYREF
  unsigned __int64 *v75; // [rsp+98h] [rbp-48h]
  unsigned __int64 *v76; // [rsp+A0h] [rbp-40h]

  v7 = (char *)a1[1] - (char *)*a1;
  v52 = 0xAAAAAAAAAAAAAAABLL * (v7 >> 3);
  v69 = -1431655765 * (v7 >> 3);
  v55 = 0xAAAAAAAAAAAAAAABLL * (((char *)a2[1] - (char *)*a2) >> 3);
  v60 = -1431655765 * (((char *)a2[1] - (char *)*a2) >> 3);
  v54 = v52 - 1431655765 * (((char *)a2[1] - (char *)*a2) >> 3);
  if ( !v54 )
    return;
  v8 = 2 * (v52 - 1431655765 * (((char *)a2[1] - (char *)*a2) >> 3)) + 1;
  if ( (unsigned __int64)v8 > 0x1FFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v9 = 4LL * v8;
  v72 = 0;
  v10 = (char *)sub_22077B0(v9);
  v11 = &v10[v9];
  src = v10;
  v12 = v10;
  v73 = &v10[v9];
  if ( &v10[v9] != v10 )
    v12 = memset(v10, 255, v9);
  v72 = v11;
  v53 = v54 + 1;
  v12[v54 + 1] = 0;
  v74 = 0;
  v75 = 0;
  v76 = 0;
  if ( v54 < 0 )
    goto LABEL_74;
  v57 = 0;
  v13 = 0;
LABEL_80:
  v20 = (__int64)v13;
  sub_248ACA0((unsigned __int64)&v74, v13, (__int64)&src);
  v14 = (char *)src;
LABEL_16:
  v21 = (unsigned int)(v53 - v57);
  v58 = v53 - v57;
  v61 = -v57;
LABEL_30:
  v23 = v58;
LABEL_19:
  j = (unsigned __int64)src;
  for ( i = *((_DWORD *)src + v23); ; i = v22 + 1 )
  {
    v26 = i - v61;
    if ( v69 <= i )
      goto LABEL_32;
    if ( v60 <= v26 )
    {
LABEL_27:
      *(_DWORD *)&v14[4 * v58 - 4] = i;
      goto LABEL_28;
    }
    v27 = 3LL * v26;
    v28 = 3 * (i - (__int64)v26);
    v29 = i;
    while ( 1 )
    {
      v21 = a4;
      v20 = (__int64)&(*a1)->m128i_i64[v27 + 1 + v28];
      if ( !a3(a4, v20, &(*a2)->m128i_u64[v27 + 1]) )
      {
LABEL_26:
        v14 = (char *)src;
        i = v29;
        goto LABEL_27;
      }
      v26 = ++v29 - v61;
      if ( v69 == v29 )
        break;
      v27 += 3;
      if ( v29 == (_DWORD)v55 + v61 )
        goto LABEL_26;
    }
    v14 = (char *)src;
    i = v29;
LABEL_32:
    j = v58 - 1;
    *(_DWORD *)&v14[4 * j] = i;
    if ( v60 <= v26 )
      break;
LABEL_28:
    v61 += 2;
    v58 += 2;
    if ( v61 > v57 )
    {
      ++v57;
      v13 = v75;
      if ( v54 < v57 )
      {
        v50 = v74;
        if ( v74 != v75 )
        {
          do
          {
            if ( *v50 )
              j_j___libc_free_0(*v50);
            v50 += 3;
          }
          while ( v50 != v13 );
          v50 = v74;
        }
        if ( v50 )
          j_j___libc_free_0((unsigned __int64)v50);
        goto LABEL_74;
      }
      if ( v75 == v76 )
        goto LABEL_80;
      v14 = (char *)src;
      if ( v75 )
      {
        v75[2] = 0;
        v15 = v72;
        *v13 = 0;
        v13[1] = 0;
        v16 = v15 - v14;
        if ( v16 )
        {
          if ( v16 > 0x7FFFFFFFFFFFFFFCLL )
            goto LABEL_91;
          v17 = (void *)sub_22077B0(v16);
        }
        else
        {
          v17 = 0;
        }
        *v13 = (unsigned __int64)v17;
        v14 = (char *)src;
        v13[1] = (unsigned __int64)v17;
        v18 = v72;
        v13[2] = (unsigned __int64)v17 + v16;
        v19 = v18 - v14;
        if ( v18 != v14 )
        {
          v20 = (__int64)v14;
          v17 = memmove(v17, v14, v18 - v14);
        }
        v13[1] = (unsigned __int64)v17 + v19;
        v13 = v75;
      }
      v75 = v13 + 3;
      goto LABEL_16;
    }
    v14 = (char *)src;
    if ( -v57 == v61 )
      goto LABEL_30;
    v21 = (unsigned int)v58;
    v20 = (unsigned int)v61;
    v22 = *((_DWORD *)src + v58 - 2);
    if ( v57 != v61 )
    {
      v23 = v58;
      if ( *((_DWORD *)src + v58) > v22 )
        goto LABEL_19;
    }
  }
  v30 = a2[1];
  v31 = *a2;
  v21 = (char *)v30 - (char *)*a2;
  if ( v21 )
  {
    if ( v21 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v32 = sub_22077B0(v21);
      v30 = a2[1];
      v63 = v32;
      v31 = *a2;
      goto LABEL_36;
    }
LABEL_91:
    sub_4261EA(v21, v20, j);
  }
  v63 = 0;
LABEL_36:
  for ( j = v63; v31 != v30; j += 24LL )
  {
    if ( j )
    {
      *(__m128i *)j = _mm_loadu_si128(v31);
      v20 = v31[1].m128i_i64[0];
      *(_QWORD *)(j + 16) = v20;
    }
    v31 = (const __m128i *)((char *)v31 + 24);
  }
  v33 = a1[1];
  v34 = *a1;
  v59 = (char *)v33 - (char *)*a1;
  if ( v59 )
  {
    if ( v59 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_91;
    v35 = sub_22077B0(v59);
    v33 = a1[1];
    v34 = *a1;
  }
  else
  {
    v35 = 0;
  }
  for ( k = (__m128i *)v35; v34 != v33; k = (__m128i *)((char *)k + 24) )
  {
    if ( k )
    {
      *k = _mm_loadu_si128(v34);
      k[1].m128i_i64[0] = v34[1].m128i_i64[0];
    }
    v34 = (const __m128i *)((char *)v34 + 24);
  }
  v67 = -1431655765 * (v75 - v74) - 1;
  if ( (int)v52 > 0 || (int)v55 > 0 )
  {
    v37 = v69;
    v65 = &v74[3 * v67];
    while ( 1 )
    {
      v38 = v37 - v60;
      v39 = *v65;
      if ( v37 - v60 + v67 )
      {
        v40 = v38 - 1;
        v41 = *(_DWORD *)(v39 + 4LL * (v54 + v38 - 1));
        v70 = v41;
        if ( v38 != v67 )
        {
          v42 = v38 + 1;
          if ( *(_DWORD *)(v39 + 4LL * (v54 + v42)) > v41 )
          {
            v70 = *(_DWORD *)(v39 + 4LL * (v54 + v42));
            v40 = v42;
          }
        }
      }
      else
      {
        v40 = v38 + 1;
        v70 = *(_DWORD *)(v39 + 4LL * (v38 + 1 + v54));
      }
      v68 = v70 - v40;
      if ( v37 > v70 && v60 > v70 - v40 )
      {
        v43 = (__int64 *)(v35 + 24LL * v37 - 24);
        v44 = v37;
        v45 = (__int64 *)(v63 + 24LL * v60 - 24);
        do
        {
          v46 = *v45;
          v47 = *v43;
          --v44;
          v43 -= 3;
          v45 -= 3;
          a5(a6, v47, v46);
        }
        while ( v44 > v70 && v68 < v44 + v60 - v37 );
      }
      if ( !v67 )
        break;
      --v67;
      v65 -= 3;
      if ( v70 <= 0 && v68 <= 0 )
        break;
      v37 = v70;
      v60 = v68;
    }
  }
  if ( v35 )
    j_j___libc_free_0(v35);
  if ( v63 )
    j_j___libc_free_0(v63);
  v48 = v75;
  v49 = v74;
  if ( v75 != v74 )
  {
    do
    {
      if ( *v49 )
        j_j___libc_free_0(*v49);
      v49 += 3;
    }
    while ( v48 != v49 );
    v49 = v74;
  }
  if ( v49 )
    j_j___libc_free_0((unsigned __int64)v49);
LABEL_74:
  if ( src )
    j_j___libc_free_0((unsigned __int64)src);
}
