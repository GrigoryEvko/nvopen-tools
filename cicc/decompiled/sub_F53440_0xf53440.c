// Function: sub_F53440
// Address: 0xf53440
//
__int64 __fastcall sub_F53440(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rax
  _DWORD *v9; // rax
  __int64 v10; // r10
  unsigned int v11; // ebx
  __int64 *v12; // rax
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rax
  __int64 v16; // r10
  _BYTE *v17; // rbx
  _BYTE *v18; // r15
  const __m128i *v19; // rax
  __m128i si128; // xmm1
  __m128i v21; // xmm2
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rcx
  __int64 v27; // r8
  bool v28; // cc
  _QWORD *v29; // rdx
  __int64 v30; // rdx
  unsigned __int64 v31; // rcx
  __int64 v32; // r13
  _BYTE *v33; // rbx
  _BYTE *v34; // r12
  __int64 v35; // rdi
  __int64 v37; // rcx
  unsigned __int64 v38; // rsi
  __int64 v39; // rax
  unsigned __int64 v40; // rdx
  __int64 v41; // rcx
  __m128i *v42; // rbx
  __int64 v43; // rdi
  __m128i *v44; // r15
  __int64 *v45; // rax
  __int64 v46; // rdx
  __int64 i; // rsi
  __int64 *v48; // rsi
  size_t v49; // rdx
  __int64 v50; // r9
  __int64 v51; // rdx
  size_t v52; // r8
  const __m128i *v53; // r10
  __m128i *v54; // rdx
  __m128i *v55; // rax
  unsigned __int64 v56; // rsi
  char *v57; // rax
  char *v58; // r8
  unsigned int v59; // edx
  unsigned int v60; // eax
  __int64 v61; // r10
  const __m128i *v62; // [rsp+8h] [rbp-158h]
  size_t na; // [rsp+20h] [rbp-140h]
  size_t n; // [rsp+20h] [rbp-140h]
  size_t nb; // [rsp+20h] [rbp-140h]
  __int64 v67; // [rsp+28h] [rbp-138h]
  __int64 v68; // [rsp+28h] [rbp-138h]
  __int64 v69; // [rsp+28h] [rbp-138h]
  const __m128i *v70; // [rsp+28h] [rbp-138h]
  __int64 *v71; // [rsp+30h] [rbp-130h] BYREF
  unsigned int v72; // [rsp+38h] [rbp-128h]
  __int64 v73; // [rsp+40h] [rbp-120h] BYREF
  __int64 v74; // [rsp+48h] [rbp-118h]
  _QWORD v75[4]; // [rsp+50h] [rbp-110h] BYREF
  __int64 v76; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v77; // [rsp+78h] [rbp-E8h]
  __int64 v78; // [rsp+80h] [rbp-E0h] BYREF
  unsigned int v79; // [rsp+88h] [rbp-D8h]
  _BYTE *v80; // [rsp+C0h] [rbp-A0h] BYREF
  __int64 v81; // [rsp+C8h] [rbp-98h]
  _BYTE v82[144]; // [rsp+D0h] [rbp-90h] BYREF

  v8 = *(_QWORD *)(*(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)) + 8LL);
  if ( (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17 <= 1 )
    v8 = **(_QWORD **)(v8 + 16);
  v9 = sub_AE2980(a2, *(_DWORD *)(v8 + 8) >> 8);
  v10 = a3;
  v76 = 0;
  v11 = v9[3];
  v12 = &v78;
  v77 = 1;
  do
  {
    *v12 = -4096;
    v12 += 2;
  }
  while ( v12 != (__int64 *)&v80 );
  v72 = v11;
  v80 = v82;
  v81 = 0x400000000LL;
  if ( v11 > 0x40 )
  {
    sub_C43690((__int64)&v71, 0, 0);
    v10 = a3;
  }
  else
  {
    v71 = 0;
  }
  v67 = v10;
  if ( !(unsigned __int8)sub_B4DE70(a1, a2, v11, &v76, &v71) )
  {
    v32 = 0;
    goto LABEL_23;
  }
  v15 = (unsigned int)v81;
  v16 = v67;
  if ( (_DWORD)v81 && !v67 )
  {
    v37 = *(unsigned int *)(a4 + 8);
    v38 = *(unsigned int *)(a4 + 12);
    v73 = 4101;
    v74 = 0;
    LODWORD(v39) = v37;
    v40 = v37 + 2;
    v41 = 8 * v37;
    if ( !v41 )
    {
      if ( v40 > v38 )
      {
        sub_C8D5F0(a4, (const void *)(a4 + 16), v40, 8u, v13, v14);
        v41 = 8LL * *(unsigned int *)(a4 + 8);
      }
      v55 = *(__m128i **)a4;
      v55->m128i_i64[(unsigned __int64)v41 / 8] = 4101;
      v55->m128i_i64[(unsigned __int64)v41 / 8 + 1] = 0;
      *(_DWORD *)(a4 + 8) += 2;
      goto LABEL_52;
    }
    if ( v40 > v38 )
    {
      sub_C8D5F0(a4, (const void *)(a4 + 16), v40, 8u, v13, v14);
      v51 = *(unsigned int *)(a4 + 8);
      v42 = *(__m128i **)a4;
      v41 = 8 * v51;
      LODWORD(v39) = *(_DWORD *)(a4 + 8);
      v40 = v51 + 2;
      v44 = (__m128i *)(*(_QWORD *)a4 + v41);
      v43 = v41 >> 3;
      if ( (unsigned __int64)v41 <= 0xF )
      {
LABEL_43:
        *(_DWORD *)(a4 + 8) = v39 + 2;
        if ( v42 != v44 )
        {
          v45 = &v42->m128i_i64[(unsigned int)v40 + v41 / 0xFFFFFFFFFFFFFFF8LL];
          if ( (unsigned int)v41 >= 8 )
          {
            *v45 = v42->m128i_i64[0];
            *(__int64 *)((char *)v45 + (unsigned int)v41 - 8) = *(__int64 *)((char *)&v42->m128i_i64[-1]
                                                                           + (unsigned int)v41);
            v56 = (unsigned __int64)(v45 + 1) & 0xFFFFFFFFFFFFFFF8LL;
            v57 = (char *)v45 - v56;
            v58 = (char *)((char *)v42 - v57);
            if ( (((_DWORD)v41 + (_DWORD)v57) & 0xFFFFFFF8) >= 8 )
            {
              v59 = (v41 + (_DWORD)v57) & 0xFFFFFFF8;
              v60 = 0;
              do
              {
                v61 = v60;
                v60 += 8;
                *(_QWORD *)(v56 + v61) = *(_QWORD *)&v58[v61];
              }
              while ( v60 < v59 );
            }
          }
          else if ( (_DWORD)v41 )
          {
            *(_BYTE *)v45 = v42->m128i_i8[0];
          }
        }
        if ( v43 )
        {
          v46 = 0;
          for ( i = 4101; ; i = *(&v73 + v46) )
          {
            v42->m128i_i64[v46++] = i;
            if ( v43 == v46 )
              break;
          }
          v48 = (__int64 *)((char *)&v73 + v41);
          if ( (__int64 *)((char *)&v73 + v41) == v75 )
            goto LABEL_52;
          v49 = (char *)v75 - (char *)v48;
        }
        else
        {
          v49 = 16;
          v48 = &v73;
        }
        memcpy(v44, v48, v49);
LABEL_52:
        v15 = (unsigned int)v81;
        v16 = 1;
        goto LABEL_10;
      }
      v52 = v41 - 16;
      v53 = (__m128i *)((char *)v42 + v41 - 16);
      if ( *(unsigned int *)(a4 + 12) >= v40 )
      {
        v54 = (__m128i *)(*(_QWORD *)a4 + v41);
      }
      else
      {
        nb = v41 - 16;
        v70 = (__m128i *)((char *)v42 + v52);
        sub_C8D5F0(a4, (const void *)(a4 + 16), v40, 8u, v52, v50);
        v54 = v44;
        v53 = v70;
        v52 = nb;
        v39 = *(unsigned int *)(a4 + 8);
        v44 = (__m128i *)(*(_QWORD *)a4 + 8 * v39);
      }
    }
    else
    {
      v42 = *(__m128i **)a4;
      v43 = v41 >> 3;
      v44 = (__m128i *)(*(_QWORD *)a4 + v41);
      if ( (unsigned __int64)v41 <= 0xF )
        goto LABEL_43;
      v52 = v41 - 16;
      v54 = (__m128i *)(*(_QWORD *)a4 + v41);
      v53 = (__m128i *)((char *)v42 + v41 - 16);
    }
    if ( v54 != v53 )
    {
      *v44 = _mm_loadu_si128(v53);
      LODWORD(v39) = *(_DWORD *)(a4 + 8);
    }
    *(_DWORD *)(a4 + 8) = v39 + 2;
    if ( v42 != v53 )
      memmove(&v42[1], v42, v52);
    v42->m128i_i64[0] = 4101;
    v42->m128i_i64[1] = 0;
    goto LABEL_52;
  }
LABEL_10:
  v17 = v80;
  v18 = &v80[24 * v15];
  v19 = (const __m128i *)&v73;
  if ( v18 != v80 )
  {
    while ( 1 )
    {
      v23 = *(unsigned int *)(a5 + 8);
      v24 = *(_QWORD *)v17;
      v25 = v23 + 1;
      if ( v23 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
      {
        v62 = v19;
        n = v16;
        v69 = *(_QWORD *)v17;
        sub_C8D5F0(a5, (const void *)(a5 + 16), v23 + 1, 8u, v24, v25);
        v23 = *(unsigned int *)(a5 + 8);
        v19 = v62;
        v16 = n;
        v24 = v69;
      }
      v26 = *(_QWORD *)a5;
      v73 = 4101;
      v74 = v16;
      *(_QWORD *)(v26 + 8 * v23) = v24;
      v27 = v16 + 1;
      ++*(_DWORD *)(a5 + 8);
      v28 = *((_DWORD *)v17 + 4) <= 0x40u;
      v75[0] = 16;
      v29 = (_QWORD *)*((_QWORD *)v17 + 1);
      if ( !v28 )
        v29 = (_QWORD *)*v29;
      v75[1] = v29;
      v30 = *(unsigned int *)(a4 + 8);
      v75[2] = 30;
      v31 = *(unsigned int *)(a4 + 12);
      v75[3] = 34;
      if ( v30 + 6 > v31 )
      {
        na = (size_t)v19;
        v68 = v16 + 1;
        sub_C8D5F0(a4, (const void *)(a4 + 16), v30 + 6, 8u, v27, v25);
        v30 = *(unsigned int *)(a4 + 8);
        v19 = (const __m128i *)na;
        v27 = v68;
      }
      v17 += 24;
      si128 = _mm_load_si128(v19 + 1);
      v21 = _mm_load_si128(v19 + 2);
      v22 = *(_QWORD *)a4 + 8 * v30;
      *(__m128i *)v22 = _mm_load_si128(v19);
      *(__m128i *)(v22 + 16) = si128;
      *(__m128i *)(v22 + 32) = v21;
      *(_DWORD *)(a4 + 8) += 6;
      if ( v18 == v17 )
        break;
      v16 = v27;
    }
  }
  if ( v72 <= 0x40 )
  {
    a2 = 0;
    if ( v72 )
      a2 = (__int64)((_QWORD)v71 << (64 - (unsigned __int8)v72)) >> (64 - (unsigned __int8)v72);
  }
  else
  {
    a2 = *v71;
  }
  sub_AF6280(a4, a2);
  v32 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
LABEL_23:
  if ( v72 > 0x40 && v71 )
    j_j___libc_free_0_0(v71);
  v33 = v80;
  v34 = &v80[24 * (unsigned int)v81];
  if ( v80 != v34 )
  {
    do
    {
      v34 -= 24;
      if ( *((_DWORD *)v34 + 4) > 0x40u )
      {
        v35 = *((_QWORD *)v34 + 1);
        if ( v35 )
          j_j___libc_free_0_0(v35);
      }
    }
    while ( v33 != v34 );
    v34 = v80;
  }
  if ( v34 != v82 )
    _libc_free(v34, a2);
  if ( (v77 & 1) == 0 )
    sub_C7D6A0(v78, 16LL * v79, 8);
  return v32;
}
