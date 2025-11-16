// Function: sub_240E820
// Address: 0x240e820
//
void __fastcall sub_240E820(__int64 a1, __int64 *a2)
{
  __int64 v3; // r14
  __int64 v4; // r13
  unsigned __int64 v5; // rdx
  __int64 v6; // rax
  char *v7; // r12
  char *v8; // rdx
  __int64 v9; // rax
  __int64 v10; // r15
  __int64 v11; // rsi
  __int64 v12; // r14
  __int64 v13; // r13
  __int64 v14; // rax
  unsigned __int64 v15; // r12
  unsigned __int64 v16; // r12
  __int64 *v17; // rdi
  __int64 v18; // r12
  const void *v19; // r13
  size_t v20; // r14
  int v21; // eax
  unsigned int v22; // r15d
  _QWORD *v23; // r9
  __int64 v24; // rax
  _QWORD *v25; // r9
  _QWORD *v26; // rcx
  unsigned __int64 *v27; // rbx
  unsigned __int64 *v28; // r12
  unsigned __int64 v29; // rax
  unsigned __int64 v30; // r14
  bool v31; // cf
  unsigned __int64 v32; // rax
  const __m128i *v33; // rax
  signed __int64 v34; // r13
  __m128i *v35; // rdx
  __m128i *v36; // r13
  const __m128i *v37; // rdi
  __int64 v38; // r14
  char *v39; // r15
  const __m128i *v40; // rax
  __m128i *v41; // r12
  __int64 v42; // rcx
  const __m128i *v43; // rcx
  unsigned __int64 *v44; // r13
  unsigned __int64 v45; // r14
  __int64 v46; // rax
  _QWORD *v47; // [rsp+0h] [rbp-80h]
  __int64 i; // [rsp+8h] [rbp-78h]
  __int64 v49; // [rsp+8h] [rbp-78h]
  _QWORD *v50; // [rsp+10h] [rbp-70h]
  unsigned __int64 v51; // [rsp+10h] [rbp-70h]
  unsigned __int64 v52; // [rsp+18h] [rbp-68h]
  __int64 *v53; // [rsp+18h] [rbp-68h]
  char *v54; // [rsp+18h] [rbp-68h]
  unsigned __int64 v55; // [rsp+20h] [rbp-60h] BYREF
  __int64 *v56; // [rsp+28h] [rbp-58h] BYREF
  char *v57; // [rsp+30h] [rbp-50h] BYREF
  char *v58; // [rsp+38h] [rbp-48h]
  char *v59; // [rsp+40h] [rbp-40h]

  *(_QWORD *)(a1 + 280) = 0;
  *(_QWORD *)(a1 + 288) = 0;
  *(_QWORD *)(a1 + 296) = 0;
  *(_QWORD *)(a1 + 304) = 0;
  *(_QWORD *)(a1 + 312) = 0;
  *(_QWORD *)(a1 + 320) = 0;
  *(_QWORD *)(a1 + 328) = 0;
  *(_QWORD *)(a1 + 336) = 0;
  *(_QWORD *)(a1 + 344) = 0;
  *(_QWORD *)(a1 + 352) = 0;
  *(_QWORD *)(a1 + 360) = 0;
  *(_QWORD *)(a1 + 368) = 0;
  *(_QWORD *)(a1 + 376) = 0;
  *(_QWORD *)(a1 + 384) = 0;
  *(_QWORD *)(a1 + 392) = 0;
  *(_QWORD *)(a1 + 400) = 0;
  *(_QWORD *)(a1 + 408) = 0;
  *(_QWORD *)(a1 + 416) = 0;
  *(_QWORD *)(a1 + 424) = 0;
  *(_QWORD *)(a1 + 432) = 0;
  *(_QWORD *)(a1 + 440) = 0;
  *(_QWORD *)(a1 + 448) = 0;
  *(_QWORD *)(a1 + 456) = 0;
  *(_QWORD *)(a1 + 464) = 0;
  *(_QWORD *)(a1 + 472) = 0;
  *(_QWORD *)(a1 + 480) = 0;
  *(_QWORD *)(a1 + 624) = a1 + 648;
  *(_QWORD *)(a1 + 872) = a1 + 856;
  *(_QWORD *)(a1 + 880) = a1 + 856;
  *(_QWORD *)(a1 + 488) = 0;
  *(_QWORD *)(a1 + 496) = 0;
  *(_QWORD *)(a1 + 504) = 0;
  *(_QWORD *)(a1 + 512) = 0;
  *(_QWORD *)(a1 + 520) = 0;
  *(_QWORD *)(a1 + 528) = 0;
  *(_QWORD *)(a1 + 536) = 0;
  *(_QWORD *)(a1 + 544) = 0;
  *(_QWORD *)(a1 + 552) = 0;
  *(_QWORD *)(a1 + 560) = 0;
  *(_QWORD *)(a1 + 568) = 0;
  *(_QWORD *)(a1 + 576) = 0;
  *(_QWORD *)(a1 + 584) = 0;
  *(_QWORD *)(a1 + 592) = 0;
  *(_QWORD *)(a1 + 600) = 0;
  *(_QWORD *)(a1 + 608) = 0;
  *(_QWORD *)(a1 + 616) = 0;
  *(_QWORD *)(a1 + 632) = 16;
  *(_DWORD *)(a1 + 640) = 0;
  *(_BYTE *)(a1 + 644) = 1;
  *(_QWORD *)(a1 + 792) = 0;
  *(_QWORD *)(a1 + 800) = 0;
  *(_QWORD *)(a1 + 808) = 0;
  *(_QWORD *)(a1 + 816) = 0;
  *(_DWORD *)(a1 + 824) = 0;
  *(_QWORD *)(a1 + 832) = 0;
  *(_QWORD *)(a1 + 840) = 0;
  *(_DWORD *)(a1 + 856) = 0;
  *(_QWORD *)(a1 + 864) = 0;
  *(_QWORD *)(a1 + 888) = 0;
  *(_QWORD *)(a1 + 896) = 0;
  *(_QWORD *)(a1 + 904) = 0;
  *(_QWORD *)(a1 + 912) = 0x800000000LL;
  *(_QWORD *)(a1 + 928) = 200;
  v3 = a2[1];
  v4 = *a2;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v5 = v3 - v4;
  if ( v3 == v4 )
  {
    v7 = 0;
  }
  else
  {
    if ( v5 > 0x7FFFFFFFFFFFFFE0LL )
      sub_4261EA(a1, a2, v5);
    v52 = v3 - v4;
    v6 = sub_22077B0(v3 - v4);
    v3 = a2[1];
    v4 = *a2;
    v5 = v52;
    v7 = (char *)v6;
  }
  v57 = v7;
  v58 = v7;
  v59 = &v7[v5];
  if ( v4 == v3 )
  {
    v8 = v7;
    v9 = 0;
  }
  else
  {
    do
    {
      if ( v7 )
      {
        *(_QWORD *)v7 = v7 + 16;
        sub_240DB00((__int64 *)v7, *(_BYTE **)v4, *(_QWORD *)v4 + *(_QWORD *)(v4 + 8));
      }
      v4 += 32;
      v7 += 32;
    }
    while ( v3 != v4 );
    v8 = v57;
    v9 = v7 - v57;
  }
  v10 = qword_4FE3970;
  v11 = qword_4FE3968;
  v58 = v7;
  if ( qword_4FE3970 != qword_4FE3968 )
  {
    v12 = qword_4FE3970 - qword_4FE3968;
    if ( v59 - v7 >= (unsigned __int64)(qword_4FE3970 - qword_4FE3968) )
    {
      v13 = qword_4FE3968;
      do
      {
        if ( v7 )
        {
          *(_QWORD *)v7 = v7 + 16;
          sub_240DB00((__int64 *)v7, *(_BYTE **)v13, *(_QWORD *)v13 + *(_QWORD *)(v13 + 8));
        }
        v13 += 32;
        v7 += 32;
      }
      while ( v10 != v13 );
      v58 += v12;
      goto LABEL_16;
    }
    v29 = v9 >> 5;
    v30 = v12 >> 5;
    if ( v30 > 0x3FFFFFFFFFFFFFFLL - v29 )
      sub_4262D8((__int64)"vector::_M_range_insert");
    if ( v30 < v29 )
      v30 = v29;
    v31 = __CFADD__(v30, v29);
    v32 = v30 + v29;
    if ( v31 )
    {
      v45 = 0x7FFFFFFFFFFFFFE0LL;
    }
    else
    {
      if ( !v32 )
      {
        v51 = 0;
        v54 = 0;
LABEL_45:
        if ( v8 == v7 )
        {
          v36 = (__m128i *)v54;
        }
        else
        {
          v33 = (const __m128i *)(v8 + 16);
          v34 = v7 - v8;
          v35 = (__m128i *)v54;
          v36 = (__m128i *)&v54[v34];
          do
          {
            if ( v35 )
            {
              v35->m128i_i64[0] = (__int64)v35[1].m128i_i64;
              v37 = (const __m128i *)v33[-1].m128i_i64[0];
              if ( v37 == v33 )
              {
                v35[1] = _mm_loadu_si128(v33);
              }
              else
              {
                v35->m128i_i64[0] = (__int64)v37;
                v35[1].m128i_i64[0] = v33->m128i_i64[0];
              }
              v35->m128i_i64[1] = v33[-1].m128i_i64[1];
              v33[-1].m128i_i64[0] = (__int64)v33;
              v33[-1].m128i_i64[1] = 0;
              v33->m128i_i8[0] = 0;
            }
            v35 += 2;
            v33 += 2;
          }
          while ( v35 != v36 );
        }
        v38 = v11;
        do
        {
          if ( v36 )
          {
            v36->m128i_i64[0] = (__int64)v36[1].m128i_i64;
            sub_240DB00(v36->m128i_i64, *(_BYTE **)v38, *(_QWORD *)v38 + *(_QWORD *)(v38 + 8));
          }
          v38 += 32;
          v36 += 2;
        }
        while ( v10 != v38 );
        v39 = v58;
        if ( v58 == v7 )
        {
          v41 = v36;
        }
        else
        {
          v40 = (const __m128i *)(v7 + 16);
          v41 = (__m128i *)((char *)v36 + v58 - v7);
          do
          {
            if ( v36 )
            {
              v36->m128i_i64[0] = (__int64)v36[1].m128i_i64;
              v43 = (const __m128i *)v40[-1].m128i_i64[0];
              if ( v43 == v40 )
              {
                v36[1] = _mm_loadu_si128(v40);
              }
              else
              {
                v36->m128i_i64[0] = (__int64)v43;
                v36[1].m128i_i64[0] = v40->m128i_i64[0];
              }
              v42 = v40[-1].m128i_i64[1];
              v40[-1].m128i_i64[0] = (__int64)v40;
              v40[-1].m128i_i64[1] = 0;
              v36->m128i_i64[1] = v42;
              v40->m128i_i8[0] = 0;
            }
            v36 += 2;
            v40 += 2;
          }
          while ( v36 != v41 );
        }
        v44 = (unsigned __int64 *)v57;
        if ( v39 != v57 )
        {
          do
          {
            if ( (unsigned __int64 *)*v44 != v44 + 2 )
              j_j___libc_free_0(*v44);
            v44 += 4;
          }
          while ( v39 != (char *)v44 );
          v44 = (unsigned __int64 *)v57;
        }
        if ( v44 )
          j_j___libc_free_0((unsigned __int64)v44);
        v58 = (char *)v41;
        v57 = v54;
        v59 = (char *)v51;
        goto LABEL_16;
      }
      if ( v32 > 0x3FFFFFFFFFFFFFFLL )
        v32 = 0x3FFFFFFFFFFFFFFLL;
      v45 = 32 * v32;
    }
    v49 = qword_4FE3968;
    v46 = sub_22077B0(v45);
    v8 = v57;
    v11 = v49;
    v54 = (char *)v46;
    v51 = v46 + v45;
    goto LABEL_45;
  }
LABEL_16:
  sub_CA41E0(&v56);
  sub_23CA6B0(&v55, &v57, v56);
  v14 = v55;
  v15 = *(_QWORD *)(a1 + 792);
  v55 = 0;
  *(_QWORD *)(a1 + 792) = v14;
  if ( v15 )
  {
    sub_23C6FB0(v15);
    j_j___libc_free_0(v15);
    v16 = v55;
    if ( v55 )
    {
      sub_23C6FB0(v55);
      j_j___libc_free_0(v16);
    }
  }
  v17 = v56;
  if ( v56 && !_InterlockedSub((volatile signed __int32 *)v56 + 2, 1u) )
    (*(void (__fastcall **)(__int64 *))(*v17 + 8))(v17);
  v18 = qword_4FE35C8;
  v53 = (__int64 *)(a1 + 896);
  for ( i = qword_4FE35D0; i != v18; v18 += 32 )
  {
    while ( 1 )
    {
      v19 = *(const void **)v18;
      v20 = *(_QWORD *)(v18 + 8);
      v21 = sub_C92610();
      v22 = sub_C92740((__int64)v53, v19, v20, v21);
      v23 = (_QWORD *)(*(_QWORD *)(a1 + 896) + 8LL * v22);
      if ( *v23 )
        break;
LABEL_28:
      v50 = v23;
      v24 = sub_C7D670(v20 + 9, 8);
      v25 = v50;
      v26 = (_QWORD *)v24;
      if ( v20 )
      {
        v47 = (_QWORD *)v24;
        memcpy((void *)(v24 + 8), v19, v20);
        v25 = v50;
        v26 = v47;
      }
      *((_BYTE *)v26 + v20 + 8) = 0;
      v18 += 32;
      *v26 = v20;
      *v25 = v26;
      ++*(_DWORD *)(a1 + 908);
      sub_C929D0(v53, v22);
      if ( i == v18 )
        goto LABEL_31;
    }
    if ( *v23 == -8 )
    {
      --*(_DWORD *)(a1 + 912);
      goto LABEL_28;
    }
  }
LABEL_31:
  v27 = (unsigned __int64 *)v58;
  v28 = (unsigned __int64 *)v57;
  if ( v58 != v57 )
  {
    do
    {
      if ( (unsigned __int64 *)*v28 != v28 + 2 )
        j_j___libc_free_0(*v28);
      v28 += 4;
    }
    while ( v27 != v28 );
    v28 = (unsigned __int64 *)v57;
  }
  if ( v28 )
    j_j___libc_free_0((unsigned __int64)v28);
}
