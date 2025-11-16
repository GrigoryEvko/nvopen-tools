// Function: sub_23B0250
// Address: 0x23b0250
//
__int64 __fastcall sub_23B0250(__int64 a1, const void *a2, size_t a3, __int64 a4, __int64 a5, __int64 a6)
{
  __m128i *v7; // r12
  __int64 v8; // rbx
  unsigned __int64 v9; // rdx
  int v10; // eax
  unsigned int v11; // r14d
  __int64 *v12; // rbx
  __int64 result; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  bool v17; // cf
  unsigned __int64 v18; // rax
  __int64 v19; // rax
  const __m128i *v20; // r14
  __m128i *i; // rbx
  unsigned __int64 *v22; // rsi
  __m128i v23; // xmm0
  int *v24; // rdi
  __int64 v25; // rax
  __int64 v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rax
  unsigned __int64 v30; // r15
  int *v31; // rcx
  __int64 v32; // r15
  unsigned __int64 v33; // r15
  char *v34; // rcx
  const void *v35; // rsi
  __int64 v36; // r15
  __m128i *v37; // r14
  unsigned __int64 v38; // rdi
  unsigned __int64 v39; // rdi
  unsigned __int64 v40; // rbx
  unsigned __int64 v41; // r15
  unsigned __int64 v42; // rdi
  unsigned __int64 v43; // rdi
  __int64 v44; // rax
  __int64 v45; // [rsp+8h] [rbp-98h]
  __int64 v46; // [rsp+10h] [rbp-90h]
  __m128i *v47; // [rsp+18h] [rbp-88h]
  __int64 v48; // [rsp+18h] [rbp-88h]
  __int64 v49; // [rsp+20h] [rbp-80h]
  __int64 v50; // [rsp+28h] [rbp-78h]
  unsigned __int64 v51; // [rsp+30h] [rbp-70h]
  unsigned __int64 v52; // [rsp+38h] [rbp-68h]

  v7 = *(__m128i **)(a1 + 8);
  v52 = *(_QWORD *)a1;
  v8 = (__int64)v7->m128i_i64 - *(_QWORD *)a1;
  v9 = 0x8E38E38E38E38E39LL * (v8 >> 4);
  v51 = v9;
  if ( v7 != *(__m128i **)(a1 + 16) )
  {
    if ( v7 )
    {
      v7->m128i_i64[0] = a1;
      v7->m128i_i32[2] = v9;
      v7[1].m128i_i64[0] = a4;
      v7[1].m128i_i64[1] = 0;
      v7[2].m128i_i64[0] = a5;
      v7[2].m128i_i64[1] = a6;
      v7[3].m128i_i32[2] = 0;
      v7[4].m128i_i64[0] = 0;
      v7[4].m128i_i64[1] = (__int64)&v7[3].m128i_i64[1];
      v7[5].m128i_i64[0] = (__int64)&v7[3].m128i_i64[1];
      v7[5].m128i_i64[1] = 0;
      v7[6].m128i_i64[0] = 0;
      v7[6].m128i_i64[1] = 0;
      v7[7].m128i_i64[0] = 0;
      v7[7].m128i_i64[1] = 0;
      v7[8].m128i_i64[0] = 0;
      v7[8].m128i_i64[1] = 0;
      v7 = *(__m128i **)(a1 + 8);
    }
    *(_QWORD *)(a1 + 8) = v7 + 9;
    goto LABEL_5;
  }
  v15 = 0xE38E38E38E38E3LL;
  if ( v51 == 0xE38E38E38E38E3LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v16 = 1;
  if ( v51 )
    v16 = 0x8E38E38E38E38E39LL * (v8 >> 4);
  v17 = __CFADD__(v51, v16);
  v18 = v51 + v16;
  v49 = v18;
  if ( v17 )
  {
    v43 = 0x7FFFFFFFFFFFFFB0LL;
    v49 = 0xE38E38E38E38E3LL;
LABEL_57:
    v45 = a6;
    v46 = a5;
    v48 = a4;
    v44 = sub_22077B0(v43);
    a4 = v48;
    a5 = v46;
    v50 = v44;
    a6 = v45;
    goto LABEL_17;
  }
  if ( v18 )
  {
    if ( v18 <= 0xE38E38E38E38E3LL )
      v15 = v18;
    v49 = v15;
    v43 = 144 * v15;
    goto LABEL_57;
  }
  v50 = 0;
LABEL_17:
  v19 = v50 + v8;
  if ( v50 + v8 )
  {
    *(_QWORD *)v19 = a1;
    *(_QWORD *)(v19 + 16) = a4;
    *(_DWORD *)(v19 + 8) = v51;
    v15 = v19 + 56;
    *(_QWORD *)(v19 + 24) = 0;
    *(_QWORD *)(v19 + 32) = a5;
    *(_QWORD *)(v19 + 40) = a6;
    *(_DWORD *)(v19 + 56) = 0;
    *(_QWORD *)(v19 + 64) = 0;
    *(_QWORD *)(v19 + 72) = v19 + 56;
    *(_QWORD *)(v19 + 80) = v19 + 56;
    *(_QWORD *)(v19 + 88) = 0;
    *(_QWORD *)(v19 + 96) = 0;
    *(_QWORD *)(v19 + 104) = 0;
    *(_QWORD *)(v19 + 112) = 0;
    *(_QWORD *)(v19 + 120) = 0;
    *(_QWORD *)(v19 + 128) = 0;
    *(_QWORD *)(v19 + 136) = 0;
  }
  v20 = (const __m128i *)v52;
  if ( v7 == (__m128i *)v52 )
  {
    v47 = (__m128i *)(v50 + 144);
  }
  else
  {
    for ( i = (__m128i *)v50; ; i += 9 )
    {
      if ( i )
      {
        v22 = &i[3].m128i_u64[1];
        i->m128i_i64[0] = v20->m128i_i64[0];
        i->m128i_i32[2] = v20->m128i_i32[2];
        i[1].m128i_i64[0] = v20[1].m128i_i64[0];
        i[1].m128i_i64[1] = v20[1].m128i_i64[1];
        v23 = _mm_loadu_si128(v20 + 2);
        i[3].m128i_i32[2] = 0;
        i[4].m128i_i64[0] = 0;
        i[4].m128i_i64[1] = (__int64)&i[3].m128i_i64[1];
        i[5].m128i_i64[0] = (__int64)&i[3].m128i_i64[1];
        i[5].m128i_i64[1] = 0;
        i[2] = v23;
        v24 = (int *)v20[4].m128i_i64[0];
        if ( v24 )
        {
          v25 = sub_23B0130(v24, (__int64)v22);
          v26 = v25;
          do
          {
            v27 = v25;
            v25 = *(_QWORD *)(v25 + 16);
          }
          while ( v25 );
          i[4].m128i_i64[1] = v27;
          v28 = v26;
          do
          {
            v15 = v28;
            v28 = *(_QWORD *)(v28 + 24);
          }
          while ( v28 );
          i[5].m128i_i64[0] = v15;
          v29 = v20[5].m128i_i64[1];
          i[4].m128i_i64[0] = v26;
          i[5].m128i_i64[1] = v29;
        }
        v30 = v20[6].m128i_i64[1] - v20[6].m128i_i64[0];
        i[6].m128i_i64[0] = 0;
        i[6].m128i_i64[1] = 0;
        i[7].m128i_i64[0] = 0;
        if ( v30 )
        {
          if ( v30 > 0x7FFFFFFFFFFFFFFCLL )
            goto LABEL_63;
          v24 = (int *)v30;
          v31 = (int *)sub_22077B0(v30);
        }
        else
        {
          v31 = 0;
        }
        i[6].m128i_i64[0] = (__int64)v31;
        i[7].m128i_i64[0] = (__int64)v31 + v30;
        i[6].m128i_i64[1] = (__int64)v31;
        v22 = (unsigned __int64 *)v20[6].m128i_i64[0];
        v32 = v20[6].m128i_i64[1] - (_QWORD)v22;
        if ( (unsigned __int64 *)v20[6].m128i_i64[1] != v22 )
        {
          v24 = v31;
          v31 = (int *)memmove(v31, v22, v20[6].m128i_i64[1] - (_QWORD)v22);
        }
        i[6].m128i_i64[1] = (__int64)v31 + v32;
        v33 = v20[8].m128i_i64[0] - v20[7].m128i_i64[1];
        i[7].m128i_i64[1] = 0;
        i[8].m128i_i64[0] = 0;
        i[8].m128i_i64[1] = 0;
        if ( v33 )
        {
          if ( v33 > 0x7FFFFFFFFFFFFFFCLL )
LABEL_63:
            sub_4261EA(v24, v22, v15);
          v34 = (char *)sub_22077B0(v33);
        }
        else
        {
          v33 = 0;
          v34 = 0;
        }
        i[7].m128i_i64[1] = (__int64)v34;
        i[8].m128i_i64[1] = (__int64)&v34[v33];
        i[8].m128i_i64[0] = (__int64)v34;
        v35 = (const void *)v20[7].m128i_i64[1];
        v36 = v20[8].m128i_i64[0] - (_QWORD)v35;
        if ( (const void *)v20[8].m128i_i64[0] != v35 )
          v34 = (char *)memmove(v34, v35, v20[8].m128i_i64[0] - (_QWORD)v35);
        i[8].m128i_i64[0] = (__int64)&v34[v36];
      }
      v20 += 9;
      if ( v7 == v20 )
        break;
    }
    v37 = (__m128i *)v52;
    v47 = i + 18;
    do
    {
      v38 = v37[7].m128i_u64[1];
      if ( v38 )
        j_j___libc_free_0(v38);
      v39 = v37[6].m128i_u64[0];
      if ( v39 )
        j_j___libc_free_0(v39);
      v40 = v37[4].m128i_u64[0];
      while ( v40 )
      {
        v41 = v40;
        sub_23AFE30(*(_QWORD **)(v40 + 24));
        v42 = *(_QWORD *)(v40 + 40);
        v40 = *(_QWORD *)(v40 + 16);
        if ( v42 != v41 + 56 )
          j_j___libc_free_0(v42);
        j_j___libc_free_0(v41);
      }
      v37 += 9;
    }
    while ( v7 != v37 );
  }
  if ( v52 )
    j_j___libc_free_0(v52);
  *(_QWORD *)(a1 + 8) = v47;
  *(_QWORD *)a1 = v50;
  *(_QWORD *)(a1 + 16) = v50 + 144 * v49;
LABEL_5:
  v10 = sub_C92610();
  v11 = sub_C92740(a1 + 24, a2, a3, v10);
  v12 = (__int64 *)(*(_QWORD *)(a1 + 24) + 8LL * v11);
  result = *v12;
  if ( *v12 )
  {
    if ( result != -8 )
      return result;
    --*(_DWORD *)(a1 + 40);
  }
  v14 = sub_23AE710(16, 8, a2, a3);
  if ( v14 )
  {
    *(_QWORD *)v14 = a3;
    *(_DWORD *)(v14 + 8) = v51;
  }
  *v12 = v14;
  ++*(_DWORD *)(a1 + 36);
  return sub_C929D0((__int64 *)(a1 + 24), v11);
}
