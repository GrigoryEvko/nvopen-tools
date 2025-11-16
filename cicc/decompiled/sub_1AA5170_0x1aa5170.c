// Function: sub_1AA5170
// Address: 0x1aa5170
//
unsigned __int64 *__fastcall sub_1AA5170(unsigned __int64 *a1, __int64 a2, unsigned __int64 a3, unsigned __int64 a4)
{
  unsigned __int64 *v4; // r14
  __int64 v8; // r15
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // rax
  char *v13; // r9
  __int64 v14; // rax
  char *v15; // r10
  __int64 v16; // rbx
  __m128i *v17; // rax
  __m128i *v18; // r10
  __int64 v19; // rcx
  char *v20; // rsi
  unsigned __int64 v21; // rcx
  unsigned __int64 v22; // rax
  __int64 v23; // r9
  unsigned __int64 v24; // r10
  __int64 v25; // rdi
  __int64 i; // r8
  bool v27; // cf
  __int64 v28; // rax
  char *v29; // rdx
  unsigned __int64 v30; // rax
  unsigned __int64 v31; // r13
  unsigned __int64 v32; // rsi
  unsigned __int64 v33; // rax
  bool v34; // cf
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  unsigned __int64 *result; // rax
  char *v38; // r13
  unsigned __int64 v39; // r14
  bool v40; // cc
  char *v41; // rax
  unsigned __int64 v42; // rax
  unsigned __int64 v43; // rdx
  unsigned __int64 v44; // rax
  unsigned __int64 v45; // rdx
  __int64 v46; // [rsp+0h] [rbp-50h]
  unsigned __int64 v47; // [rsp+8h] [rbp-48h]
  __m128i *src; // [rsp+10h] [rbp-40h]
  void *srca; // [rsp+10h] [rbp-40h]
  __m128i *v50; // [rsp+18h] [rbp-38h]
  __m128i *v51; // [rsp+18h] [rbp-38h]

  v4 = a1;
  v8 = *(unsigned int *)(a2 + 8);
  if ( !*(_DWORD *)(a2 + 8) )
  {
    sub_1AA4710(*(char **)a2, *(char **)a2, (unsigned __int8 (__fastcall *)(__m128i *, __int8 *))sub_1AA3490);
    j_j___libc_free_0(0, 0);
    v41 = *(char **)a2;
    *a1 = a3;
    v42 = *((_QWORD *)v41 + 3);
    v43 = v42;
    if ( a3 >= v42 )
      v43 = a3;
    if ( a4 >= v42 )
      v42 = a4;
    a1[1] = v43;
    if ( v42 >= a3 )
      a3 = v42;
    v21 = a3;
    goto LABEL_40;
  }
  v9 = 0;
  v10 = 0;
  do
  {
    v11 = 16;
    v12 = v9 + *(_QWORD *)a2;
    if ( *(_QWORD *)(v12 + 24) >= 0x10u )
      v11 = *(_QWORD *)(v12 + 24);
    ++v10;
    v9 += 56;
    *(_QWORD *)(v12 + 24) = v11;
  }
  while ( v10 != v8 );
  v13 = *(char **)a2;
  v14 = 56LL * *(unsigned int *)(a2 + 8);
  v15 = (char *)(*(_QWORD *)a2 + v14);
  if ( v14 )
  {
    v50 = (__m128i *)(*(_QWORD *)a2 + v14);
    src = *(__m128i **)a2;
    v47 = a3;
    v16 = 0x6DB6DB6DB6DB6DB7LL * (v14 >> 3);
    while ( 1 )
    {
      v46 = 56 * v16;
      v17 = (__m128i *)sub_2207800(56 * v16, &unk_435FF63);
      if ( v17 )
        break;
      v16 >>= 1;
      if ( !v16 )
      {
        v15 = (char *)v50;
        v13 = (char *)src;
        a3 = v47;
        goto LABEL_57;
      }
    }
    v18 = v50;
    v19 = v16;
    a3 = v47;
    v51 = v17;
    sub_1AA5050(src, v18, v17, v19, (unsigned __int8 (__fastcall *)(__m128i *, __int8 *))sub_1AA3490);
    j_j___libc_free_0(v51, v46);
    v20 = *(char **)a2;
    *a1 = v47;
    v21 = *((_QWORD *)v20 + 3);
    v22 = v21;
    if ( v47 >= v21 )
      v22 = v47;
    if ( a4 >= v21 )
      v21 = a4;
    a1[1] = v22;
    if ( v21 < v47 )
      v21 = v47;
  }
  else
  {
LABEL_57:
    sub_1AA4710(v13, v15, (unsigned __int8 (__fastcall *)(__m128i *, __int8 *))sub_1AA3490);
    j_j___libc_free_0(0, 0);
    v20 = *(char **)a2;
    *a1 = a3;
    v44 = *((_QWORD *)v20 + 3);
    v45 = v44;
    if ( a3 >= v44 )
      v45 = a3;
    if ( a4 >= v44 )
      v44 = a4;
    a1[1] = v45;
    if ( v44 < a3 )
      v44 = a3;
    v21 = v44;
  }
  v23 = v8 - 1;
  v24 = 2 * a3;
  v25 = 0;
  srca = (void *)a4;
  for ( i = 0; ; ++i )
  {
    v29 = &v20[v25];
    v30 = *(_QWORD *)&v20[v25 + 8];
    if ( i == v23 )
      break;
    v31 = a3;
    if ( *(_QWORD *)&v20[v25 + 80] >= a3 )
      v31 = *(_QWORD *)&v20[v25 + 80];
    v32 = v31;
    if ( v30 <= 4 )
    {
      *((_QWORD *)v29 + 5) = v21;
      v27 = v24 < 0x10;
      v28 = 16;
LABEL_17:
      if ( !v27 )
        v28 = 2 * a3;
      v25 += 56;
      v21 += (v31 + v28 - 1) / v31 * v31;
      goto LABEL_20;
    }
    if ( v30 <= 0x10 )
    {
      *((_QWORD *)v29 + 5) = v21;
      v27 = v24 < 0x20;
      v28 = 32;
      goto LABEL_17;
    }
LABEL_29:
    if ( v30 > 0x80 )
    {
      if ( v30 > 0x200 )
      {
        v39 = v30 + 256;
        v40 = v30 <= 0x1000;
        v33 = v30 + 128;
        if ( !v40 )
          v33 = v39;
      }
      else
      {
        v33 = v30 + 64;
      }
    }
    else
    {
      v33 = v30 + 32;
    }
    *((_QWORD *)v29 + 5) = v21;
    if ( v33 < v24 )
      v33 = 2 * a3;
    v25 += 56;
    v21 += (v32 + v33 - 1) / v32 * v32;
    if ( i == v23 )
    {
      v4 = a1;
      a4 = (unsigned __int64)srca;
      goto LABEL_40;
    }
LABEL_20:
    v20 = *(char **)a2;
  }
  if ( v30 <= 4 )
  {
    v4 = a1;
    a4 = (unsigned __int64)srca;
    *((_QWORD *)v29 + 5) = v21;
    v34 = v24 < 0x10;
    v35 = 16;
    goto LABEL_37;
  }
  if ( v30 > 0x10 )
  {
    v32 = a3;
    goto LABEL_29;
  }
  v4 = a1;
  a4 = (unsigned __int64)srca;
  *((_QWORD *)v29 + 5) = v21;
  v34 = v24 < 0x20;
  v35 = 32;
LABEL_37:
  if ( v34 )
    v24 = v35;
  v21 += a3 * ((a3 + v24 - 1) / a3);
LABEL_40:
  v36 = v21 % a4;
  result = v4;
  v38 = (char *)(v21 + a4 - v21 % a4);
  if ( v36 )
    v21 = (unsigned __int64)v38;
  v4[2] = v21;
  return result;
}
