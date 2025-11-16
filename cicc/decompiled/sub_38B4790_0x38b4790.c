// Function: sub_38B4790
// Address: 0x38b4790
//
char __fastcall sub_38B4790(__int64 a1, __int64 a2)
{
  char v3; // r12
  char result; // al
  __int64 i; // r15
  __int64 v7; // rax
  __int64 *v8; // rdi
  char v9; // al
  __m128i *v10; // rsi
  __int64 *v11; // r15
  const __m128i *v12; // r12
  const __m128i *v13; // rdx
  __m128i *v14; // rax
  __m128i *v15; // rcx
  unsigned int *v16; // r12
  __int64 v17; // rbx
  __int64 v18; // r13
  unsigned __int64 *v19; // rax
  __int64 v20; // rdx
  _QWORD *v21; // rax
  int *v22; // rsi
  __int64 v23; // rdx
  __m128i *v24; // r8
  unsigned __int64 v26; // [rsp+18h] [rbp-C8h]
  __int64 j; // [rsp+20h] [rbp-C0h]
  unsigned __int64 v28; // [rsp+28h] [rbp-B8h]
  unsigned __int64 *v29; // [rsp+30h] [rbp-B0h]
  __int32 v30; // [rsp+40h] [rbp-A0h]
  __int64 v31; // [rsp+48h] [rbp-98h]
  unsigned int *v32; // [rsp+48h] [rbp-98h]
  char v33; // [rsp+48h] [rbp-98h]
  unsigned int v34; // [rsp+54h] [rbp-8Ch] BYREF
  __int64 v35; // [rsp+58h] [rbp-88h] BYREF
  __m128i v36; // [rsp+60h] [rbp-80h] BYREF
  __int64 v37; // [rsp+70h] [rbp-70h]
  __int64 v38; // [rsp+78h] [rbp-68h]
  __int64 v39; // [rsp+80h] [rbp-60h] BYREF
  int v40; // [rsp+88h] [rbp-58h] BYREF
  _QWORD *v41; // [rsp+90h] [rbp-50h]
  int *v42; // [rsp+98h] [rbp-48h]
  int *v43; // [rsp+A0h] [rbp-40h]
  __int64 v44; // [rsp+A8h] [rbp-38h]

  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  v3 = sub_388AF10(a1, 16, "expected ':' in refs");
  result = v3 | sub_388AF10(a1, 12, "expected '(' in refs");
  if ( result )
    return result;
  v40 = 0;
  v42 = &v40;
  v43 = &v40;
  v41 = 0;
  v44 = 0;
  for ( i = a1 + 8; ; *(_DWORD *)(a1 + 64) = sub_3887100(i) )
  {
    v7 = *(_QWORD *)(a1 + 56);
    v8 = (__int64 *)a1;
    v35 = 0;
    v31 = v7;
    v9 = sub_388F790(a1, &v35, &v34);
    if ( v9 )
      goto LABEL_31;
    if ( (v35 & 0xFFFFFFFFFFFFFFF8LL) == (qword_5052688 & 0xFFFFFFFFFFFFFFF8LL) )
    {
      v21 = v41;
      if ( v41 )
      {
        v22 = &v40;
        do
        {
          while ( 1 )
          {
            v8 = (__int64 *)v21[2];
            v23 = v21[3];
            if ( *((_DWORD *)v21 + 8) >= v34 )
              break;
            v21 = (_QWORD *)v21[3];
            if ( !v23 )
              goto LABEL_42;
          }
          v22 = (int *)v21;
          v21 = (_QWORD *)v21[2];
        }
        while ( v8 );
LABEL_42:
        if ( v22 != &v40 && v34 >= v22[8] )
          goto LABEL_45;
      }
      else
      {
        v22 = &v40;
      }
      v8 = &v39;
      v36.m128i_i64[0] = (__int64)&v34;
      v22 = (int *)sub_38B4270(&v39, (__int64)v22, (unsigned int **)&v36);
LABEL_45:
      v36.m128i_i32[0] = (__int64)(*(_QWORD *)(a2 + 8) - *(_QWORD *)a2) >> 3;
      v36.m128i_i64[1] = v31;
      v24 = (__m128i *)*((_QWORD *)v22 + 6);
      if ( v24 == *((__m128i **)v22 + 7) )
      {
        v8 = (__int64 *)(v22 + 10);
        sub_3894FE0((unsigned __int64 *)v22 + 5, *((const __m128i **)v22 + 6), &v36);
      }
      else
      {
        if ( v24 )
        {
          *v24 = _mm_loadu_si128(&v36);
          v24 = (__m128i *)*((_QWORD *)v22 + 6);
        }
        *((_QWORD *)v22 + 6) = v24 + 1;
      }
    }
    v10 = *(__m128i **)(a2 + 8);
    if ( v10 == *(__m128i **)(a2 + 16) )
    {
      v8 = (__int64 *)a2;
      sub_142E0C0((char **)a2, v10->m128i_i8, &v35);
    }
    else
    {
      if ( v10 )
      {
        v10->m128i_i64[0] = v35;
        v10 = *(__m128i **)(a2 + 8);
      }
      v10 = (__m128i *)((char *)v10 + 8);
      *(_QWORD *)(a2 + 8) = v10;
    }
    if ( *(_DWORD *)(a1 + 64) != 4 )
      break;
  }
  v11 = (__int64 *)a2;
  for ( j = (__int64)v42; (int *)j != &v40; j = sub_220EEE0(j) )
  {
    v12 = *(const __m128i **)(j + 48);
    v13 = *(const __m128i **)(j + 40);
    v30 = *(_DWORD *)(j + 32);
    v26 = (char *)v12 - (char *)v13;
    if ( v12 == v13 )
    {
      v28 = 0;
      if ( v12 == v13 )
        goto LABEL_29;
    }
    else
    {
      if ( v26 > 0x7FFFFFFFFFFFFFF0LL )
        sub_4261EA(v8, v10, v13);
      v28 = sub_22077B0(v26);
      v12 = *(const __m128i **)(j + 48);
      v13 = *(const __m128i **)(j + 40);
      if ( v13 == v12 )
        goto LABEL_27;
    }
    v14 = (__m128i *)v28;
    v15 = (__m128i *)(v28 + (char *)v12 - (char *)v13);
    v32 = (unsigned int *)v15;
    do
    {
      if ( v14 )
        *v14 = _mm_loadu_si128(v13);
      ++v14;
      ++v13;
    }
    while ( v14 != v15 );
    v16 = (unsigned int *)v28;
    if ( (__m128i *)v28 == v15 )
    {
LABEL_28:
      v10 = (__m128i *)v26;
      j_j___libc_free_0(v28);
      goto LABEL_29;
    }
    do
    {
      while ( 1 )
      {
        v36.m128i_i64[1] = 0;
        v37 = 0;
        v17 = *v16;
        v36.m128i_i32[0] = v30;
        v18 = *((_QWORD *)v16 + 1);
        v38 = 0;
        v19 = (unsigned __int64 *)sub_3891660((_QWORD *)(a1 + 1224), v36.m128i_i32);
        if ( v36.m128i_i64[1] )
        {
          v29 = v19;
          j_j___libc_free_0(v36.m128i_u64[1]);
          v19 = v29;
        }
        v20 = *v11;
        v36.m128i_i64[1] = v18;
        v36.m128i_i64[0] = v20 + 8 * v17;
        v10 = (__m128i *)v19[6];
        if ( v10 != (__m128i *)v19[7] )
          break;
        v16 += 4;
        sub_3895160(v19 + 5, v10, &v36);
        if ( v32 == v16 )
          goto LABEL_27;
      }
      if ( v10 )
      {
        *v10 = _mm_loadu_si128(&v36);
        v10 = (__m128i *)v19[6];
      }
      ++v10;
      v16 += 4;
      v19[6] = (unsigned __int64)v10;
    }
    while ( v32 != v16 );
LABEL_27:
    if ( v28 )
      goto LABEL_28;
LABEL_29:
    v8 = (__int64 *)j;
  }
  v9 = sub_388AF10(a1, 13, "expected ')' in refs");
LABEL_31:
  v33 = v9;
  sub_3889030(v41);
  return v33;
}
