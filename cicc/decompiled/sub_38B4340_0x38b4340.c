// Function: sub_38B4340
// Address: 0x38b4340
//
__int64 __fastcall sub_38B4340(__int64 a1, _QWORD *a2)
{
  __int64 v3; // rbx
  int i; // eax
  _QWORD *v6; // rdi
  unsigned __int8 v7; // al
  __m128i *v8; // rsi
  const __m128i *v9; // r12
  const __m128i *v10; // rdx
  __m128i *v11; // rax
  __m128i *v12; // rcx
  unsigned int *v13; // r12
  __int64 v14; // rbx
  __int64 v15; // r13
  unsigned __int64 *v16; // rax
  __int64 v17; // rdx
  unsigned int v18; // edi
  _QWORD *v19; // rax
  __int64 v20; // r14
  int *v21; // rsi
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // rax
  __m128i *v25; // r8
  unsigned __int64 v27; // [rsp+18h] [rbp-C8h]
  __int64 j; // [rsp+20h] [rbp-C0h]
  unsigned __int64 v29; // [rsp+28h] [rbp-B8h]
  unsigned __int64 *v30; // [rsp+30h] [rbp-B0h]
  __int32 v31; // [rsp+44h] [rbp-9Ch]
  unsigned int *v32; // [rsp+48h] [rbp-98h]
  unsigned __int8 v33; // [rsp+48h] [rbp-98h]
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

  v3 = a1 + 8;
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  if ( (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_388AF10(a1, 12, "expected '(' in typeIdInfo") )
  {
    return 1;
  }
  v40 = 0;
  v41 = 0;
  v44 = 0;
  v42 = &v40;
  v43 = &v40;
  for ( i = *(_DWORD *)(a1 + 64); ; *(_DWORD *)(a1 + 64) = i )
  {
    v35 = 0;
    if ( i == 371 )
    {
      v18 = *(_DWORD *)(a1 + 104);
      v19 = v41;
      v20 = *(_QWORD *)(a1 + 56);
      v21 = &v40;
      v34 = v18;
      if ( !v41 )
        goto LABEL_43;
      do
      {
        while ( 1 )
        {
          v22 = v19[2];
          v23 = v19[3];
          if ( v18 <= *((_DWORD *)v19 + 8) )
            break;
          v19 = (_QWORD *)v19[3];
          if ( !v23 )
            goto LABEL_41;
        }
        v21 = (int *)v19;
        v19 = (_QWORD *)v19[2];
      }
      while ( v22 );
LABEL_41:
      if ( v21 == &v40 || v18 < v21[8] )
      {
LABEL_43:
        v36.m128i_i64[0] = (__int64)&v34;
        v21 = (int *)sub_38B4270(&v39, (__int64)v21, (unsigned int **)&v36);
      }
      v24 = a2[1] - *a2;
      v36.m128i_i64[1] = v20;
      v36.m128i_i32[0] = v24 >> 3;
      v25 = (__m128i *)*((_QWORD *)v21 + 6);
      if ( v25 == *((__m128i **)v21 + 7) )
      {
        sub_3894FE0((unsigned __int64 *)v21 + 5, *((const __m128i **)v21 + 6), &v36);
      }
      else
      {
        if ( v25 )
        {
          *v25 = _mm_loadu_si128(&v36);
          v25 = (__m128i *)*((_QWORD *)v21 + 6);
        }
        *((_QWORD *)v21 + 6) = v25 + 1;
      }
      v6 = (_QWORD *)v3;
      *(_DWORD *)(a1 + 64) = sub_3887100(v3);
    }
    else
    {
      v6 = (_QWORD *)a1;
      v7 = sub_388BD80(a1, &v35);
      if ( v7 )
        goto LABEL_33;
    }
    v8 = (__m128i *)a2[1];
    if ( v8 == (__m128i *)a2[2] )
    {
      v6 = a2;
      sub_9CA200((__int64)a2, v8, &v35);
    }
    else
    {
      if ( v8 )
      {
        v8->m128i_i64[0] = v35;
        v8 = (__m128i *)a2[1];
      }
      v8 = (__m128i *)((char *)v8 + 8);
      a2[1] = v8;
    }
    if ( *(_DWORD *)(a1 + 64) != 4 )
      break;
    i = sub_3887100(v3);
  }
  for ( j = (__int64)v42; (int *)j != &v40; j = sub_220EEE0(j) )
  {
    v9 = *(const __m128i **)(j + 48);
    v10 = *(const __m128i **)(j + 40);
    v31 = *(_DWORD *)(j + 32);
    v27 = (char *)v9 - (char *)v10;
    if ( v9 == v10 )
    {
      v29 = 0;
      if ( v9 == v10 )
        goto LABEL_31;
    }
    else
    {
      if ( v27 > 0x7FFFFFFFFFFFFFF0LL )
        sub_4261EA(v6, v8, v10);
      v29 = sub_22077B0(v27);
      v9 = *(const __m128i **)(j + 48);
      v10 = *(const __m128i **)(j + 40);
      if ( v9 == v10 )
        goto LABEL_29;
    }
    v11 = (__m128i *)v29;
    v12 = (__m128i *)(v29 + (char *)v9 - (char *)v10);
    v32 = (unsigned int *)v12;
    do
    {
      if ( v11 )
        *v11 = _mm_loadu_si128(v10);
      ++v11;
      ++v10;
    }
    while ( v12 != v11 );
    v13 = (unsigned int *)v29;
    if ( (__m128i *)v29 == v12 )
    {
LABEL_30:
      v8 = (__m128i *)v27;
      j_j___libc_free_0(v29);
      goto LABEL_31;
    }
    do
    {
      while ( 1 )
      {
        v36.m128i_i64[1] = 0;
        v37 = 0;
        v14 = *v13;
        v36.m128i_i32[0] = v31;
        v15 = *((_QWORD *)v13 + 1);
        v38 = 0;
        v16 = (unsigned __int64 *)sub_38917E0((_QWORD *)(a1 + 1344), v36.m128i_i32);
        if ( v36.m128i_i64[1] )
        {
          v30 = v16;
          j_j___libc_free_0(v36.m128i_u64[1]);
          v16 = v30;
        }
        v17 = *a2;
        v36.m128i_i64[1] = v15;
        v36.m128i_i64[0] = v17 + 8 * v14;
        v8 = (__m128i *)v16[6];
        if ( v8 != (__m128i *)v16[7] )
          break;
        v13 += 4;
        sub_38952E0(v16 + 5, v8, &v36);
        if ( v32 == v13 )
          goto LABEL_29;
      }
      if ( v8 )
      {
        *v8 = _mm_loadu_si128(&v36);
        v8 = (__m128i *)v16[6];
      }
      ++v8;
      v13 += 4;
      v16[6] = (unsigned __int64)v8;
    }
    while ( v32 != v13 );
LABEL_29:
    if ( v29 )
      goto LABEL_30;
LABEL_31:
    v6 = (_QWORD *)j;
  }
  v7 = sub_388AF10(a1, 13, "expected ')' in typeIdInfo");
LABEL_33:
  v33 = v7;
  sub_3889030(v41);
  return v33;
}
