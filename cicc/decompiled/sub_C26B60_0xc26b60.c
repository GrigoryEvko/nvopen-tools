// Function: sub_C26B60
// Address: 0xc26b60
//
__m128i *__fastcall sub_C26B60(__int64 a1, _BYTE *a2, const __m128i *a3)
{
  __int64 v4; // r14
  _BYTE *v5; // r8
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdx
  bool v9; // cf
  unsigned __int64 v10; // rax
  signed __int64 v11; // rdx
  __int64 v12; // r13
  char *v13; // rcx
  __m128i *result; // rax
  __m128i v15; // xmm0
  __m128i v16; // xmm1
  char *v17; // r9
  __int64 v18; // r15
  signed __int64 v19; // r10
  char *v20; // r14
  __int64 v21; // rsi
  char *v22; // rax
  __int64 v23; // r13
  __int64 v24; // rax
  __int64 v25; // [rsp+8h] [rbp-48h]
  __int64 v26; // [rsp+10h] [rbp-40h]
  _BYTE *v27; // [rsp+10h] [rbp-40h]
  signed __int64 v28; // [rsp+10h] [rbp-40h]
  char *v29; // [rsp+18h] [rbp-38h]
  _BYTE *v30; // [rsp+18h] [rbp-38h]
  char *v31; // [rsp+18h] [rbp-38h]
  _BYTE *v32; // [rsp+18h] [rbp-38h]

  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(_BYTE **)a1;
  v6 = 0xCCCCCCCCCCCCCCCDLL * ((v4 - *(_QWORD *)a1) >> 3);
  if ( v6 == 0x333333333333333LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v6 )
    v7 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) >> 3);
  v9 = __CFADD__(v7, v6);
  v10 = v7 - 0x3333333333333333LL * ((__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) >> 3);
  v11 = a2 - v5;
  if ( v9 )
  {
    v23 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v10 )
    {
      v12 = 0;
      v13 = 0;
      goto LABEL_7;
    }
    if ( v10 > 0x333333333333333LL )
      v10 = 0x333333333333333LL;
    v23 = 40 * v10;
  }
  v28 = a2 - v5;
  v32 = *(_BYTE **)a1;
  v24 = sub_22077B0(v23);
  v5 = v32;
  v11 = v28;
  v13 = (char *)v24;
  v12 = v24 + v23;
LABEL_7:
  result = (__m128i *)&v13[v11];
  if ( &v13[v11] )
  {
    v15 = _mm_loadu_si128(a3);
    v16 = _mm_loadu_si128(a3 + 1);
    result[2].m128i_i64[0] = a3[2].m128i_i64[0];
    *result = v15;
    result[1] = v16;
  }
  v17 = &v13[v11 + 40];
  v18 = *(_QWORD *)(a1 + 16);
  v19 = v4 - (_QWORD)a2;
  v20 = &v17[v4 - (_QWORD)a2];
  if ( v11 > 0 )
  {
    v25 = v19;
    v26 = (__int64)&v13[v11 + 40];
    v30 = v5;
    v22 = (char *)memmove(v13, v5, v11);
    v5 = v30;
    v19 = v25;
    v17 = (char *)v26;
    v13 = v22;
    v21 = v18 - (_QWORD)v30;
    if ( v25 <= 0 )
      goto LABEL_14;
    goto LABEL_16;
  }
  if ( v19 > 0 )
  {
LABEL_16:
    v27 = v5;
    v31 = v13;
    result = (__m128i *)memcpy(v17, a2, v19);
    v5 = v27;
    v13 = v31;
    if ( !v27 )
      goto LABEL_12;
    goto LABEL_13;
  }
  if ( v5 )
  {
LABEL_13:
    v21 = v18 - (_QWORD)v5;
LABEL_14:
    v29 = v13;
    result = (__m128i *)j_j___libc_free_0(v5, v21);
    v13 = v29;
  }
LABEL_12:
  *(_QWORD *)a1 = v13;
  *(_QWORD *)(a1 + 8) = v20;
  *(_QWORD *)(a1 + 16) = v12;
  return result;
}
