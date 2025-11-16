// Function: sub_9C2BD0
// Address: 0x9c2bd0
//
__m128i *__fastcall sub_9C2BD0(__int64 a1, _BYTE *a2, const __m128i *a3)
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
  char *v16; // r9
  __int64 v17; // r15
  signed __int64 v18; // r10
  char *v19; // r14
  __int64 v20; // rsi
  char *v21; // rax
  __int64 v22; // r13
  __int64 v23; // rax
  __int64 v24; // [rsp+8h] [rbp-48h]
  __int64 v25; // [rsp+10h] [rbp-40h]
  _BYTE *v26; // [rsp+10h] [rbp-40h]
  signed __int64 v27; // [rsp+10h] [rbp-40h]
  char *v28; // [rsp+18h] [rbp-38h]
  _BYTE *v29; // [rsp+18h] [rbp-38h]
  char *v30; // [rsp+18h] [rbp-38h]
  _BYTE *v31; // [rsp+18h] [rbp-38h]

  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(_BYTE **)a1;
  v6 = 0xAAAAAAAAAAAAAAABLL * ((v4 - *(_QWORD *)a1) >> 3);
  if ( v6 == 0x555555555555555LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v6 )
    v7 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) >> 3);
  v9 = __CFADD__(v7, v6);
  v10 = v7 - 0x5555555555555555LL * ((__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) >> 3);
  v11 = a2 - v5;
  if ( v9 )
  {
    v22 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v10 )
    {
      v12 = 0;
      v13 = 0;
      goto LABEL_7;
    }
    if ( v10 > 0x555555555555555LL )
      v10 = 0x555555555555555LL;
    v22 = 24 * v10;
  }
  v27 = a2 - v5;
  v31 = *(_BYTE **)a1;
  v23 = sub_22077B0(v22);
  v5 = v31;
  v11 = v27;
  v13 = (char *)v23;
  v12 = v23 + v22;
LABEL_7:
  result = (__m128i *)&v13[v11];
  if ( &v13[v11] )
  {
    v15 = _mm_loadu_si128(a3);
    result[1].m128i_i64[0] = a3[1].m128i_i64[0];
    *result = v15;
  }
  v16 = &v13[v11 + 24];
  v17 = *(_QWORD *)(a1 + 16);
  v18 = v4 - (_QWORD)a2;
  v19 = &v16[v4 - (_QWORD)a2];
  if ( v11 > 0 )
  {
    v24 = v18;
    v25 = (__int64)&v13[v11 + 24];
    v29 = v5;
    v21 = (char *)memmove(v13, v5, v11);
    v5 = v29;
    v18 = v24;
    v16 = (char *)v25;
    v13 = v21;
    v20 = v17 - (_QWORD)v29;
    if ( v24 <= 0 )
      goto LABEL_14;
    goto LABEL_16;
  }
  if ( v18 > 0 )
  {
LABEL_16:
    v26 = v5;
    v30 = v13;
    result = (__m128i *)memcpy(v16, a2, v18);
    v5 = v26;
    v13 = v30;
    if ( !v26 )
      goto LABEL_12;
    goto LABEL_13;
  }
  if ( v5 )
  {
LABEL_13:
    v20 = v17 - (_QWORD)v5;
LABEL_14:
    v28 = v13;
    result = (__m128i *)j_j___libc_free_0(v5, v20);
    v13 = v28;
  }
LABEL_12:
  *(_QWORD *)a1 = v13;
  *(_QWORD *)(a1 + 8) = v19;
  *(_QWORD *)(a1 + 16) = v12;
  return result;
}
