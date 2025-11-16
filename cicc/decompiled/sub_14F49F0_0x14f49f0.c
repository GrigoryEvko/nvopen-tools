// Function: sub_14F49F0
// Address: 0x14f49f0
//
__m128i *__fastcall sub_14F49F0(__int64 a1, _BYTE *a2, const __m128i *a3)
{
  __int64 v3; // rbx
  __int64 v4; // r14
  _BYTE *v5; // r8
  __int64 v6; // rax
  __int64 v8; // rdx
  bool v10; // cf
  unsigned __int64 v11; // rax
  char *v12; // rcx
  signed __int64 v13; // rdx
  __int64 v14; // rbx
  __m128i *result; // rax
  char *v16; // r9
  __int64 v17; // r15
  signed __int64 v18; // r10
  char *v19; // r14
  __int64 v20; // rsi
  char *v21; // rax
  __int64 v22; // rbx
  __int64 v23; // rax
  __int64 v24; // [rsp+8h] [rbp-48h]
  __int64 v25; // [rsp+10h] [rbp-40h]
  _BYTE *v26; // [rsp+10h] [rbp-40h]
  signed __int64 v27; // [rsp+10h] [rbp-40h]
  char *v28; // [rsp+18h] [rbp-38h]
  _BYTE *v29; // [rsp+18h] [rbp-38h]
  char *v30; // [rsp+18h] [rbp-38h]
  _BYTE *v31; // [rsp+18h] [rbp-38h]

  v3 = 0x7FFFFFFFFFFFFFFLL;
  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(_BYTE **)a1;
  v6 = (v4 - *(_QWORD *)a1) >> 4;
  if ( v6 == 0x7FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  if ( v6 )
    v8 = (__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) >> 4;
  v10 = __CFADD__(v8, v6);
  v11 = v8 + v6;
  v12 = (char *)v10;
  v13 = a2 - v5;
  if ( v10 )
  {
    v22 = 0x7FFFFFFFFFFFFFF0LL;
  }
  else
  {
    if ( !v11 )
    {
      v14 = 0;
      goto LABEL_7;
    }
    if ( v11 <= 0x7FFFFFFFFFFFFFFLL )
      v3 = v11;
    v22 = 16 * v3;
  }
  v27 = a2 - v5;
  v31 = *(_BYTE **)a1;
  v23 = sub_22077B0(v22);
  v5 = v31;
  v13 = v27;
  v12 = (char *)v23;
  v14 = v23 + v22;
LABEL_7:
  result = (__m128i *)&v12[v13];
  if ( &v12[v13] )
    *result = _mm_loadu_si128(a3);
  v16 = &v12[v13 + 16];
  v17 = *(_QWORD *)(a1 + 16);
  v18 = v4 - (_QWORD)a2;
  v19 = &v16[v4 - (_QWORD)a2];
  if ( v13 > 0 )
  {
    v24 = v18;
    v25 = (__int64)&v12[v13 + 16];
    v29 = v5;
    v21 = (char *)memmove(v12, v5, v13);
    v5 = v29;
    v18 = v24;
    v16 = (char *)v25;
    v12 = v21;
    v20 = v17 - (_QWORD)v29;
    if ( v24 <= 0 )
      goto LABEL_14;
    goto LABEL_16;
  }
  if ( v18 > 0 )
  {
LABEL_16:
    v26 = v5;
    v30 = v12;
    result = (__m128i *)memcpy(v16, a2, v18);
    v5 = v26;
    v12 = v30;
    if ( !v26 )
      goto LABEL_12;
    goto LABEL_13;
  }
  if ( v5 )
  {
LABEL_13:
    v20 = v17 - (_QWORD)v5;
LABEL_14:
    v28 = v12;
    result = (__m128i *)j_j___libc_free_0(v5, v20);
    v12 = v28;
  }
LABEL_12:
  *(_QWORD *)a1 = v12;
  *(_QWORD *)(a1 + 8) = v19;
  *(_QWORD *)(a1 + 16) = v14;
  return result;
}
