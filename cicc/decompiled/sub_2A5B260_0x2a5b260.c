// Function: sub_2A5B260
// Address: 0x2a5b260
//
void __fastcall sub_2A5B260(__int64 a1, _BYTE *a2, const __m128i *a3)
{
  __int64 v4; // r14
  _BYTE *v5; // r8
  __int64 v6; // rax
  __int64 v7; // rdx
  bool v8; // cf
  unsigned __int64 v9; // rax
  signed __int64 v10; // rdx
  unsigned __int64 v11; // r13
  char *v12; // rcx
  __m128i *v13; // rax
  __m128i v14; // xmm0
  __m128i v15; // xmm1
  __m128i v16; // xmm2
  char *v17; // r9
  signed __int64 v18; // r10
  char *v19; // r14
  char *v20; // rax
  unsigned __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // [rsp+8h] [rbp-48h]
  __int64 v24; // [rsp+10h] [rbp-40h]
  _BYTE *v25; // [rsp+10h] [rbp-40h]
  signed __int64 v26; // [rsp+10h] [rbp-40h]
  char *v27; // [rsp+18h] [rbp-38h]
  _BYTE *v28; // [rsp+18h] [rbp-38h]
  char *v29; // [rsp+18h] [rbp-38h]
  _BYTE *v30; // [rsp+18h] [rbp-38h]

  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(_BYTE **)a1;
  v6 = 0x6DB6DB6DB6DB6DB7LL * ((v4 - *(_QWORD *)a1) >> 3);
  if ( v6 == 0x249249249249249LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v6 )
    v7 = 0x6DB6DB6DB6DB6DB7LL * ((__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) >> 3);
  v8 = __CFADD__(v7, v6);
  v9 = v7 + v6;
  v10 = a2 - v5;
  if ( v8 )
  {
    v21 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v9 )
    {
      v11 = 0;
      v12 = 0;
      goto LABEL_7;
    }
    if ( v9 > 0x249249249249249LL )
      v9 = 0x249249249249249LL;
    v21 = 56 * v9;
  }
  v26 = a2 - v5;
  v30 = *(_BYTE **)a1;
  v22 = sub_22077B0(v21);
  v5 = v30;
  v10 = v26;
  v12 = (char *)v22;
  v11 = v22 + v21;
LABEL_7:
  v13 = (__m128i *)&v12[v10];
  if ( &v12[v10] )
  {
    v14 = _mm_loadu_si128(a3);
    v15 = _mm_loadu_si128(a3 + 1);
    v16 = _mm_loadu_si128(a3 + 2);
    v13[3].m128i_i64[0] = a3[3].m128i_i64[0];
    *v13 = v14;
    v13[1] = v15;
    v13[2] = v16;
  }
  v17 = &v12[v10 + 56];
  v18 = v4 - (_QWORD)a2;
  v19 = &v17[v4 - (_QWORD)a2];
  if ( v10 > 0 )
  {
    v23 = v18;
    v24 = (__int64)&v12[v10 + 56];
    v28 = v5;
    v20 = (char *)memmove(v12, v5, v10);
    v5 = v28;
    v18 = v23;
    v17 = (char *)v24;
    v12 = v20;
    if ( v23 <= 0 )
      goto LABEL_13;
LABEL_15:
    v25 = v5;
    v29 = v12;
    memcpy(v17, a2, v18);
    v5 = v25;
    v12 = v29;
    if ( !v25 )
      goto LABEL_12;
    goto LABEL_13;
  }
  if ( v18 > 0 )
    goto LABEL_15;
  if ( v5 )
  {
LABEL_13:
    v27 = v12;
    j_j___libc_free_0((unsigned __int64)v5);
    v12 = v27;
  }
LABEL_12:
  *(_QWORD *)a1 = v12;
  *(_QWORD *)(a1 + 8) = v19;
  *(_QWORD *)(a1 + 16) = v11;
}
