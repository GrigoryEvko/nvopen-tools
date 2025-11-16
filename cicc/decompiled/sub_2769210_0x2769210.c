// Function: sub_2769210
// Address: 0x2769210
//
void __fastcall sub_2769210(__int64 a1, _BYTE *a2, __m128i *a3)
{
  _BYTE *v3; // r9
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rcx
  bool v7; // cf
  unsigned __int64 v8; // rax
  signed __int64 v9; // r10
  unsigned __int64 v10; // r15
  char *v11; // r14
  __m128i *v12; // rbx
  __int64 v13; // rdx
  __m128i v14; // xmm0
  __int64 v15; // rsi
  __m128i v16; // xmm1
  __m128i v17; // xmm2
  __int64 v18; // r11
  __int64 v19; // r8
  __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rsi
  __m128i v24; // xmm3
  __m128i v25; // xmm4
  char *v26; // r11
  signed __int64 v27; // rcx
  char *v28; // rbx
  unsigned __int64 v29; // r15
  __int64 v30; // rax
  __int64 v31; // [rsp+8h] [rbp-58h]
  signed __int64 v32; // [rsp+10h] [rbp-50h]
  __int64 v33; // [rsp+10h] [rbp-50h]
  __int64 v34; // [rsp+10h] [rbp-50h]
  signed __int64 v35; // [rsp+10h] [rbp-50h]
  _BYTE *v36; // [rsp+18h] [rbp-48h]
  __int64 v37; // [rsp+18h] [rbp-48h]
  __int64 v38; // [rsp+18h] [rbp-48h]
  _BYTE *v39; // [rsp+18h] [rbp-48h]
  __int64 v40; // [rsp+20h] [rbp-40h]
  _BYTE *v41; // [rsp+20h] [rbp-40h]
  _BYTE *v42; // [rsp+20h] [rbp-40h]

  v3 = *(_BYTE **)a1;
  v40 = *(_QWORD *)(a1 + 8);
  v4 = 0xCCCCCCCCCCCCCCCDLL * ((v40 - *(_QWORD *)a1) >> 4);
  if ( v4 == 0x199999999999999LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v5 = 1;
  if ( v4 )
    v5 = 0xCCCCCCCCCCCCCCCDLL * ((v40 - *(_QWORD *)a1) >> 4);
  v7 = __CFADD__(v5, v4);
  v8 = v5 - 0x3333333333333333LL * ((v40 - *(_QWORD *)a1) >> 4);
  v9 = a2 - v3;
  if ( v7 )
  {
    v29 = 0x7FFFFFFFFFFFFFD0LL;
LABEL_19:
    v35 = a2 - v3;
    v39 = *(_BYTE **)a1;
    v30 = sub_22077B0(v29);
    v3 = v39;
    v9 = v35;
    v11 = (char *)v30;
    v10 = v30 + v29;
    goto LABEL_7;
  }
  if ( v8 )
  {
    if ( v8 > 0x199999999999999LL )
      v8 = 0x199999999999999LL;
    v29 = 80 * v8;
    goto LABEL_19;
  }
  v10 = 0;
  v11 = 0;
LABEL_7:
  v12 = (__m128i *)&v11[v9];
  if ( &v11[v9] )
  {
    v12->m128i_i64[0] = 0;
    v12->m128i_i64[1] = 0;
    v12[1].m128i_i64[0] = 0;
    v12[1].m128i_i64[1] = 0;
    v12[2].m128i_i64[0] = 0;
    v12[2].m128i_i64[1] = 0;
    v12[3].m128i_i64[0] = 0;
    v12[3].m128i_i64[1] = 0;
    v12[4].m128i_i64[0] = 0;
    v12[4].m128i_i64[1] = 0;
    v32 = v9;
    v36 = v3;
    sub_2768EA0((__int64 *)&v11[v9], 0);
    v3 = v36;
    v9 = v32;
    if ( a3->m128i_i64[0] )
    {
      v13 = v12[1].m128i_i64[1];
      v14 = _mm_loadu_si128(a3);
      v15 = v12[2].m128i_i64[0];
      v16 = _mm_loadu_si128(a3 + 1);
      v37 = v12[1].m128i_i64[0];
      v17 = _mm_loadu_si128(a3 + 2);
      v18 = v12->m128i_i64[0];
      a3->m128i_i64[1] = v12->m128i_i64[1];
      v19 = v12[2].m128i_i64[1];
      v33 = v13;
      v20 = v12[4].m128i_i64[0];
      v21 = v12[4].m128i_i64[1];
      v31 = v15;
      v22 = v12[3].m128i_i64[0];
      v23 = v12[3].m128i_i64[1];
      a3[1].m128i_i64[0] = v37;
      v24 = _mm_loadu_si128(a3 + 3);
      a3->m128i_i64[0] = v18;
      v25 = _mm_loadu_si128(a3 + 4);
      a3[2].m128i_i64[1] = v19;
      a3[1].m128i_i64[1] = v33;
      *v12 = v14;
      a3[2].m128i_i64[0] = v31;
      v12[1] = v16;
      v12[2] = v17;
      v12[3] = v24;
      v12[4] = v25;
      a3[3].m128i_i64[0] = v22;
      a3[3].m128i_i64[1] = v23;
      a3[4].m128i_i64[0] = v20;
      a3[4].m128i_i64[1] = v21;
    }
  }
  v26 = &v11[v9 + 80];
  v27 = v40 - (_QWORD)a2;
  v28 = &v26[v40 - (_QWORD)a2];
  if ( v9 > 0 )
  {
    v34 = (__int64)&v11[v9 + 80];
    v38 = v40 - (_QWORD)a2;
    v41 = v3;
    memmove(v11, v3, v9);
    v3 = v41;
    v27 = v38;
    v26 = (char *)v34;
    if ( v38 <= 0 )
      goto LABEL_14;
LABEL_16:
    v42 = v3;
    memcpy(v26, a2, v27);
    v3 = v42;
    if ( !v42 )
      goto LABEL_13;
    goto LABEL_14;
  }
  if ( v27 > 0 )
    goto LABEL_16;
  if ( v3 )
LABEL_14:
    j_j___libc_free_0((unsigned __int64)v3);
LABEL_13:
  *(_QWORD *)a1 = v11;
  *(_QWORD *)(a1 + 8) = v28;
  *(_QWORD *)(a1 + 16) = v10;
}
