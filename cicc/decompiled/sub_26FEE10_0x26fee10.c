// Function: sub_26FEE10
// Address: 0x26fee10
//
bool __fastcall sub_26FEE10(__int64 *a1, unsigned __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  __int64 *v6; // rax
  unsigned __int8 *v7; // rax
  _BYTE *v8; // rdx
  const char *v9; // rax
  __int64 v10; // r12
  char *v11; // rbx
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // r13
  __int64 v14; // r14
  const char *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdx
  unsigned __int64 v19; // rax
  __int64 v20; // rsi
  __m128i *v21; // rsi
  unsigned __int64 v22; // rbx
  const char *v23; // rax
  __int64 v24; // rdx
  unsigned __int64 v25; // rsi
  _QWORD *v26; // rdx
  _QWORD *v27; // rcx
  _QWORD *v28; // rax
  __int64 v29; // r8
  __int64 v30; // rdi
  __int64 v31; // rax
  unsigned __int64 v32; // rdi
  _QWORD *v33; // r9
  __int64 v34; // r8
  __int64 v35; // rcx
  _QWORD *v36; // [rsp+0h] [rbp-90h]
  __int64 v39; // [rsp+18h] [rbp-78h]
  _BYTE *v42; // [rsp+30h] [rbp-60h]
  __int64 v43; // [rsp+38h] [rbp-58h]
  __m128i v44; // [rsp+40h] [rbp-50h] BYREF
  __m128i v45[4]; // [rsp+50h] [rbp-40h] BYREF

  v5 = *(_QWORD *)(a3 + 24);
  v39 = a3 + 8;
  v36 = (_QWORD *)(a5 + 8);
  while ( v39 != v5 )
  {
    v6 = *(__int64 **)(v5 + 32);
    if ( (*(_BYTE *)(*v6 + 80) & 1) == 0 )
      return 0;
    if ( !(unsigned int)sub_B92110(*v6) )
      return 0;
    v7 = sub_E02A50(**(_QWORD **)(v5 + 32), *(_QWORD *)(v5 + 40) + a4, *a1);
    v43 = (__int64)v7;
    v42 = v8;
    if ( !v7 )
      return 0;
    v9 = sub_BD5D20((__int64)v7);
    v10 = a1[51];
    v11 = (char *)v9;
    v13 = v12;
    if ( a1[50] != v10 )
    {
      v14 = a1[50];
      while ( !sub_1099960(v14, v11, v13) )
      {
        v14 += 72;
        if ( v10 == v14 )
          goto LABEL_10;
      }
      return 0;
    }
LABEL_10:
    v16 = sub_BD5D20(v43);
    if ( v17 == 18
      && !(*(_QWORD *)v16 ^ 0x75705F6178635F5FLL | *((_QWORD *)v16 + 1) ^ 0x75747269765F6572LL)
      && *((_WORD *)v16 + 8) == 27745 )
    {
      goto LABEL_20;
    }
    if ( (_BYTE)qword_4FF90E8 )
      goto LABEL_22;
    if ( !sub_B2FC80(v43) )
    {
      v18 = *(_QWORD *)(v43 + 80);
      if ( !v18 )
        BUG();
      v19 = *(_QWORD *)(v18 + 24) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v19 == v18 + 24 )
        goto LABEL_54;
      if ( !v19 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v19 - 24) - 30 > 0xA )
LABEL_54:
        BUG();
      if ( *(_BYTE *)(v19 - 24) == 36 )
        goto LABEL_20;
LABEL_22:
      v20 = 0;
      if ( *v42 < 4u )
        v20 = (__int64)v42;
      sub_26FACF0((__int64)&v44, v20, v5 + 32);
      v21 = (__m128i *)a2[1];
      if ( v21 == (__m128i *)a2[2] )
      {
        sub_26FEC80(a2, v21, &v44);
      }
      else
      {
        if ( v21 )
        {
          *v21 = _mm_loadu_si128(&v44);
          v21[1] = _mm_loadu_si128(v45);
          v21 = (__m128i *)a2[1];
        }
        a2[1] = (unsigned __int64)&v21[2];
      }
      goto LABEL_20;
    }
    if ( !a5 )
      goto LABEL_22;
    sub_B2F930(&v44, v43);
    v22 = sub_B2F650(v44.m128i_i64[0], v44.m128i_i64[1]);
    if ( (__m128i *)v44.m128i_i64[0] != v45 )
      j_j___libc_free_0(v44.m128i_u64[0]);
    v23 = sub_BD5D20(v43);
    v25 = sub_B2F650((__int64)v23, v24);
    v26 = *(_QWORD **)(a5 + 16);
    if ( !v26 )
    {
      v32 = *(unsigned __int8 *)(a5 + 343);
      goto LABEL_47;
    }
    v27 = v36;
    v28 = *(_QWORD **)(a5 + 16);
    do
    {
      while ( 1 )
      {
        v29 = v28[2];
        v30 = v28[3];
        if ( v22 <= v28[4] )
          break;
        v28 = (_QWORD *)v28[3];
        if ( !v30 )
          goto LABEL_36;
      }
      v27 = v28;
      v28 = (_QWORD *)v28[2];
    }
    while ( v29 );
LABEL_36:
    v31 = *(unsigned __int8 *)(a5 + 343);
    if ( v36 == v27 || v22 < v27[4] )
    {
      v32 = *(unsigned __int8 *)(a5 + 343);
      if ( v25 == v22 )
        goto LABEL_47;
    }
    else
    {
      v32 = v31 | (unsigned __int64)(v27 + 4) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v31 & 0xFFFFFFFFFFFFFFF8LL | (unsigned __int64)(v27 + 4) & 0xFFFFFFFFFFFFFFF8LL || v25 == v22 )
        goto LABEL_47;
      v32 = *(unsigned __int8 *)(a5 + 343);
    }
    v33 = v36;
    do
    {
      while ( 1 )
      {
        v34 = v26[2];
        v35 = v26[3];
        if ( v25 <= v26[4] )
          break;
        v26 = (_QWORD *)v26[3];
        if ( !v35 )
          goto LABEL_43;
      }
      v33 = v26;
      v26 = (_QWORD *)v26[2];
    }
    while ( v34 );
LABEL_43:
    if ( v36 != v33 && v25 >= v33[4] )
      v32 = v31 | (unsigned __int64)(v33 + 4) & 0xFFFFFFFFFFFFFFF8LL;
LABEL_47:
    if ( !(unsigned __int8)sub_26F5F30(v32) )
      goto LABEL_22;
LABEL_20:
    v5 = sub_220EF30(v5);
  }
  return a2[1] != *a2;
}
