// Function: sub_33E4710
// Address: 0x33e4710
//
__m128i *__fastcall sub_33E4710(_QWORD *a1, const __m128i *a2)
{
  _QWORD *v2; // r8
  __int64 v5; // rbx
  unsigned __int16 v6; // r14
  unsigned __int64 v7; // rsi
  unsigned __int16 v8; // dx
  unsigned __int64 v9; // rdi
  bool v10; // cl
  __int64 v11; // rax
  _QWORD *v12; // r9
  char v13; // r12
  __m128i *v14; // rbx
  __int64 v16; // rax
  unsigned __int16 v17; // ax
  _QWORD *v18; // [rsp+0h] [rbp-40h]
  _QWORD *v19; // [rsp+8h] [rbp-38h]
  _QWORD *v20; // [rsp+8h] [rbp-38h]

  v2 = a1 + 1;
  v5 = a1[2];
  if ( !v5 )
  {
    if ( v2 == (_QWORD *)a1[3] )
    {
      v12 = a1 + 1;
      v13 = 1;
      goto LABEL_14;
    }
    v6 = a2->m128i_i16[0];
    v5 = (__int64)(a1 + 1);
    v7 = a2->m128i_u64[1];
LABEL_18:
    v20 = v2;
    v16 = sub_220EF80(v5);
    v12 = (_QWORD *)v5;
    v2 = v20;
    v8 = *(_WORD *)(v16 + 32);
    v9 = *(_QWORD *)(v16 + 40);
    v5 = v16;
    goto LABEL_10;
  }
  v6 = a2->m128i_i16[0];
  v7 = a2->m128i_u64[1];
  while ( 1 )
  {
    v8 = *(_WORD *)(v5 + 32);
    v9 = *(_QWORD *)(v5 + 40);
    v10 = v6 < v8;
    if ( v6 == v8 )
      v10 = v7 < v9;
    v11 = *(_QWORD *)(v5 + 24);
    if ( v10 )
      v11 = *(_QWORD *)(v5 + 16);
    if ( !v11 )
      break;
    v5 = v11;
  }
  v12 = (_QWORD *)v5;
  if ( v10 )
  {
    if ( a1[3] == v5 )
    {
      v12 = (_QWORD *)v5;
      v13 = 1;
      if ( v2 == (_QWORD *)v5 )
        goto LABEL_14;
      goto LABEL_20;
    }
    goto LABEL_18;
  }
LABEL_10:
  if ( v8 == v6 )
  {
    if ( v7 <= v9 )
      return (__m128i *)v5;
  }
  else if ( v8 >= v6 )
  {
    return (__m128i *)v5;
  }
  if ( v12 )
  {
    v13 = 1;
    if ( v2 == v12 )
    {
LABEL_14:
      v18 = v12;
      v19 = v2;
      v14 = (__m128i *)sub_22077B0(0x30u);
      v14[2] = _mm_loadu_si128(a2);
      sub_220F040(v13, (__int64)v14, v18, v19);
      ++a1[5];
      return v14;
    }
LABEL_20:
    v17 = *((_WORD *)v12 + 16);
    v13 = v6 < v17;
    if ( v6 == v17 )
      v13 = v7 < v12[5];
    goto LABEL_14;
  }
  return 0;
}
