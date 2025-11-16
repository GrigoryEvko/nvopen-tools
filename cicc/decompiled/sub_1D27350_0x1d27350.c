// Function: sub_1D27350
// Address: 0x1d27350
//
__m128i *__fastcall sub_1D27350(_QWORD *a1, const __m128i *a2)
{
  _QWORD *v2; // r8
  _QWORD *v5; // rbx
  unsigned __int8 v6; // r14
  unsigned __int64 v7; // rsi
  unsigned __int8 v8; // dl
  unsigned __int64 v9; // rdi
  bool v10; // cl
  _QWORD *v11; // rax
  _QWORD *v12; // r9
  unsigned int v13; // r12d
  __m128i *v14; // rbx
  __int64 v16; // rax
  unsigned __int8 v17; // al
  _QWORD *v18; // [rsp+0h] [rbp-40h]
  _QWORD *v19; // [rsp+8h] [rbp-38h]
  _QWORD *v20; // [rsp+8h] [rbp-38h]

  v2 = a1 + 1;
  v5 = (_QWORD *)a1[2];
  if ( !v5 )
  {
    if ( v2 == (_QWORD *)a1[3] )
    {
      v12 = a1 + 1;
      v13 = 1;
      goto LABEL_14;
    }
    v6 = a2->m128i_i8[0];
    v5 = a1 + 1;
    v7 = a2->m128i_u64[1];
LABEL_18:
    v20 = v2;
    v16 = sub_220EF80(v5);
    v12 = v5;
    v2 = v20;
    v8 = *(_BYTE *)(v16 + 32);
    v9 = *(_QWORD *)(v16 + 40);
    v5 = (_QWORD *)v16;
    goto LABEL_10;
  }
  v6 = a2->m128i_i8[0];
  v7 = a2->m128i_u64[1];
  while ( 1 )
  {
    v8 = *((_BYTE *)v5 + 32);
    v9 = v5[5];
    v10 = v6 < v8;
    if ( v6 == v8 )
      v10 = v7 < v9;
    v11 = (_QWORD *)v5[3];
    if ( v10 )
      v11 = (_QWORD *)v5[2];
    if ( !v11 )
      break;
    v5 = v11;
  }
  v12 = v5;
  if ( v10 )
  {
    if ( (_QWORD *)a1[3] == v5 )
    {
      v12 = v5;
      v13 = 1;
      if ( v2 == v5 )
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
      v14 = (__m128i *)sub_22077B0(48);
      v14[2] = _mm_loadu_si128(a2);
      sub_220F040(v13, v14, v18, v19);
      ++a1[5];
      return v14;
    }
LABEL_20:
    v17 = *((_BYTE *)v12 + 32);
    LOBYTE(v13) = v6 < v17;
    if ( v6 == v17 )
      LOBYTE(v13) = v7 < v12[5];
    v13 = (unsigned __int8)v13;
    goto LABEL_14;
  }
  return 0;
}
