// Function: sub_2CE0EA0
// Address: 0x2ce0ea0
//
__m128i *__fastcall sub_2CE0EA0(_QWORD *a1, const __m128i *a2)
{
  _QWORD *v2; // r8
  _QWORD *v3; // rbx
  unsigned __int64 v4; // r14
  _QWORD *v5; // rax
  unsigned __int64 v6; // rdx
  bool v7; // cl
  _QWORD *v8; // r9
  char v9; // r12
  __m128i *v10; // rbx
  __int64 v12; // rax
  unsigned __int64 v13; // rax
  _QWORD *v14; // [rsp+0h] [rbp-40h]
  _QWORD *v15; // [rsp+8h] [rbp-38h]

  v2 = a1 + 1;
  v3 = (_QWORD *)a1[2];
  if ( !v3 )
  {
    if ( v2 == (_QWORD *)a1[3] )
    {
      v8 = a1 + 1;
      v9 = 1;
      goto LABEL_14;
    }
    v4 = a2->m128i_i64[0];
    v3 = a1 + 1;
LABEL_18:
    v12 = sub_220EF80((__int64)v3);
    v8 = v3;
    v2 = a1 + 1;
    v6 = *(_QWORD *)(v12 + 32);
    v3 = (_QWORD *)v12;
    goto LABEL_10;
  }
  v4 = a2->m128i_i64[0];
  while ( 1 )
  {
    v6 = v3[4];
    v7 = v6 > v4;
    if ( v4 == v6 )
      v7 = a2->m128i_i64[1] < v3[5];
    v5 = (_QWORD *)v3[3];
    if ( v7 )
      v5 = (_QWORD *)v3[2];
    if ( !v5 )
      break;
    v3 = v5;
  }
  v8 = v3;
  if ( v7 )
  {
    if ( (_QWORD *)a1[3] == v3 )
    {
      v8 = v3;
      v9 = 1;
      if ( v2 == v3 )
        goto LABEL_14;
      goto LABEL_20;
    }
    goto LABEL_18;
  }
LABEL_10:
  if ( v6 == v4 )
  {
    if ( v3[5] >= a2->m128i_i64[1] )
      return (__m128i *)v3;
  }
  else if ( v6 >= v4 )
  {
    return (__m128i *)v3;
  }
  if ( v8 )
  {
    v9 = 1;
    if ( v2 == v8 )
    {
LABEL_14:
      v14 = v8;
      v15 = v2;
      v10 = (__m128i *)sub_22077B0(0x30u);
      v10[2] = _mm_loadu_si128(a2);
      sub_220F040(v9, (__int64)v10, v14, v15);
      ++a1[5];
      return v10;
    }
LABEL_20:
    v13 = v8[4];
    v9 = v4 < v13;
    if ( v4 == v13 )
      v9 = a2->m128i_i64[1] < v8[5];
    goto LABEL_14;
  }
  return 0;
}
