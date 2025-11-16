// Function: sub_18A8930
// Address: 0x18a8930
//
__m128i *__fastcall sub_18A8930(_QWORD *a1, const __m128i *a2)
{
  _QWORD *v2; // r8
  _QWORD *v3; // r12
  unsigned __int64 v4; // rbx
  _QWORD *v5; // rax
  char v6; // cl
  unsigned __int64 v7; // rdx
  _QWORD *v8; // rcx
  __m128i *v9; // r15
  _BOOL4 v10; // r9d
  __int64 v12; // rax
  _BOOL4 v13; // [rsp+4h] [rbp-3Ch]
  _QWORD *v14; // [rsp+8h] [rbp-38h]

  v2 = a1 + 1;
  v3 = (_QWORD *)a1[2];
  if ( !v3 )
  {
    v3 = a1 + 1;
    if ( v2 == (_QWORD *)a1[3] )
    {
      v10 = 1;
      goto LABEL_18;
    }
    v4 = a2->m128i_i64[0];
LABEL_21:
    v12 = sub_220EF80(v3);
    v2 = a1 + 1;
    v7 = *(_QWORD *)(v12 + 32);
    if ( v4 <= v7 )
    {
      v8 = v3;
      v3 = (_QWORD *)v12;
      goto LABEL_11;
    }
    goto LABEL_17;
  }
  v4 = a2->m128i_i64[0];
  while ( 1 )
  {
    v7 = v3[4];
    if ( v4 < v7 || v4 == v7 && a2->m128i_i64[1] < v3[5] )
      break;
    v5 = (_QWORD *)v3[3];
    v6 = 0;
    if ( !v5 )
      goto LABEL_9;
LABEL_6:
    v3 = v5;
  }
  v5 = (_QWORD *)v3[2];
  v6 = 1;
  if ( v5 )
    goto LABEL_6;
LABEL_9:
  if ( v6 )
  {
    if ( (_QWORD *)a1[3] == v3 )
      goto LABEL_17;
    goto LABEL_21;
  }
  v8 = v3;
  if ( v4 > v7 )
    goto LABEL_17;
LABEL_11:
  if ( v4 != v7 || v3[5] >= a2->m128i_i64[1] )
    return (__m128i *)v3;
  if ( !v8 )
    return 0;
  v3 = v8;
LABEL_17:
  v10 = 1;
  if ( v2 != v3 && v4 >= v3[4] )
  {
    v10 = 0;
    if ( v4 == v3[4] )
      v10 = a2->m128i_i64[1] < v3[5];
  }
LABEL_18:
  v13 = v10;
  v14 = v2;
  v9 = (__m128i *)sub_22077B0(48);
  v9[2] = _mm_loadu_si128(a2);
  sub_220F040(v13, v9, v3, v14);
  ++a1[5];
  return v9;
}
