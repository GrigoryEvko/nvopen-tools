// Function: sub_F18EE0
// Address: 0xf18ee0
//
__int64 __fastcall sub_F18EE0(__int64 a1, __m128i *a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  _QWORD *v4; // r13
  int v5; // r8d
  __int64 v6; // rax
  __m128i v7; // xmm1
  __int64 v8; // rbx
  unsigned __int64 v9; // rax
  __int8 v10; // al
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rax
  unsigned int v13; // [rsp+Ch] [rbp-34h]

  result = sub_F18D30(a1, (unsigned __int64 *)a2);
  if ( v3 )
  {
    v4 = (_QWORD *)v3;
    v5 = 1;
    if ( result || v3 == a1 + 8 )
      goto LABEL_3;
    v9 = *(_QWORD *)(v3 + 32);
    if ( a2->m128i_i64[0] < v9 )
      goto LABEL_17;
    if ( a2->m128i_i64[0] != v9 )
    {
LABEL_8:
      v5 = 0;
LABEL_3:
      v13 = v5;
      v6 = sub_22077B0(72);
      v7 = _mm_loadu_si128(a2 + 1);
      v8 = v6;
      *(__m128i *)(v6 + 32) = _mm_loadu_si128(a2);
      *(__m128i *)(v6 + 48) = v7;
      *(_QWORD *)(v6 + 64) = a2[2].m128i_i64[0];
      sub_220F040(v13, v6, v4, a1 + 8);
      ++*(_QWORD *)(a1 + 40);
      return v8;
    }
    v10 = a2[1].m128i_i8[8];
    if ( *(_BYTE *)(v3 + 56) )
    {
      if ( !v10 )
        goto LABEL_17;
      v11 = a2->m128i_u64[1];
      v12 = v4[5];
      if ( v11 < v12 || v11 == v12 && a2[1].m128i_i64[0] < v4[6] )
        goto LABEL_17;
      if ( v11 > v12 || v4[6] < a2[1].m128i_i64[0] )
        goto LABEL_8;
    }
    else if ( v10 )
    {
      goto LABEL_8;
    }
    if ( a2[2].m128i_i64[0] >= v4[8] )
      goto LABEL_8;
LABEL_17:
    v5 = 1;
    goto LABEL_3;
  }
  return result;
}
