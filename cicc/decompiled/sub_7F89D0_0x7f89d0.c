// Function: sub_7F89D0
// Address: 0x7f89d0
//
_QWORD *__fastcall sub_7F89D0(char *a1, __int64 *a2, __int64 a3, __m128i *a4)
{
  __int64 v5; // rdi
  __int64 i; // rax
  _QWORD *v7; // r14
  __m128i *v8; // r12
  __m128i *v9; // rax
  __m128i *j; // r15
  __int64 v11; // rbx
  __m128i *v12; // rax

  sub_7F8930(a1, a2);
  v5 = *a2;
  for ( i = *(_QWORD *)(*a2 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v7 = **(_QWORD ***)(i + 168);
  if ( v7 && a4 )
  {
    v8 = (__m128i *)a4[1].m128i_i64[0];
    a4[1].m128i_i64[0] = 0;
    v9 = sub_7F1240(a4, v7[1]);
    v9[1].m128i_i64[0] = (__int64)v8;
    a4 = v9;
    for ( j = v9; ; j = v12 )
    {
      v7 = (_QWORD *)*v7;
      if ( !v7 || !v8 )
        break;
      v11 = v8[1].m128i_i64[0];
      v8[1].m128i_i64[0] = 0;
      v12 = sub_7F1240(v8, v7[1]);
      v12[1].m128i_i64[0] = v11;
      j[1].m128i_i64[0] = (__int64)v12;
      v8 = (__m128i *)v12[1].m128i_i64[0];
    }
    v5 = *a2;
  }
  return sub_7F88E0(v5, a4);
}
