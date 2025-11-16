// Function: sub_772C20
// Address: 0x772c20
//
__int64 __fastcall sub_772C20(__int64 a1, __int64 a2, __int64 a3, const __m128i **a4, __m128i *a5)
{
  const __m128i *v5; // rax
  __int8 v6; // dl
  __int64 v8; // rdi
  __int64 v9; // rdx
  char v10; // al

  v5 = *a4;
  *a5 = _mm_loadu_si128(*a4);
  a5[1].m128i_i64[0] = v5[1].m128i_i64[0];
  v6 = a5->m128i_i8[0];
  if ( a5->m128i_i8[0] != 48 )
  {
    if ( v6 == 28 )
    {
      a5->m128i_i8[0] = 23;
      a5->m128i_i64[1] = *(_QWORD *)(*(_QWORD *)(v5->m128i_i64[1] + 128) + 128LL);
      return 1;
    }
    if ( v6 != 6 )
      return 1;
    v8 = a5->m128i_i64[1];
    goto LABEL_12;
  }
  v9 = a5->m128i_i64[1];
  v10 = *(_BYTE *)(v9 + 8);
  if ( v10 == 1 )
  {
    a5->m128i_i8[0] = 2;
    a5->m128i_i64[1] = *(_QWORD *)(v9 + 32);
    return 1;
  }
  else
  {
    if ( v10 != 2 )
    {
      if ( v10 )
        sub_721090();
      a5->m128i_i8[0] = 6;
      v8 = *(_QWORD *)(v9 + 32);
      a5->m128i_i64[1] = v8;
LABEL_12:
      a5->m128i_i64[1] = sub_8D21C0(v8);
      return 1;
    }
    a5->m128i_i8[0] = 59;
    a5->m128i_i64[1] = *(_QWORD *)(v9 + 32);
    return 1;
  }
}
