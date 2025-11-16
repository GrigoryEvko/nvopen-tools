// Function: sub_73EC50
// Address: 0x73ec50
//
const __m128i *__fastcall sub_73EC50(const __m128i *a1)
{
  const __m128i *v1; // r12
  char v2; // di
  char v3; // al
  const __m128i *i; // rdx
  bool v5; // cc
  __m128i **v6; // rbx
  __m128i *v7; // rax
  __int8 v8; // al
  __int64 v10; // [rsp+8h] [rbp-18h] BYREF

  v1 = a1;
  v2 = a1[8].m128i_i8[12];
  if ( v2 == 12 )
  {
    v3 = 12;
    if ( (unsigned __int8)(v1[11].m128i_i8[8] - 2) <= 1u )
      return (const __m128i *)v1[10].m128i_i64[0];
  }
  else
  {
    v3 = v2;
  }
  i = v1;
  v5 = (unsigned __int8)v3 <= 0xCu;
  if ( v3 == 12 )
    goto LABEL_9;
LABEL_4:
  if ( !v5 )
  {
    if ( v3 != 13 )
      return v1;
    for ( i = (const __m128i *)i[10].m128i_i64[1]; ; i = (const __m128i *)i[10].m128i_i64[0] )
    {
      v3 = i[8].m128i_i8[12];
      v5 = (unsigned __int8)v3 <= 0xCu;
      if ( v3 != 12 )
        goto LABEL_4;
LABEL_9:
      if ( i[11].m128i_i8[8] == 3 )
        break;
LABEL_7:
      ;
    }
    v6 = (__m128i **)&v10;
    while ( 1 )
    {
      if ( v2 == 12 && v1[11].m128i_i8[8] == 3 )
      {
        *v6 = (__m128i *)v1[10].m128i_i64[0];
        return (const __m128i *)v10;
      }
      v7 = (__m128i *)sub_7259C0(v2);
      *v6 = v7;
      sub_73BCD0(v1, v7, 0);
      v8 = v1[8].m128i_i8[12];
      if ( v8 == 12 )
        goto LABEL_11;
      if ( (unsigned __int8)v8 <= 0xCu )
        break;
      if ( v8 != 13 )
        goto LABEL_19;
      v1 = (const __m128i *)v1[10].m128i_i64[1];
      v6 = (__m128i **)&(*v6)[10].m128i_i64[1];
LABEL_12:
      v2 = v1[8].m128i_i8[12];
    }
    if ( v8 != 6 && v8 != 8 )
LABEL_19:
      sub_721090();
LABEL_11:
    v1 = (const __m128i *)v1[10].m128i_i64[0];
    v6 = (__m128i **)&(*v6)[10];
    goto LABEL_12;
  }
  if ( v3 == 6 || v3 == 8 )
    goto LABEL_7;
  return v1;
}
