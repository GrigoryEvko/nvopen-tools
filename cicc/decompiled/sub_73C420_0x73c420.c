// Function: sub_73C420
// Address: 0x73c420
//
__m128i *__fastcall sub_73C420(const __m128i *a1, __int64 a2)
{
  __m128i *v2; // r14
  __m128i *v3; // r13
  const __m128i *v4; // rbx
  __int32 v5; // r12d
  __m128i *v6; // r15

  v2 = 0;
  v3 = 0;
  v4 = a1;
  if ( a1[8].m128i_i8[12] != 12 )
    goto LABEL_15;
LABEL_2:
  v5 = 0;
  do
  {
    while ( v4[8].m128i_i8[14] >= 0 || v5 || HIDWORD(qword_4F077B4) && qword_4F077A8 <= 0x9C3Fu )
    {
      v4 = (const __m128i *)v4[10].m128i_i64[0];
      if ( v4[8].m128i_i8[12] != 12 )
        goto LABEL_9;
    }
    v5 = v4[8].m128i_i32[2];
    v4 = (const __m128i *)v4[10].m128i_i64[0];
  }
  while ( v4[8].m128i_i8[12] == 12 );
LABEL_9:
  v6 = (__m128i *)sub_7259C0(8);
  sub_73C230(v4, v6);
  sub_72A160((__int64)v6);
  if ( v5 )
  {
    v6[8].m128i_i8[14] |= 0x80u;
    v6[8].m128i_i32[2] = v5;
  }
  if ( v2 )
  {
LABEL_12:
    v3[10].m128i_i64[0] = (__int64)v6;
    goto LABEL_13;
  }
  while ( 1 )
  {
    v2 = v6;
LABEL_13:
    v4 = (const __m128i *)v4[10].m128i_i64[0];
    if ( !(unsigned int)sub_8D3410(v4) )
      break;
    v3 = v6;
    if ( v4[8].m128i_i8[12] == 12 )
      goto LABEL_2;
LABEL_15:
    v6 = (__m128i *)sub_7259C0(8);
    sub_73C230(v4, v6);
    sub_72A160((__int64)v6);
    if ( v2 )
      goto LABEL_12;
  }
  v6[10].m128i_i64[0] = a2;
  return v2;
}
