// Function: sub_854590
// Address: 0x854590
//
_BOOL8 __fastcall sub_854590(int a1)
{
  _BOOL4 v1; // r14d
  __int64 v3; // rax
  bool v4; // zf
  _QWORD *m128i_i64; // r13
  __m128i *v6; // rdi
  _QWORD *v7; // rax
  __m128i *v8; // rax
  __int64 *v9; // rsi
  __m128i *v10; // rdx

  v1 = 0;
  if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 12) & 4) != 0 )
    return v1;
  sub_853C60(1);
  v3 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  v4 = a1 == 0;
  m128i_i64 = 0;
  v6 = *(__m128i **)(v3 + 440);
  if ( v4 )
  {
    if ( v6 )
    {
      if ( qword_4F074B0 )
        sub_854000(v6);
      v6 = 0;
    }
  }
  else if ( v6 )
  {
    v7 = *(_QWORD **)(v3 + 440);
    do
    {
      m128i_i64 = v7;
      v7 = (_QWORD *)*v7;
    }
    while ( v7 );
  }
  v8 = (__m128i *)qword_4D03E88;
  if ( !qword_4D03E88 )
  {
    *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 440) = v6;
    return v6 != 0;
  }
  v9 = 0;
  do
  {
    while ( 1 )
    {
      v10 = v8;
      v8 = (__m128i *)v8->m128i_i64[0];
      if ( *(_DWORD *)(v10->m128i_i64[1] + 12) == 1 )
        break;
      v9 = (__int64 *)v10;
LABEL_10:
      if ( !v8 )
        goto LABEL_18;
    }
    if ( v9 )
      *v9 = (__int64)v8;
    else
      qword_4D03E88 = v8;
    v10->m128i_i64[0] = 0;
    if ( !v6 )
      v6 = v10;
    if ( !m128i_i64 )
    {
      m128i_i64 = v10->m128i_i64;
      goto LABEL_10;
    }
    *m128i_i64 = v10;
    m128i_i64 = v10->m128i_i64;
  }
  while ( v8 );
LABEL_18:
  v1 = v6 != 0;
  v4 = qword_4D03E88 == 0;
  *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 440) = v6;
  if ( v4 )
    return v1;
  sub_854430();
  return v6 != 0;
}
