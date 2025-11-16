// Function: sub_8D4EC0
// Address: 0x8d4ec0
//
__m128i *__fastcall sub_8D4EC0(__int64 a1, __int64 a2, __m128i *a3)
{
  int v4; // r14d
  __int64 v5; // r12
  const __m128i *v6; // rbx
  char v7; // al
  char v8; // di
  __m128i *v9; // r13
  __int8 v10; // al
  unsigned int v12; // eax
  unsigned int v13; // [rsp+Ch] [rbp-34h]

  v4 = 0;
  v5 = a2;
  v6 = (const __m128i *)a1;
  if ( (*(_BYTE *)(a1 + 140) & 0xFB) == 8 )
    v4 = sub_8D4C10(a1, dword_4F077C4 != 2);
  v7 = *(_BYTE *)(a2 + 140);
  if ( (v7 & 0xFB) == 8 )
  {
    v12 = sub_8D4C10(a2, dword_4F077C4 != 2);
    v8 = *(_BYTE *)(a1 + 140);
    v13 = v12;
    v7 = *(_BYTE *)(a2 + 140);
    if ( v8 != 12 )
      goto LABEL_6;
  }
  else
  {
    v8 = *(_BYTE *)(a1 + 140);
    v13 = 0;
    if ( v8 != 12 )
      goto LABEL_8;
  }
  do
  {
    v6 = (const __m128i *)v6[10].m128i_i64[0];
    v8 = v6[8].m128i_i8[12];
  }
  while ( v8 == 12 );
LABEL_6:
  if ( v7 == 12 )
  {
    do
      v5 = *(_QWORD *)(v5 + 160);
    while ( *(_BYTE *)(v5 + 140) == 12 );
  }
LABEL_8:
  if ( a3 == v6 )
  {
    v9 = a3;
    return sub_73C570(v9, v4 | v13);
  }
  v9 = (__m128i *)sub_7259C0(v8);
  sub_73C230(v6, v9);
  v10 = v6[8].m128i_i8[12];
  if ( v10 != 8 )
  {
    if ( v10 == 13 )
    {
      v9[10].m128i_i64[1] = sub_8D4EC0(v6[10].m128i_i64[1], *(_QWORD *)(v5 + 168), a3);
      return sub_73C570(v9, v4 | v13);
    }
    if ( v10 != 6 )
      sub_721090();
  }
  v9[10].m128i_i64[0] = sub_8D4EC0(v6[10].m128i_i64[0], *(_QWORD *)(v5 + 160), a3);
  return sub_73C570(v9, v4 | v13);
}
