// Function: sub_7AE360
// Address: 0x7ae360
//
void __fastcall sub_7AE360(__int64 a1)
{
  __m128i *v1; // rbx
  __int64 v2; // rdx
  unsigned __int64 v3; // rax
  __m128i *v4; // rdx
  __int64 v5; // rdx
  __m128i *v6; // rax
  __int64 v7; // r14
  size_t v8; // r15
  char *v9; // rax
  char *v10; // rax
  __int64 v11; // rdx
  __int64 **v12; // rax
  __int64 v13; // rax
  __int64 *v14; // rax
  __m128i *v15; // rax
  __m128i *v16; // rax

  if ( unk_4D03E88 && !(unk_4D03D18 | unk_4D03D20) && (*(_BYTE *)(unk_4F061C0 + 56LL) & 8) == 0 )
  {
    v11 = qword_4F08558;
    if ( qword_4F08558 )
      qword_4F08558 = *(_QWORD *)qword_4F08558;
    else
      v11 = sub_823970(112);
    *(_QWORD *)v11 = 0;
    v12 = (__int64 **)unk_4D03E88;
    *(_QWORD *)(v11 + 40) = 0;
    *(_WORD *)(v11 + 24) = 0;
    *(_QWORD *)(v11 + 28) = 0;
    *(_BYTE *)(v11 + 26) = 3;
    *(_QWORD *)(v11 + 48) = v12;
    *(_QWORD *)(v11 + 8) = v12[7];
    if ( *(_BYTE *)(a1 + 24) )
    {
      do
      {
        *((_BYTE *)v12 + 72) &= ~8u;
        v14 = *v12;
        if ( !v14 )
          break;
        *((_BYTE *)v14 + 72) &= ~8u;
        v12 = (__int64 **)*v14;
      }
      while ( v12 );
    }
    v13 = *(_QWORD *)(v11 + 8);
    *(_DWORD *)(v11 + 28) = 0;
    *(_WORD *)(v11 + 24) = 0;
    *(_QWORD *)(v11 + 16) = v13;
    if ( *(_QWORD *)(a1 + 8) )
      **(_QWORD **)(a1 + 16) = v11;
    else
      *(_QWORD *)(a1 + 8) = v11;
    *(_QWORD *)(a1 + 16) = v11;
    unk_4D03E88 = 0;
  }
  v1 = (__m128i *)qword_4F08558;
  if ( qword_4F08558 )
    qword_4F08558 = *(_QWORD *)qword_4F08558;
  else
    v1 = (__m128i *)sub_823970(112);
  v1->m128i_i64[0] = 0;
  v1[2].m128i_i64[1] = 0;
  v2 = *(_QWORD *)&dword_4F063F8;
  v1[1].m128i_i8[10] = 0;
  v3 = word_4F06418[0];
  v1->m128i_i64[1] = v2;
  v1[1].m128i_i16[4] = v3;
  v1[1].m128i_i64[0] = qword_4F063F0;
  v1[1].m128i_i32[3] = dword_4F06650[0];
  v1[2].m128i_i32[0] = unk_4F0664C;
  v4 = v1;
  if ( !*(_BYTE *)(a1 + 24) )
    v4 = (__m128i *)unk_4F06640;
  v1[2].m128i_i64[1] = (__int64)v4;
  if ( unk_4D03D20 )
  {
    v1[1].m128i_i8[10] = 4;
    v7 = unk_4F06408 - (_QWORD)qword_4F06410;
    v8 = unk_4F06408 - (_QWORD)qword_4F06410 + 1LL;
    v9 = (char *)sub_823970(unk_4F06408 - (_QWORD)qword_4F06410 + 2LL);
    v10 = strncpy(v9, qword_4F06410, v8);
    v10[v7 + 1] = 0;
    v1[3].m128i_i64[0] = (__int64)v10;
    v1[3].m128i_i64[1] = (__int64)&v10[v7];
    goto LABEL_11;
  }
  if ( (unsigned __int16)v3 > 0x12u )
  {
    if ( (_WORD)v3 == 137 )
    {
      v1[1].m128i_i8[10] = 6;
      v1[3].m128i_i64[0] = unk_4F061F0;
      goto LABEL_11;
    }
  }
  else
  {
    v5 = 294914;
    if ( _bittest64(&v5, v3) )
    {
      v1[1].m128i_i8[10] = 1;
      v1[3] = _mm_loadu_si128((const __m128i *)&qword_4D04A00);
      v1[4] = _mm_loadu_si128((const __m128i *)&unk_4D04A10);
      v1[5] = _mm_loadu_si128(&xmmword_4D04A20);
      v1[6] = _mm_loadu_si128((const __m128i *)&unk_4D04A30);
      goto LABEL_11;
    }
  }
  if ( (unsigned __int16)(v3 - 4) <= 3u || (unsigned __int16)(v3 - 181) <= 1u || (_WORD)v3 == 2 )
  {
    v1[1].m128i_i8[10] = 2;
    v6 = (__m128i *)sub_7ADFF0();
    v1[3].m128i_i64[0] = (__int64)v6;
    sub_72A510(xmmword_4F06300, v6);
    if ( *(_QWORD *)(a1 + 8) )
      goto LABEL_12;
LABEL_18:
    *(_QWORD *)(a1 + 8) = v1;
    goto LABEL_13;
  }
  if ( (_WORD)v3 == 8 )
  {
    v1[1].m128i_i8[10] = 8;
    v15 = (__m128i *)sub_7ADFF0();
    v1[3].m128i_i64[0] = (__int64)v15;
    sub_72A510(xmmword_4F06300, v15);
    v16 = (__m128i *)sub_7ADFF0();
    v1[3].m128i_i64[1] = (__int64)v16;
    sub_72A510(xmmword_4F06220, v16);
    v1[4].m128i_i64[0] = (__int64)qword_4F06218;
    v1[4].m128i_i64[1] = *(_QWORD *)(qword_4D04A00 + 8) + 11LL;
    v1[5].m128i_i64[0] = unk_4F06210;
  }
LABEL_11:
  if ( !*(_QWORD *)(a1 + 8) )
    goto LABEL_18;
LABEL_12:
  **(_QWORD **)(a1 + 16) = v1;
LABEL_13:
  *(_QWORD *)(a1 + 16) = v1;
}
