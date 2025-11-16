// Function: sub_1DF3220
// Address: 0x1df3220
//
__int64 __fastcall sub_1DF3220(__m128i *a1, _QWORD *a2)
{
  _QWORD *v4; // rsi
  __int64 (*v5)(); // rdx
  __int64 v6; // rax
  __int64 (*v7)(); // rdx
  __int64 v8; // rax
  const __m128i *v9; // rax
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // r13
  __int8 v19; // al
  __int64 v20; // rdi
  __int64 (*v21)(); // rax
  __int64 v23; // r13
  _QWORD *v24; // rbx
  unsigned int v25; // r14d
  int v26; // eax

  v4 = (_QWORD *)a2[2];
  a1[14].m128i_i64[1] = (__int64)v4;
  v5 = *(__int64 (**)())(*v4 + 40LL);
  v6 = 0;
  if ( v5 != sub_1D00B00 )
  {
    v6 = ((__int64 (__fastcall *)(_QWORD *))v5)(v4);
    v4 = (_QWORD *)a1[14].m128i_i64[1];
  }
  a1[15].m128i_i64[0] = v6;
  v7 = *(__int64 (**)())(*v4 + 112LL);
  v8 = 0;
  if ( v7 != sub_1D00B10 )
  {
    v8 = ((__int64 (__fastcall *)(_QWORD *))v7)(v4);
    v4 = (_QWORD *)a1[14].m128i_i64[1];
  }
  a1[15].m128i_i64[1] = v8;
  v9 = (const __m128i *)v4[20];
  a1[16] = _mm_loadu_si128(v9);
  a1[17] = _mm_loadu_si128(v9 + 1);
  a1[18] = _mm_loadu_si128(v9 + 2);
  a1[19] = _mm_loadu_si128(v9 + 3);
  a1[20].m128i_i64[0] = v9[4].m128i_i64[0];
  sub_1F4B6B0(&a1[22].m128i_u64[1], v4);
  v10 = (__int64 *)a1->m128i_i64[1];
  a1[20].m128i_i64[1] = a2[5];
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
LABEL_24:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_4FC6A0C )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_24;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_4FC6A0C);
  v14 = (__int64 *)a1->m128i_i64[1];
  a1[21].m128i_i64[0] = v13;
  v15 = *v14;
  v16 = v14[1];
  if ( v15 == v16 )
LABEL_25:
    BUG();
  while ( *(_UNKNOWN **)v15 != &unk_4FC820C )
  {
    v15 += 16;
    if ( v16 == v15 )
      goto LABEL_25;
  }
  v17 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v15 + 8) + 104LL))(*(_QWORD *)(v15 + 8), &unk_4FC820C);
  a1[22].m128i_i64[0] = 0;
  a1[21].m128i_i64[1] = v17;
  v18 = *a2 + 112LL;
  v19 = sub_1560180(v18, 34);
  if ( !v19 )
    v19 = sub_1560180(v18, 17);
  v20 = a1[15].m128i_i64[0];
  a1[40].m128i_i8[0] = v19;
  v21 = *(__int64 (**)())(*(_QWORD *)v20 + 488LL);
  if ( v21 == sub_1DF1850 )
    return 0;
  if ( !(unsigned __int8)v21() )
    return 0;
  v23 = a2[41];
  v24 = a2 + 40;
  if ( v24 == (_QWORD *)v23 )
    return 0;
  v25 = 0;
  do
  {
    v26 = sub_1DF2010((__int64)a1, v23);
    v23 = *(_QWORD *)(v23 + 8);
    v25 |= v26;
  }
  while ( v24 != (_QWORD *)v23 );
  return v25;
}
