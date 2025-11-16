// Function: sub_352A730
// Address: 0x352a730
//
__int64 __fastcall sub_352A730(__m128i *a1, _QWORD *a2)
{
  __int64 v3; // rdi
  __int64 (*v5)(void); // rdx
  __int64 v6; // rax
  __int64 v7; // rax
  _QWORD *v8; // rsi
  const __m128i *v9; // rax
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 *v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 *v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 *v25; // rax
  __int64 (*v26)(); // rax
  __int64 v28; // r13
  _QWORD *v29; // rbx
  unsigned int v30; // r14d
  int v31; // eax

  v3 = a2[2];
  a1[12].m128i_i64[1] = v3;
  v5 = *(__int64 (**)(void))(*(_QWORD *)v3 + 128LL);
  v6 = 0;
  if ( v5 != sub_2DAC790 )
  {
    v6 = v5();
    v3 = a1[12].m128i_i64[1];
  }
  a1[13].m128i_i64[0] = v6;
  v7 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v3 + 200LL))(v3);
  v8 = (_QWORD *)a1[12].m128i_i64[1];
  a1[13].m128i_i64[1] = v7;
  v9 = (const __m128i *)v8[25];
  a1[14] = _mm_loadu_si128(v9);
  a1[15] = _mm_loadu_si128(v9 + 1);
  a1[16] = _mm_loadu_si128(v9 + 2);
  a1[17] = _mm_loadu_si128(v9 + 3);
  a1[18] = _mm_loadu_si128(v9 + 4);
  sub_2FF7BB0(a1 + 42, v8);
  v10 = (__int64 *)a1->m128i_i64[1];
  a1[19].m128i_i64[0] = a2[4];
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
LABEL_37:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_50208AC )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_37;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_50208AC);
  v14 = (__int64 *)a1->m128i_i64[1];
  a1[19].m128i_i64[1] = v13 + 200;
  v15 = *v14;
  v16 = v14[1];
  if ( v15 == v16 )
LABEL_35:
    BUG();
  while ( *(_UNKNOWN **)v15 != &unk_502234C )
  {
    v15 += 16;
    if ( v16 == v15 )
      goto LABEL_35;
  }
  v17 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v15 + 8) + 104LL))(*(_QWORD *)(v15 + 8), &unk_502234C);
  v18 = (__int64 *)a1->m128i_i64[1];
  a1[20].m128i_i64[0] = v17 + 200;
  v19 = *v18;
  v20 = v18[1];
  if ( v19 == v20 )
LABEL_36:
    BUG();
  while ( *(_UNKNOWN **)v19 != &unk_4F87C64 )
  {
    v19 += 16;
    if ( v20 == v19 )
      goto LABEL_36;
  }
  v21 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v19 + 8) + 104LL))(
                      *(_QWORD *)(v19 + 8),
                      &unk_4F87C64)
                  + 176);
  a1[21].m128i_i64[1] = v21;
  if ( v21 )
  {
    v21 = *(_QWORD *)(v21 + 8);
    if ( v21 )
    {
      v22 = (__int64 *)a1->m128i_i64[1];
      v23 = *v22;
      v24 = v22[1];
      if ( v23 == v24 )
LABEL_34:
        BUG();
      while ( *(_UNKNOWN **)v23 != &unk_503BDA8 )
      {
        v23 += 16;
        if ( v24 == v23 )
          goto LABEL_34;
      }
      v25 = (__int64 *)(*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v23 + 8) + 104LL))(
                         *(_QWORD *)(v23 + 8),
                         &unk_503BDA8);
      v21 = sub_3503E60(v25);
    }
  }
  a1[21].m128i_i64[0] = v21;
  a1[20].m128i_i64[1] = 0;
  sub_2F5FFA0((__m128i *)a1[22].m128i_i64, (__int64)a2);
  v26 = *(__int64 (**)())(*(_QWORD *)a1[13].m128i_i64[0] + 712LL);
  if ( v26 == sub_2FDC610 )
    return 0;
  if ( !(unsigned __int8)v26() )
    return 0;
  v28 = a2[41];
  v29 = a2 + 40;
  if ( v29 == (_QWORD *)v28 )
    return 0;
  v30 = 0;
  do
  {
    v31 = sub_35290F0((__int64)a1, v28);
    v28 = *(_QWORD *)(v28 + 8);
    v30 |= v31;
  }
  while ( v29 != (_QWORD *)v28 );
  return v30;
}
