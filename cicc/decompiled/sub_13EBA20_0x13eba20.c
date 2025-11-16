// Function: sub_13EBA20
// Address: 0x13eba20
//
__int64 __fastcall sub_13EBA20(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __m128i *v6; // rdx
  __int64 v7; // r13
  __m128i si128; // xmm0
  __int64 v9; // rax
  size_t v10; // rdx
  _BYTE *v11; // rdi
  const char *v12; // rsi
  unsigned __int64 v13; // rax
  __int64 *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 *v19; // rdx
  __int64 *v20; // r13
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rdi
  __int64 v24; // rbx
  __int64 v25; // rdx
  __int64 v26; // rax
  unsigned __int64 v28; // rax
  __int64 v29; // rax
  size_t v30; // [rsp+8h] [rbp-28h]

  v5 = sub_16BA580(a1, a2, a3);
  v6 = *(__m128i **)(v5 + 24);
  v7 = v5;
  if ( *(_QWORD *)(v5 + 16) - (_QWORD)v6 <= 0x11u )
  {
    v7 = sub_16E7EE0(v5, "LVI for function '", 18);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_428A440);
    v6[1].m128i_i16[0] = 10016;
    *v6 = si128;
    *(_QWORD *)(v5 + 24) += 18LL;
  }
  v9 = sub_1649960(a2);
  v11 = *(_BYTE **)(v7 + 24);
  v12 = (const char *)v9;
  v13 = *(_QWORD *)(v7 + 16) - (_QWORD)v11;
  if ( v13 < v10 )
  {
    v29 = sub_16E7EE0(v7, v12);
    v11 = *(_BYTE **)(v29 + 24);
    v7 = v29;
    v13 = *(_QWORD *)(v29 + 16) - (_QWORD)v11;
  }
  else if ( v10 )
  {
    v30 = v10;
    memcpy(v11, v12, v10);
    v11 = (_BYTE *)(v30 + *(_QWORD *)(v7 + 24));
    v28 = *(_QWORD *)(v7 + 16) - (_QWORD)v11;
    *(_QWORD *)(v7 + 24) = v11;
    if ( v28 > 2 )
      goto LABEL_6;
    goto LABEL_17;
  }
  if ( v13 > 2 )
  {
LABEL_6:
    v11[2] = 10;
    *(_WORD *)v11 = 14887;
    *(_QWORD *)(v7 + 24) += 3LL;
    goto LABEL_7;
  }
LABEL_17:
  sub_16E7EE0(v7, "':\n", 3);
LABEL_7:
  v14 = *(__int64 **)(a1 + 8);
  v15 = *v14;
  v16 = v14[1];
  if ( v15 == v16 )
LABEL_24:
    BUG();
  while ( *(_UNKNOWN **)v15 != &unk_4F99130 )
  {
    v15 += 16;
    if ( v16 == v15 )
      goto LABEL_24;
  }
  v17 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v15 + 8) + 104LL))(*(_QWORD *)(v15 + 8), &unk_4F99130);
  v18 = sub_13EB2D0(v17);
  v19 = *(__int64 **)(a1 + 8);
  v20 = (__int64 *)v18;
  v21 = *v19;
  v22 = v19[1];
  if ( v21 == v22 )
LABEL_23:
    BUG();
  while ( *(_UNKNOWN **)v21 != &unk_4F9E06C )
  {
    v21 += 16;
    if ( v22 == v21 )
      goto LABEL_23;
  }
  v23 = *(_QWORD *)(v21 + 8);
  v24 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v23 + 104LL))(v23, &unk_4F9E06C);
  v26 = sub_16BA580(v23, &unk_4F9E06C, v25);
  sub_13EB990(v20, a2, v24 + 160, v26);
  return 0;
}
