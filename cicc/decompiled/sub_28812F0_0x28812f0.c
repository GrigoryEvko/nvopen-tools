// Function: sub_28812F0
// Address: 0x28812f0
//
_BYTE *__fastcall sub_28812F0(__int64 a1, __int64 a2, __int64 (__fastcall *a3)(__int64, char *, __int64), __int64 a4)
{
  __int64 v6; // rax
  size_t v7; // rdx
  _BYTE *v8; // rdi
  unsigned __int8 *v9; // rsi
  unsigned __int64 v10; // rax
  size_t v11; // r13
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rdi
  _BYTE *v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rdi
  _BYTE *v22; // rax
  __int64 v23; // rdi
  _BYTE *result; // rax
  unsigned __int64 v25; // rax
  __m128i *v26; // rdx
  __int64 v27; // r13
  _BYTE *v28; // rax

  v6 = a3(a4, "LoopUnrollPass]", 14);
  v8 = *(_BYTE **)(a2 + 32);
  v9 = (unsigned __int8 *)v6;
  v10 = *(_QWORD *)(a2 + 24);
  v11 = v7;
  if ( v7 > v10 - (unsigned __int64)v8 )
  {
    sub_CB6200(a2, v9, v7);
    v8 = *(_BYTE **)(a2 + 32);
    v10 = *(_QWORD *)(a2 + 24);
  }
  else if ( v7 )
  {
    memcpy(v8, v9, v7);
    v25 = *(_QWORD *)(a2 + 24);
    v8 = (_BYTE *)(v11 + *(_QWORD *)(a2 + 32));
    *(_QWORD *)(a2 + 32) = v8;
    if ( v25 > (unsigned __int64)v8 )
      goto LABEL_4;
    goto LABEL_41;
  }
  if ( v10 > (unsigned __int64)v8 )
  {
LABEL_4:
    *(_QWORD *)(a2 + 32) = v8 + 1;
    *v8 = 60;
    goto LABEL_5;
  }
LABEL_41:
  sub_CB5D20(a2, 60);
LABEL_5:
  if ( !*(_BYTE *)(a1 + 1) )
    goto LABEL_11;
  v12 = *(_QWORD *)(a2 + 32);
  v13 = a2;
  if ( !*(_BYTE *)a1 )
  {
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v12) > 2 )
    {
      *(_BYTE *)(v12 + 2) = 45;
      *(_WORD *)v12 = 28526;
      v12 = *(_QWORD *)(a2 + 32) + 3LL;
      *(_QWORD *)(a2 + 32) = v12;
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v12) > 7 )
        goto LABEL_10;
      goto LABEL_53;
    }
    v13 = sub_CB6200(a2, "no-", 3u);
    v12 = *(_QWORD *)(v13 + 32);
  }
  if ( (unsigned __int64)(*(_QWORD *)(v13 + 24) - v12) > 7 )
  {
LABEL_10:
    *(_QWORD *)v12 = 0x3B6C616974726170LL;
    *(_QWORD *)(v13 + 32) += 8LL;
    goto LABEL_11;
  }
LABEL_53:
  sub_CB6200(v13, "partial;", 8u);
LABEL_11:
  if ( !*(_BYTE *)(a1 + 3) )
    goto LABEL_17;
  v14 = *(_QWORD *)(a2 + 32);
  v15 = a2;
  if ( !*(_BYTE *)(a1 + 2) )
  {
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v14) > 2 )
    {
      *(_BYTE *)(v14 + 2) = 45;
      *(_WORD *)v14 = 28526;
      v14 = *(_QWORD *)(a2 + 32) + 3LL;
      *(_QWORD *)(a2 + 32) = v14;
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v14) > 7 )
        goto LABEL_16;
      goto LABEL_55;
    }
    v15 = sub_CB6200(a2, "no-", 3u);
    v14 = *(_QWORD *)(v15 + 32);
  }
  if ( (unsigned __int64)(*(_QWORD *)(v15 + 24) - v14) > 7 )
  {
LABEL_16:
    *(_QWORD *)v14 = 0x3B676E696C656570LL;
    *(_QWORD *)(v15 + 32) += 8LL;
    goto LABEL_17;
  }
LABEL_55:
  sub_CB6200(v15, (unsigned __int8 *)"peeling;", 8u);
LABEL_17:
  if ( !*(_BYTE *)(a1 + 5) )
    goto LABEL_23;
  v16 = *(_QWORD *)(a2 + 32);
  v17 = a2;
  if ( !*(_BYTE *)(a1 + 4) )
  {
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v16) > 2 )
    {
      *(_BYTE *)(v16 + 2) = 45;
      *(_WORD *)v16 = 28526;
      v16 = *(_QWORD *)(a2 + 32) + 3LL;
      *(_QWORD *)(a2 + 32) = v16;
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v16) > 7 )
        goto LABEL_22;
      goto LABEL_51;
    }
    v17 = sub_CB6200(a2, "no-", 3u);
    v16 = *(_QWORD *)(v17 + 32);
  }
  if ( (unsigned __int64)(*(_QWORD *)(v17 + 24) - v16) > 7 )
  {
LABEL_22:
    *(_QWORD *)v16 = 0x3B656D69746E7572LL;
    *(_QWORD *)(v17 + 32) += 8LL;
    goto LABEL_23;
  }
LABEL_51:
  sub_CB6200(v17, "runtime;", 8u);
LABEL_23:
  if ( !*(_BYTE *)(a1 + 7) )
    goto LABEL_29;
  v18 = *(_BYTE **)(a2 + 32);
  v19 = a2;
  if ( !*(_BYTE *)(a1 + 6) )
  {
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v18 > 2u )
    {
      v18[2] = 45;
      *(_WORD *)v18 = 28526;
      v18 = (_BYTE *)(*(_QWORD *)(a2 + 32) + 3LL);
      *(_QWORD *)(a2 + 32) = v18;
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v18 > 0xAu )
        goto LABEL_28;
      goto LABEL_52;
    }
    v19 = sub_CB6200(a2, "no-", 3u);
    v18 = *(_BYTE **)(v19 + 32);
  }
  if ( *(_QWORD *)(v19 + 24) - (_QWORD)v18 > 0xAu )
  {
LABEL_28:
    qmemcpy(v18, "upperbound;", 11);
    *(_QWORD *)(v19 + 32) += 11LL;
    goto LABEL_29;
  }
LABEL_52:
  sub_CB6200(v19, "upperbound;", 0xBu);
LABEL_29:
  if ( !*(_BYTE *)(a1 + 9) )
    goto LABEL_35;
  v20 = *(_QWORD *)(a2 + 32);
  v21 = a2;
  if ( !*(_BYTE *)(a1 + 8) )
  {
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v20) > 2 )
    {
      *(_BYTE *)(v20 + 2) = 45;
      *(_WORD *)v20 = 28526;
      v20 = *(_QWORD *)(a2 + 32) + 3LL;
      *(_QWORD *)(a2 + 32) = v20;
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v20) > 0xF )
        goto LABEL_34;
      goto LABEL_54;
    }
    v21 = sub_CB6200(a2, "no-", 3u);
    v20 = *(_QWORD *)(v21 + 32);
  }
  if ( (unsigned __int64)(*(_QWORD *)(v21 + 24) - v20) > 0xF )
  {
LABEL_34:
    *(__m128i *)v20 = _mm_load_si128((const __m128i *)&xmmword_4397080);
    *(_QWORD *)(v21 + 32) += 16LL;
    goto LABEL_35;
  }
LABEL_54:
  sub_CB6200(v21, "profile-peeling;", 0x10u);
LABEL_35:
  if ( !*(_BYTE *)(a1 + 16) )
    goto LABEL_36;
  v26 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v26 > 0xFu )
  {
    v27 = a2;
    *v26 = _mm_load_si128((const __m128i *)&xmmword_4397090);
    *(_QWORD *)(a2 + 32) += 16LL;
    if ( !*(_BYTE *)(a1 + 16) )
      goto LABEL_47;
LABEL_49:
    sub_CB59D0(v27, *(unsigned int *)(a1 + 12));
    v28 = *(_BYTE **)(v27 + 32);
    if ( (unsigned __int64)v28 < *(_QWORD *)(v27 + 24) )
      goto LABEL_48;
    goto LABEL_50;
  }
  v27 = sub_CB6200(a2, "full-unroll-max=", 0x10u);
  if ( *(_BYTE *)(a1 + 16) )
    goto LABEL_49;
LABEL_47:
  sub_F03F40(v27);
  v28 = *(_BYTE **)(v27 + 32);
  if ( (unsigned __int64)v28 < *(_QWORD *)(v27 + 24) )
  {
LABEL_48:
    *(_QWORD *)(v27 + 32) = v28 + 1;
    *v28 = 59;
    goto LABEL_36;
  }
LABEL_50:
  sub_CB5D20(v27, 59);
LABEL_36:
  v22 = *(_BYTE **)(a2 + 32);
  if ( (unsigned __int64)v22 >= *(_QWORD *)(a2 + 24) )
  {
    v23 = sub_CB5D20(a2, 79);
  }
  else
  {
    v23 = a2;
    *(_QWORD *)(a2 + 32) = v22 + 1;
    *v22 = 79;
  }
  sub_CB59F0(v23, *(int *)(a1 + 20));
  result = *(_BYTE **)(a2 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(a2 + 24) )
    return (_BYTE *)sub_CB5D20(a2, 62);
  *(_QWORD *)(a2 + 32) = result + 1;
  *result = 62;
  return result;
}
