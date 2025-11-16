// Function: sub_278AD50
// Address: 0x278ad50
//
_BYTE *__fastcall sub_278AD50(_BYTE *a1, __int64 a2, __int64 (__fastcall *a3)(__int64, char *, __int64), __int64 a4)
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
  __m128i si128; // xmm0
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rdi
  _BYTE *result; // rax
  unsigned __int64 v24; // rax

  v6 = a3(a4, "GVNPass]", 7);
  v8 = *(_BYTE **)(a2 + 32);
  v9 = (unsigned __int8 *)v6;
  v10 = *(_QWORD *)(a2 + 24);
  v11 = v7;
  if ( v10 - (unsigned __int64)v8 < v7 )
  {
    sub_CB6200(a2, v9, v7);
    v8 = *(_BYTE **)(a2 + 32);
    v10 = *(_QWORD *)(a2 + 24);
  }
  else if ( v7 )
  {
    memcpy(v8, v9, v7);
    v24 = *(_QWORD *)(a2 + 24);
    v8 = (_BYTE *)(v11 + *(_QWORD *)(a2 + 32));
    *(_QWORD *)(a2 + 32) = v8;
    if ( (unsigned __int64)v8 < v24 )
      goto LABEL_4;
    goto LABEL_38;
  }
  if ( (unsigned __int64)v8 < v10 )
  {
LABEL_4:
    *(_QWORD *)(a2 + 32) = v8 + 1;
    *v8 = 60;
    goto LABEL_5;
  }
LABEL_38:
  sub_CB5D20(a2, 60);
LABEL_5:
  if ( !a1[1] )
    goto LABEL_11;
  v12 = *(_QWORD *)(a2 + 32);
  v13 = a2;
  if ( !*a1 )
  {
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v12) > 2 )
    {
      *(_BYTE *)(v12 + 2) = 45;
      *(_WORD *)v12 = 28526;
      v12 = *(_QWORD *)(a2 + 32) + 3LL;
      *(_QWORD *)(a2 + 32) = v12;
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v12) > 3 )
        goto LABEL_10;
      goto LABEL_41;
    }
    v13 = sub_CB6200(a2, "no-", 3u);
    v12 = *(_QWORD *)(v13 + 32);
  }
  if ( (unsigned __int64)(*(_QWORD *)(v13 + 24) - v12) > 3 )
  {
LABEL_10:
    *(_DWORD *)v12 = 996504176;
    *(_QWORD *)(v13 + 32) += 4LL;
    goto LABEL_11;
  }
LABEL_41:
  sub_CB6200(v13, (unsigned __int8 *)"pre;", 4u);
LABEL_11:
  if ( !a1[5] )
    goto LABEL_17;
  v14 = *(_QWORD *)(a2 + 32);
  v15 = a2;
  if ( !a1[4] )
  {
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v14) > 2 )
    {
      *(_BYTE *)(v14 + 2) = 45;
      *(_WORD *)v14 = 28526;
      v14 = *(_QWORD *)(a2 + 32) + 3LL;
      *(_QWORD *)(a2 + 32) = v14;
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v14) > 8 )
        goto LABEL_16;
      goto LABEL_42;
    }
    v15 = sub_CB6200(a2, "no-", 3u);
    v14 = *(_QWORD *)(v15 + 32);
  }
  if ( (unsigned __int64)(*(_QWORD *)(v15 + 24) - v14) > 8 )
  {
LABEL_16:
    *(_BYTE *)(v14 + 8) = 59;
    *(_QWORD *)v14 = 0x6572702D64616F6CLL;
    *(_QWORD *)(v15 + 32) += 9LL;
    goto LABEL_17;
  }
LABEL_42:
  sub_CB6200(v15, (unsigned __int8 *)"load-pre;", 9u);
LABEL_17:
  if ( !a1[9] )
    goto LABEL_23;
  v16 = *(_QWORD *)(a2 + 32);
  v17 = a2;
  if ( !a1[8] )
  {
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v16) > 2 )
    {
      *(_BYTE *)(v16 + 2) = 45;
      *(_WORD *)v16 = 28526;
      v16 = *(_QWORD *)(a2 + 32) + 3LL;
      *(_QWORD *)(a2 + 32) = v16;
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v16) > 0x17 )
        goto LABEL_22;
      goto LABEL_45;
    }
    v17 = sub_CB6200(a2, "no-", 3u);
    v16 = *(_QWORD *)(v17 + 32);
  }
  if ( (unsigned __int64)(*(_QWORD *)(v17 + 24) - v16) > 0x17 )
  {
LABEL_22:
    si128 = _mm_load_si128((const __m128i *)&xmmword_4393B70);
    *(_QWORD *)(v16 + 16) = 0x3B6572702D64616FLL;
    *(__m128i *)v16 = si128;
    *(_QWORD *)(v17 + 32) += 24LL;
    goto LABEL_23;
  }
LABEL_45:
  sub_CB6200(v17, "split-backedge-load-pre;", 0x18u);
LABEL_23:
  if ( !a1[11] )
    goto LABEL_29;
  v19 = *(_QWORD *)(a2 + 32);
  v20 = a2;
  if ( !a1[10] )
  {
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v19) > 2 )
    {
      *(_BYTE *)(v19 + 2) = 45;
      *(_WORD *)v19 = 28526;
      v19 = *(_QWORD *)(a2 + 32) + 3LL;
      *(_QWORD *)(a2 + 32) = v19;
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v19) > 6 )
        goto LABEL_28;
      goto LABEL_44;
    }
    v20 = sub_CB6200(a2, "no-", 3u);
    v19 = *(_QWORD *)(v20 + 32);
  }
  if ( (unsigned __int64)(*(_QWORD *)(v20 + 24) - v19) > 6 )
  {
LABEL_28:
    *(_DWORD *)v19 = 1684890989;
    *(_WORD *)(v19 + 4) = 28773;
    *(_BYTE *)(v19 + 6) = 59;
    *(_QWORD *)(v20 + 32) += 7LL;
    goto LABEL_29;
  }
LABEL_44:
  sub_CB6200(v20, "memdep;", 7u);
LABEL_29:
  if ( !a1[13] )
    goto LABEL_35;
  v21 = *(_QWORD *)(a2 + 32);
  v22 = a2;
  if ( !a1[12] )
  {
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v21) > 2 )
    {
      *(_BYTE *)(v21 + 2) = 45;
      *(_WORD *)v21 = 28526;
      v21 = *(_QWORD *)(a2 + 32) + 3LL;
      *(_QWORD *)(a2 + 32) = v21;
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v21) > 8 )
        goto LABEL_34;
      goto LABEL_43;
    }
    v22 = sub_CB6200(a2, "no-", 3u);
    v21 = *(_QWORD *)(v22 + 32);
  }
  if ( (unsigned __int64)(*(_QWORD *)(v22 + 24) - v21) > 8 )
  {
LABEL_34:
    *(_BYTE *)(v21 + 8) = 97;
    *(_QWORD *)v21 = 0x737379726F6D656DLL;
    *(_QWORD *)(v22 + 32) += 9LL;
    goto LABEL_35;
  }
LABEL_43:
  sub_CB6200(v22, (unsigned __int8 *)"memoryssa", 9u);
LABEL_35:
  result = *(_BYTE **)(a2 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(a2 + 24) )
    return (_BYTE *)sub_CB5D20(a2, 62);
  *(_QWORD *)(a2 + 32) = result + 1;
  *result = 62;
  return result;
}
