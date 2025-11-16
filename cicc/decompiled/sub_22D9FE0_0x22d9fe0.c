// Function: sub_22D9FE0
// Address: 0x22d9fe0
//
__int64 __fastcall sub_22D9FE0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v7; // r12
  __m128i *v8; // rax
  __m128i si128; // xmm0
  const char *v10; // rax
  size_t v11; // rdx
  _BYTE *v12; // rdi
  unsigned __int8 *v13; // rsi
  _BYTE *v14; // rax
  size_t v15; // rbx
  __int64 v16; // rax
  __int64 v17; // r12
  __int64 v18; // r15
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rbx
  __int64 v23; // r13
  __int64 v24; // rdx
  _BYTE *v26; // rax
  __int64 i; // [rsp+8h] [rbp-38h]

  v7 = *a2;
  v8 = *(__m128i **)(*a2 + 32);
  if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v8 <= 0x18u )
  {
    v7 = sub_CB6200(v7, "PHI Values for function: ", 0x19u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_4366BF0);
    v8[1].m128i_i8[8] = 32;
    v8[1].m128i_i64[0] = 0x3A6E6F6974636E75LL;
    *v8 = si128;
    *(_QWORD *)(v7 + 32) += 25LL;
  }
  v10 = sub_BD5D20(a3);
  v12 = *(_BYTE **)(v7 + 32);
  v13 = (unsigned __int8 *)v10;
  v14 = *(_BYTE **)(v7 + 24);
  v15 = v11;
  if ( v14 - v12 < v11 )
  {
    v7 = sub_CB6200(v7, v13, v11);
    v14 = *(_BYTE **)(v7 + 24);
    v12 = *(_BYTE **)(v7 + 32);
  }
  else if ( v11 )
  {
    memcpy(v12, v13, v11);
    v26 = *(_BYTE **)(v7 + 24);
    v12 = (_BYTE *)(*(_QWORD *)(v7 + 32) + v15);
    *(_QWORD *)(v7 + 32) = v12;
    if ( v26 != v12 )
      goto LABEL_6;
    goto LABEL_22;
  }
  if ( v14 != v12 )
  {
LABEL_6:
    *v12 = 10;
    ++*(_QWORD *)(v7 + 32);
    goto LABEL_7;
  }
LABEL_22:
  sub_CB6200(v7, (unsigned __int8 *)"\n", 1u);
LABEL_7:
  v16 = sub_BC1CD0(a4, &unk_4FDBCF8, a3);
  v17 = *(_QWORD *)(a3 + 80);
  v18 = v16 + 8;
  for ( i = a3 + 72; i != v17; v17 = *(_QWORD *)(v17 + 8) )
  {
    v19 = v17 - 24;
    if ( !v17 )
      v19 = 0;
    v20 = sub_AA5930(v19);
    v22 = v21;
    v23 = v20;
    if ( v20 != v21 )
    {
      while ( 1 )
      {
        sub_22D9C50(v18, v23);
        if ( !v23 )
          goto LABEL_18;
        v24 = *(_QWORD *)(v23 + 32);
        if ( !v24 )
          BUG();
        if ( *(_BYTE *)(v24 - 24) != 84 )
          break;
        v23 = v24 - 24;
        if ( v22 == v24 - 24 )
          goto LABEL_19;
      }
      if ( v22 )
      {
        sub_22D9C50(v18, 0);
LABEL_18:
        BUG();
      }
    }
LABEL_19:
    ;
  }
  sub_22D5800(v18, *a2);
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
