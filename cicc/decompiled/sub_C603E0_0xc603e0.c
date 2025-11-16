// Function: sub_C603E0
// Address: 0xc603e0
//
__int64 __fastcall sub_C603E0(const void ***a1)
{
  char *v2; // rbx
  _BYTE *v3; // rdi
  char *v4; // rcx
  _BYTE *v5; // rax
  _BYTE *v6; // rax
  bool v7; // cf
  char *v8; // rax
  const void **v9; // rax
  __int64 v10; // r8
  char *v11; // rsi
  char *v12; // rdx
  const void *v13; // rcx
  __int64 v15; // rax
  __m128i *v16; // rdx
  __int64 v17; // r13
  __m128i si128; // xmm0
  _BYTE *v19; // rdi
  const void *v20; // rsi
  size_t v21; // r12
  _BYTE *v22; // rax
  __int64 v23; // rax
  __int64 v24[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = (char *)(*a1)[1];
  v3 = **a1;
  if ( v2 )
  {
    v4 = &v2[(_QWORD)v3];
    v5 = v3;
    while ( (unsigned __int8)(*v5 - 48) <= 9u )
    {
      if ( ++v5 == v4 )
        goto LABEL_7;
    }
    v6 = (_BYTE *)(v5 - v4);
    v7 = __CFADD__(v2, v6);
    v8 = &v6[(_QWORD)v2];
    if ( v7 )
      v2 = v8;
  }
LABEL_7:
  if ( (unsigned __int8)sub_C93CC0(v3, v2, 10, v24) )
  {
    v15 = sub_CB72A0(v3, v2);
    v16 = *(__m128i **)(v15 + 32);
    v17 = v15;
    if ( *(_QWORD *)(v15 + 24) - (_QWORD)v16 <= 0x18u )
    {
      v23 = sub_CB6200(v15, "Failed to parse int at : ", 25);
      v19 = *(_BYTE **)(v23 + 32);
      v17 = v23;
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F66710);
      v16[1].m128i_i8[8] = 32;
      v16[1].m128i_i64[0] = 0x3A20746120746E69LL;
      *v16 = si128;
      v19 = (_BYTE *)(*(_QWORD *)(v15 + 32) + 25LL);
      *(_QWORD *)(v15 + 32) = v19;
    }
    v20 = **a1;
    v21 = (size_t)(*a1)[1];
    v22 = *(_BYTE **)(v17 + 24);
    if ( v22 - v19 < v21 )
    {
      v17 = sub_CB6200(v17, v20, v21);
      v22 = *(_BYTE **)(v17 + 24);
      v19 = *(_BYTE **)(v17 + 32);
    }
    else if ( v21 )
    {
      memcpy(v19, v20, v21);
      v22 = *(_BYTE **)(v17 + 24);
      v19 = (_BYTE *)(v21 + *(_QWORD *)(v17 + 32));
      *(_QWORD *)(v17 + 32) = v19;
    }
    if ( v22 == v19 )
    {
      sub_CB6200(v17, "\n", 1);
    }
    else
    {
      *v19 = 10;
      ++*(_QWORD *)(v17 + 32);
    }
    return -1;
  }
  else
  {
    v9 = *a1;
    v10 = v24[0];
    v11 = 0;
    v12 = (char *)(*a1)[1];
    v13 = **a1;
    if ( v12 >= v2 )
    {
      v12 = v2;
      v11 = (char *)((_BYTE *)(*a1)[1] - v2);
    }
    v9[1] = v11;
    *v9 = &v12[(_QWORD)v13];
  }
  return v10;
}
