// Function: sub_38E1B70
// Address: 0x38e1b70
//
__m128i *__fastcall sub_38E1B70(__int64 a1, _DWORD *a2, size_t a3)
{
  const char **v4; // r14
  __int64 v5; // rax
  __int64 v6; // r15
  int v7; // eax
  void *v8; // rcx
  __int64 v9; // r12
  const char **v10; // rbx
  size_t v11; // rax
  const char *v12; // rdi
  const char *v13; // r12
  __int64 v14; // r12
  _BYTE *v15; // rax
  __m128i *v16; // rdi
  __int64 v17; // rax
  unsigned __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  __m128i si128; // xmm0
  __m128i v23; // xmm0
  __int64 v24; // rax
  __int64 v25; // rax
  const char **v26; // [rsp+8h] [rbp-48h]
  void *s1; // [rsp+18h] [rbp-38h]
  const char *s1a; // [rsp+18h] [rbp-38h]
  void *s1b; // [rsp+18h] [rbp-38h]

  v4 = *(const char ***)(a1 + 128);
  v5 = 2LL * *(_QWORD *)(a1 + 120);
  v26 = &v4[v5];
  v6 = (v5 * 8) >> 4;
  if ( v5 > 0 )
  {
    while ( 1 )
    {
      v9 = v6 >> 1;
      v10 = &v4[2 * (v6 >> 1)];
      if ( !*v10 )
        break;
      s1a = *v10;
      v11 = strlen(*v10);
      v12 = s1a;
      v8 = (void *)v11;
      if ( a3 >= v11 )
      {
        if ( !v11 )
          goto LABEL_5;
        s1 = (void *)v11;
        v7 = memcmp(v12, a2, v11);
        v8 = s1;
        if ( !v7 )
          goto LABEL_5;
LABEL_12:
        if ( v7 >= 0 )
          goto LABEL_13;
LABEL_7:
        v4 = v10 + 2;
        v6 = v6 - v9 - 1;
        if ( v6 <= 0 )
          goto LABEL_14;
      }
      else
      {
        if ( !a3 )
          goto LABEL_13;
        s1b = (void *)v11;
        v7 = memcmp(v12, a2, a3);
        v8 = s1b;
        if ( v7 )
          goto LABEL_12;
LABEL_6:
        if ( a3 > (unsigned __int64)v8 )
          goto LABEL_7;
LABEL_13:
        v6 >>= 1;
        if ( v9 <= 0 )
          goto LABEL_14;
      }
    }
    v8 = 0;
LABEL_5:
    if ( (void *)a3 == v8 )
      goto LABEL_13;
    goto LABEL_6;
  }
LABEL_14:
  if ( v26 == v4 )
    goto LABEL_17;
  v13 = *v4;
  if ( *v4 )
  {
    if ( strlen(*v4) == a3 && (!a3 || !memcmp(v13, a2, a3)) )
      return (__m128i *)v4[1];
LABEL_17:
    if ( a3 != 4 )
      goto LABEL_18;
LABEL_33:
    if ( *a2 == 1886152040 )
      return xmmword_452E800;
LABEL_18:
    v14 = (__int64)sub_16E8CB0();
    v15 = *(_BYTE **)(v14 + 24);
    if ( *(_BYTE **)(v14 + 16) == v15 )
    {
      v14 = sub_16E7EE0(v14, "'", 1u);
      v16 = *(__m128i **)(v14 + 24);
      v18 = *(_QWORD *)(v14 + 16) - (_QWORD)v16;
      if ( a3 > v18 )
      {
LABEL_20:
        v19 = sub_16E7EE0(v14, (char *)a2, a3);
        v16 = *(__m128i **)(v19 + 24);
        v14 = v19;
        if ( *(_QWORD *)(v19 + 16) - (_QWORD)v16 <= 0x2Eu )
          goto LABEL_21;
        goto LABEL_29;
      }
    }
    else
    {
      *v15 = 39;
      v16 = (__m128i *)(*(_QWORD *)(v14 + 24) + 1LL);
      v17 = *(_QWORD *)(v14 + 16);
      *(_QWORD *)(v14 + 24) = v16;
      v18 = v17 - (_QWORD)v16;
      if ( a3 > v18 )
        goto LABEL_20;
    }
    if ( a3 )
    {
      memcpy(v16, a2, a3);
      v25 = *(_QWORD *)(v14 + 16);
      v16 = (__m128i *)(a3 + *(_QWORD *)(v14 + 24));
      *(_QWORD *)(v14 + 24) = v16;
      v18 = v25 - (_QWORD)v16;
    }
    if ( v18 <= 0x2E )
    {
LABEL_21:
      v14 = sub_16E7EE0(v14, "' is not a recognized processor for this target", 0x2Fu);
      v20 = *(_QWORD *)(v14 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(v14 + 16) - v20) > 0x15 )
      {
LABEL_22:
        si128 = _mm_load_si128((const __m128i *)&xmmword_3F82980);
        *(_DWORD *)(v20 + 16) = 1919906675;
        *(_WORD *)(v20 + 20) = 2601;
        *(__m128i *)v20 = si128;
        *(_QWORD *)(v14 + 24) += 22LL;
        return xmmword_452E800;
      }
LABEL_30:
      sub_16E7EE0(v14, " (ignoring processor)\n", 0x16u);
      return xmmword_452E800;
    }
LABEL_29:
    *v16 = _mm_load_si128((const __m128i *)&xmmword_3F82940);
    v23 = _mm_load_si128((const __m128i *)&xmmword_3F82970);
    qmemcpy(&v16[2], "for this target", 15);
    v16[1] = v23;
    v20 = *(_QWORD *)(v14 + 24) + 47LL;
    v24 = *(_QWORD *)(v14 + 16);
    *(_QWORD *)(v14 + 24) = v20;
    if ( (unsigned __int64)(v24 - v20) > 0x15 )
      goto LABEL_22;
    goto LABEL_30;
  }
  if ( a3 )
  {
    if ( a3 != 4 )
      goto LABEL_18;
    goto LABEL_33;
  }
  return (__m128i *)v4[1];
}
