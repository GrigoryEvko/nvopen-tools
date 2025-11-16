// Function: sub_30F9EE0
// Address: 0x30f9ee0
//
__int64 __fastcall sub_30F9EE0(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v5; // r12
  __m128i *v6; // rdx
  __m128i si128; // xmm0
  const char *v8; // rax
  size_t v9; // rdx
  _WORD *v10; // rdi
  unsigned __int8 *v11; // rsi
  unsigned __int64 v12; // rax
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // r12
  __int64 i; // r13
  __int64 v17; // rdi
  __m128i *v18; // rdx
  __m128i v19; // xmm0
  __int64 *v20; // r12
  __int64 *v21; // r13
  __int64 v22; // rdi
  __int64 v23; // r15
  _WORD *v24; // rdx
  __int64 *v25; // rax
  __int64 *v26; // rdx
  __int64 v27; // rdi
  void *v28; // rdx
  __int64 v29; // rdi
  _BYTE *v30; // rax
  bool v31; // zf
  __int64 v33; // rdi
  void *v34; // rdx
  __int64 v35; // r15
  __int64 v36; // r8
  unsigned __int64 v37; // rax
  __int64 *v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r9
  __int64 v41; // rax
  __int64 *v42; // rax
  __int64 v43; // rax
  unsigned __int64 v44; // rdx
  unsigned __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // [rsp-10h] [rbp-D0h]
  __int64 v48; // [rsp-8h] [rbp-C8h]
  __int64 v50; // [rsp+18h] [rbp-A8h]
  size_t v51; // [rsp+18h] [rbp-A8h]
  __int64 *v52; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v53; // [rsp+28h] [rbp-98h]
  _BYTE v54[32]; // [rsp+30h] [rbp-90h] BYREF
  __int64 v55; // [rsp+50h] [rbp-70h] BYREF
  __int64 *v56; // [rsp+58h] [rbp-68h]
  __int64 v57; // [rsp+60h] [rbp-60h]
  int v58; // [rsp+68h] [rbp-58h]
  char v59; // [rsp+6Ch] [rbp-54h]
  char v60; // [rsp+70h] [rbp-50h] BYREF

  v5 = *a2;
  v6 = *(__m128i **)(*a2 + 32);
  if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v6 <= 0x31u )
  {
    v5 = sub_CB6200(v5, "Memory Dereferencibility of pointers in function '", 0x32u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_44CE560);
    v6[3].m128i_i16[0] = 10016;
    *v6 = si128;
    v6[1] = _mm_load_si128((const __m128i *)&xmmword_44CE570);
    v6[2] = _mm_load_si128((const __m128i *)&xmmword_44CE580);
    *(_QWORD *)(v5 + 32) += 50LL;
  }
  v8 = sub_BD5D20(a3);
  v10 = *(_WORD **)(v5 + 32);
  v11 = (unsigned __int8 *)v8;
  v12 = *(_QWORD *)(v5 + 24) - (_QWORD)v10;
  if ( v12 < v9 )
  {
    v46 = sub_CB6200(v5, v11, v9);
    v10 = *(_WORD **)(v46 + 32);
    v5 = v46;
    v12 = *(_QWORD *)(v46 + 24) - (_QWORD)v10;
  }
  else if ( v9 )
  {
    v51 = v9;
    memcpy(v10, v11, v9);
    v10 = (_WORD *)(v51 + *(_QWORD *)(v5 + 32));
    v45 = *(_QWORD *)(v5 + 24) - (_QWORD)v10;
    *(_QWORD *)(v5 + 32) = v10;
    if ( v45 > 1 )
      goto LABEL_6;
    goto LABEL_66;
  }
  if ( v12 > 1 )
  {
LABEL_6:
    *v10 = 2599;
    *(_QWORD *)(v5 + 32) += 2LL;
    goto LABEL_7;
  }
LABEL_66:
  sub_CB6200(v5, (unsigned __int8 *)"'\n", 2u);
LABEL_7:
  v55 = 0;
  v13 = a3 + 72;
  v52 = (__int64 *)v54;
  v53 = 0x400000000LL;
  v56 = (__int64 *)&v60;
  v57 = 4;
  v58 = 0;
  v59 = 1;
  v14 = sub_B2BEC0(a3);
  v15 = *(_QWORD *)(a3 + 80);
  v50 = v14;
  if ( a3 + 72 == v15 )
  {
    i = 0;
  }
  else
  {
    if ( !v15 )
      BUG();
    while ( 1 )
    {
      i = *(_QWORD *)(v15 + 32);
      if ( i != v15 + 24 )
        break;
      v15 = *(_QWORD *)(v15 + 8);
      if ( v13 == v15 )
        goto LABEL_13;
      if ( !v15 )
        BUG();
    }
  }
  while ( v15 != v13 )
  {
    if ( !i )
      BUG();
    if ( *(_BYTE *)(i - 24) == 61 )
    {
      v35 = *(_QWORD *)(i - 56);
      if ( sub_D30730(v35, *(_QWORD *)(i - 16), v50, i - 24, 0, 0, 0) )
      {
        v43 = (unsigned int)v53;
        v44 = (unsigned int)v53 + 1LL;
        if ( v44 > HIDWORD(v53) )
        {
          sub_C8D5F0((__int64)&v52, v54, v44, 8u, v36, v47);
          v43 = (unsigned int)v53;
        }
        v52[v43] = v35;
        LODWORD(v53) = v53 + 1;
      }
      _BitScanReverse64(&v37, 1LL << (*(_WORD *)(i - 22) >> 1));
      if ( sub_D305E0(v35, *(_QWORD *)(i - 16), 63 - (v37 ^ 0x3F), v50, i - 24, 0, 0, 0) )
      {
        if ( !v59 )
          goto LABEL_64;
        v42 = v56;
        v39 = HIDWORD(v57);
        v38 = &v56[HIDWORD(v57)];
        if ( v56 != v38 )
        {
          while ( v35 != *v42 )
          {
            if ( v38 == ++v42 )
              goto LABEL_59;
          }
          goto LABEL_46;
        }
LABEL_59:
        if ( HIDWORD(v57) < (unsigned int)v57 )
        {
          ++HIDWORD(v57);
          *v38 = v35;
          ++v55;
        }
        else
        {
LABEL_64:
          sub_C8CC70((__int64)&v55, v35, (__int64)v38, v39, v48, v40);
        }
      }
    }
LABEL_46:
    for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v15 + 32) )
    {
      v41 = v15 - 24;
      if ( !v15 )
        v41 = 0;
      if ( i != v41 + 48 )
        break;
      v15 = *(_QWORD *)(v15 + 8);
      if ( v13 == v15 )
        goto LABEL_13;
      if ( !v15 )
        BUG();
    }
  }
LABEL_13:
  v17 = *a2;
  v18 = *(__m128i **)(*a2 + 32);
  if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v18 <= 0x22u )
  {
    sub_CB6200(v17, "The following are dereferenceable:\n", 0x23u);
  }
  else
  {
    v19 = _mm_load_si128((const __m128i *)&xmmword_428B1C0);
    v18[2].m128i_i8[2] = 10;
    v18[2].m128i_i16[0] = 14949;
    *v18 = v19;
    v18[1] = _mm_load_si128((const __m128i *)&xmmword_428B1D0);
    *(_QWORD *)(v17 + 32) += 35LL;
  }
  v20 = v52;
  v21 = &v52[(unsigned int)v53];
  if ( v21 != v52 )
  {
    do
    {
      v22 = *a2;
      v23 = *v20;
      v24 = *(_WORD **)(*a2 + 32);
      if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v24 <= 1u )
      {
        sub_CB6200(v22, (unsigned __int8 *)"  ", 2u);
      }
      else
      {
        *v24 = 8224;
        *(_QWORD *)(v22 + 32) += 2LL;
      }
      sub_A69870(v23, (_BYTE *)*a2, 0);
      if ( v59 )
      {
        v25 = v56;
        v26 = &v56[HIDWORD(v57)];
        if ( v56 != v26 )
        {
          while ( v23 != *v25 )
          {
            if ( v26 == ++v25 )
              goto LABEL_34;
          }
LABEL_23:
          v27 = *a2;
          v28 = *(void **)(*a2 + 32);
          if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v28 <= 9u )
          {
            sub_CB6200(v27, "\t(aligned)", 0xAu);
          }
          else
          {
            qmemcpy(v28, "\t(aligned)", 10);
            *(_QWORD *)(v27 + 32) += 10LL;
          }
LABEL_25:
          v29 = *a2;
          v30 = *(_BYTE **)(*a2 + 32);
          if ( *(_BYTE **)(*a2 + 24) == v30 )
            goto LABEL_36;
          goto LABEL_26;
        }
      }
      else if ( sub_C8CA60((__int64)&v55, v23) )
      {
        goto LABEL_23;
      }
LABEL_34:
      v33 = *a2;
      v34 = *(void **)(*a2 + 32);
      if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v34 <= 0xBu )
      {
        sub_CB6200(v33, "\t(unaligned)", 0xCu);
        goto LABEL_25;
      }
      qmemcpy(v34, "\t(unaligned)", 12);
      *(_QWORD *)(v33 + 32) += 12LL;
      v29 = *a2;
      v30 = *(_BYTE **)(*a2 + 32);
      if ( *(_BYTE **)(*a2 + 24) == v30 )
      {
LABEL_36:
        sub_CB6200(v29, (unsigned __int8 *)"\n", 1u);
        goto LABEL_27;
      }
LABEL_26:
      *v30 = 10;
      ++*(_QWORD *)(v29 + 32);
LABEL_27:
      ++v20;
    }
    while ( v21 != v20 );
  }
  v31 = v59 == 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_QWORD *)a1 = 1;
  if ( v31 )
    _libc_free((unsigned __int64)v56);
  if ( v52 != (__int64 *)v54 )
    _libc_free((unsigned __int64)v52);
  return a1;
}
