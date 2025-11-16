// Function: sub_2EEFF60
// Address: 0x2eeff60
//
__int64 __fastcall sub_2EEFF60(__int64 a1, char *a2, __int64 *a3)
{
  __int64 v6; // rdi
  _BYTE *v7; // rax
  int v8; // eax
  int v9; // eax
  __int64 v10; // r15
  _WORD *v11; // rdx
  size_t v12; // rax
  _BYTE *v13; // rdi
  size_t v14; // rbx
  unsigned __int64 v15; // rax
  _BYTE *v16; // rbx
  __int64 v17; // rdi
  __int64 v18; // rsi
  __int64 v19; // r12
  __m128i *v20; // rdx
  __m128i si128; // xmm0
  size_t v22; // rax
  _BYTE *v23; // rdi
  size_t v24; // r15
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rax
  void *v27; // rdx
  const char *v28; // rax
  size_t v29; // rdx
  _BYTE *v30; // rdi
  unsigned __int8 *v31; // rsi
  unsigned __int64 v32; // rax
  size_t v33; // r13
  __int64 v35; // rax
  unsigned __int64 v36; // rax
  unsigned int v37; // eax
  __int64 v38; // rax
  char *src; // [rsp+8h] [rbp-38h]

  v6 = *(_QWORD *)(a1 + 16);
  v7 = *(_BYTE **)(v6 + 32);
  if ( (unsigned __int64)v7 >= *(_QWORD *)(v6 + 24) )
  {
    sub_CB5D20(v6, 10);
  }
  else
  {
    *(_QWORD *)(v6 + 32) = v7 + 1;
    *v7 = 10;
  }
  v8 = *(_DWORD *)(a1 + 664);
  if ( !v8 )
  {
    if ( !qword_5022360 )
      sub_C7D570((__int64 *)&qword_5022360, (__int64 (*)(void))sub_BC3580, (__int64)sub_BC3540);
    if ( &_pthread_key_create )
    {
      v37 = pthread_mutex_lock(qword_5022360);
      if ( v37 )
        sub_4264C5(v37);
    }
    v8 = *(_DWORD *)(a1 + 664);
  }
  v9 = v8 + 1;
  *(_DWORD *)(a1 + 664) = v9;
  if ( v9 == 1 )
  {
    if ( !*(_QWORD *)(a1 + 24) )
    {
LABEL_14:
      v17 = *(_QWORD *)(a1 + 640);
      v18 = *(_QWORD *)(a1 + 16);
      if ( v17 )
        sub_2E11F00(v17, v18);
      else
        sub_2E823F0((__int64)a3, v18, *(_QWORD *)(a1 + 656));
      goto LABEL_16;
    }
    v10 = *(_QWORD *)(a1 + 16);
    v11 = *(_WORD **)(v10 + 32);
    if ( *(_QWORD *)(v10 + 24) - (_QWORD)v11 <= 1u )
    {
      v10 = sub_CB6200(*(_QWORD *)(a1 + 16), (unsigned __int8 *)"# ", 2u);
    }
    else
    {
      *v11 = 8227;
      *(_QWORD *)(v10 + 32) += 2LL;
    }
    if ( *(_QWORD *)(a1 + 24) )
    {
      src = *(char **)(a1 + 24);
      v12 = strlen(src);
      v13 = *(_BYTE **)(v10 + 32);
      v14 = v12;
      v15 = *(_QWORD *)(v10 + 24);
      if ( v14 <= v15 - (unsigned __int64)v13 )
      {
        if ( v14 )
        {
          memcpy(v13, src, v14);
          v16 = (_BYTE *)(*(_QWORD *)(v10 + 32) + v14);
          v15 = *(_QWORD *)(v10 + 24);
          *(_QWORD *)(v10 + 32) = v16;
          v13 = v16;
        }
        goto LABEL_12;
      }
      v10 = sub_CB6200(v10, (unsigned __int8 *)src, v14);
    }
    v13 = *(_BYTE **)(v10 + 32);
    v15 = *(_QWORD *)(v10 + 24);
LABEL_12:
    if ( v15 <= (unsigned __int64)v13 )
    {
      sub_CB5D20(v10, 10);
    }
    else
    {
      *(_QWORD *)(v10 + 32) = v13 + 1;
      *v13 = 10;
    }
    goto LABEL_14;
  }
LABEL_16:
  v19 = *(_QWORD *)(a1 + 16);
  v20 = *(__m128i **)(v19 + 32);
  if ( *(_QWORD *)(v19 + 24) - (_QWORD)v20 <= 0x15u )
  {
    v19 = sub_CB6200(v19, "*** Bad machine code: ", 0x16u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_4453DA0);
    v20[1].m128i_i32[0] = 1701080931;
    v20[1].m128i_i16[2] = 8250;
    *v20 = si128;
    *(_QWORD *)(v19 + 32) += 22LL;
  }
  if ( !a2 )
    goto LABEL_29;
  v22 = strlen(a2);
  v23 = *(_BYTE **)(v19 + 32);
  v24 = v22;
  v25 = *(_QWORD *)(v19 + 24) - (_QWORD)v23;
  if ( v24 > v25 )
  {
    v19 = sub_CB6200(v19, (unsigned __int8 *)a2, v24);
LABEL_29:
    v23 = *(_BYTE **)(v19 + 32);
    v25 = *(_QWORD *)(v19 + 24) - (_QWORD)v23;
    goto LABEL_30;
  }
  if ( v24 )
  {
    memcpy(v23, a2, v24);
    v23 = (_BYTE *)(v24 + *(_QWORD *)(v19 + 32));
    v26 = *(_QWORD *)(v19 + 24) - (_QWORD)v23;
    *(_QWORD *)(v19 + 32) = v23;
    if ( v26 <= 4 )
      goto LABEL_22;
LABEL_31:
    *(_DWORD *)v23 = 707406368;
    v23[4] = 10;
    v27 = (void *)(*(_QWORD *)(v19 + 32) + 5LL);
    v35 = *(_QWORD *)(v19 + 24);
    *(_QWORD *)(v19 + 32) = v27;
    if ( (unsigned __int64)(v35 - (_QWORD)v27) > 0xE )
      goto LABEL_23;
    goto LABEL_32;
  }
LABEL_30:
  if ( v25 > 4 )
    goto LABEL_31;
LABEL_22:
  v19 = sub_CB6200(v19, (unsigned __int8 *)" ***\n", 5u);
  v27 = *(void **)(v19 + 32);
  if ( *(_QWORD *)(v19 + 24) - (_QWORD)v27 > 0xEu )
  {
LABEL_23:
    qmemcpy(v27, "- function:    ", 15);
    *(_QWORD *)(v19 + 32) += 15LL;
    goto LABEL_24;
  }
LABEL_32:
  v19 = sub_CB6200(v19, "- function:    ", 0xFu);
LABEL_24:
  v28 = sub_2E791E0(a3);
  v30 = *(_BYTE **)(v19 + 32);
  v31 = (unsigned __int8 *)v28;
  v32 = *(_QWORD *)(v19 + 24);
  v33 = v29;
  if ( v29 > v32 - (unsigned __int64)v30 )
  {
    v38 = sub_CB6200(v19, v31, v29);
    v30 = *(_BYTE **)(v38 + 32);
    v19 = v38;
    v32 = *(_QWORD *)(v38 + 24);
  }
  else if ( v29 )
  {
    memcpy(v30, v31, v29);
    v36 = *(_QWORD *)(v19 + 24);
    v30 = (_BYTE *)(v33 + *(_QWORD *)(v19 + 32));
    *(_QWORD *)(v19 + 32) = v30;
    if ( v36 > (unsigned __int64)v30 )
      goto LABEL_27;
    return sub_CB5D20(v19, 10);
  }
  if ( v32 > (unsigned __int64)v30 )
  {
LABEL_27:
    *(_QWORD *)(v19 + 32) = v30 + 1;
    *v30 = 10;
    return (__int64)(v30 + 1);
  }
  return sub_CB5D20(v19, 10);
}
