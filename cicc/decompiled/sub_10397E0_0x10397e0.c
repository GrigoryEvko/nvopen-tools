// Function: sub_10397E0
// Address: 0x10397e0
//
unsigned __int64 *__fastcall sub_10397E0(__int64 *a1, __int64 a2, char a3, void *a4, size_t a5)
{
  __int64 *v8; // rax
  unsigned __int64 v9; // rax
  unsigned __int8 *v10; // rdi
  unsigned __int64 *result; // rax
  __int64 v12; // rsi
  unsigned __int64 *v13; // rbx
  __int64 v14; // rax
  _WORD *v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // rdi
  _BYTE *v18; // rax
  _QWORD *v19; // rax
  __m128i *v20; // rdx
  __int64 v21; // rdi
  __m128i si128; // xmm0
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // r14
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  void *v29; // rdi
  unsigned __int64 v30; // rax
  unsigned __int64 *v33; // [rsp+10h] [rbp-80h]
  _QWORD *v34; // [rsp+18h] [rbp-78h]
  __int64 v35; // [rsp+18h] [rbp-78h]
  unsigned __int64 *v36; // [rsp+20h] [rbp-70h] BYREF
  unsigned __int64 *v37; // [rsp+28h] [rbp-68h]
  __int64 v38; // [rsp+30h] [rbp-60h]
  unsigned __int8 *v39; // [rsp+40h] [rbp-50h] BYREF
  size_t v40; // [rsp+48h] [rbp-48h]
  _QWORD v41[8]; // [rsp+50h] [rbp-40h] BYREF

  v34 = (_QWORD *)sub_BD5C60(a2);
  sub_10391D0((__int64)&v39, a3);
  v35 = sub_A78730(v34, "memprof", 7u, v39, v40);
  v8 = (__int64 *)sub_BD5C60(a2);
  v9 = sub_A7B440((__int64 *)(a2 + 72), v8, -1, v35);
  v10 = v39;
  *(_QWORD *)(a2 + 72) = v9;
  if ( v10 != (unsigned __int8 *)v41 )
    j_j___libc_free_0(v10, v41[0] + 1LL);
  result = &qword_4F8F3C0;
  if ( !LOBYTE(qword_4F8F408[8]) )
    return result;
  v12 = *a1;
  v36 = 0;
  v37 = 0;
  v38 = 0;
  sub_10394D0((__int64)a1, v12, (__int64)&v36);
  result = v37;
  v13 = v36;
  v33 = v37;
  if ( v36 == v37 )
    goto LABEL_30;
  do
  {
    v19 = sub_CB72A0();
    v20 = (__m128i *)v19[4];
    v21 = (__int64)v19;
    if ( v19[3] - (_QWORD)v20 <= 0x3Cu )
    {
      v21 = sub_CB6200((__int64)v19, "MemProf hinting: Total size for full allocation context hash ", 0x3Du);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F8E310);
      qmemcpy(&v20[3], "context hash ", 13);
      *v20 = si128;
      v20[1] = _mm_load_si128((const __m128i *)&xmmword_3F8E320);
      v20[2] = _mm_load_si128((const __m128i *)&xmmword_3F8E330);
      v19[4] += 61LL;
    }
    v23 = sub_CB59D0(v21, *v13);
    v24 = *(_QWORD *)(v23 + 32);
    v25 = v23;
    if ( (unsigned __int64)(*(_QWORD *)(v23 + 24) - v24) <= 4 )
    {
      v27 = sub_CB6200(v23, (unsigned __int8 *)" and ", 5u);
      v29 = *(void **)(v27 + 32);
      v25 = v27;
    }
    else
    {
      *(_DWORD *)v24 = 1684955424;
      *(_BYTE *)(v24 + 4) = 32;
      v29 = (void *)(*(_QWORD *)(v23 + 32) + 5LL);
      *(_QWORD *)(v23 + 32) = v29;
    }
    v30 = *(_QWORD *)(v25 + 24) - (_QWORD)v29;
    if ( v30 >= a5 )
    {
      if ( a5 )
      {
        memcpy(v29, a4, a5);
        v28 = *(_QWORD *)(v25 + 24);
        v29 = (void *)(a5 + *(_QWORD *)(v25 + 32));
        *(_QWORD *)(v25 + 32) = v29;
        v30 = v28 - (_QWORD)v29;
      }
      if ( v30 > 0xB )
      {
LABEL_10:
        qmemcpy(v29, " alloc type ", 12);
        *(_QWORD *)(v25 + 32) += 12LL;
        goto LABEL_11;
      }
    }
    else
    {
      v26 = sub_CB6200(v25, (unsigned __int8 *)a4, a5);
      v29 = *(void **)(v26 + 32);
      v25 = v26;
      if ( *(_QWORD *)(v26 + 24) - (_QWORD)v29 > 0xBu )
        goto LABEL_10;
    }
    v25 = sub_CB6200(v25, " alloc type ", 0xCu);
LABEL_11:
    sub_10391D0((__int64)&v39, a3);
    v14 = sub_CB6200(v25, v39, v40);
    v15 = *(_WORD **)(v14 + 32);
    v16 = v14;
    if ( *(_QWORD *)(v14 + 24) - (_QWORD)v15 <= 1u )
    {
      v16 = sub_CB6200(v14, (unsigned __int8 *)": ", 2u);
    }
    else
    {
      *v15 = 8250;
      *(_QWORD *)(v14 + 32) += 2LL;
    }
    v17 = sub_CB59D0(v16, v13[1]);
    v18 = *(_BYTE **)(v17 + 32);
    if ( *(_BYTE **)(v17 + 24) == v18 )
    {
      sub_CB6200(v17, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v18 = 10;
      ++*(_QWORD *)(v17 + 32);
    }
    if ( v39 != (unsigned __int8 *)v41 )
      j_j___libc_free_0(v39, v41[0] + 1LL);
    v13 += 2;
  }
  while ( v33 != v13 );
  result = v36;
  v33 = v36;
LABEL_30:
  if ( v33 )
    return (unsigned __int64 *)j_j___libc_free_0(v33, v38 - (_QWORD)v33);
  return result;
}
