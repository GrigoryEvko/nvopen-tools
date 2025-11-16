// Function: sub_2273BC0
// Address: 0x2273bc0
//
__int64 __fastcall sub_2273BC0(__int64 a1, _QWORD *a2)
{
  unsigned int v2; // r13d
  __int64 v3; // r8
  __int64 v5; // rcx
  _QWORD *v7; // rax
  __m128i *v8; // rdx
  __int64 v9; // r13
  __m128i si128; // xmm0
  void *v11; // rdi
  _QWORD *v12; // rbx
  _QWORD *v13; // r12
  __int64 v14; // rax
  unsigned __int8 *v15; // rsi
  size_t v16; // r12
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rax
  const char *v19; // rsi
  __int64 v20; // rax
  __m128i *v21; // rdx
  __m128i v22; // xmm0
  __int64 v23; // rax
  __int64 v24; // rax
  unsigned __int64 v25; // [rsp+0h] [rbp-80h] BYREF
  unsigned __int64 v26; // [rsp+8h] [rbp-78h] BYREF
  unsigned __int8 *v27[2]; // [rsp+10h] [rbp-70h] BYREF
  __int64 v28; // [rsp+20h] [rbp-60h] BYREF
  _QWORD *v29; // [rsp+30h] [rbp-50h] BYREF
  _QWORD *v30; // [rsp+38h] [rbp-48h]
  __int64 v31; // [rsp+40h] [rbp-40h]
  __int64 v32; // [rsp+48h] [rbp-38h]
  __int64 v33; // [rsp+50h] [rbp-30h]

  v2 = 0;
  v3 = a2[18];
  if ( !v3 )
    return v2;
  v5 = a2[17];
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  sub_2396290(&v25, a1, &v29, v5, v3);
  if ( (v25 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    v25 = v25 & 0xFFFFFFFFFFFFFFFELL | 1;
    v7 = sub_CB72A0();
    v8 = (__m128i *)v7[4];
    v9 = (__int64)v7;
    if ( v7[3] - (_QWORD)v8 <= 0x10u )
    {
      v14 = sub_CB6200((__int64)v7, "Could not parse -", 0x11u);
      v11 = *(void **)(v14 + 32);
      v9 = v14;
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_4365C60);
      v8[1].m128i_i8[0] = 45;
      *v8 = si128;
      v11 = (void *)(v7[4] + 17LL);
      v7[4] = v11;
    }
    v15 = (unsigned __int8 *)a2[3];
    v16 = a2[4];
    v17 = *(_QWORD *)(v9 + 24) - (_QWORD)v11;
    if ( v17 < v16 )
    {
      v23 = sub_CB6200(v9, v15, v16);
      v11 = *(void **)(v23 + 32);
      v9 = v23;
      v17 = *(_QWORD *)(v23 + 24) - (_QWORD)v11;
    }
    else if ( v16 )
    {
      memcpy(v11, v15, v16);
      v24 = *(_QWORD *)(v9 + 24);
      v11 = (void *)(v16 + *(_QWORD *)(v9 + 32));
      *(_QWORD *)(v9 + 32) = v11;
      v17 = v24 - (_QWORD)v11;
    }
    if ( v17 <= 0xA )
    {
      v9 = sub_CB6200(v9, " pipeline: ", 0xBu);
    }
    else
    {
      qmemcpy(v11, " pipeline: ", 11);
      *(_QWORD *)(v9 + 32) += 11LL;
    }
    v18 = v25;
    v25 = 0;
    v26 = v18 | 1;
    sub_C64870((__int64)v27, (__int64 *)&v26);
    v19 = (const char *)v27[0];
    v20 = sub_CB6200(v9, v27[0], (size_t)v27[1]);
    v21 = *(__m128i **)(v20 + 32);
    if ( *(_QWORD *)(v20 + 24) - (_QWORD)v21 <= 0x1Bu )
    {
      v19 = "... I'm going to ignore it.\n";
      sub_CB6200(v20, "... I'm going to ignore it.\n", 0x1Cu);
    }
    else
    {
      v22 = _mm_load_si128((const __m128i *)&xmmword_4365C70);
      qmemcpy(&v21[1], " ignore it.\n", 12);
      *v21 = v22;
      *(_QWORD *)(v20 + 32) += 28LL;
    }
    if ( (__int64 *)v27[0] != &v28 )
    {
      v19 = (const char *)(v28 + 1);
      j_j___libc_free_0((unsigned __int64)v27[0]);
    }
    if ( (v26 & 1) != 0 || (v26 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_C63C30(&v26, (__int64)v19);
    if ( (v25 & 1) != 0 || (v2 = 0, (v25 & 0xFFFFFFFFFFFFFFFELL) != 0) )
      sub_C63C30(&v25, (__int64)v19);
  }
  else
  {
    v2 = 1;
  }
  v12 = v30;
  v13 = v29;
  if ( v30 != v29 )
  {
    do
    {
      if ( *v13 )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v13 + 8LL))(*v13);
      ++v13;
    }
    while ( v12 != v13 );
    v13 = v29;
  }
  if ( !v13 )
    return v2;
  j_j___libc_free_0((unsigned __int64)v13);
  return v2;
}
