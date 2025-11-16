// Function: sub_C9DED0
// Address: 0xc9ded0
//
__int64 *__fastcall sub_C9DED0(__int64 *a1)
{
  char **v2; // rbx
  __int64 v3; // rax
  __int64 v4; // rbx
  char *v6; // rsi
  __int64 v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdi
  char *v14; // r15
  char *v15; // r14
  __int64 v16; // rax
  __int64 v17; // r13
  __int64 v18; // rax
  __m128i *v19; // rdx
  __int64 v20; // rdi
  __int64 v21; // rax
  __m128i *v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rbx
  int v25; // [rsp+0h] [rbp-40h] BYREF
  __int64 v26; // [rsp+8h] [rbp-38h]

  if ( !qword_4F84F60 )
    sub_C7D570(&qword_4F84F60, sub_CA0780, (__int64)sub_C9FD10);
  v2 = (char **)qword_4F84F60;
  if ( !*(_QWORD *)(qword_4F84F60 + 8) )
  {
    v3 = sub_22077B0(96);
    v4 = v3;
    if ( v3 )
      sub_CB6EE0(v3, 2, 0, 0, 0);
LABEL_6:
    *a1 = v4;
    return a1;
  }
  v6 = "-";
  v7 = qword_4F84F60;
  if ( !(unsigned int)sub_2241AC0(qword_4F84F60, "-") )
  {
    v11 = sub_22077B0(96);
    v4 = v11;
    if ( v11 )
      sub_CB6EE0(v11, 1, 0, 0, 0);
    goto LABEL_6;
  }
  v25 = 0;
  v12 = sub_2241E40(v7, "-", v8, v9, v10);
  v13 = 96;
  v26 = v12;
  v14 = *v2;
  v15 = v2[1];
  v16 = sub_22077B0(96);
  v17 = v16;
  if ( v16 )
  {
    v6 = v14;
    v13 = v16;
    sub_CB7060(v16, v14, v15, &v25, 7);
  }
  if ( v25 )
  {
    v18 = sub_CB72A0(v13, v6);
    v19 = *(__m128i **)(v18 + 32);
    v20 = v18;
    if ( *(_QWORD *)(v18 + 24) - (_QWORD)v19 <= 0x1Fu )
    {
      v20 = sub_CB6200(v18, "Error opening info-output-file '", 32);
    }
    else
    {
      *v19 = _mm_load_si128((const __m128i *)&xmmword_3F67A00);
      v19[1] = _mm_load_si128((const __m128i *)&xmmword_3F67A10);
      *(_QWORD *)(v18 + 32) += 32LL;
    }
    v21 = sub_CB6200(v20, *v2, v2[1]);
    v22 = *(__m128i **)(v21 + 32);
    if ( *(_QWORD *)(v21 + 24) - (_QWORD)v22 <= 0xFu )
    {
      sub_CB6200(v21, " for appending!\n", 16);
    }
    else
    {
      *v22 = _mm_load_si128((const __m128i *)&xmmword_3F67A20);
      *(_QWORD *)(v21 + 32) += 16LL;
    }
    v23 = sub_22077B0(96);
    v24 = v23;
    if ( v23 )
      sub_CB6EE0(v23, 2, 0, 0, 0);
    *a1 = v24;
    if ( v17 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v17 + 8LL))(v17);
  }
  else
  {
    *a1 = v17;
  }
  return a1;
}
