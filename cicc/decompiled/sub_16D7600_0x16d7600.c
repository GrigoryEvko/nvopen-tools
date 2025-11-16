// Function: sub_16D7600
// Address: 0x16d7600
//
__int64 *__fastcall sub_16D7600(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rbx
  char *v11; // rsi
  __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdi
  const char *v19; // r15
  __int64 v20; // r14
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // r13
  __int64 v24; // rax
  __m128i *v25; // rdx
  __int64 v26; // rdi
  __int64 v27; // rax
  __m128i *v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rbx
  int v31; // [rsp+0h] [rbp-40h] BYREF
  __int64 v32; // [rsp+8h] [rbp-38h]

  if ( !qword_4FA1630 )
    sub_16C1EA0((__int64)&qword_4FA1630, sub_16D5DD0, (__int64)sub_16D6250, a4, a5, a6);
  v7 = qword_4FA1630;
  if ( !*(_QWORD *)(qword_4FA1630 + 8) )
  {
    v8 = sub_22077B0(80);
    v9 = v8;
    if ( v8 )
      sub_16E8970(v8, 2, 0, 0);
LABEL_6:
    *a1 = v9;
    return a1;
  }
  v11 = "-";
  v12 = qword_4FA1630;
  if ( !(unsigned int)sub_2241AC0(qword_4FA1630, "-") )
  {
    v16 = sub_22077B0(80);
    v9 = v16;
    if ( v16 )
      sub_16E8970(v16, 1, 0, 0);
    goto LABEL_6;
  }
  v31 = 0;
  v17 = sub_2241E40(v12, "-", v13, v14, v15);
  v18 = 80;
  v32 = v17;
  v19 = *(const char **)v7;
  v20 = *(_QWORD *)(v7 + 8);
  v21 = sub_22077B0(80);
  v23 = v21;
  if ( v21 )
  {
    v11 = (char *)v19;
    v18 = v21;
    sub_16E8AF0(v21, v19, v20, &v31, 3);
  }
  if ( v31 )
  {
    v24 = sub_16E8CB0(v18, v11, v22);
    v25 = *(__m128i **)(v24 + 24);
    v26 = v24;
    if ( *(_QWORD *)(v24 + 16) - (_QWORD)v25 <= 0x1Fu )
    {
      v26 = sub_16E7EE0(v24, "Error opening info-output-file '", 32);
    }
    else
    {
      *v25 = _mm_load_si128((const __m128i *)&xmmword_3F67A00);
      v25[1] = _mm_load_si128((const __m128i *)&xmmword_3F67A10);
      *(_QWORD *)(v24 + 24) += 32LL;
    }
    v27 = sub_16E7EE0(v26, *(const char **)v7, *(_QWORD *)(v7 + 8));
    v28 = *(__m128i **)(v27 + 24);
    if ( *(_QWORD *)(v27 + 16) - (_QWORD)v28 <= 0xFu )
    {
      sub_16E7EE0(v27, " for appending!\n", 16);
    }
    else
    {
      *v28 = _mm_load_si128((const __m128i *)&xmmword_3F67A20);
      *(_QWORD *)(v27 + 24) += 16LL;
    }
    v29 = sub_22077B0(80);
    v30 = v29;
    if ( v29 )
      sub_16E8970(v29, 2, 0, 0);
    *a1 = v30;
    if ( v23 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v23 + 8LL))(v23);
  }
  else
  {
    *a1 = v23;
  }
  return a1;
}
