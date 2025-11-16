// Function: sub_C62320
// Address: 0xc62320
//
void __fastcall sub_C62320(__int64 a1, char **a2)
{
  unsigned __int64 v2; // rax
  char **v4; // rdi
  char *v6; // rdx
  const char *v7; // rsi
  __int64 v8; // rax
  char *v9; // r15
  char *v10; // rdi
  size_t v11; // rax
  __int64 v12; // rax
  int v13; // eax
  __int64 v14; // rdi
  _DWORD *v15; // rax
  __int64 v16; // rax
  __m128i *v17; // rdx
  __int64 v18; // rdi
  __m128i si128; // xmm0
  __int64 v20; // rax
  __m128i *v21; // rdx
  __m128i v22; // xmm0
  __int64 v23; // rax
  __m128i *v24; // rdx
  __int64 v25; // r12
  __m128i v26; // xmm0
  __m128i *v27; // rdi
  unsigned __int64 v28; // rax
  __m128i v29; // xmm0
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  unsigned __int64 v33; // [rsp-D8h] [rbp-D8h]
  size_t v34; // [rsp-D8h] [rbp-D8h]
  unsigned __int64 v35; // [rsp-D0h] [rbp-D0h]
  int v36; // [rsp-BCh] [rbp-BCh] BYREF
  char **v37; // [rsp-A8h] [rbp-A8h]
  const char *v38; // [rsp-A0h] [rbp-A0h]
  __int64 v39[2]; // [rsp-98h] [rbp-98h] BYREF
  _QWORD v40[2]; // [rsp-88h] [rbp-88h] BYREF
  char *v41; // [rsp-78h] [rbp-78h] BYREF
  unsigned __int64 v42; // [rsp-70h] [rbp-70h]
  _BYTE v43[104]; // [rsp-68h] [rbp-68h] BYREF

  v2 = (unsigned __int64)a2[1];
  if ( !v2 )
    return;
  v4 = &v41;
  v6 = *a2;
  v7 = (const char *)v39;
  v42 = v2;
  LOBYTE(v39[0]) = 61;
  v41 = v6;
  v8 = sub_C931B0(&v41, v39, 1, 0);
  if ( v8 == -1
    || (v4 = (char **)(v8 + 1), v9 = v41, v8 + 1 > v42)
    || (v7 = (const char *)(v42 - (_QWORD)v4), v4 = (char **)((char *)v4 + (_QWORD)v41), v37 = v4, (v38 = v7) == 0) )
  {
    v16 = sub_CB72A0(v4, v7);
    v17 = *(__m128i **)(v16 + 32);
    v18 = v16;
    if ( *(_QWORD *)(v16 + 24) - (_QWORD)v17 <= 0x13u )
    {
      v18 = sub_CB6200(v16, "DebugCounter Error: ", 20);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F66740);
      v17[1].m128i_i32[0] = 540701295;
      *v17 = si128;
      *(_QWORD *)(v16 + 32) += 20LL;
    }
    v20 = sub_CB6200(v18, *a2, a2[1]);
    v21 = *(__m128i **)(v20 + 32);
    if ( *(_QWORD *)(v20 + 24) - (_QWORD)v21 <= 0x19u )
    {
      sub_CB6200(v20, " does not have an = in it\n", 26);
    }
    else
    {
      v22 = _mm_load_si128((const __m128i *)&xmmword_3F66750);
      qmemcpy(&v21[1], "n = in it\n", 10);
      *v21 = v22;
      *(_QWORD *)(v20 + 32) += 26LL;
    }
    return;
  }
  v35 = v8;
  v33 = v42;
  v41 = v43;
  v42 = 0x300000000LL;
  if ( (unsigned __int8)sub_C606D0(v4, (size_t)v7, (__int64)&v41) )
    goto LABEL_6;
  v11 = v33;
  if ( v35 <= v33 )
    v11 = v35;
  v39[0] = (__int64)v40;
  v34 = v11;
  sub_C5F830(v39, v9, (__int64)&v9[v11]);
  v7 = (const char *)v39;
  v12 = sub_C61310(a1 + 32, (__int64)v39);
  if ( v12 == a1 + 40 )
  {
    v14 = v39[0];
    v36 = 0;
    if ( (_QWORD *)v39[0] == v40 )
      goto LABEL_23;
  }
  else
  {
    v13 = *(_DWORD *)(v12 + 64);
    v14 = v39[0];
    v36 = v13;
    if ( (_QWORD *)v39[0] == v40 )
      goto LABEL_13;
  }
  v7 = (const char *)(v40[0] + 1LL);
  j_j___libc_free_0(v14, v40[0] + 1LL);
  v13 = v36;
LABEL_13:
  if ( !v13 )
  {
LABEL_23:
    v23 = sub_CB72A0(v14, v7);
    v24 = *(__m128i **)(v23 + 32);
    v25 = v23;
    if ( *(_QWORD *)(v23 + 24) - (_QWORD)v24 <= 0x13u )
    {
      v7 = "DebugCounter Error: ";
      v30 = sub_CB6200(v23, "DebugCounter Error: ", 20);
      v27 = *(__m128i **)(v30 + 32);
      v25 = v30;
    }
    else
    {
      v26 = _mm_load_si128((const __m128i *)&xmmword_3F66740);
      v24[1].m128i_i32[0] = 540701295;
      *v24 = v26;
      v27 = (__m128i *)(*(_QWORD *)(v23 + 32) + 20LL);
      *(_QWORD *)(v23 + 32) = v27;
    }
    v28 = *(_QWORD *)(v25 + 24) - (_QWORD)v27;
    if ( v34 > v28 )
    {
      v7 = v9;
      v31 = sub_CB6200(v25, v9, v34);
      v27 = *(__m128i **)(v31 + 32);
      v25 = v31;
      v28 = *(_QWORD *)(v31 + 24) - (_QWORD)v27;
    }
    else if ( v34 )
    {
      v7 = v9;
      memcpy(v27, v9, v34);
      v32 = *(_QWORD *)(v25 + 24);
      v27 = (__m128i *)(v34 + *(_QWORD *)(v25 + 32));
      *(_QWORD *)(v25 + 32) = v27;
      v28 = v32 - (_QWORD)v27;
    }
    if ( v28 <= 0x1C )
    {
      v7 = " is not a registered counter\n";
      sub_CB6200(v25, " is not a registered counter\n", 29);
    }
    else
    {
      v29 = _mm_load_si128((const __m128i *)&xmmword_3F66760);
      qmemcpy(&v27[1], "ered counter\n", 13);
      *v27 = v29;
      *(_QWORD *)(v25 + 32) += 29LL;
    }
LABEL_6:
    v10 = v41;
    if ( v41 == v43 )
      return;
LABEL_15:
    _libc_free(v10, v7);
    return;
  }
  *((_BYTE *)sub_C60B10() + 104) = 1;
  v15 = sub_C620E0(a1, &v36);
  v7 = (const char *)&v41;
  *((_BYTE *)v15 + 16) = 1;
  sub_C5F8E0((__int64)(v15 + 14), &v41);
  v10 = v41;
  if ( v41 != v43 )
    goto LABEL_15;
}
