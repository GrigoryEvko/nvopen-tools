// Function: sub_2531550
// Address: 0x2531550
//
__m128i *__fastcall sub_2531550(__m128i *a1, __int64 a2, void **a3, char a4, void **a5, __int64 a6)
{
  unsigned __int8 *v10; // rdi
  size_t v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // r8
  void *v14; // rax
  const char *v15; // rsi
  __int64 v16; // rdx
  __int64 v18; // rax
  size_t v19; // rdx
  __int64 (__fastcall **v20)(); // rax
  void *v21; // rax
  __int64 v22; // rax
  void *v23; // rax
  __int64 v24; // rax
  void *v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  void *v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  size_t v31; // rdx
  __int64 v32; // [rsp+8h] [rbp-F8h]
  size_t v33; // [rsp+10h] [rbp-F0h]
  unsigned int v35; // [rsp+2Ch] [rbp-D4h] BYREF
  __int64 v36[2]; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v37; // [rsp+40h] [rbp-C0h] BYREF
  unsigned __int8 *v38; // [rsp+50h] [rbp-B0h] BYREF
  size_t n; // [rsp+58h] [rbp-A8h]
  _QWORD src[2]; // [rsp+60h] [rbp-A0h] BYREF
  __int64 *v41; // [rsp+70h] [rbp-90h] BYREF
  __int64 (__fastcall **v42)(); // [rsp+78h] [rbp-88h]
  __int16 v43; // [rsp+90h] [rbp-70h]

  if ( !*(_QWORD *)(a6 + 8) )
  {
    sub_CA0F50(v36, a3);
    v41 = v36;
    v43 = 260;
    sub_C67360((__int64 *)&v38, (__int64)&v41, &v35);
    v10 = *(unsigned __int8 **)a6;
    if ( v38 == (unsigned __int8 *)src )
    {
      v31 = n;
      if ( n )
      {
        if ( n == 1 )
          *v10 = src[0];
        else
          memcpy(v10, src, n);
        v31 = n;
        v10 = *(unsigned __int8 **)a6;
      }
      *(_QWORD *)(a6 + 8) = v31;
      v10[v31] = 0;
      v10 = v38;
      goto LABEL_6;
    }
    v11 = n;
    v12 = src[0];
    if ( v10 == (unsigned __int8 *)(a6 + 16) )
    {
      *(_QWORD *)a6 = v38;
      *(_QWORD *)(a6 + 8) = v11;
      *(_QWORD *)(a6 + 16) = v12;
    }
    else
    {
      v13 = *(_QWORD *)(a6 + 16);
      *(_QWORD *)a6 = v38;
      *(_QWORD *)(a6 + 8) = v11;
      *(_QWORD *)(a6 + 16) = v12;
      if ( v10 )
      {
        v38 = v10;
        src[0] = v13;
LABEL_6:
        n = 0;
        *v10 = 0;
        if ( v38 != (unsigned __int8 *)src )
          j_j___libc_free_0((unsigned __int64)v38);
        if ( (__int64 *)v36[0] != &v37 )
          j_j___libc_free_0(v36[0]);
        goto LABEL_10;
      }
    }
    v38 = (unsigned __int8 *)src;
    v10 = (unsigned __int8 *)src;
    goto LABEL_6;
  }
  v41 = (__int64 *)a6;
  v43 = 260;
  v18 = sub_C83360((__int64)&v41, (int *)&v35, 0, 2, 1, 0x1B6u);
  n = v19;
  v32 = v18;
  v33 = v19;
  LODWORD(v38) = v18;
  v20 = sub_2241E50();
  LODWORD(v41) = 17;
  v42 = v20;
  if ( (*(unsigned __int8 (__fastcall **)(size_t, __int64, __int64 **))(*(_QWORD *)v33 + 48LL))(v33, v32, &v41)
    || (*((unsigned __int8 (__fastcall **)(__int64 (__fastcall **)(), unsigned __int8 **, _QWORD))*v42 + 7))(
         v42,
         &v38,
         (unsigned int)v41) )
  {
    v21 = sub_CB72A0();
    v22 = sub_904010((__int64)v21, "file exists, overwriting");
    sub_904010(v22, "\n");
  }
  else
  {
    if ( (_DWORD)v38 )
    {
      v23 = sub_CB72A0();
      v24 = sub_904010((__int64)v23, "error writing into file");
      sub_904010(v24, "\n");
      sub_25072F0(a1->m128i_i64, byte_3F871B3);
      return a1;
    }
    v28 = sub_CB72A0();
    v29 = sub_904010((__int64)v28, "writing to the newly created file ");
    v30 = sub_CB6200(v29, *(unsigned __int8 **)a6, *(_QWORD *)(a6 + 8));
    sub_904010(v30, "\n");
  }
LABEL_10:
  sub_CB6EE0((__int64)&v41, v35, 1, 0, 0);
  if ( v35 == -1 )
  {
    v25 = sub_CB72A0();
    v26 = sub_904010((__int64)v25, "error opening file '");
    v27 = sub_CB6200(v26, *(unsigned __int8 **)a6, *(_QWORD *)(a6 + 8));
    sub_904010(v27, "' for writing!\n");
    v15 = byte_3F871B3;
    sub_25072F0(a1->m128i_i64, byte_3F871B3);
  }
  else
  {
    sub_2514F80((__int64)&v41, a2, a4, a5);
    v14 = sub_CB72A0();
    v15 = " done. \n";
    sub_904010((__int64)v14, " done. \n");
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    if ( *(_QWORD *)a6 == a6 + 16 )
    {
      a1[1] = _mm_loadu_si128((const __m128i *)(a6 + 16));
    }
    else
    {
      a1->m128i_i64[0] = *(_QWORD *)a6;
      a1[1].m128i_i64[0] = *(_QWORD *)(a6 + 16);
    }
    v16 = *(_QWORD *)(a6 + 8);
    *(_QWORD *)a6 = a6 + 16;
    *(_QWORD *)(a6 + 8) = 0;
    a1->m128i_i64[1] = v16;
    *(_BYTE *)(a6 + 16) = 0;
  }
  sub_CB5B00((int *)&v41, (__int64)v15);
  return a1;
}
