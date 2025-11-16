// Function: sub_1048250
// Address: 0x1048250
//
__int64 __fastcall sub_1048250(__int64 a1, __int64 **a2, void **a3, char a4, void **a5, __int64 a6)
{
  unsigned __int8 *v10; // rdi
  size_t v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // r8
  __int64 v14; // rdx
  __int64 v15; // r8
  int v16; // r9d
  __int64 v17; // rax
  __int64 v18; // r13
  __int64 v19; // rcx
  unsigned __int64 v20; // rsi
  void *v21; // rax
  const char *v22; // rsi
  __int64 v23; // rdx
  __int64 v25; // rax
  size_t v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // rax
  void *v30; // rax
  __int64 v31; // rax
  void *v32; // rax
  __int64 v33; // rax
  void *v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  void *v37; // rax
  __int64 v38; // rax
  __int64 v39; // rax
  size_t v40; // rdx
  __int64 v41; // [rsp+0h] [rbp-100h]
  size_t v42; // [rsp+8h] [rbp-F8h]
  __int64 v43; // [rsp+10h] [rbp-F0h]
  __int64 i; // [rsp+18h] [rbp-E8h]
  unsigned int v46; // [rsp+2Ch] [rbp-D4h] BYREF
  __int64 *v47; // [rsp+30h] [rbp-D0h] BYREF
  __int64 **v48; // [rsp+38h] [rbp-C8h]
  __int64 v49; // [rsp+40h] [rbp-C0h] BYREF
  unsigned __int8 *v50; // [rsp+50h] [rbp-B0h] BYREF
  size_t n; // [rsp+58h] [rbp-A8h]
  _QWORD src[2]; // [rsp+60h] [rbp-A0h] BYREF
  __int64 **v53; // [rsp+70h] [rbp-90h] BYREF
  __int64 v54; // [rsp+78h] [rbp-88h]
  __int16 v55; // [rsp+90h] [rbp-70h]

  v43 = a1 + 16;
  if ( !*(_QWORD *)(a6 + 8) )
  {
    sub_CA0F50((__int64 *)&v47, a3);
    v53 = &v47;
    v55 = 260;
    sub_C67360((__int64 *)&v50, (__int64)&v53, &v46);
    v10 = *(unsigned __int8 **)a6;
    if ( v50 == (unsigned __int8 *)src )
    {
      v40 = n;
      if ( n )
      {
        if ( n == 1 )
          *v10 = src[0];
        else
          memcpy(v10, src, n);
        v40 = n;
        v10 = *(unsigned __int8 **)a6;
      }
      *(_QWORD *)(a6 + 8) = v40;
      v10[v40] = 0;
      v10 = v50;
      goto LABEL_6;
    }
    v11 = n;
    v12 = src[0];
    if ( v10 == (unsigned __int8 *)(a6 + 16) )
    {
      *(_QWORD *)a6 = v50;
      *(_QWORD *)(a6 + 8) = v11;
      *(_QWORD *)(a6 + 16) = v12;
    }
    else
    {
      v13 = *(_QWORD *)(a6 + 16);
      *(_QWORD *)a6 = v50;
      *(_QWORD *)(a6 + 8) = v11;
      *(_QWORD *)(a6 + 16) = v12;
      if ( v10 )
      {
        v50 = v10;
        src[0] = v13;
LABEL_6:
        n = 0;
        *v10 = 0;
        if ( v50 != (unsigned __int8 *)src )
          j_j___libc_free_0(v50, src[0] + 1LL);
        if ( v47 != &v49 )
          j_j___libc_free_0(v47, v49 + 1);
        goto LABEL_10;
      }
    }
    v50 = (unsigned __int8 *)src;
    v10 = (unsigned __int8 *)src;
    goto LABEL_6;
  }
  v53 = (__int64 **)a6;
  v55 = 260;
  v25 = sub_C83360((__int64)&v53, (int *)&v46, 0, 2, 1, 0x1B6u);
  n = v26;
  v41 = v25;
  v42 = v26;
  LODWORD(v50) = v25;
  v29 = sub_2241E50(&v53, v25, v26, v27, v28);
  LODWORD(v53) = 17;
  v54 = v29;
  if ( (*(unsigned __int8 (__fastcall **)(size_t, __int64, __int64 ***))(*(_QWORD *)v42 + 48LL))(v42, v41, &v53)
    || (*(unsigned __int8 (__fastcall **)(__int64, unsigned __int8 **, _QWORD))(*(_QWORD *)v54 + 56LL))(
         v54,
         &v50,
         (unsigned int)v53) )
  {
    v30 = sub_CB72A0();
    v31 = sub_904010((__int64)v30, "file exists, overwriting");
    sub_904010(v31, "\n");
  }
  else
  {
    if ( (_DWORD)v50 )
    {
      v32 = sub_CB72A0();
      v33 = sub_904010((__int64)v32, "error writing into file");
      sub_904010(v33, "\n");
      *(_QWORD *)(a1 + 8) = 0;
      *(_BYTE *)(a1 + 16) = 0;
      *(_QWORD *)a1 = v43;
      return a1;
    }
    v37 = sub_CB72A0();
    v38 = sub_904010((__int64)v37, "writing to the newly created file ");
    v39 = sub_CB6200(v38, *(unsigned __int8 **)a6, *(_QWORD *)(a6 + 8));
    sub_904010(v39, "\n");
  }
LABEL_10:
  sub_CB6EE0((__int64)&v53, v46, 1, 0, 0);
  if ( v46 == -1 )
  {
    v34 = sub_CB72A0();
    v35 = sub_904010((__int64)v34, "error opening file '");
    v36 = sub_CB6200(v35, *(unsigned __int8 **)a6, *(_QWORD *)(a6 + 8));
    v22 = "' for writing!\n";
    sub_904010(v36, "' for writing!\n");
    *(_QWORD *)(a1 + 8) = 0;
    *(_BYTE *)(a1 + 16) = 0;
    *(_QWORD *)a1 = v43;
  }
  else
  {
    v48 = a2;
    BYTE1(v49) = a4;
    v47 = (__int64 *)&v53;
    LOBYTE(v49) = 0;
    sub_CA0F50((__int64 *)&v50, a5);
    sub_103F8F0((__int64)&v47, (__int64)&v50);
    v17 = **v48;
    v18 = *(_QWORD *)(v17 + 80);
    v19 = v17 + 72;
    for ( i = v17 + 72; i != v18; v18 = *(_QWORD *)(v18 + 8) )
    {
      v20 = v18 - 24;
      if ( !v18 )
        v20 = 0;
      sub_1044070((__int64)&v47, v20, v14, v19, v15, v16);
    }
    sub_904010((__int64)v47, "}\n");
    if ( v50 != (unsigned __int8 *)src )
      j_j___libc_free_0(v50, src[0] + 1LL);
    v21 = sub_CB72A0();
    v22 = " done. \n";
    sub_904010((__int64)v21, " done. \n");
    *(_QWORD *)a1 = v43;
    if ( *(_QWORD *)a6 == a6 + 16 )
    {
      *(__m128i *)(a1 + 16) = _mm_loadu_si128((const __m128i *)(a6 + 16));
    }
    else
    {
      *(_QWORD *)a1 = *(_QWORD *)a6;
      *(_QWORD *)(a1 + 16) = *(_QWORD *)(a6 + 16);
    }
    v23 = *(_QWORD *)(a6 + 8);
    *(_QWORD *)a6 = a6 + 16;
    *(_QWORD *)(a6 + 8) = 0;
    *(_QWORD *)(a1 + 8) = v23;
    *(_BYTE *)(a6 + 16) = 0;
  }
  sub_CB5B00((int *)&v53, (__int64)v22);
  return a1;
}
