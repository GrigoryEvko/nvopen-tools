// Function: sub_2DB9490
// Address: 0x2db9490
//
__int64 __fastcall sub_2DB9490(__int64 a1, __int64 *a2, void **a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 *v9; // rdi
  size_t v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // r8
  const char *v13; // rsi
  _QWORD *v14; // rax
  _QWORD *v15; // rdx
  __int64 v16; // rdx
  __int64 v18; // rax
  size_t v19; // rdx
  __int64 (__fastcall **v20)(); // rax
  _QWORD *v21; // rax
  __m128i *v22; // rdx
  __int64 v23; // rdi
  __m128i v24; // xmm0
  _BYTE *v25; // rax
  _QWORD *v26; // rax
  __m128i *v27; // rdx
  __int64 v28; // rdi
  __m128i si128; // xmm0
  _BYTE *v30; // rax
  _QWORD *v31; // rax
  __m128i *v32; // rdx
  __int64 v33; // rdi
  __m128i v34; // xmm0
  __int64 v35; // rax
  void *v36; // rdx
  _QWORD *v37; // rax
  __m128i *v38; // rdx
  __int64 v39; // rdi
  __m128i v40; // xmm0
  size_t v41; // rdx
  __int64 v42; // [rsp+8h] [rbp-F8h]
  size_t v43; // [rsp+10h] [rbp-F0h]
  unsigned int v44; // [rsp+2Ch] [rbp-D4h] BYREF
  __int64 v45[2]; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v46; // [rsp+40h] [rbp-C0h] BYREF
  unsigned __int8 *v47; // [rsp+50h] [rbp-B0h] BYREF
  size_t n; // [rsp+58h] [rbp-A8h]
  _QWORD src[2]; // [rsp+60h] [rbp-A0h] BYREF
  __int64 *v50; // [rsp+70h] [rbp-90h] BYREF
  __int64 (__fastcall **v51)(); // [rsp+78h] [rbp-88h]
  __int16 v52; // [rsp+90h] [rbp-70h]

  if ( !*(_QWORD *)(a6 + 8) )
  {
    sub_CA0F50(v45, a3);
    v50 = v45;
    v52 = 260;
    sub_C67360((__int64 *)&v47, (__int64)&v50, &v44);
    v9 = *(unsigned __int8 **)a6;
    if ( v47 == (unsigned __int8 *)src )
    {
      v41 = n;
      if ( n )
      {
        if ( n == 1 )
          *v9 = src[0];
        else
          memcpy(v9, src, n);
        v41 = n;
        v9 = *(unsigned __int8 **)a6;
      }
      *(_QWORD *)(a6 + 8) = v41;
      v9[v41] = 0;
      v9 = v47;
      goto LABEL_6;
    }
    v10 = n;
    v11 = src[0];
    if ( v9 == (unsigned __int8 *)(a6 + 16) )
    {
      *(_QWORD *)a6 = v47;
      *(_QWORD *)(a6 + 8) = v10;
      *(_QWORD *)(a6 + 16) = v11;
    }
    else
    {
      v12 = *(_QWORD *)(a6 + 16);
      *(_QWORD *)a6 = v47;
      *(_QWORD *)(a6 + 8) = v10;
      *(_QWORD *)(a6 + 16) = v11;
      if ( v9 )
      {
        v47 = v9;
        src[0] = v12;
LABEL_6:
        n = 0;
        *v9 = 0;
        if ( v47 != (unsigned __int8 *)src )
          j_j___libc_free_0((unsigned __int64)v47);
        if ( (__int64 *)v45[0] != &v46 )
          j_j___libc_free_0(v45[0]);
        goto LABEL_10;
      }
    }
    v47 = (unsigned __int8 *)src;
    v9 = (unsigned __int8 *)src;
    goto LABEL_6;
  }
  v50 = (__int64 *)a6;
  v52 = 260;
  v18 = sub_C83360((__int64)&v50, (int *)&v44, 0, 2, 1, 0x1B6u);
  n = v19;
  v42 = v18;
  v43 = v19;
  LODWORD(v47) = v18;
  v20 = sub_2241E50();
  LODWORD(v50) = 17;
  v51 = v20;
  if ( !(*(unsigned __int8 (__fastcall **)(size_t, __int64, __int64 **))(*(_QWORD *)v43 + 48LL))(v43, v42, &v50)
    && !(*((unsigned __int8 (__fastcall **)(__int64 (__fastcall **)(), unsigned __int8 **, _QWORD))*v51 + 7))(
          v51,
          &v47,
          (unsigned int)v50) )
  {
    if ( (_DWORD)v47 )
    {
      v26 = sub_CB72A0();
      v27 = (__m128i *)v26[4];
      v28 = (__int64)v26;
      if ( v26[3] - (_QWORD)v27 <= 0x16u )
      {
        v28 = sub_CB6200((__int64)v26, "error writing into file", 0x17u);
        v30 = *(_BYTE **)(v28 + 32);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CBC0);
        v27[1].m128i_i32[0] = 1713401716;
        v27[1].m128i_i16[2] = 27753;
        v27[1].m128i_i8[6] = 101;
        *v27 = si128;
        v30 = (_BYTE *)(v26[4] + 23LL);
        *(_QWORD *)(v28 + 32) = v30;
      }
      if ( *(_BYTE **)(v28 + 24) == v30 )
      {
        sub_CB6200(v28, (unsigned __int8 *)"\n", 1u);
      }
      else
      {
        *v30 = 10;
        ++*(_QWORD *)(v28 + 32);
      }
      *(_BYTE *)(a1 + 16) = 0;
      *(_QWORD *)a1 = a1 + 16;
      *(_QWORD *)(a1 + 8) = 0;
      return a1;
    }
    v37 = sub_CB72A0();
    v38 = (__m128i *)v37[4];
    v39 = (__int64)v37;
    if ( v37[3] - (_QWORD)v38 <= 0x21u )
    {
      v39 = sub_CB6200((__int64)v37, "writing to the newly created file ", 0x22u);
    }
    else
    {
      v40 = _mm_load_si128((const __m128i *)&xmmword_3F8CBD0);
      v38[2].m128i_i16[0] = 8293;
      *v38 = v40;
      v38[1] = _mm_load_si128((const __m128i *)&xmmword_3F8CBE0);
      v37[4] += 34LL;
    }
    v23 = sub_CB6200(v39, *(unsigned __int8 **)a6, *(_QWORD *)(a6 + 8));
    v25 = *(_BYTE **)(v23 + 32);
    if ( *(_BYTE **)(v23 + 24) != v25 )
      goto LABEL_23;
LABEL_38:
    sub_CB6200(v23, (unsigned __int8 *)"\n", 1u);
    goto LABEL_10;
  }
  v21 = sub_CB72A0();
  v22 = (__m128i *)v21[4];
  v23 = (__int64)v21;
  if ( v21[3] - (_QWORD)v22 <= 0x17u )
  {
    v23 = sub_CB6200((__int64)v21, "file exists, overwriting", 0x18u);
    v25 = *(_BYTE **)(v23 + 32);
  }
  else
  {
    v24 = _mm_load_si128((const __m128i *)&xmmword_3F8CBB0);
    v22[1].m128i_i64[0] = 0x676E697469727772LL;
    *v22 = v24;
    v25 = (_BYTE *)(v21[4] + 24LL);
    *(_QWORD *)(v23 + 32) = v25;
  }
  if ( *(_BYTE **)(v23 + 24) == v25 )
    goto LABEL_38;
LABEL_23:
  *v25 = 10;
  ++*(_QWORD *)(v23 + 32);
LABEL_10:
  sub_CB6EE0((__int64)&v50, v44, 1, 0, 0);
  if ( v44 == -1 )
  {
    v31 = sub_CB72A0();
    v32 = (__m128i *)v31[4];
    v33 = (__int64)v31;
    if ( v31[3] - (_QWORD)v32 <= 0x13u )
    {
      v33 = sub_CB6200((__int64)v31, "error opening file '", 0x14u);
    }
    else
    {
      v34 = _mm_load_si128((const __m128i *)&xmmword_3F8CBF0);
      v32[1].m128i_i32[0] = 656434540;
      *v32 = v34;
      v31[4] += 20LL;
    }
    v13 = *(const char **)a6;
    v35 = sub_CB6200(v33, *(unsigned __int8 **)a6, *(_QWORD *)(a6 + 8));
    v36 = *(void **)(v35 + 32);
    if ( *(_QWORD *)(v35 + 24) - (_QWORD)v36 <= 0xEu )
    {
      v13 = "' for writing!\n";
      sub_CB6200(v35, "' for writing!\n", 0xFu);
    }
    else
    {
      qmemcpy(v36, "' for writing!\n", 15);
      *(_QWORD *)(v35 + 32) += 15LL;
    }
    *(_BYTE *)(a1 + 16) = 0;
    *(_QWORD *)a1 = a1 + 16;
    *(_QWORD *)(a1 + 8) = 0;
  }
  else
  {
    v13 = (const char *)a2;
    sub_2DB8E00((__int64)&v50, a2);
    v14 = sub_CB72A0();
    v15 = (_QWORD *)v14[4];
    if ( v14[3] - (_QWORD)v15 <= 7u )
    {
      v13 = " done. \n";
      sub_CB6200((__int64)v14, " done. \n", 8u);
    }
    else
    {
      *v15 = 0xA202E656E6F6420LL;
      v14[4] += 8LL;
    }
    *(_QWORD *)a1 = a1 + 16;
    if ( *(_QWORD *)a6 == a6 + 16 )
    {
      *(__m128i *)(a1 + 16) = _mm_loadu_si128((const __m128i *)(a6 + 16));
    }
    else
    {
      *(_QWORD *)a1 = *(_QWORD *)a6;
      *(_QWORD *)(a1 + 16) = *(_QWORD *)(a6 + 16);
    }
    v16 = *(_QWORD *)(a6 + 8);
    *(_QWORD *)a6 = a6 + 16;
    *(_QWORD *)(a6 + 8) = 0;
    *(_QWORD *)(a1 + 8) = v16;
    *(_BYTE *)(a6 + 16) = 0;
  }
  sub_CB5B00((int *)&v50, (__int64)v13);
  return a1;
}
