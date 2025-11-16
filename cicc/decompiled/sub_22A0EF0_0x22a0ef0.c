// Function: sub_22A0EF0
// Address: 0x22a0ef0
//
__int64 __fastcall sub_22A0EF0(__int64 a1, __int64 a2, void **a3, char a4, void **a5, __int64 a6)
{
  unsigned __int8 *v10; // rdi
  size_t v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // r8
  const char *v14; // rsi
  __int64 v15; // rdi
  _WORD *v16; // rdx
  _QWORD *v17; // rax
  _QWORD *v18; // rdx
  __int64 v19; // rdx
  __int64 v21; // rax
  size_t v22; // rdx
  __int64 (__fastcall **v23)(); // rax
  _QWORD *v24; // rax
  __m128i *v25; // rdx
  __int64 v26; // rdi
  __m128i v27; // xmm0
  _BYTE *v28; // rax
  _QWORD *v29; // rax
  __m128i *v30; // rdx
  __int64 v31; // rdi
  __m128i si128; // xmm0
  _BYTE *v33; // rax
  _QWORD *v34; // rax
  __m128i *v35; // rdx
  __int64 v36; // rdi
  __m128i v37; // xmm0
  __int64 v38; // rax
  void *v39; // rdx
  _QWORD *v40; // rax
  __m128i *v41; // rdx
  __int64 v42; // rdi
  __m128i v43; // xmm0
  size_t v44; // rdx
  __int64 v45; // [rsp+8h] [rbp-F8h]
  size_t v46; // [rsp+10h] [rbp-F0h]
  unsigned int v48; // [rsp+2Ch] [rbp-D4h] BYREF
  __int64 v49[2]; // [rsp+30h] [rbp-D0h] BYREF
  _BYTE v50[16]; // [rsp+40h] [rbp-C0h] BYREF
  char *v51; // [rsp+50h] [rbp-B0h] BYREF
  size_t n; // [rsp+58h] [rbp-A8h]
  _QWORD src[2]; // [rsp+60h] [rbp-A0h] BYREF
  __int64 *v54; // [rsp+70h] [rbp-90h] BYREF
  __int64 (__fastcall **v55)(); // [rsp+78h] [rbp-88h]
  __int16 v56; // [rsp+90h] [rbp-70h]

  if ( !*(_QWORD *)(a6 + 8) )
  {
    sub_CA0F50(v49, a3);
    v54 = v49;
    v56 = 260;
    sub_C67360((__int64 *)&v51, (__int64)&v54, &v48);
    v10 = *(unsigned __int8 **)a6;
    if ( v51 == (char *)src )
    {
      v44 = n;
      if ( n )
      {
        if ( n == 1 )
          *v10 = src[0];
        else
          memcpy(v10, src, n);
        v44 = n;
        v10 = *(unsigned __int8 **)a6;
      }
      *(_QWORD *)(a6 + 8) = v44;
      v10[v44] = 0;
      v10 = (unsigned __int8 *)v51;
      goto LABEL_6;
    }
    v11 = n;
    v12 = src[0];
    if ( v10 == (unsigned __int8 *)(a6 + 16) )
    {
      *(_QWORD *)a6 = v51;
      *(_QWORD *)(a6 + 8) = v11;
      *(_QWORD *)(a6 + 16) = v12;
    }
    else
    {
      v13 = *(_QWORD *)(a6 + 16);
      *(_QWORD *)a6 = v51;
      *(_QWORD *)(a6 + 8) = v11;
      *(_QWORD *)(a6 + 16) = v12;
      if ( v10 )
      {
        v51 = (char *)v10;
        src[0] = v13;
LABEL_6:
        n = 0;
        *v10 = 0;
        if ( v51 != (char *)src )
          j_j___libc_free_0((unsigned __int64)v51);
        if ( (_BYTE *)v49[0] != v50 )
          j_j___libc_free_0(v49[0]);
        goto LABEL_10;
      }
    }
    v51 = (char *)src;
    v10 = (unsigned __int8 *)src;
    goto LABEL_6;
  }
  v54 = (__int64 *)a6;
  v56 = 260;
  v21 = sub_C83360((__int64)&v54, (int *)&v48, 0, 2, 1, 0x1B6u);
  n = v22;
  v45 = v21;
  v46 = v22;
  LODWORD(v51) = v21;
  v23 = sub_2241E50();
  LODWORD(v54) = 17;
  v55 = v23;
  if ( !(*(unsigned __int8 (__fastcall **)(size_t, __int64, __int64 **))(*(_QWORD *)v46 + 48LL))(v46, v45, &v54)
    && !(*((unsigned __int8 (__fastcall **)(__int64 (__fastcall **)(), char **, _QWORD))*v55 + 7))(
          v55,
          &v51,
          (unsigned int)v54) )
  {
    if ( (_DWORD)v51 )
    {
      v29 = sub_CB72A0();
      v30 = (__m128i *)v29[4];
      v31 = (__int64)v29;
      if ( v29[3] - (_QWORD)v30 <= 0x16u )
      {
        v31 = sub_CB6200((__int64)v29, "error writing into file", 0x17u);
        v33 = *(_BYTE **)(v31 + 32);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CBC0);
        v30[1].m128i_i32[0] = 1713401716;
        v30[1].m128i_i16[2] = 27753;
        v30[1].m128i_i8[6] = 101;
        *v30 = si128;
        v33 = (_BYTE *)(v29[4] + 23LL);
        *(_QWORD *)(v31 + 32) = v33;
      }
      if ( *(_BYTE **)(v31 + 24) == v33 )
      {
        sub_CB6200(v31, (unsigned __int8 *)"\n", 1u);
      }
      else
      {
        *v33 = 10;
        ++*(_QWORD *)(v31 + 32);
      }
      *(_BYTE *)(a1 + 16) = 0;
      *(_QWORD *)a1 = a1 + 16;
      *(_QWORD *)(a1 + 8) = 0;
      return a1;
    }
    v40 = sub_CB72A0();
    v41 = (__m128i *)v40[4];
    v42 = (__int64)v40;
    if ( v40[3] - (_QWORD)v41 <= 0x21u )
    {
      v42 = sub_CB6200((__int64)v40, "writing to the newly created file ", 0x22u);
    }
    else
    {
      v43 = _mm_load_si128((const __m128i *)&xmmword_3F8CBD0);
      v41[2].m128i_i16[0] = 8293;
      *v41 = v43;
      v41[1] = _mm_load_si128((const __m128i *)&xmmword_3F8CBE0);
      v40[4] += 34LL;
    }
    v26 = sub_CB6200(v42, *(unsigned __int8 **)a6, *(_QWORD *)(a6 + 8));
    v28 = *(_BYTE **)(v26 + 32);
    if ( *(_BYTE **)(v26 + 24) != v28 )
      goto LABEL_27;
LABEL_42:
    sub_CB6200(v26, (unsigned __int8 *)"\n", 1u);
    goto LABEL_10;
  }
  v24 = sub_CB72A0();
  v25 = (__m128i *)v24[4];
  v26 = (__int64)v24;
  if ( v24[3] - (_QWORD)v25 <= 0x17u )
  {
    v26 = sub_CB6200((__int64)v24, "file exists, overwriting", 0x18u);
    v28 = *(_BYTE **)(v26 + 32);
  }
  else
  {
    v27 = _mm_load_si128((const __m128i *)&xmmword_3F8CBB0);
    v25[1].m128i_i64[0] = 0x676E697469727772LL;
    *v25 = v27;
    v28 = (_BYTE *)(v24[4] + 24LL);
    *(_QWORD *)(v26 + 32) = v28;
  }
  if ( *(_BYTE **)(v26 + 24) == v28 )
    goto LABEL_42;
LABEL_27:
  *v28 = 10;
  ++*(_QWORD *)(v26 + 32);
LABEL_10:
  sub_CB6EE0((__int64)&v54, v48, 1, 0, 0);
  if ( v48 == -1 )
  {
    v34 = sub_CB72A0();
    v35 = (__m128i *)v34[4];
    v36 = (__int64)v34;
    if ( v34[3] - (_QWORD)v35 <= 0x13u )
    {
      v36 = sub_CB6200((__int64)v34, "error opening file '", 0x14u);
    }
    else
    {
      v37 = _mm_load_si128((const __m128i *)&xmmword_3F8CBF0);
      v35[1].m128i_i32[0] = 656434540;
      *v35 = v37;
      v34[4] += 20LL;
    }
    v14 = *(const char **)a6;
    v38 = sub_CB6200(v36, *(unsigned __int8 **)a6, *(_QWORD *)(a6 + 8));
    v39 = *(void **)(v38 + 32);
    if ( *(_QWORD *)(v38 + 24) - (_QWORD)v39 <= 0xEu )
    {
      v14 = "' for writing!\n";
      sub_CB6200(v38, "' for writing!\n", 0xFu);
    }
    else
    {
      qmemcpy(v39, "' for writing!\n", 15);
      *(_QWORD *)(v38 + 32) += 15LL;
    }
    *(_BYTE *)(a1 + 16) = 0;
    *(_QWORD *)a1 = a1 + 16;
    *(_QWORD *)(a1 + 8) = 0;
  }
  else
  {
    v49[1] = a2;
    v49[0] = (__int64)&v54;
    v50[1] = a4;
    v50[0] = 0;
    sub_CA0F50((__int64 *)&v51, a5);
    v14 = (const char *)&v51;
    sub_229D920(v49, &v51);
    sub_22A07D0((__int64)v49);
    v15 = v49[0];
    v16 = *(_WORD **)(v49[0] + 32);
    if ( *(_QWORD *)(v49[0] + 24) - (_QWORD)v16 <= 1u )
    {
      v14 = "}\n";
      sub_CB6200(v49[0], "}\n", 2u);
    }
    else
    {
      *v16 = 2685;
      *(_QWORD *)(v15 + 32) += 2LL;
    }
    if ( v51 != (char *)src )
    {
      v14 = (const char *)(src[0] + 1LL);
      j_j___libc_free_0((unsigned __int64)v51);
    }
    v17 = sub_CB72A0();
    v18 = (_QWORD *)v17[4];
    if ( v17[3] - (_QWORD)v18 <= 7u )
    {
      v14 = " done. \n";
      sub_CB6200((__int64)v17, " done. \n", 8u);
    }
    else
    {
      *v18 = 0xA202E656E6F6420LL;
      v17[4] += 8LL;
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
    v19 = *(_QWORD *)(a6 + 8);
    *(_QWORD *)a6 = a6 + 16;
    *(_QWORD *)(a6 + 8) = 0;
    *(_QWORD *)(a1 + 8) = v19;
    *(_BYTE *)(a6 + 16) = 0;
  }
  sub_CB5B00((int *)&v54, (__int64)v14);
  return a1;
}
