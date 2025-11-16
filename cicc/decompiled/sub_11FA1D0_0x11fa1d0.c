// Function: sub_11FA1D0
// Address: 0x11fa1d0
//
__int64 __fastcall sub_11FA1D0(__int64 a1, __int64 **a2, void **a3, char a4, void **a5, __m128i *a6)
{
  __m128i *v10; // rdi
  size_t v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // r8
  const char *v14; // rsi
  _QWORD *v15; // rax
  _QWORD *v16; // rdx
  __int64 v17; // rdx
  __int64 v19; // rax
  size_t v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // rax
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
  __int64 v50; // [rsp+40h] [rbp-C0h] BYREF
  __m128i *v51; // [rsp+50h] [rbp-B0h] BYREF
  size_t n; // [rsp+58h] [rbp-A8h]
  _QWORD src[2]; // [rsp+60h] [rbp-A0h] BYREF
  __m128i *v54; // [rsp+70h] [rbp-90h] BYREF
  __int64 v55; // [rsp+78h] [rbp-88h]
  __int16 v56; // [rsp+90h] [rbp-70h]

  if ( !a6->m128i_i64[1] )
  {
    sub_CA0F50(v49, a3);
    v54 = (__m128i *)v49;
    v56 = 260;
    sub_C67360((__int64 *)&v51, (__int64)&v54, &v48);
    v10 = (__m128i *)a6->m128i_i64[0];
    if ( v51 == (__m128i *)src )
    {
      v44 = n;
      if ( n )
      {
        if ( n == 1 )
          v10->m128i_i8[0] = src[0];
        else
          memcpy(v10, src, n);
        v44 = n;
        v10 = (__m128i *)a6->m128i_i64[0];
      }
      a6->m128i_i64[1] = v44;
      v10->m128i_i8[v44] = 0;
      v10 = v51;
      goto LABEL_6;
    }
    v11 = n;
    v12 = src[0];
    if ( v10 == &a6[1] )
    {
      a6->m128i_i64[0] = (__int64)v51;
      a6->m128i_i64[1] = v11;
      a6[1].m128i_i64[0] = v12;
    }
    else
    {
      v13 = a6[1].m128i_i64[0];
      a6->m128i_i64[0] = (__int64)v51;
      a6->m128i_i64[1] = v11;
      a6[1].m128i_i64[0] = v12;
      if ( v10 )
      {
        v51 = v10;
        src[0] = v13;
LABEL_6:
        n = 0;
        v10->m128i_i8[0] = 0;
        if ( v51 != (__m128i *)src )
          j_j___libc_free_0(v51, src[0] + 1LL);
        if ( (__int64 *)v49[0] != &v50 )
          j_j___libc_free_0(v49[0], v50 + 1);
        goto LABEL_10;
      }
    }
    v51 = (__m128i *)src;
    v10 = (__m128i *)src;
    goto LABEL_6;
  }
  v54 = a6;
  v56 = 260;
  v19 = sub_C83360((__int64)&v54, (int *)&v48, 0, 2, 1, 0x1B6u);
  n = v20;
  v45 = v19;
  v46 = v20;
  LODWORD(v51) = v19;
  v23 = sub_2241E50(&v54, v19, v20, v21, v22);
  LODWORD(v54) = 17;
  v55 = v23;
  if ( !(*(unsigned __int8 (__fastcall **)(size_t, __int64, __m128i **))(*(_QWORD *)v46 + 48LL))(v46, v45, &v54)
    && !(*(unsigned __int8 (__fastcall **)(__int64, __m128i **, _QWORD))(*(_QWORD *)v55 + 56LL))(
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
    v26 = sub_CB6200(v42, (unsigned __int8 *)a6->m128i_i64[0], a6->m128i_u64[1]);
    v28 = *(_BYTE **)(v26 + 32);
    if ( *(_BYTE **)(v26 + 24) != v28 )
      goto LABEL_23;
LABEL_38:
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
    goto LABEL_38;
LABEL_23:
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
    v14 = (const char *)a6->m128i_i64[0];
    v38 = sub_CB6200(v36, (unsigned __int8 *)a6->m128i_i64[0], a6->m128i_u64[1]);
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
    v14 = (const char *)a2;
    sub_11F98E0((__int64 **)&v54, a2, a4, a5);
    v15 = sub_CB72A0();
    v16 = (_QWORD *)v15[4];
    if ( v15[3] - (_QWORD)v16 <= 7u )
    {
      v14 = " done. \n";
      sub_CB6200((__int64)v15, " done. \n", 8u);
    }
    else
    {
      *v16 = 0xA202E656E6F6420LL;
      v15[4] += 8LL;
    }
    *(_QWORD *)a1 = a1 + 16;
    if ( (__m128i *)a6->m128i_i64[0] == &a6[1] )
    {
      *(__m128i *)(a1 + 16) = _mm_loadu_si128(a6 + 1);
    }
    else
    {
      *(_QWORD *)a1 = a6->m128i_i64[0];
      *(_QWORD *)(a1 + 16) = a6[1].m128i_i64[0];
    }
    v17 = a6->m128i_i64[1];
    a6->m128i_i64[0] = (__int64)a6[1].m128i_i64;
    a6->m128i_i64[1] = 0;
    *(_QWORD *)(a1 + 8) = v17;
    a6[1].m128i_i8[0] = 0;
  }
  sub_CB5B00((int *)&v54, (__int64)v14);
  return a1;
}
