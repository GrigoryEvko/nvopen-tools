// Function: sub_FE7580
// Address: 0xfe7580
//
__int64 __fastcall sub_FE7580(__int64 a1, size_t a2, void **a3, char a4, void **a5, __int64 a6)
{
  unsigned __int8 *v10; // rdi
  size_t v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // r8
  const char *v14; // rsi
  __int64 **v15; // r13
  __int64 i; // r13
  __int64 **v17; // rdi
  __int64 *v18; // rdx
  _QWORD *v19; // rax
  _QWORD *v20; // rdx
  __int64 v21; // rdx
  __int64 v23; // rax
  size_t v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // rax
  _QWORD *v28; // rax
  __m128i *v29; // rdx
  __int64 v30; // rdi
  __m128i si128; // xmm0
  _BYTE *v32; // rax
  _QWORD *v33; // rax
  __m128i *v34; // rdx
  __int64 v35; // rdi
  __m128i v36; // xmm0
  _BYTE *v37; // rax
  _QWORD *v38; // rax
  __m128i *v39; // rdx
  __int64 v40; // rdi
  __m128i v41; // xmm0
  __int64 v42; // rax
  void *v43; // rdx
  _QWORD *v44; // rax
  __m128i *v45; // rdx
  __int64 v46; // rdi
  __m128i v47; // xmm0
  size_t v48; // rdx
  __int64 v49; // [rsp+8h] [rbp-108h]
  size_t v50; // [rsp+10h] [rbp-100h]
  __int64 v52; // [rsp+18h] [rbp-F8h]
  unsigned int v53; // [rsp+2Ch] [rbp-E4h] BYREF
  __int64 v54[2]; // [rsp+30h] [rbp-E0h] BYREF
  _QWORD v55[2]; // [rsp+40h] [rbp-D0h] BYREF
  __int64 **p_src; // [rsp+50h] [rbp-C0h] BYREF
  size_t n; // [rsp+58h] [rbp-B8h]
  __int64 src; // [rsp+60h] [rbp-B0h] BYREF
  char v59; // [rsp+68h] [rbp-A8h]
  __int64 v60; // [rsp+70h] [rbp-A0h]
  __int64 *v61; // [rsp+80h] [rbp-90h] BYREF
  __int64 v62; // [rsp+88h] [rbp-88h]
  __int16 v63; // [rsp+A0h] [rbp-70h]

  if ( !*(_QWORD *)(a6 + 8) )
  {
    sub_CA0F50(v54, a3);
    v61 = v54;
    v63 = 260;
    sub_C67360((__int64 *)&p_src, (__int64)&v61, &v53);
    v10 = *(unsigned __int8 **)a6;
    if ( p_src == (__int64 **)&src )
    {
      v48 = n;
      if ( n )
      {
        if ( n == 1 )
          *v10 = src;
        else
          memcpy(v10, &src, n);
        v48 = n;
        v10 = *(unsigned __int8 **)a6;
      }
      *(_QWORD *)(a6 + 8) = v48;
      v10[v48] = 0;
      v10 = (unsigned __int8 *)p_src;
      goto LABEL_6;
    }
    v11 = n;
    v12 = src;
    if ( v10 == (unsigned __int8 *)(a6 + 16) )
    {
      *(_QWORD *)a6 = p_src;
      *(_QWORD *)(a6 + 8) = v11;
      *(_QWORD *)(a6 + 16) = v12;
    }
    else
    {
      v13 = *(_QWORD *)(a6 + 16);
      *(_QWORD *)a6 = p_src;
      *(_QWORD *)(a6 + 8) = v11;
      *(_QWORD *)(a6 + 16) = v12;
      if ( v10 )
      {
        p_src = (__int64 **)v10;
        src = v13;
LABEL_6:
        n = 0;
        *v10 = 0;
        if ( p_src != (__int64 **)&src )
          j_j___libc_free_0(p_src, src + 1);
        if ( (_QWORD *)v54[0] != v55 )
          j_j___libc_free_0(v54[0], v55[0] + 1LL);
        goto LABEL_10;
      }
    }
    p_src = (__int64 **)&src;
    v10 = (unsigned __int8 *)&src;
    goto LABEL_6;
  }
  v61 = (__int64 *)a6;
  v63 = 260;
  v23 = sub_C83360((__int64)&v61, (int *)&v53, 0, 2, 1, 0x1B6u);
  n = v24;
  v49 = v23;
  v50 = v24;
  LODWORD(p_src) = v23;
  v27 = sub_2241E50(&v61, v23, v24, v25, v26);
  LODWORD(v61) = 17;
  v62 = v27;
  if ( (*(unsigned __int8 (__fastcall **)(size_t, __int64, __int64 **))(*(_QWORD *)v50 + 48LL))(v50, v49, &v61)
    || (*(unsigned __int8 (__fastcall **)(__int64, __int64 ***, _QWORD))(*(_QWORD *)v62 + 56LL))(
         v62,
         &p_src,
         (unsigned int)v61) )
  {
    v28 = sub_CB72A0();
    v29 = (__m128i *)v28[4];
    v30 = (__int64)v28;
    if ( v28[3] - (_QWORD)v29 <= 0x17u )
    {
      v30 = sub_CB6200((__int64)v28, "file exists, overwriting", 0x18u);
      v32 = *(_BYTE **)(v30 + 32);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CBB0);
      v29[1].m128i_i64[0] = 0x676E697469727772LL;
      *v29 = si128;
      v32 = (_BYTE *)(v28[4] + 24LL);
      *(_QWORD *)(v30 + 32) = v32;
    }
    if ( *(_BYTE **)(v30 + 24) == v32 )
    {
LABEL_46:
      sub_CB6200(v30, (unsigned __int8 *)"\n", 1u);
      goto LABEL_10;
    }
  }
  else
  {
    if ( (_DWORD)p_src )
    {
      v33 = sub_CB72A0();
      v34 = (__m128i *)v33[4];
      v35 = (__int64)v33;
      if ( v33[3] - (_QWORD)v34 <= 0x16u )
      {
        v35 = sub_CB6200((__int64)v33, "error writing into file", 0x17u);
        v37 = *(_BYTE **)(v35 + 32);
      }
      else
      {
        v36 = _mm_load_si128((const __m128i *)&xmmword_3F8CBC0);
        v34[1].m128i_i32[0] = 1713401716;
        v34[1].m128i_i16[2] = 27753;
        v34[1].m128i_i8[6] = 101;
        *v34 = v36;
        v37 = (_BYTE *)(v33[4] + 23LL);
        *(_QWORD *)(v35 + 32) = v37;
      }
      if ( *(_BYTE **)(v35 + 24) == v37 )
      {
        sub_CB6200(v35, (unsigned __int8 *)"\n", 1u);
      }
      else
      {
        *v37 = 10;
        ++*(_QWORD *)(v35 + 32);
      }
      *(_BYTE *)(a1 + 16) = 0;
      *(_QWORD *)a1 = a1 + 16;
      *(_QWORD *)(a1 + 8) = 0;
      return a1;
    }
    v44 = sub_CB72A0();
    v45 = (__m128i *)v44[4];
    v46 = (__int64)v44;
    if ( v44[3] - (_QWORD)v45 <= 0x21u )
    {
      v46 = sub_CB6200((__int64)v44, "writing to the newly created file ", 0x22u);
    }
    else
    {
      v47 = _mm_load_si128((const __m128i *)&xmmword_3F8CBD0);
      v45[2].m128i_i16[0] = 8293;
      *v45 = v47;
      v45[1] = _mm_load_si128((const __m128i *)&xmmword_3F8CBE0);
      v44[4] += 34LL;
    }
    v30 = sub_CB6200(v46, *(unsigned __int8 **)a6, *(_QWORD *)(a6 + 8));
    v32 = *(_BYTE **)(v30 + 32);
    if ( *(_BYTE **)(v30 + 24) == v32 )
      goto LABEL_46;
  }
  *v32 = 10;
  ++*(_QWORD *)(v30 + 32);
LABEL_10:
  sub_CB6EE0((__int64)&v61, v53, 1, 0, 0);
  if ( v53 == -1 )
  {
    v38 = sub_CB72A0();
    v39 = (__m128i *)v38[4];
    v40 = (__int64)v38;
    if ( v38[3] - (_QWORD)v39 <= 0x13u )
    {
      v40 = sub_CB6200((__int64)v38, "error opening file '", 0x14u);
    }
    else
    {
      v41 = _mm_load_si128((const __m128i *)&xmmword_3F8CBF0);
      v39[1].m128i_i32[0] = 656434540;
      *v39 = v41;
      v38[4] += 20LL;
    }
    v42 = sub_CB6200(v40, *(unsigned __int8 **)a6, *(_QWORD *)(a6 + 8));
    v43 = *(void **)(v42 + 32);
    if ( *(_QWORD *)(v42 + 24) - (_QWORD)v43 <= 0xEu )
    {
      sub_CB6200(v42, "' for writing!\n", 0xFu);
    }
    else
    {
      qmemcpy(v43, "' for writing!\n", 15);
      *(_QWORD *)(v42 + 32) += 15LL;
    }
    *(_QWORD *)a1 = a1 + 16;
    v14 = byte_3F871B3;
    sub_FDB1F0((__int64 *)a1, byte_3F871B3, (__int64)byte_3F871B3);
  }
  else
  {
    n = a2;
    v59 = a4;
    p_src = &v61;
    LOBYTE(src) = 0;
    v60 = 0;
    sub_CA0F50(v54, a5);
    v14 = (const char *)v54;
    sub_FDE800(&p_src, v54);
    v15 = (__int64 **)n;
    v52 = sub_FDC440(*(__int64 **)n) + 72;
    for ( i = *(_QWORD *)(sub_FDC440(*v15) + 80); i != v52; i = *(_QWORD *)(i + 8) )
    {
      v14 = (const char *)(i - 24);
      if ( !i )
        v14 = 0;
      sub_FE2580((__int64)&p_src, (unsigned __int64)v14);
    }
    v17 = p_src;
    v18 = p_src[4];
    if ( (unsigned __int64)((char *)p_src[3] - (char *)v18) <= 1 )
    {
      v14 = "}\n";
      sub_CB6200((__int64)p_src, "}\n", 2u);
    }
    else
    {
      *(_WORD *)v18 = 2685;
      v17[4] = (__int64 *)((char *)v17[4] + 2);
    }
    if ( (_QWORD *)v54[0] != v55 )
    {
      v14 = (const char *)(v55[0] + 1LL);
      j_j___libc_free_0(v54[0], v55[0] + 1LL);
    }
    v19 = sub_CB72A0();
    v20 = (_QWORD *)v19[4];
    if ( v19[3] - (_QWORD)v20 <= 7u )
    {
      v14 = " done. \n";
      sub_CB6200((__int64)v19, " done. \n", 8u);
    }
    else
    {
      *v20 = 0xA202E656E6F6420LL;
      v19[4] += 8LL;
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
    v21 = *(_QWORD *)(a6 + 8);
    *(_QWORD *)a6 = a6 + 16;
    *(_QWORD *)(a6 + 8) = 0;
    *(_QWORD *)(a1 + 8) = v21;
    *(_BYTE *)(a6 + 16) = 0;
  }
  sub_CB5B00((int *)&v61, (__int64)v14);
  return a1;
}
