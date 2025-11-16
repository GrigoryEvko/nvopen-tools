// Function: sub_315D7A0
// Address: 0x315d7a0
//
__int64 __fastcall sub_315D7A0(__int64 a1, __int64 ***a2, void **a3, char a4, void **a5, __int64 a6)
{
  __m128i *v10; // rdi
  size_t v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // r8
  const char *v14; // rsi
  __int64 v15; // rax
  __int64 v16; // r13
  __int64 ***v17; // rdi
  __int64 **v18; // rdx
  _QWORD *v19; // rax
  _QWORD *v20; // rdx
  __int64 v21; // rdx
  __int64 v23; // rax
  size_t v24; // rdx
  __int64 (__fastcall **v25)(); // rax
  _QWORD *v26; // rax
  __m128i *v27; // rdx
  __int64 v28; // rdi
  __m128i si128; // xmm0
  _BYTE *v30; // rax
  _QWORD *v31; // rax
  __m128i *v32; // rdx
  __int64 v33; // rdi
  __m128i v34; // xmm0
  _BYTE *v35; // rax
  _QWORD *v36; // rax
  __m128i *v37; // rdx
  __int64 v38; // rdi
  __m128i v39; // xmm0
  __int64 v40; // rax
  void *v41; // rdx
  _QWORD *v42; // rax
  __m128i *v43; // rdx
  __int64 v44; // rdi
  __m128i v45; // xmm0
  size_t v46; // rdx
  __int64 v47; // [rsp+8h] [rbp-F8h]
  size_t v48; // [rsp+10h] [rbp-F0h]
  __int64 i; // [rsp+18h] [rbp-E8h]
  unsigned int v51; // [rsp+2Ch] [rbp-D4h] BYREF
  __int64 *****v52; // [rsp+30h] [rbp-D0h] BYREF
  __int64 ***v53; // [rsp+38h] [rbp-C8h]
  _BYTE v54[16]; // [rsp+40h] [rbp-C0h] BYREF
  __m128i *v55; // [rsp+50h] [rbp-B0h] BYREF
  size_t n; // [rsp+58h] [rbp-A8h]
  _QWORD src[2]; // [rsp+60h] [rbp-A0h] BYREF
  __int64 ****v58; // [rsp+70h] [rbp-90h] BYREF
  __int64 (__fastcall **v59)(); // [rsp+78h] [rbp-88h]
  __int16 v60; // [rsp+90h] [rbp-70h]

  if ( !*(_QWORD *)(a6 + 8) )
  {
    sub_CA0F50((__int64 *)&v52, a3);
    v58 = (__int64 ****)&v52;
    v60 = 260;
    sub_C67360((__int64 *)&v55, (__int64)&v58, &v51);
    v10 = *(__m128i **)a6;
    if ( v55 == (__m128i *)src )
    {
      v46 = n;
      if ( n )
      {
        if ( n == 1 )
          v10->m128i_i8[0] = src[0];
        else
          memcpy(v10, src, n);
        v46 = n;
        v10 = *(__m128i **)a6;
      }
      *(_QWORD *)(a6 + 8) = v46;
      v10->m128i_i8[v46] = 0;
      v10 = v55;
      goto LABEL_6;
    }
    v11 = n;
    v12 = src[0];
    if ( v10 == (__m128i *)(a6 + 16) )
    {
      *(_QWORD *)a6 = v55;
      *(_QWORD *)(a6 + 8) = v11;
      *(_QWORD *)(a6 + 16) = v12;
    }
    else
    {
      v13 = *(_QWORD *)(a6 + 16);
      *(_QWORD *)a6 = v55;
      *(_QWORD *)(a6 + 8) = v11;
      *(_QWORD *)(a6 + 16) = v12;
      if ( v10 )
      {
        v55 = v10;
        src[0] = v13;
LABEL_6:
        n = 0;
        v10->m128i_i8[0] = 0;
        if ( v55 != (__m128i *)src )
          j_j___libc_free_0((unsigned __int64)v55);
        if ( v52 != (__int64 *****)v54 )
          j_j___libc_free_0((unsigned __int64)v52);
        goto LABEL_10;
      }
    }
    v55 = (__m128i *)src;
    v10 = (__m128i *)src;
    goto LABEL_6;
  }
  v58 = (__int64 ****)a6;
  v60 = 260;
  v23 = sub_C83360((__int64)&v58, (int *)&v51, 0, 2, 1, 0x1B6u);
  n = v24;
  v47 = v23;
  v48 = v24;
  LODWORD(v55) = v23;
  v25 = sub_2241E50();
  LODWORD(v58) = 17;
  v59 = v25;
  if ( (*(unsigned __int8 (__fastcall **)(size_t, __int64, __int64 *****))(*(_QWORD *)v48 + 48LL))(v48, v47, &v58)
    || (*((unsigned __int8 (__fastcall **)(__int64 (__fastcall **)(), __m128i **, _QWORD))*v59 + 7))(
         v59,
         &v55,
         (unsigned int)v58) )
  {
    v26 = sub_CB72A0();
    v27 = (__m128i *)v26[4];
    v28 = (__int64)v26;
    if ( v26[3] - (_QWORD)v27 <= 0x17u )
    {
      v28 = sub_CB6200((__int64)v26, "file exists, overwriting", 0x18u);
      v30 = *(_BYTE **)(v28 + 32);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CBB0);
      v27[1].m128i_i64[0] = 0x676E697469727772LL;
      *v27 = si128;
      v30 = (_BYTE *)(v26[4] + 24LL);
      *(_QWORD *)(v28 + 32) = v30;
    }
    if ( *(_BYTE **)(v28 + 24) == v30 )
    {
LABEL_46:
      sub_CB6200(v28, (unsigned __int8 *)"\n", 1u);
      goto LABEL_10;
    }
  }
  else
  {
    if ( (_DWORD)v55 )
    {
      v31 = sub_CB72A0();
      v32 = (__m128i *)v31[4];
      v33 = (__int64)v31;
      if ( v31[3] - (_QWORD)v32 <= 0x16u )
      {
        v33 = sub_CB6200((__int64)v31, "error writing into file", 0x17u);
        v35 = *(_BYTE **)(v33 + 32);
      }
      else
      {
        v34 = _mm_load_si128((const __m128i *)&xmmword_3F8CBC0);
        v32[1].m128i_i32[0] = 1713401716;
        v32[1].m128i_i16[2] = 27753;
        v32[1].m128i_i8[6] = 101;
        *v32 = v34;
        v35 = (_BYTE *)(v31[4] + 23LL);
        *(_QWORD *)(v33 + 32) = v35;
      }
      if ( v35 == *(_BYTE **)(v33 + 24) )
      {
        sub_CB6200(v33, (unsigned __int8 *)"\n", 1u);
      }
      else
      {
        *v35 = 10;
        ++*(_QWORD *)(v33 + 32);
      }
      *(_BYTE *)(a1 + 16) = 0;
      *(_QWORD *)a1 = a1 + 16;
      *(_QWORD *)(a1 + 8) = 0;
      return a1;
    }
    v42 = sub_CB72A0();
    v43 = (__m128i *)v42[4];
    v44 = (__int64)v42;
    if ( v42[3] - (_QWORD)v43 <= 0x21u )
    {
      v44 = sub_CB6200((__int64)v42, "writing to the newly created file ", 0x22u);
    }
    else
    {
      v45 = _mm_load_si128((const __m128i *)&xmmword_3F8CBD0);
      v43[2].m128i_i16[0] = 8293;
      *v43 = v45;
      v43[1] = _mm_load_si128((const __m128i *)&xmmword_3F8CBE0);
      v42[4] += 34LL;
    }
    v28 = sub_CB6200(v44, *(unsigned __int8 **)a6, *(_QWORD *)(a6 + 8));
    v30 = *(_BYTE **)(v28 + 32);
    if ( *(_BYTE **)(v28 + 24) == v30 )
      goto LABEL_46;
  }
  *v30 = 10;
  ++*(_QWORD *)(v28 + 32);
LABEL_10:
  sub_CB6EE0((__int64)&v58, v51, 1, 0, 0);
  if ( v51 == -1 )
  {
    v36 = sub_CB72A0();
    v37 = (__m128i *)v36[4];
    v38 = (__int64)v36;
    if ( v36[3] - (_QWORD)v37 <= 0x13u )
    {
      v38 = sub_CB6200((__int64)v36, "error opening file '", 0x14u);
    }
    else
    {
      v39 = _mm_load_si128((const __m128i *)&xmmword_3F8CBF0);
      v37[1].m128i_i32[0] = 656434540;
      *v37 = v39;
      v36[4] += 20LL;
    }
    v14 = *(const char **)a6;
    v40 = sub_CB6200(v38, *(unsigned __int8 **)a6, *(_QWORD *)(a6 + 8));
    v41 = *(void **)(v40 + 32);
    if ( *(_QWORD *)(v40 + 24) - (_QWORD)v41 <= 0xEu )
    {
      v14 = "' for writing!\n";
      sub_CB6200(v40, "' for writing!\n", 0xFu);
    }
    else
    {
      qmemcpy(v41, "' for writing!\n", 15);
      *(_QWORD *)(v40 + 32) += 15LL;
    }
    *(_BYTE *)(a1 + 16) = 0;
    *(_QWORD *)a1 = a1 + 16;
    *(_QWORD *)(a1 + 8) = 0;
  }
  else
  {
    v53 = a2;
    v54[1] = a4;
    v52 = &v58;
    v54[0] = 0;
    sub_CA0F50((__int64 *)&v55, a5);
    v14 = (const char *)&v55;
    sub_31584C0((__int64 ****)&v52, &v55);
    v15 = ***v53;
    v16 = *(_QWORD *)(v15 + 80);
    for ( i = v15 + 72; i != v16; v16 = *(_QWORD *)(v16 + 8) )
    {
      v14 = (const char *)(v16 - 24);
      if ( !v16 )
        v14 = 0;
      sub_315CC60((__int64 *)&v52, (unsigned __int64)v14);
    }
    v17 = (__int64 ***)v52;
    v18 = (__int64 **)v52[4];
    if ( (unsigned __int64)((char *)v52[3] - (char *)v18) <= 1 )
    {
      v14 = "}\n";
      sub_CB6200((__int64)v52, "}\n", 2u);
    }
    else
    {
      *(_WORD *)v18 = 2685;
      v17[4] = (__int64 **)((char *)v17[4] + 2);
    }
    if ( v55 != (__m128i *)src )
    {
      v14 = (const char *)(src[0] + 1LL);
      j_j___libc_free_0((unsigned __int64)v55);
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
  sub_CB5B00((int *)&v58, (__int64)v14);
  return a1;
}
