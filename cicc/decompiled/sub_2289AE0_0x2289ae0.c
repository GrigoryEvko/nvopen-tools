// Function: sub_2289AE0
// Address: 0x2289ae0
//
__int64 __fastcall sub_2289AE0(__int64 a1, __int64 a2, void **a3, char a4, void **a5, __int64 a6)
{
  unsigned __int8 *v10; // rdi
  size_t v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // r8
  const char *v14; // rsi
  _QWORD *v15; // rax
  _QWORD *v16; // rdx
  __int64 v17; // rdx
  __int64 v19; // rax
  size_t v20; // rdx
  __int64 (__fastcall **v21)(); // rax
  _QWORD *v22; // rax
  __m128i *v23; // rdx
  __int64 v24; // rdi
  __m128i v25; // xmm0
  _BYTE *v26; // rax
  _QWORD *v27; // rax
  __m128i *v28; // rdx
  __int64 v29; // rdi
  __m128i si128; // xmm0
  _BYTE *v31; // rax
  _QWORD *v32; // rax
  __m128i *v33; // rdx
  __int64 v34; // rdi
  __m128i v35; // xmm0
  __int64 v36; // rax
  void *v37; // rdx
  _QWORD *v38; // rax
  __m128i *v39; // rdx
  __int64 v40; // rdi
  __m128i v41; // xmm0
  size_t v42; // rdx
  __int64 v43; // [rsp+8h] [rbp-F8h]
  size_t v44; // [rsp+10h] [rbp-F0h]
  unsigned int v46; // [rsp+2Ch] [rbp-D4h] BYREF
  __int64 v47[2]; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v48; // [rsp+40h] [rbp-C0h] BYREF
  unsigned __int8 *v49; // [rsp+50h] [rbp-B0h] BYREF
  size_t n; // [rsp+58h] [rbp-A8h]
  _QWORD src[2]; // [rsp+60h] [rbp-A0h] BYREF
  __int64 *v52; // [rsp+70h] [rbp-90h] BYREF
  __int64 (__fastcall **v53)(); // [rsp+78h] [rbp-88h]
  __int16 v54; // [rsp+90h] [rbp-70h]

  if ( !*(_QWORD *)(a6 + 8) )
  {
    sub_CA0F50(v47, a3);
    v52 = v47;
    v54 = 260;
    sub_C67360((__int64 *)&v49, (__int64)&v52, &v46);
    v10 = *(unsigned __int8 **)a6;
    if ( v49 == (unsigned __int8 *)src )
    {
      v42 = n;
      if ( n )
      {
        if ( n == 1 )
          *v10 = src[0];
        else
          memcpy(v10, src, n);
        v42 = n;
        v10 = *(unsigned __int8 **)a6;
      }
      *(_QWORD *)(a6 + 8) = v42;
      v10[v42] = 0;
      v10 = v49;
      goto LABEL_6;
    }
    v11 = n;
    v12 = src[0];
    if ( v10 == (unsigned __int8 *)(a6 + 16) )
    {
      *(_QWORD *)a6 = v49;
      *(_QWORD *)(a6 + 8) = v11;
      *(_QWORD *)(a6 + 16) = v12;
    }
    else
    {
      v13 = *(_QWORD *)(a6 + 16);
      *(_QWORD *)a6 = v49;
      *(_QWORD *)(a6 + 8) = v11;
      *(_QWORD *)(a6 + 16) = v12;
      if ( v10 )
      {
        v49 = v10;
        src[0] = v13;
LABEL_6:
        n = 0;
        *v10 = 0;
        if ( v49 != (unsigned __int8 *)src )
          j_j___libc_free_0((unsigned __int64)v49);
        if ( (__int64 *)v47[0] != &v48 )
          j_j___libc_free_0(v47[0]);
        goto LABEL_10;
      }
    }
    v49 = (unsigned __int8 *)src;
    v10 = (unsigned __int8 *)src;
    goto LABEL_6;
  }
  v52 = (__int64 *)a6;
  v54 = 260;
  v19 = sub_C83360((__int64)&v52, (int *)&v46, 0, 2, 1, 0x1B6u);
  n = v20;
  v43 = v19;
  v44 = v20;
  LODWORD(v49) = v19;
  v21 = sub_2241E50();
  LODWORD(v52) = 17;
  v53 = v21;
  if ( !(*(unsigned __int8 (__fastcall **)(size_t, __int64, __int64 **))(*(_QWORD *)v44 + 48LL))(v44, v43, &v52)
    && !(*((unsigned __int8 (__fastcall **)(__int64 (__fastcall **)(), unsigned __int8 **, _QWORD))*v53 + 7))(
          v53,
          &v49,
          (unsigned int)v52) )
  {
    if ( (_DWORD)v49 )
    {
      v27 = sub_CB72A0();
      v28 = (__m128i *)v27[4];
      v29 = (__int64)v27;
      if ( v27[3] - (_QWORD)v28 <= 0x16u )
      {
        v29 = sub_CB6200((__int64)v27, "error writing into file", 0x17u);
        v31 = *(_BYTE **)(v29 + 32);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CBC0);
        v28[1].m128i_i32[0] = 1713401716;
        v28[1].m128i_i16[2] = 27753;
        v28[1].m128i_i8[6] = 101;
        *v28 = si128;
        v31 = (_BYTE *)(v27[4] + 23LL);
        *(_QWORD *)(v29 + 32) = v31;
      }
      if ( *(_BYTE **)(v29 + 24) == v31 )
      {
        sub_CB6200(v29, (unsigned __int8 *)"\n", 1u);
      }
      else
      {
        *v31 = 10;
        ++*(_QWORD *)(v29 + 32);
      }
      *(_BYTE *)(a1 + 16) = 0;
      *(_QWORD *)a1 = a1 + 16;
      *(_QWORD *)(a1 + 8) = 0;
      return a1;
    }
    v38 = sub_CB72A0();
    v39 = (__m128i *)v38[4];
    v40 = (__int64)v38;
    if ( v38[3] - (_QWORD)v39 <= 0x21u )
    {
      v40 = sub_CB6200((__int64)v38, "writing to the newly created file ", 0x22u);
    }
    else
    {
      v41 = _mm_load_si128((const __m128i *)&xmmword_3F8CBD0);
      v39[2].m128i_i16[0] = 8293;
      *v39 = v41;
      v39[1] = _mm_load_si128((const __m128i *)&xmmword_3F8CBE0);
      v38[4] += 34LL;
    }
    v24 = sub_CB6200(v40, *(unsigned __int8 **)a6, *(_QWORD *)(a6 + 8));
    v26 = *(_BYTE **)(v24 + 32);
    if ( *(_BYTE **)(v24 + 24) != v26 )
      goto LABEL_23;
LABEL_38:
    sub_CB6200(v24, (unsigned __int8 *)"\n", 1u);
    goto LABEL_10;
  }
  v22 = sub_CB72A0();
  v23 = (__m128i *)v22[4];
  v24 = (__int64)v22;
  if ( v22[3] - (_QWORD)v23 <= 0x17u )
  {
    v24 = sub_CB6200((__int64)v22, "file exists, overwriting", 0x18u);
    v26 = *(_BYTE **)(v24 + 32);
  }
  else
  {
    v25 = _mm_load_si128((const __m128i *)&xmmword_3F8CBB0);
    v23[1].m128i_i64[0] = 0x676E697469727772LL;
    *v23 = v25;
    v26 = (_BYTE *)(v22[4] + 24LL);
    *(_QWORD *)(v24 + 32) = v26;
  }
  if ( *(_BYTE **)(v24 + 24) == v26 )
    goto LABEL_38;
LABEL_23:
  *v26 = 10;
  ++*(_QWORD *)(v24 + 32);
LABEL_10:
  sub_CB6EE0((__int64)&v52, v46, 1, 0, 0);
  if ( v46 == -1 )
  {
    v32 = sub_CB72A0();
    v33 = (__m128i *)v32[4];
    v34 = (__int64)v32;
    if ( v32[3] - (_QWORD)v33 <= 0x13u )
    {
      v34 = sub_CB6200((__int64)v32, "error opening file '", 0x14u);
    }
    else
    {
      v35 = _mm_load_si128((const __m128i *)&xmmword_3F8CBF0);
      v33[1].m128i_i32[0] = 656434540;
      *v33 = v35;
      v32[4] += 20LL;
    }
    v14 = *(const char **)a6;
    v36 = sub_CB6200(v34, *(unsigned __int8 **)a6, *(_QWORD *)(a6 + 8));
    v37 = *(void **)(v36 + 32);
    if ( *(_QWORD *)(v36 + 24) - (_QWORD)v37 <= 0xEu )
    {
      v14 = "' for writing!\n";
      sub_CB6200(v36, "' for writing!\n", 0xFu);
    }
    else
    {
      qmemcpy(v37, "' for writing!\n", 15);
      *(_QWORD *)(v36 + 32) += 15LL;
    }
    *(_BYTE *)(a1 + 16) = 0;
    *(_QWORD *)a1 = a1 + 16;
    *(_QWORD *)(a1 + 8) = 0;
  }
  else
  {
    v14 = (const char *)a2;
    sub_2289010((__int64)&v52, a2, a4, a5);
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
    if ( *(_QWORD *)a6 == a6 + 16 )
    {
      *(__m128i *)(a1 + 16) = _mm_loadu_si128((const __m128i *)(a6 + 16));
    }
    else
    {
      *(_QWORD *)a1 = *(_QWORD *)a6;
      *(_QWORD *)(a1 + 16) = *(_QWORD *)(a6 + 16);
    }
    v17 = *(_QWORD *)(a6 + 8);
    *(_QWORD *)a6 = a6 + 16;
    *(_QWORD *)(a6 + 8) = 0;
    *(_QWORD *)(a1 + 8) = v17;
    *(_BYTE *)(a6 + 16) = 0;
  }
  sub_CB5B00((int *)&v52, (__int64)v14);
  return a1;
}
