// Function: sub_E9FB40
// Address: 0xe9fb40
//
void __fastcall sub_E9FB40(_QWORD *a1, __int64 a2, const char **a3, __int64 a4)
{
  __int64 v4; // rsi
  _QWORD *v5; // r13
  _QWORD *v6; // rax
  unsigned __int64 v7; // rdx
  const char **v8; // r15
  const char **v9; // r12
  size_t v10; // rbx
  size_t v11; // rax
  _QWORD *v12; // rax
  __m128i *v13; // rdx
  __m128i si128; // xmm0
  _QWORD *v15; // r12
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  _BYTE *v19; // rsi
  _QWORD *v20; // r9
  __int64 v21; // rdx
  _QWORD *v22; // rdx
  void *v23; // rax
  _BYTE *v24; // rsi
  __int64 v25; // rbx
  _QWORD *v26; // rdi
  _BYTE *v27; // rax
  _QWORD *v28; // rax
  __m128i *v29; // rdx
  __m128i v30; // xmm0
  const char **v31; // rbx
  void *v32; // rax
  __int64 v33; // rdi
  _QWORD *v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  _QWORD *v39; // rdi
  _BYTE *v40; // rax
  _QWORD *v41; // rax
  __m128i *v42; // rdx
  __m128i v43; // xmm0
  const char **v44; // [rsp+8h] [rbp-C8h]
  __int64 v45; // [rsp+10h] [rbp-C0h]
  int v47; // [rsp+28h] [rbp-A8h]
  int v48; // [rsp+2Ch] [rbp-A4h]
  __int64 v49[2]; // [rsp+30h] [rbp-A0h] BYREF
  _QWORD v50[2]; // [rsp+40h] [rbp-90h] BYREF
  __int64 v51[2]; // [rsp+50h] [rbp-80h] BYREF
  _QWORD v52[2]; // [rsp+60h] [rbp-70h] BYREF
  void *v53; // [rsp+70h] [rbp-60h] BYREF
  const char *v54; // [rsp+78h] [rbp-58h]
  _QWORD *v55; // [rsp+80h] [rbp-50h]
  _QWORD *v56; // [rsp+88h] [rbp-48h]
  int v57; // [rsp+90h] [rbp-40h]

  if ( !byte_4F8A329 )
  {
    v4 = 2 * a2;
    v5 = &a1[v4];
    if ( &a1[v4] == a1 )
    {
      v47 = 0;
    }
    else
    {
      v6 = a1;
      v7 = 0;
      do
      {
        if ( v7 < v6[1] )
          v7 = v6[1];
        v6 += 2;
      }
      while ( v5 != v6 );
      v47 = v7;
    }
    v8 = a3;
    v9 = &a3[8 * a4];
    if ( v9 == a3 )
    {
      v48 = 0;
    }
    else
    {
      v10 = 0;
      do
      {
        v11 = strlen(*v8);
        if ( v10 < v11 )
          v10 = v11;
        v8 += 8;
      }
      while ( v9 != v8 );
      v48 = v10;
    }
    v12 = sub_CB72A0();
    v13 = (__m128i *)v12[4];
    if ( v12[3] - (_QWORD)v13 <= 0x20u )
    {
      sub_CB6200((__int64)v12, "Available CPUs for this target:\n\n", 0x21u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F82890);
      v13[2].m128i_i8[0] = 10;
      *v13 = si128;
      v13[1] = _mm_load_si128((const __m128i *)&xmmword_3F828A0);
      v12[4] += 33LL;
    }
    if ( v5 != a1 )
    {
      v44 = v9;
      v15 = a1;
      while ( v15[1] == 12 && *(_QWORD *)*v15 == 0x616C2D656C707061LL && *(_DWORD *)(*v15 + 8LL) == 1953719668 )
      {
LABEL_24:
        v15 += 2;
        if ( v5 == v15 )
        {
          v9 = v44;
          goto LABEL_31;
        }
      }
      v23 = sub_CB72A0();
      v24 = (_BYTE *)*v15;
      v25 = (__int64)v23;
      if ( *v15 )
      {
        v16 = v15[1];
        v51[0] = (__int64)v52;
        sub_E9F6D0(v51, v24, (__int64)&v24[v16]);
        v19 = (_BYTE *)*v15;
        v20 = (_QWORD *)v51[0];
        if ( *v15 )
        {
          v21 = v15[1];
          v45 = v51[0];
          v49[0] = (__int64)v50;
          sub_E9F6D0(v49, v19, (__int64)&v19[v21]);
          v22 = (_QWORD *)v49[0];
          v20 = (_QWORD *)v45;
LABEL_20:
          v55 = v20;
          v54 = "  %-*s - Select the %s processor.\n";
          v56 = v22;
          v53 = &unk_49E41B0;
          v57 = v47;
          sub_CB6620(v25, (__int64)&v53, (__int64)v22, v17, v18, (__int64)v20);
          if ( (_QWORD *)v49[0] != v50 )
            j_j___libc_free_0(v49[0], v50[0] + 1LL);
          if ( (_QWORD *)v51[0] != v52 )
            j_j___libc_free_0(v51[0], v52[0] + 1LL);
          goto LABEL_24;
        }
      }
      else
      {
        v51[1] = 0;
        v51[0] = (__int64)v52;
        v20 = v52;
        LOBYTE(v52[0]) = 0;
      }
      v49[0] = (__int64)v50;
      v22 = v50;
      v49[1] = 0;
      LOBYTE(v50[0]) = 0;
      goto LABEL_20;
    }
LABEL_31:
    v26 = sub_CB72A0();
    v27 = (_BYTE *)v26[4];
    if ( (unsigned __int64)v27 >= v26[3] )
    {
      sub_CB5D20((__int64)v26, 10);
    }
    else
    {
      v26[4] = v27 + 1;
      *v27 = 10;
    }
    v28 = sub_CB72A0();
    v29 = (__m128i *)v28[4];
    if ( v28[3] - (_QWORD)v29 <= 0x24u )
    {
      sub_CB6200((__int64)v28, "Available features for this target:\n\n", 0x25u);
    }
    else
    {
      v30 = _mm_load_si128((const __m128i *)&xmmword_3F828B0);
      v29[2].m128i_i32[0] = 171603045;
      v29[2].m128i_i8[4] = 10;
      *v29 = v30;
      v29[1] = _mm_load_si128((const __m128i *)&xmmword_3F828C0);
      v28[4] += 37LL;
    }
    if ( v9 != a3 )
    {
      v31 = a3;
      do
      {
        v32 = sub_CB72A0();
        v31 += 8;
        v54 = "  %-*s - %s.\n";
        v33 = (__int64)v32;
        v34 = *(v31 - 7);
        v53 = &unk_49E41B0;
        v55 = v34;
        v56 = *(v31 - 8);
        v57 = v48;
        sub_CB6620(v33, (__int64)&v53, v35, v36, v37, v38);
      }
      while ( v9 != v31 );
    }
    v39 = sub_CB72A0();
    v40 = (_BYTE *)v39[4];
    if ( (unsigned __int64)v40 >= v39[3] )
    {
      sub_CB5D20((__int64)v39, 10);
    }
    else
    {
      v39[4] = v40 + 1;
      *v40 = 10;
    }
    v41 = sub_CB72A0();
    v42 = (__m128i *)v41[4];
    if ( v41[3] - (_QWORD)v42 <= 0x74u )
    {
      sub_CB6200(
        (__int64)v41,
        "Use +feature to enable a feature, or -feature to disable it.\n"
        "For example, llc -mcpu=mycpu -mattr=+feature1,-feature2\n",
        0x75u);
      byte_4F8A329 = 1;
    }
    else
    {
      v43 = _mm_load_si128((const __m128i *)&xmmword_3F828D0);
      v42[7].m128i_i32[0] = 845509237;
      v42[7].m128i_i8[4] = 10;
      *v42 = v43;
      byte_4F8A329 = 1;
      v42[1] = _mm_load_si128((const __m128i *)&xmmword_3F828E0);
      v42[2] = _mm_load_si128((const __m128i *)&xmmword_3F828F0);
      v42[3] = _mm_load_si128((const __m128i *)&xmmword_3F82900);
      v42[4] = _mm_load_si128((const __m128i *)&xmmword_3F82910);
      v42[5] = _mm_load_si128((const __m128i *)&xmmword_3F82920);
      v42[6] = _mm_load_si128((const __m128i *)&xmmword_3F82930);
      v41[4] += 117LL;
    }
  }
}
