// Function: sub_1680D70
// Address: 0x1680d70
//
unsigned __int64 __fastcall sub_1680D70(const char *a1, __int64 a2, const char **a3, __int64 a4)
{
  __int64 v4; // rsi
  const char *v5; // r14
  const char **v6; // r13
  const char **v8; // r15
  size_t v9; // rbx
  size_t v10; // rax
  const char **v11; // r15
  const char **v12; // r12
  size_t v13; // rbx
  size_t v14; // rax
  __int64 v15; // rax
  __m128i *v16; // rdx
  __int64 v17; // rdi
  __m128i si128; // xmm0
  const char **v19; // rbx
  __int64 v20; // rax
  __int64 v21; // rdi
  _BYTE *v22; // rax
  _BYTE *v23; // rdx
  __int64 v24; // rax
  __m128i *v25; // rdx
  __int64 v26; // rdi
  __m128i v27; // xmm0
  const char **v28; // r13
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rdi
  _BYTE *v32; // rax
  _BYTE *v33; // rdx
  __int64 v34; // rax
  __m128i *v35; // rdx
  __int64 v36; // rdi
  unsigned __int64 result; // rax
  __m128i v38; // xmm0
  int v40; // [rsp+18h] [rbp-68h]
  int v41; // [rsp+1Ch] [rbp-64h]
  void *v42; // [rsp+20h] [rbp-60h] BYREF
  const char *v43; // [rsp+28h] [rbp-58h]
  __int64 v44; // [rsp+30h] [rbp-50h]
  __int64 v45; // [rsp+38h] [rbp-48h]
  int v46; // [rsp+40h] [rbp-40h]

  v4 = a2 << 6;
  v5 = a1;
  v6 = (const char **)&a1[v4];
  if ( &a1[v4] == a1 )
  {
    v41 = 0;
  }
  else
  {
    v8 = (const char **)a1;
    v9 = 0;
    do
    {
      a1 = *v8;
      v10 = strlen(*v8);
      if ( v9 < v10 )
        v9 = v10;
      v8 += 8;
    }
    while ( v6 != v8 );
    v41 = v9;
  }
  v11 = a3;
  v12 = &a3[8 * a4];
  if ( v12 == a3 )
  {
    v40 = 0;
  }
  else
  {
    v13 = 0;
    do
    {
      a1 = *v11;
      v14 = strlen(*v11);
      if ( v13 < v14 )
        v13 = v14;
      v11 += 8;
    }
    while ( v11 != v12 );
    v40 = v13;
  }
  v15 = sub_16E8CB0(a1, v4, a3);
  v16 = *(__m128i **)(v15 + 24);
  v17 = v15;
  if ( *(_QWORD *)(v15 + 16) - (_QWORD)v16 <= 0x20u )
  {
    v4 = (__int64)"Available CPUs for this target:\n\n";
    sub_16E7EE0(v15, "Available CPUs for this target:\n\n", 33);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F82890);
    v16[2].m128i_i8[0] = 10;
    *v16 = si128;
    v16[1] = _mm_load_si128((const __m128i *)&xmmword_3F828A0);
    *(_QWORD *)(v15 + 24) += 33LL;
  }
  if ( v6 != (const char **)v5 )
  {
    v19 = (const char **)v5;
    do
    {
      v20 = sub_16E8CB0(v17, v4, v16);
      v19 += 8;
      v4 = (__int64)&v42;
      v43 = "  %-*s - %s.\n";
      v17 = v20;
      v42 = &unk_49EE538;
      v44 = (__int64)*(v19 - 7);
      v45 = (__int64)*(v19 - 8);
      v46 = v41;
      sub_16E8450(v20, &v42);
    }
    while ( v6 != v19 );
  }
  v21 = sub_16E8CB0(v17, v4, v16);
  v22 = *(_BYTE **)(v21 + 24);
  if ( (unsigned __int64)v22 >= *(_QWORD *)(v21 + 16) )
  {
    v4 = 10;
    sub_16E7DE0(v21, 10);
  }
  else
  {
    v23 = v22 + 1;
    *(_QWORD *)(v21 + 24) = v22 + 1;
    *v22 = 10;
  }
  v24 = sub_16E8CB0(v21, v4, v23);
  v25 = *(__m128i **)(v24 + 24);
  v26 = v24;
  if ( *(_QWORD *)(v24 + 16) - (_QWORD)v25 <= 0x24u )
  {
    v4 = (__int64)"Available features for this target:\n\n";
    sub_16E7EE0(v24, "Available features for this target:\n\n", 37);
  }
  else
  {
    v27 = _mm_load_si128((const __m128i *)&xmmword_3F828B0);
    v25[2].m128i_i32[0] = 171603045;
    v25[2].m128i_i8[4] = 10;
    *v25 = v27;
    v25[1] = _mm_load_si128((const __m128i *)&xmmword_3F828C0);
    *(_QWORD *)(v24 + 24) += 37LL;
  }
  if ( v12 != a3 )
  {
    v28 = a3;
    do
    {
      v29 = sub_16E8CB0(v26, v4, v25);
      v28 += 8;
      v4 = (__int64)&v42;
      v43 = "  %-*s - %s.\n";
      v26 = v29;
      v30 = (__int64)*(v28 - 7);
      v42 = &unk_49EE538;
      v44 = v30;
      v45 = (__int64)*(v28 - 8);
      v46 = v40;
      sub_16E8450(v26, &v42);
    }
    while ( v12 != v28 );
  }
  v31 = sub_16E8CB0(v26, v4, v25);
  v32 = *(_BYTE **)(v31 + 24);
  if ( (unsigned __int64)v32 >= *(_QWORD *)(v31 + 16) )
  {
    v4 = 10;
    sub_16E7DE0(v31, 10);
  }
  else
  {
    v33 = v32 + 1;
    *(_QWORD *)(v31 + 24) = v32 + 1;
    *v32 = 10;
  }
  v34 = sub_16E8CB0(v31, v4, v33);
  v35 = *(__m128i **)(v34 + 24);
  v36 = v34;
  result = *(_QWORD *)(v34 + 16) - (_QWORD)v35;
  if ( result <= 0x74 )
    return sub_16E7EE0(
             v36,
             "Use +feature to enable a feature, or -feature to disable it.\n"
             "For example, llc -mcpu=mycpu -mattr=+feature1,-feature2\n",
             117);
  v38 = _mm_load_si128((const __m128i *)&xmmword_3F828D0);
  v35[7].m128i_i32[0] = 845509237;
  v35[7].m128i_i8[4] = 10;
  *v35 = v38;
  v35[1] = _mm_load_si128((const __m128i *)&xmmword_3F828E0);
  v35[2] = _mm_load_si128((const __m128i *)&xmmword_3F828F0);
  v35[3] = _mm_load_si128((const __m128i *)&xmmword_3F82900);
  v35[4] = _mm_load_si128((const __m128i *)&xmmword_3F82910);
  v35[5] = _mm_load_si128((const __m128i *)&xmmword_3F82920);
  v35[6] = _mm_load_si128((const __m128i *)&xmmword_3F82930);
  *(_QWORD *)(v36 + 24) += 117LL;
  return result;
}
