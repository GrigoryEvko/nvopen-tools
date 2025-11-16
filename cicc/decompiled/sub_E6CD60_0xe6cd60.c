// Function: sub_E6CD60
// Address: 0xe6cd60
//
unsigned __int64 __fastcall sub_E6CD60(__int64 a1, void **a2, int a3, unsigned int a4, int a5, __int64 a6, __int64 a7)
{
  __int64 *v7; // r15
  __m128i *v8; // r9
  size_t v9; // r13
  int v10; // eax
  unsigned int v11; // r11d
  size_t **v12; // rbx
  __int64 v14; // rax
  unsigned int v15; // r11d
  __int64 v16; // r10
  size_t **v17; // rax
  size_t *v18; // rdx
  __m128i *v19; // rsi
  unsigned int v20; // [rsp+0h] [rbp-D0h]
  __int64 v21; // [rsp+0h] [rbp-D0h]
  __m128i *src; // [rsp+8h] [rbp-C8h]
  unsigned int srca; // [rsp+8h] [rbp-C8h]
  __m128i *v28; // [rsp+30h] [rbp-A0h]
  size_t v29; // [rsp+38h] [rbp-98h]
  __m128i *v30; // [rsp+50h] [rbp-80h] BYREF
  size_t n; // [rsp+58h] [rbp-78h]
  __m128i v32; // [rsp+60h] [rbp-70h] BYREF
  __m128i *v33; // [rsp+70h] [rbp-60h]
  size_t v34; // [rsp+78h] [rbp-58h]
  __m128i v35; // [rsp+80h] [rbp-50h] BYREF
  char v36; // [rsp+90h] [rbp-40h]

  v7 = (__int64 *)(a1 + 2240);
  sub_CA0F50((__int64 *)&v30, a2);
  v8 = v30;
  v33 = &v35;
  if ( v30 == &v32 )
  {
    v8 = &v35;
    v35 = _mm_load_si128(&v32);
  }
  else
  {
    v33 = v30;
    v35.m128i_i64[0] = v32.m128i_i64[0];
  }
  v9 = n;
  v30 = &v32;
  v28 = v8;
  src = v8;
  v34 = n;
  n = 0;
  v32.m128i_i8[0] = 0;
  v36 = 1;
  v29 = v34;
  v10 = sub_C92610();
  v11 = sub_C92740((__int64)v7, v28, v29, v10);
  v12 = (size_t **)(*(_QWORD *)(a1 + 2240) + 8LL * v11);
  if ( *v12 )
  {
    if ( *v12 != (size_t *)-8LL )
      goto LABEL_5;
    --*(_DWORD *)(a1 + 2256);
  }
  v20 = v11;
  v14 = sub_C7D670(v9 + 17, 8);
  v15 = v20;
  v16 = v14;
  if ( v9 )
  {
    v19 = src;
    srca = v20;
    v21 = v14;
    memcpy((void *)(v14 + 16), v19, v9);
    v15 = srca;
    v16 = v21;
  }
  *(_BYTE *)(v16 + v9 + 16) = 0;
  *(_QWORD *)v16 = v9;
  *(_BYTE *)(v16 + 8) = 1;
  *v12 = (size_t *)v16;
  ++*(_DWORD *)(a1 + 2252);
  v12 = (size_t **)(*(_QWORD *)(a1 + 2240) + 8LL * (unsigned int)sub_C929D0(v7, v15));
  if ( *v12 == (size_t *)-8LL || !*v12 )
  {
    v17 = v12 + 1;
    do
    {
      do
      {
        v18 = *v17;
        v12 = v17++;
      }
      while ( !v18 );
    }
    while ( v18 == (size_t *)-8LL );
  }
LABEL_5:
  if ( v33 != &v35 )
    j_j___libc_free_0(v33, v35.m128i_i64[0] + 1);
  if ( v30 != &v32 )
    j_j___libc_free_0(v30, v32.m128i_i64[0] + 1);
  return sub_E6CBD0((_QWORD *)a1, *v12 + 2, **v12, a3, a4, a5, a6, 1u, 1, *(_QWORD *)(a7 + 16));
}
