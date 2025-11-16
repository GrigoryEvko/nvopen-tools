// Function: sub_38BEB70
// Address: 0x38beb70
//
__int64 __fastcall sub_38BEB70(__int64 a1, __int64 a2, int a3, int a4, int a5, __int64 a6, __int64 a7)
{
  __int64 v7; // r15
  __m128i *v9; // r14
  size_t v10; // rbx
  unsigned int v11; // r9d
  size_t **v12; // r13
  __int64 v14; // rax
  unsigned int v15; // r9d
  __int64 v16; // r8
  void *v17; // rdi
  size_t *v18; // rax
  __int64 v19; // rax
  void *v20; // rax
  unsigned int v21; // [rsp+0h] [rbp-C0h]
  __int64 v22; // [rsp+0h] [rbp-C0h]
  __int64 v23; // [rsp+8h] [rbp-B8h]
  unsigned int v24; // [rsp+8h] [rbp-B8h]
  unsigned int v25; // [rsp+10h] [rbp-B0h]
  void *src; // [rsp+40h] [rbp-80h] BYREF
  size_t n; // [rsp+48h] [rbp-78h]
  __m128i v32; // [rsp+50h] [rbp-70h] BYREF
  __m128i *v33; // [rsp+60h] [rbp-60h]
  size_t v34; // [rsp+68h] [rbp-58h]
  __m128i v35; // [rsp+70h] [rbp-50h] BYREF
  char v36; // [rsp+80h] [rbp-40h]

  v7 = a1 + 1344;
  sub_16E2FC0((__int64 *)&src, a2);
  v9 = (__m128i *)src;
  v33 = &v35;
  if ( src == &v32 )
  {
    v9 = &v35;
    v35 = _mm_load_si128(&v32);
  }
  else
  {
    v33 = (__m128i *)src;
    v35.m128i_i64[0] = v32.m128i_i64[0];
  }
  v10 = n;
  v32.m128i_i8[0] = 0;
  n = 0;
  v34 = v10;
  src = &v32;
  v36 = 1;
  v11 = sub_16D19C0(v7, (unsigned __int8 *)v9, v10);
  v12 = (size_t **)(*(_QWORD *)(a1 + 1344) + 8LL * v11);
  if ( *v12 )
  {
    if ( *v12 != (size_t *)-8LL )
      goto LABEL_5;
    --*(_DWORD *)(a1 + 1360);
  }
  v21 = v11;
  v14 = malloc(v10 + 17);
  v15 = v21;
  v16 = v14;
  if ( !v14 )
  {
    if ( v10 == -17 )
    {
      v19 = malloc(1u);
      v15 = v21;
      v16 = 0;
      if ( v19 )
      {
        v17 = (void *)(v19 + 16);
        v16 = v19;
        goto LABEL_20;
      }
    }
    v22 = v16;
    v24 = v15;
    sub_16BD1C0("Allocation failed", 1u);
    v15 = v24;
    v16 = v22;
  }
  v17 = (void *)(v16 + 16);
  if ( v10 + 1 > 1 )
  {
LABEL_20:
    v23 = v16;
    v25 = v15;
    v20 = memcpy(v17, v9, v10);
    v16 = v23;
    v15 = v25;
    v17 = v20;
  }
  *((_BYTE *)v17 + v10) = 0;
  *(_QWORD *)v16 = v10;
  *(_BYTE *)(v16 + 8) = 1;
  *v12 = (size_t *)v16;
  ++*(_DWORD *)(a1 + 1356);
  v12 = (size_t **)(*(_QWORD *)(a1 + 1344) + 8LL * (unsigned int)sub_16D1CD0(v7, v15));
  if ( *v12 == (size_t *)-8LL || !*v12 )
  {
    do
    {
      do
      {
        v18 = v12[1];
        ++v12;
      }
      while ( !v18 );
    }
    while ( v18 == (size_t *)-8LL );
  }
LABEL_5:
  if ( v33 != &v35 )
    j_j___libc_free_0((unsigned __int64)v33);
  if ( src != &v32 )
    j_j___libc_free_0((unsigned __int64)src);
  return sub_38BE590(a1, (unsigned __int8 *)*v12 + 16, **v12, a3, a4, 3u, a5, a6, 1, *(_QWORD *)(a7 + 8));
}
