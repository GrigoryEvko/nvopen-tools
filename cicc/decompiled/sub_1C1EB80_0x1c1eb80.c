// Function: sub_1C1EB80
// Address: 0x1c1eb80
//
__int64 __fastcall sub_1C1EB80(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v5; // rax
  const __m128i *v6; // r15
  __int64 v7; // r14
  const __m128i *v8; // rbx
  __m128i *v9; // rsi
  __m128i *v10; // rax
  __m128i *v11; // rdx
  const __m128i *v12; // rax
  __int64 result; // rax
  __m128i *v14; // r13
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 v17; // rax
  __m128i *v18; // rdx
  void *v19; // rdi
  void *src; // [rsp+0h] [rbp-70h] BYREF
  __m128i *v21; // [rsp+8h] [rbp-68h]
  __int8 *v22; // [rsp+10h] [rbp-60h]
  const __m128i *v23; // [rsp+20h] [rbp-50h] BYREF
  __int64 v24; // [rsp+28h] [rbp-48h]
  __int64 v25; // [rsp+30h] [rbp-40h]

  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    v5 = a3[1];
    v6 = (const __m128i *)*a3;
    src = 0;
    v21 = 0;
    v22 = 0;
    v7 = 36 * v5;
    v8 = (const __m128i *)((char *)v6 + 36 * v5);
    if ( (unsigned __int64)(36 * v5) > 0x7FFFFFFFFFFFFFF8LL )
      sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
    v9 = 0;
    if ( v7 )
    {
      v10 = (__m128i *)sub_22077B0(v7);
      src = v10;
      v9 = v10;
      v22 = &v10->m128i_i8[v7];
      if ( v6 != v8 )
      {
        v11 = v10;
        v12 = v6;
        do
        {
          if ( v11 )
          {
            *v11 = _mm_loadu_si128(v12);
            v11[1] = _mm_loadu_si128(v12 + 1);
            v11[2].m128i_i32[0] = v12[2].m128i_i32[0];
          }
          v12 = (const __m128i *)((char *)v12 + 36);
          v11 = (__m128i *)((char *)v11 + 36);
        }
        while ( v8 != v12 );
        v9 = (__m128i *)((char *)v9 + 4 * ((unsigned __int64)((char *)v8 - (char *)v6 - 36) >> 2) + 36);
      }
    }
    v21 = v9;
    v23 = 0;
    v24 = 0;
    v25 = 0;
    result = sub_1C1E4C0(a1, a2, (__int64)&src, &v23, 0);
    if ( v23 )
      result = j_j___libc_free_0(v23, v25 - (_QWORD)v23);
    if ( src )
      return j_j___libc_free_0(src, v22 - (_BYTE *)src);
  }
  else
  {
    src = 0;
    v21 = 0;
    v22 = 0;
    v23 = 0;
    v24 = 0;
    v25 = 0;
    result = sub_1C1E4C0(a1, a2, (__int64)&src, &v23, 0);
    if ( v23 )
      result = j_j___libc_free_0(v23, v25 - (_QWORD)v23);
    v14 = (__m128i *)src;
    if ( src == v21 )
    {
      *a3 = 0;
      a3[1] = 0;
    }
    else
    {
      v15 = sub_16E4080(a1);
      v16 = ((char *)v21 - (_BYTE *)src) >> 2;
      v17 = sub_145CBF0(*(__int64 **)(v15 + 8), (char *)v21 - (_BYTE *)src, 4);
      v18 = v21;
      v14 = (__m128i *)src;
      *a3 = v17;
      v19 = (void *)v17;
      result = 0x8E38E38E38E38E39LL;
      a3[1] = 0x8E38E38E38E38E39LL * v16;
      if ( v14 != v18 )
        result = (__int64)memmove(v19, v14, (char *)v18 - (char *)v14);
    }
    if ( v14 )
      return j_j___libc_free_0(v14, v22 - (__int8 *)v14);
  }
  return result;
}
