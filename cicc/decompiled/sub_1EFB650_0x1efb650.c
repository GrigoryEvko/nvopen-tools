// Function: sub_1EFB650
// Address: 0x1efb650
//
__int64 __fastcall sub_1EFB650(size_t a1)
{
  unsigned __int64 v2; // rax
  __int64 *v3; // rbx
  __int64 result; // rax
  __int64 *i; // r13
  __int64 *v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rax
  const __m128i *v9; // r14
  __m128i *v10; // r13
  __m128i *v11; // rbx
  __int64 v12; // rsi
  __m128i *v13; // r13
  unsigned __int64 v14; // rdi
  __int64 v15; // [rsp+0h] [rbp-40h] BYREF
  __int64 v16; // [rsp+8h] [rbp-38h]
  __m128i *v17; // [rsp+10h] [rbp-30h]

  v2 = *(unsigned int *)(a1 + 544);
  if ( v2 > 2 )
  {
    v7 = *(_QWORD *)(a1 + 536);
    v8 = 40 * v2;
    v9 = (const __m128i *)(v7 + v8);
    v10 = (__m128i *)(v7 + 40);
    sub_1EFA190(&v15, (__m128i *)(v7 + 40), 0xCCCCCCCCCCCCCCCDLL * ((v8 - 40) >> 3));
    if ( v17 )
      sub_1EF99C0(v10, v9, v17, v16);
    else
      sub_1EF8CB0(v10, (__int64)v9);
    v11 = v17;
    v12 = 40 * v16;
    v13 = (__m128i *)((char *)v17 + 40 * v16);
    if ( v17 != v13 )
    {
      do
      {
        v14 = v11[1].m128i_u64[0];
        v11 = (__m128i *)((char *)v11 + 40);
        _libc_free(v14);
      }
      while ( v13 != v11 );
      v13 = v17;
      v12 = 40 * v16;
    }
    j_j___libc_free_0(v13, v12);
    v2 = *(unsigned int *)(a1 + 544);
  }
  v3 = *(__int64 **)(a1 + 536);
  result = 5 * v2;
  for ( i = &v3[result]; i != v3; result = sub_1EFA6F0(a1, v6) )
  {
    v6 = v3;
    v3 += 5;
  }
  return result;
}
