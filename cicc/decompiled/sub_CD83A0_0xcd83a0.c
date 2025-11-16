// Function: sub_CD83A0
// Address: 0xcd83a0
//
__int64 __fastcall sub_CD83A0(__int64 a1, __int64 a2, const __m128i **a3)
{
  const __m128i *v6; // rax
  const __m128i *v7; // r15
  __int64 v8; // r14
  const __m128i *v9; // rbx
  __m128i *v10; // rsi
  __m128i *v11; // rax
  __m128i *v12; // rdx
  const __m128i *v13; // rax
  __int64 result; // rax
  _BYTE *v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 *v18; // r8
  unsigned __int64 v19; // r12
  unsigned __int64 v20; // rdx
  __m128i *v21; // r8
  _BYTE *v22; // rsi
  void *src; // [rsp+0h] [rbp-70h] BYREF
  __m128i *v24; // [rsp+8h] [rbp-68h]
  __int8 *v25; // [rsp+10h] [rbp-60h]
  const __m128i *v26; // [rsp+20h] [rbp-50h] BYREF
  __int64 v27; // [rsp+28h] [rbp-48h]
  __int64 v28; // [rsp+30h] [rbp-40h]

  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    v6 = a3[1];
    v7 = *a3;
    src = 0;
    v24 = 0;
    v25 = 0;
    v8 = 36LL * (_QWORD)v6;
    v9 = (const __m128i *)((char *)v7 + 36 * (_QWORD)v6);
    if ( (unsigned __int64)(36LL * (_QWORD)v6) > 0x7FFFFFFFFFFFFFF8LL )
      sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
    v10 = 0;
    if ( v8 )
    {
      v11 = (__m128i *)sub_22077B0(v8);
      src = v11;
      v10 = v11;
      v25 = &v11->m128i_i8[v8];
      if ( v7 != v9 )
      {
        v12 = v11;
        v13 = v7;
        do
        {
          if ( v12 )
          {
            *v12 = _mm_loadu_si128(v13);
            v12[1] = _mm_loadu_si128(v13 + 1);
            v12[2].m128i_i32[0] = v13[2].m128i_i32[0];
          }
          v13 = (const __m128i *)((char *)v13 + 36);
          v12 = (__m128i *)((char *)v12 + 36);
        }
        while ( v9 != v13 );
        v10 = (__m128i *)((char *)v10 + 4 * ((unsigned __int64)((char *)v9 - (char *)v7 - 36) >> 2) + 36);
      }
    }
    v24 = v10;
    v26 = 0;
    v27 = 0;
    v28 = 0;
    result = sub_CD7CE0(a1, a2, (__int64)&src, &v26, 0);
    if ( v26 )
      result = j_j___libc_free_0(v26, v28 - (_QWORD)v26);
    v15 = src;
    if ( src )
      return j_j___libc_free_0(v15, v25 - v15);
  }
  else
  {
    src = 0;
    v24 = 0;
    v25 = 0;
    v26 = 0;
    v27 = 0;
    v28 = 0;
    result = sub_CD7CE0(a1, a2, (__int64)&src, &v26, 0);
    if ( v26 )
      result = j_j___libc_free_0(v26, v28 - (_QWORD)v26);
    v15 = src;
    if ( src == v24 )
    {
      *a3 = 0;
      a3[1] = 0;
    }
    else
    {
      v16 = sub_CB0A70(a1);
      v17 = (char *)v24 - (_BYTE *)src;
      v18 = *(__int64 **)(v16 + 8);
      result = *v18;
      v19 = 0x8E38E38E38E38E39LL * (((char *)v24 - (_BYTE *)src) >> 2);
      v18[10] += (char *)v24 - (_BYTE *)src;
      v20 = v17 + ((result + 3) & 0xFFFFFFFFFFFFFFFCLL);
      if ( v18[1] >= v20 && result )
      {
        *v18 = v20;
        v21 = (__m128i *)((result + 3) & 0xFFFFFFFFFFFFFFFCLL);
      }
      else
      {
        result = sub_9D1E70((__int64)v18, v17, v17, 2);
        v21 = (__m128i *)result;
      }
      v15 = v24;
      v22 = src;
      *a3 = v21;
      a3[1] = (const __m128i *)v19;
      if ( v22 != v15 )
      {
        result = (__int64)memmove(v21, v22, v15 - v22);
        v15 = src;
      }
    }
    if ( v15 )
      return j_j___libc_free_0(v15, v25 - v15);
  }
  return result;
}
