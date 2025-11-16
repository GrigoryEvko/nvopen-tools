// Function: sub_E82ED0
// Address: 0xe82ed0
//
unsigned __int64 __fastcall sub_E82ED0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 v5; // rax
  unsigned __int64 v6; // r9
  __int64 v7; // rdi
  const void *v8; // rsi
  unsigned __int64 result; // rax
  __int64 v10; // r8
  __int64 v11; // rdx
  unsigned __int64 v12; // rcx
  const __m128i *v13; // r12
  __m128i *v14; // rdx
  size_t v15; // r12
  __int64 v16; // rdi
  void *v17; // r13
  size_t v18; // r12
  _BYTE *v19; // rdi
  unsigned __int64 v20; // r12
  __int64 v21; // [rsp+0h] [rbp-1E0h]
  const __m128i *v22; // [rsp+0h] [rbp-1E0h]
  unsigned __int64 v23; // [rsp+8h] [rbp-1D8h]
  __int64 v24; // [rsp+8h] [rbp-1D8h]
  _BYTE *v25; // [rsp+20h] [rbp-1C0h] BYREF
  __int64 v26; // [rsp+28h] [rbp-1B8h]
  _BYTE v27[96]; // [rsp+30h] [rbp-1B0h] BYREF
  void *src; // [rsp+90h] [rbp-150h] BYREF
  size_t n; // [rsp+98h] [rbp-148h]
  __int64 v30; // [rsp+A0h] [rbp-140h]
  _BYTE v31[312]; // [rsp+A8h] [rbp-138h] BYREF

  v25 = v27;
  v4 = sub_E8BB10(a1, 0);
  v30 = 256;
  v26 = 0x400000000LL;
  v5 = *(_QWORD *)(a1 + 296);
  src = v31;
  n = 0;
  (*(void (__fastcall **)(_QWORD, __int64, void **, _BYTE **, __int64))(**(_QWORD **)(v5 + 16) + 24LL))(
    *(_QWORD *)(v5 + 16),
    a2,
    &src,
    &v25,
    a3);
  v7 = v4 + 96;
  v8 = (const void *)(v4 + 112);
  result = (unsigned __int64)v25;
  v10 = (__int64)&v25[24 * (unsigned int)v26];
  if ( v25 != (_BYTE *)v10 )
  {
    do
    {
      *(_DWORD *)(result + 8) += *(_QWORD *)(v4 + 48);
      v11 = *(unsigned int *)(v4 + 104);
      v6 = v11 + 1;
      v12 = *(_QWORD *)(v4 + 96);
      if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(v4 + 108) )
      {
        if ( v12 > result || v12 + 24 * v11 <= result )
        {
          v22 = (const __m128i *)result;
          v24 = v10;
          sub_C8D5F0(v7, v8, v6, 0x18u, v10, v6);
          result = (unsigned __int64)v22;
          v12 = *(_QWORD *)(v4 + 96);
          v11 = *(unsigned int *)(v4 + 104);
          v10 = v24;
          v13 = v22;
        }
        else
        {
          v21 = v10;
          v20 = result - v12;
          v23 = result;
          sub_C8D5F0(v7, v8, v6, 0x18u, v10, v6);
          v12 = *(_QWORD *)(v4 + 96);
          v11 = *(unsigned int *)(v4 + 104);
          v10 = v21;
          result = v23;
          v13 = (const __m128i *)(v12 + v20);
        }
      }
      else
      {
        v13 = (const __m128i *)result;
      }
      result += 24LL;
      v14 = (__m128i *)(v12 + 24 * v11);
      *v14 = _mm_loadu_si128(v13);
      v14[1].m128i_i64[0] = v13[1].m128i_i64[0];
      ++*(_DWORD *)(v4 + 104);
    }
    while ( v10 != result );
  }
  v15 = n;
  v16 = *(_QWORD *)(v4 + 48);
  *(_QWORD *)(v4 + 32) = a3;
  *(_BYTE *)(v4 + 29) |= 1u;
  v17 = src;
  if ( v15 + v16 > *(_QWORD *)(v4 + 56) )
  {
    v8 = (const void *)(v4 + 64);
    result = sub_C8D290(v4 + 40, (const void *)(v4 + 64), v15 + v16, 1u, v10, v6);
    v16 = *(_QWORD *)(v4 + 48);
  }
  if ( v15 )
  {
    v8 = v17;
    result = (unsigned __int64)memcpy((void *)(*(_QWORD *)(v4 + 40) + v16), v17, v15);
    v16 = *(_QWORD *)(v4 + 48);
  }
  v18 = v16 + v15;
  v19 = src;
  *(_QWORD *)(v4 + 48) = v18;
  if ( v19 != v31 )
    result = _libc_free(v19, v8);
  if ( v25 != v27 )
    return _libc_free(v25, v8);
  return result;
}
