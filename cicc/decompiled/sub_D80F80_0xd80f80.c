// Function: sub_D80F80
// Address: 0xd80f80
//
__int64 __fastcall sub_D80F80(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r12
  __int64 v8; // rsi
  __int64 v10; // rax
  const __m128i *v11; // rdx
  __m128i *v12; // r13
  __int64 result; // rax
  __int64 v14; // rdi
  int v15; // r14d
  unsigned __int64 v16[5]; // [rsp+8h] [rbp-28h] BYREF

  v7 = a1 + 16;
  v8 = a1 + 16;
  v10 = sub_C8D7D0(a1, a1 + 16, a2, 0x30u, v16, a6);
  v11 = *(const __m128i **)a1;
  v12 = (__m128i *)v10;
  result = *(unsigned int *)(a1 + 8);
  v14 = *(_QWORD *)a1 + 48 * result;
  if ( *(_QWORD *)a1 != v14 )
  {
    result = (__int64)v12;
    do
    {
      if ( result )
      {
        *(__m128i *)result = _mm_loadu_si128(v11);
        *(__m128i *)(result + 16) = _mm_loadu_si128(v11 + 1);
        *(_DWORD *)(result + 40) = v11[2].m128i_i32[2];
        *(_BYTE *)(result + 44) = v11[2].m128i_i8[12];
        *(_QWORD *)(result + 32) = &unk_49DE3A8;
      }
      v11 += 3;
      result += 48;
    }
    while ( (const __m128i *)v14 != v11 );
    v14 = *(_QWORD *)a1;
  }
  v15 = v16[0];
  if ( v7 != v14 )
    result = _libc_free(v14, v8);
  *(_QWORD *)a1 = v12;
  *(_DWORD *)(a1 + 12) = v15;
  return result;
}
