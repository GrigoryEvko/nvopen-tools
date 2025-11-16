// Function: sub_983D20
// Address: 0x983d20
//
__int64 __fastcall sub_983D20(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // rsi
  __int64 v6; // rax
  const __m128i *v7; // rdx
  __m128i *v8; // r13
  __int64 result; // rax
  __int64 v10; // rdi
  int v11; // r14d
  _QWORD v12[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = a1 + 16;
  v4 = a1 + 16;
  v6 = sub_C8D7D0(a1, a1 + 16, a2, 48, v12);
  v7 = *(const __m128i **)a1;
  v8 = (__m128i *)v6;
  result = *(unsigned int *)(a1 + 8);
  v10 = *(_QWORD *)a1 + 48 * result;
  if ( *(_QWORD *)a1 != v10 )
  {
    result = (__int64)v8;
    do
    {
      if ( result )
      {
        *(__m128i *)result = _mm_loadu_si128(v7);
        *(__m128i *)(result + 16) = _mm_loadu_si128(v7 + 1);
        *(_DWORD *)(result + 40) = v7[2].m128i_i32[2];
        *(_BYTE *)(result + 44) = v7[2].m128i_i8[12];
        *(_QWORD *)(result + 32) = &unk_49D9580;
      }
      v7 += 3;
      result += 48;
    }
    while ( (const __m128i *)v10 != v7 );
    v10 = *(_QWORD *)a1;
  }
  v11 = v12[0];
  if ( v3 != v10 )
    result = _libc_free(v10, v4);
  *(_QWORD *)a1 = v8;
  *(_DWORD *)(a1 + 12) = v11;
  return result;
}
