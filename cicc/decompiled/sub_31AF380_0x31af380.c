// Function: sub_31AF380
// Address: 0x31af380
//
void __fastcall sub_31AF380(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __m128i *v8; // rax
  const __m128i *v9; // rdx
  __m128i *v10; // r13
  unsigned __int64 v11; // rdi
  int v12; // r14d
  unsigned __int64 v13[5]; // [rsp+8h] [rbp-28h] BYREF

  v6 = a1 + 16;
  v8 = (__m128i *)sub_C8D7D0(a1, a1 + 16, a2, 0x30u, v13, a6);
  v9 = *(const __m128i **)a1;
  v10 = v8;
  v11 = *(_QWORD *)a1 + 48LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v11 )
  {
    do
    {
      if ( v8 )
      {
        *v8 = _mm_loadu_si128(v9);
        v8[1] = _mm_loadu_si128(v9 + 1);
        v8[2].m128i_i32[2] = v9[2].m128i_i32[2];
        v8[2].m128i_i8[12] = v9[2].m128i_i8[12];
        v8[2].m128i_i64[0] = (__int64)&unk_4A347A0;
      }
      v9 += 3;
      v8 += 3;
    }
    while ( (const __m128i *)v11 != v9 );
    v11 = *(_QWORD *)a1;
  }
  v12 = v13[0];
  if ( v6 != v11 )
    _libc_free(v11);
  *(_QWORD *)a1 = v10;
  *(_DWORD *)(a1 + 12) = v12;
}
