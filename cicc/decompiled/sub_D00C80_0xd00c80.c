// Function: sub_D00C80
// Address: 0xd00c80
//
__int64 __fastcall sub_D00C80(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r13
  __int64 v9; // rsi
  __int64 v10; // rbx
  __int64 result; // rax
  __int64 v12; // r12
  __m128i *v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // r15
  __int64 v17; // rdi
  int v18; // r15d
  unsigned __int64 v19[7]; // [rsp+8h] [rbp-38h] BYREF

  v8 = a1 + 16;
  v9 = a1 + 16;
  v10 = sub_C8D7D0(a1, a1 + 16, a2, 0x38u, v19, a6);
  result = *(_QWORD *)a1;
  v12 = *(_QWORD *)a1 + 56LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v12 )
  {
    v13 = (__m128i *)v10;
    do
    {
      if ( v13 )
      {
        *v13 = _mm_loadu_si128((const __m128i *)result);
        v13[1].m128i_i64[0] = *(_QWORD *)(result + 16);
        v13[2].m128i_i32[0] = *(_DWORD *)(result + 32);
        v13[1].m128i_i64[1] = *(_QWORD *)(result + 24);
        v14 = *(_QWORD *)(result + 40);
        *(_DWORD *)(result + 32) = 0;
        v13[2].m128i_i64[1] = v14;
        v13[3].m128i_i8[0] = *(_BYTE *)(result + 48);
        v13[3].m128i_i8[1] = *(_BYTE *)(result + 49);
      }
      result += 56;
      v13 = (__m128i *)((char *)v13 + 56);
    }
    while ( v12 != result );
    v15 = *(unsigned int *)(a1 + 8);
    v16 = *(_QWORD *)a1;
    result = 7 * v15;
    v12 = *(_QWORD *)a1 + 56 * v15;
    if ( *(_QWORD *)a1 != v12 )
    {
      do
      {
        v12 -= 56;
        if ( *(_DWORD *)(v12 + 32) > 0x40u )
        {
          v17 = *(_QWORD *)(v12 + 24);
          if ( v17 )
            result = j_j___libc_free_0_0(v17);
        }
      }
      while ( v16 != v12 );
      v12 = *(_QWORD *)a1;
    }
  }
  v18 = v19[0];
  if ( v8 != v12 )
    result = _libc_free(v12, v9);
  *(_QWORD *)a1 = v10;
  *(_DWORD *)(a1 + 12) = v18;
  return result;
}
