// Function: sub_38C87C0
// Address: 0x38c87c0
//
void __fastcall sub_38C87C0(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rdx
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rbx
  unsigned __int64 v5; // rax
  __int64 v6; // r13
  const __m128i *v7; // rax
  unsigned __int64 *v8; // r14
  const __m128i *v9; // rdx
  unsigned __int64 v10; // rsi
  __m128i *v11; // rax
  __int32 v12; // ecx
  const __m128i *v13; // rcx
  unsigned __int64 *v14; // r15

  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation", 1u);
  v2 = ((((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
      | (*(unsigned int *)(a1 + 12) + 2LL)
      | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4;
  v3 = ((v2
       | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 8)
     | v2
     | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
     | (*(unsigned int *)(a1 + 12) + 2LL)
     | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1);
  v4 = a2;
  v5 = (v3 | (v3 >> 16) | HIDWORD(v3)) + 1;
  if ( v5 >= a2 )
    v4 = v5;
  if ( v4 > 0xFFFFFFFF )
    v4 = 0xFFFFFFFFLL;
  v6 = malloc(72 * v4);
  if ( !v6 )
    sub_16BD1C0("Allocation failed", 1u);
  v7 = *(const __m128i **)a1;
  v8 = (unsigned __int64 *)(*(_QWORD *)a1 + 72LL * *(unsigned int *)(a1 + 8));
  if ( *(unsigned __int64 **)a1 != v8 )
  {
    v9 = v7 + 1;
    v10 = v6 + 8 * ((unsigned __int64)((char *)v8 - (char *)v7 - 72) >> 3) + 72;
    v11 = (__m128i *)v6;
    do
    {
      if ( v11 )
      {
        v11->m128i_i64[0] = (__int64)v11[1].m128i_i64;
        v13 = (const __m128i *)v9[-1].m128i_i64[0];
        if ( v13 == v9 )
        {
          v11[1] = _mm_loadu_si128(v9);
        }
        else
        {
          v11->m128i_i64[0] = (__int64)v13;
          v11[1].m128i_i64[0] = v9->m128i_i64[0];
        }
        v11->m128i_i64[1] = v9[-1].m128i_i64[1];
        v12 = v9[1].m128i_i32[0];
        v9[-1].m128i_i64[0] = (__int64)v9;
        v9[-1].m128i_i64[1] = 0;
        v9->m128i_i8[0] = 0;
        v11[2].m128i_i32[0] = v12;
        v11[2].m128i_i64[1] = v9[1].m128i_i64[1];
        LOBYTE(v12) = v9[3].m128i_i8[0];
        v11[4].m128i_i8[0] = v12;
        if ( (_BYTE)v12 )
          v11[3] = _mm_loadu_si128(v9 + 2);
      }
      v11 = (__m128i *)((char *)v11 + 72);
      v9 = (const __m128i *)((char *)v9 + 72);
    }
    while ( v11 != (__m128i *)v10 );
    v14 = *(unsigned __int64 **)a1;
    v8 = (unsigned __int64 *)(*(_QWORD *)a1 + 72LL * *(unsigned int *)(a1 + 8));
    if ( *(unsigned __int64 **)a1 != v8 )
    {
      do
      {
        v8 -= 9;
        if ( (unsigned __int64 *)*v8 != v8 + 2 )
          j_j___libc_free_0(*v8);
      }
      while ( v8 != v14 );
      v8 = *(unsigned __int64 **)a1;
    }
  }
  if ( v8 != (unsigned __int64 *)(a1 + 16) )
    _libc_free((unsigned __int64)v8);
  *(_QWORD *)a1 = v6;
  *(_DWORD *)(a1 + 12) = v4;
}
