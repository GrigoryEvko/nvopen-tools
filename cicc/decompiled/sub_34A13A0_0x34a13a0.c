// Function: sub_34A13A0
// Address: 0x34a13a0
//
void __fastcall sub_34A13A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rbx
  bool v6; // zf
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 i; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __m128i *v13; // rax
  __m128i *v14; // rdi
  __int64 v15; // rdx
  unsigned __int64 v16; // rdi
  __m128i *v17; // [rsp+8h] [rbp-28h] BYREF

  v5 = a2;
  v6 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v6 )
  {
    v7 = *(_QWORD *)(a1 + 16);
    v8 = 72LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v7 = a1 + 16;
    v8 = 576;
  }
  for ( i = v7 + v8; i != v7; v7 += 72 )
  {
    if ( v7 )
    {
      *(_QWORD *)v7 = 0;
      *(_BYTE *)(v7 + 24) = 0;
      *(_QWORD *)(v7 + 32) = 0;
    }
  }
  if ( a2 != a3 )
  {
    do
    {
      if ( *(_QWORD *)v5 || *(_BYTE *)(v5 + 24) && (*(_QWORD *)(v5 + 8) || *(_QWORD *)(v5 + 16)) || *(_QWORD *)(v5 + 32) )
      {
        sub_34A1150(a1, v5, (__int64 *)&v17);
        v13 = v17;
        *v17 = _mm_loadu_si128((const __m128i *)v5);
        v14 = v17;
        v13[1] = _mm_loadu_si128((const __m128i *)(v5 + 16));
        v15 = *(_QWORD *)(v5 + 32);
        v13[2].m128i_i64[0] = v15;
        v14[2].m128i_i64[1] = (__int64)&v14[3].m128i_i64[1];
        v14[3].m128i_i64[0] = 0x200000000LL;
        if ( *(_DWORD *)(v5 + 48) )
          sub_349D9E0((__int64)&v14[2].m128i_i64[1], (char **)(v5 + 40), v15, v10, v11, v12);
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
        v16 = *(_QWORD *)(v5 + 40);
        if ( v16 != v5 + 56 )
          _libc_free(v16);
      }
      v5 += 72;
    }
    while ( a3 != v5 );
  }
}
