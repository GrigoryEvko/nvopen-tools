// Function: sub_32A8340
// Address: 0x32a8340
//
__int64 __fastcall sub_32A8340(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  unsigned int v6; // r8d
  __int64 v7; // rdx
  __int64 v8; // rax
  int v9; // edx
  __int64 v10; // rax
  __m128i v12; // [rsp-28h] [rbp-28h]

  if ( *(_DWORD *)a4 != *(_DWORD *)(a1 + 24) )
    return 0;
  v5 = **(_QWORD **)(a1 + 40);
  if ( *(_DWORD *)(a4 + 8) == *(_DWORD *)(v5 + 24)
    && ((v7 = *(_QWORD *)(a4 + 16),
         v12 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v5 + 40)),
         *(_QWORD *)v7 = v12.m128i_i64[0],
         *(_DWORD *)(v7 + 8) = v12.m128i_i32[2],
         !*(_BYTE *)(a4 + 28))
     || *(_DWORD *)(a4 + 24) == (*(_DWORD *)(a4 + 24) & *(_DWORD *)(v5 + 28)))
    && (v6 = sub_32657E0(a4 + 32, *(_QWORD *)(*(_QWORD *)(a1 + 40) + 40LL)), (_BYTE)v6)
    && (!*(_BYTE *)(a4 + 52) || *(_DWORD *)(a4 + 48) == (*(_DWORD *)(a4 + 48) & *(_DWORD *)(a1 + 28)))
    && (v8 = *(_QWORD *)(a1 + 56)) != 0 )
  {
    v9 = 1;
    do
    {
      while ( *(_DWORD *)(v8 + 8) != a2 )
      {
        v8 = *(_QWORD *)(v8 + 32);
        if ( !v8 )
          return v9 ^ 1u;
      }
      if ( !v9 )
        return 0;
      v10 = *(_QWORD *)(v8 + 32);
      if ( !v10 )
        return v6;
      if ( a2 == *(_DWORD *)(v10 + 8) )
        return 0;
      v8 = *(_QWORD *)(v10 + 32);
      v9 = 0;
    }
    while ( v8 );
    return v9 ^ 1u;
  }
  else
  {
    return 0;
  }
}
