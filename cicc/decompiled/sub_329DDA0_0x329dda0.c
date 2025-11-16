// Function: sub_329DDA0
// Address: 0x329dda0
//
__int64 __fastcall sub_329DDA0(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r14
  bool v7; // zf
  __int64 v8; // rax
  __m128i v9; // xmm0
  __int64 v10; // rax
  __int64 v11; // rax
  int v12; // edx
  __int64 v13; // rax
  __m128i v14; // xmm0
  __int64 v15; // rax

  if ( *(_DWORD *)a4 != *(_DWORD *)(a1 + 24) )
    return 0;
  v5 = a4 + 8;
  v7 = (unsigned __int8)sub_32700B0(a4 + 8, **(_QWORD **)(a1 + 40), *(_DWORD *)(*(_QWORD *)(a1 + 40) + 8LL)) == 0;
  v8 = *(_QWORD *)(a1 + 40);
  if ( v7 )
  {
    if ( !(unsigned __int8)sub_32700B0(v5, *(_QWORD *)(v8 + 40), *(_DWORD *)(v8 + 48)) )
      return 0;
    v14 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a1 + 40));
    v15 = *(_QWORD *)(a4 + 64);
    *(_QWORD *)v15 = v14.m128i_i64[0];
    *(_DWORD *)(v15 + 8) = v14.m128i_i32[2];
  }
  else
  {
    v9 = _mm_loadu_si128((const __m128i *)(v8 + 40));
    v10 = *(_QWORD *)(a4 + 64);
    *(_QWORD *)v10 = v9.m128i_i64[0];
    *(_DWORD *)(v10 + 8) = v9.m128i_i32[2];
  }
  if ( *(_BYTE *)(a4 + 76) && *(_DWORD *)(a4 + 72) != (*(_DWORD *)(a4 + 72) & *(_DWORD *)(a1 + 28)) )
    return 0;
  v11 = *(_QWORD *)(a1 + 56);
  if ( !v11 )
    return 0;
  v12 = 1;
  while ( 1 )
  {
    while ( *(_DWORD *)(v11 + 8) != a2 )
    {
      v11 = *(_QWORD *)(v11 + 32);
      if ( !v11 )
        return v12 ^ 1u;
    }
    if ( !v12 )
      return 0;
    v13 = *(_QWORD *)(v11 + 32);
    if ( !v13 )
      break;
    if ( a2 == *(_DWORD *)(v13 + 8) )
      return 0;
    v11 = *(_QWORD *)(v13 + 32);
    v12 = 0;
    if ( !v11 )
      return v12 ^ 1u;
  }
  return 1;
}
