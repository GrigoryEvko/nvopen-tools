// Function: sub_329EB20
// Address: 0x329eb20
//
char __fastcall sub_329EB20(__int64 a1, __int64 a2, __int64 a3)
{
  char result; // al
  _DWORD *v5; // rdx
  __int64 v6; // rax
  int v7; // edx
  __int64 v8; // rcx
  __int64 v9; // rcx
  __int64 v10; // rax
  int v11; // ecx
  __m128i v12; // [rsp-38h] [rbp-38h]
  __m128i v13; // [rsp-28h] [rbp-28h]

  if ( *(_DWORD *)a3 != *(_DWORD *)(a1 + 24) )
    return 0;
  v5 = *(_DWORD **)(a1 + 40);
  v6 = *(_QWORD *)v5;
  if ( *(_DWORD *)(a3 + 8) != *(_DWORD *)(*(_QWORD *)v5 + 24LL) )
    return 0;
  v7 = v5[2];
  v8 = *(_QWORD *)(a3 + 16);
  v13 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v6 + 40));
  *(_QWORD *)v8 = v13.m128i_i64[0];
  *(_DWORD *)(v8 + 8) = v13.m128i_i32[2];
  v9 = *(_QWORD *)(a3 + 24);
  v12 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v6 + 40) + 40LL));
  *(_QWORD *)v9 = v12.m128i_i64[0];
  *(_DWORD *)(v9 + 8) = v12.m128i_i32[2];
  if ( *(_BYTE *)(a3 + 36) )
  {
    if ( *(_DWORD *)(a3 + 32) != (*(_DWORD *)(a3 + 32) & *(_DWORD *)(v6 + 28)) )
      return 0;
  }
  v10 = *(_QWORD *)(v6 + 56);
  if ( !v10 )
    return 0;
  v11 = 1;
  do
  {
    if ( v7 == *(_DWORD *)(v10 + 8) )
    {
      if ( !v11 )
        return 0;
      v10 = *(_QWORD *)(v10 + 32);
      if ( !v10 )
        goto LABEL_17;
      if ( *(_DWORD *)(v10 + 8) == v7 )
        return 0;
      v11 = 0;
    }
    v10 = *(_QWORD *)(v10 + 32);
  }
  while ( v10 );
  if ( v11 == 1 )
    return 0;
LABEL_17:
  result = sub_32657E0(a3 + 40, *(_QWORD *)(*(_QWORD *)(a1 + 40) + 40LL));
  if ( !result )
    return 0;
  if ( *(_BYTE *)(a3 + 60) )
    return (*(_DWORD *)(a3 + 56) & *(_DWORD *)(a1 + 28)) == *(_DWORD *)(a3 + 56);
  return result;
}
