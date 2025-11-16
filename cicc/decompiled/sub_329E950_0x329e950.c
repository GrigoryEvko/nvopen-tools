// Function: sub_329E950
// Address: 0x329e950
//
__int64 __fastcall sub_329E950(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  _QWORD *v5; // rdx
  int v6; // eax
  __int64 v7; // r14
  __int64 v8; // r14
  char v9; // al
  __int64 v10; // rax
  __m128i v11; // xmm0
  __int64 v12; // rax
  __int64 v13; // rax
  int v14; // edx
  __int64 v15; // rax
  __m128i v16; // xmm0
  __int64 v17; // rax
  __m128i v18; // xmm0
  __int64 v19; // rax
  __int64 v20; // [rsp-70h] [rbp-70h]
  __int64 v21; // [rsp-70h] [rbp-70h]
  __m128i v22; // [rsp-38h] [rbp-38h]

  if ( *(_DWORD *)a4 != *(_DWORD *)(a1 + 24) )
    return 0;
  v5 = *(_QWORD **)(a1 + 40);
  v6 = *(_DWORD *)(a4 + 8);
  v7 = *v5;
  if ( v6 == *(_DWORD *)(*v5 + 24LL) )
  {
    v20 = a4;
    v9 = sub_32657E0(a4 + 16, **(_QWORD **)(v7 + 40));
    a4 = v20;
    if ( v9 )
    {
      v10 = *(_QWORD *)(v20 + 32);
      v22 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v7 + 40) + 40LL));
      *(_QWORD *)v10 = v22.m128i_i64[0];
      *(_DWORD *)(v10 + 8) = v22.m128i_i32[2];
      if ( !*(_BYTE *)(v20 + 44) || *(_DWORD *)(v20 + 40) == (*(_DWORD *)(v20 + 40) & *(_DWORD *)(v7 + 28)) )
      {
        v11 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a1 + 40) + 40LL));
        v12 = *(_QWORD *)(v20 + 48);
        *(_QWORD *)v12 = v11.m128i_i64[0];
        *(_DWORD *)(v12 + 8) = v11.m128i_i32[2];
        goto LABEL_11;
      }
    }
    v5 = *(_QWORD **)(a1 + 40);
    v6 = *(_DWORD *)(v20 + 8);
  }
  v8 = v5[5];
  if ( *(_DWORD *)(v8 + 24) != v6 )
    return 0;
  v21 = a4;
  if ( !(unsigned __int8)sub_32657E0(a4 + 16, **(_QWORD **)(v8 + 40)) )
    return 0;
  a4 = v21;
  v16 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v8 + 40) + 40LL));
  v17 = *(_QWORD *)(v21 + 32);
  *(_QWORD *)v17 = v16.m128i_i64[0];
  *(_DWORD *)(v17 + 8) = v16.m128i_i32[2];
  if ( *(_BYTE *)(v21 + 44) )
  {
    if ( *(_DWORD *)(v21 + 40) != (*(_DWORD *)(v21 + 40) & *(_DWORD *)(v8 + 28)) )
      return 0;
  }
  v18 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a1 + 40));
  v19 = *(_QWORD *)(v21 + 48);
  *(_QWORD *)v19 = v18.m128i_i64[0];
  *(_DWORD *)(v19 + 8) = v18.m128i_i32[2];
LABEL_11:
  if ( *(_BYTE *)(a4 + 60) && *(_DWORD *)(a4 + 56) != (*(_DWORD *)(a4 + 56) & *(_DWORD *)(a1 + 28)) )
    return 0;
  v13 = *(_QWORD *)(a1 + 56);
  if ( !v13 )
    return 0;
  v14 = 1;
  while ( 1 )
  {
    while ( *(_DWORD *)(v13 + 8) != a2 )
    {
      v13 = *(_QWORD *)(v13 + 32);
      if ( !v13 )
        return v14 ^ 1u;
    }
    if ( !v14 )
      return 0;
    v15 = *(_QWORD *)(v13 + 32);
    if ( !v15 )
      break;
    if ( a2 == *(_DWORD *)(v15 + 8) )
      return 0;
    v13 = *(_QWORD *)(v15 + 32);
    v14 = 0;
    if ( !v13 )
      return v14 ^ 1u;
  }
  return 1;
}
