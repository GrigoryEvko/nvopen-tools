// Function: sub_32A64A0
// Address: 0x32a64a0
//
bool __fastcall sub_32A64A0(__int64 a1, __int64 a2, __int64 a3)
{
  bool result; // al
  __int64 *v5; // rdx
  __int64 v6; // rax
  int v7; // ecx
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // rax
  int v11; // ecx
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rdx
  __m128i v17; // [rsp-78h] [rbp-78h]
  __m128i v18; // [rsp-68h] [rbp-68h]
  __m128i v19; // [rsp-48h] [rbp-48h]
  __m128i v20; // [rsp-38h] [rbp-38h]

  if ( *(_DWORD *)a3 != *(_DWORD *)(a1 + 24) )
    return 0;
  v5 = *(__int64 **)(a1 + 40);
  v6 = *v5;
  v7 = *((_DWORD *)v5 + 2);
  v8 = *(_QWORD *)(a3 + 40);
  *(_QWORD *)v8 = v6;
  *(_DWORD *)(v8 + 8) = v7;
  if ( *(_DWORD *)(a3 + 8) != *(_DWORD *)(v6 + 24) )
    goto LABEL_18;
  v13 = *(_QWORD *)(a3 + 16);
  v20 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v6 + 40));
  *(_QWORD *)v13 = v20.m128i_i64[0];
  *(_DWORD *)(v13 + 8) = v20.m128i_i32[2];
  v14 = *(_QWORD *)(a3 + 24);
  v19 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v6 + 40) + 40LL));
  *(_QWORD *)v14 = v19.m128i_i64[0];
  *(_DWORD *)(v14 + 8) = v19.m128i_i32[2];
  if ( *(_BYTE *)(a3 + 36) )
  {
    if ( *(_DWORD *)(a3 + 32) != (*(_DWORD *)(a3 + 32) & *(_DWORD *)(v6 + 28)) )
      goto LABEL_18;
  }
  if ( !(unsigned __int8)sub_32657E0(a3 + 48, *(_QWORD *)(*(_QWORD *)(a1 + 40) + 40LL)) )
  {
LABEL_18:
    v9 = *(_QWORD *)(a1 + 40);
    v10 = *(_QWORD *)(v9 + 40);
    v11 = *(_DWORD *)(v9 + 48);
    v12 = *(_QWORD *)(a3 + 40);
    *(_QWORD *)v12 = v10;
    *(_DWORD *)(v12 + 8) = v11;
    if ( *(_DWORD *)(a3 + 8) != *(_DWORD *)(v10 + 24) )
      return 0;
    v15 = *(_QWORD *)(a3 + 16);
    v18 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v10 + 40));
    *(_QWORD *)v15 = v18.m128i_i64[0];
    *(_DWORD *)(v15 + 8) = v18.m128i_i32[2];
    v16 = *(_QWORD *)(a3 + 24);
    v17 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v10 + 40) + 40LL));
    *(_QWORD *)v16 = v17.m128i_i64[0];
    *(_DWORD *)(v16 + 8) = v17.m128i_i32[2];
    if ( *(_BYTE *)(a3 + 36) )
    {
      if ( *(_DWORD *)(a3 + 32) != (*(_DWORD *)(a3 + 32) & *(_DWORD *)(v10 + 28)) )
        return 0;
    }
    if ( !(unsigned __int8)sub_32657E0(a3 + 48, **(_QWORD **)(a1 + 40)) )
      return 0;
  }
  result = 1;
  if ( *(_BYTE *)(a3 + 68) )
    return (*(_DWORD *)(a3 + 64) & *(_DWORD *)(a1 + 28)) == *(_DWORD *)(a3 + 64);
  return result;
}
