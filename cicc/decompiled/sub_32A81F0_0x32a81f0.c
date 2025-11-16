// Function: sub_32A81F0
// Address: 0x32a81f0
//
bool __fastcall sub_32A81F0(__int64 a1, __int64 a2, __int64 a3)
{
  bool result; // al
  _QWORD *v5; // rdx
  int v6; // eax
  __int64 v7; // r13
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rax
  __m128i v11; // [rsp-48h] [rbp-48h]
  __m128i v12; // [rsp-38h] [rbp-38h]

  if ( *(_DWORD *)a3 != *(_DWORD *)(a1 + 24) )
    return 0;
  v5 = *(_QWORD **)(a1 + 40);
  v6 = *(_DWORD *)(a3 + 8);
  v7 = *v5;
  if ( v6 == *(_DWORD *)(*v5 + 24LL) )
  {
    if ( (unsigned __int8)sub_32657E0(a3 + 16, **(_QWORD **)(v7 + 40)) )
    {
      v9 = *(_QWORD *)(a3 + 32);
      v12 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v7 + 40) + 40LL));
      *(_QWORD *)v9 = v12.m128i_i64[0];
      *(_DWORD *)(v9 + 8) = v12.m128i_i32[2];
      if ( !*(_BYTE *)(a3 + 44) || *(_DWORD *)(a3 + 40) == (*(_DWORD *)(a3 + 40) & *(_DWORD *)(v7 + 28)) )
      {
        if ( (unsigned __int8)sub_32657E0(a3 + 48, *(_QWORD *)(*(_QWORD *)(a1 + 40) + 40LL)) )
          goto LABEL_16;
      }
    }
    v5 = *(_QWORD **)(a1 + 40);
    v6 = *(_DWORD *)(a3 + 8);
  }
  v8 = v5[5];
  if ( v6 != *(_DWORD *)(v8 + 24) )
    return 0;
  if ( !(unsigned __int8)sub_32657E0(a3 + 16, **(_QWORD **)(v8 + 40)) )
    return 0;
  v10 = *(_QWORD *)(a3 + 32);
  v11 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v8 + 40) + 40LL));
  *(_QWORD *)v10 = v11.m128i_i64[0];
  *(_DWORD *)(v10 + 8) = v11.m128i_i32[2];
  if ( *(_BYTE *)(a3 + 44) )
  {
    if ( *(_DWORD *)(a3 + 40) != (*(_DWORD *)(a3 + 40) & *(_DWORD *)(v8 + 28)) )
      return 0;
  }
  if ( !(unsigned __int8)sub_32657E0(a3 + 48, **(_QWORD **)(a1 + 40)) )
    return 0;
LABEL_16:
  result = 1;
  if ( *(_BYTE *)(a3 + 68) )
    return (*(_DWORD *)(a3 + 64) & *(_DWORD *)(a1 + 28)) == *(_DWORD *)(a3 + 64);
  return result;
}
