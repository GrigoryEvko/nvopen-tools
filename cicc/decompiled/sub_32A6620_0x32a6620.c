// Function: sub_32A6620
// Address: 0x32a6620
//
bool __fastcall sub_32A6620(__int64 a1, __int64 a2, __int64 a3)
{
  bool result; // al
  __int64 *v5; // rax
  __int64 v6; // r13
  int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // r13
  int v11; // edx
  __int64 v12; // rax
  __m128i v13; // xmm0
  __int64 v14; // rax
  __m128i v15; // xmm0
  __int64 v16; // rax
  __m128i v17; // xmm0
  __int64 v18; // rax
  __m128i v19; // xmm0
  __int64 v20; // rax
  __m128i v21; // xmm0
  __int64 v22; // rax
  __m128i v23; // xmm0
  __int64 v24; // rax

  result = 0;
  if ( *(_DWORD *)a3 == *(_DWORD *)(a1 + 24) )
  {
    v5 = *(__int64 **)(a1 + 40);
    v6 = *v5;
    v7 = *((_DWORD *)v5 + 2);
    v8 = *(_QWORD *)(a3 + 48);
    *(_QWORD *)v8 = v6;
    *(_DWORD *)(v8 + 8) = v7;
    if ( *(_DWORD *)(a3 + 8) != *(_DWORD *)(v6 + 24) )
      goto LABEL_3;
    v13 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v6 + 40));
    v14 = *(_QWORD *)(a3 + 16);
    *(_QWORD *)v14 = v13.m128i_i64[0];
    *(_DWORD *)(v14 + 8) = v13.m128i_i32[2];
    if ( !(unsigned __int8)sub_32657E0(a3 + 24, *(_QWORD *)(*(_QWORD *)(v6 + 40) + 40LL)) )
    {
      v21 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v6 + 40) + 40LL));
      v22 = *(_QWORD *)(a3 + 16);
      *(_QWORD *)v22 = v21.m128i_i64[0];
      *(_DWORD *)(v22 + 8) = v21.m128i_i32[2];
      if ( !(unsigned __int8)sub_32657E0(a3 + 24, **(_QWORD **)(v6 + 40)) )
        goto LABEL_3;
    }
    if ( !*(_BYTE *)(a3 + 44) || *(_DWORD *)(a3 + 40) == (*(_DWORD *)(a3 + 40) & *(_DWORD *)(v6 + 28)) )
    {
      v15 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a1 + 40) + 40LL));
      v16 = *(_QWORD *)(a3 + 56);
      *(_QWORD *)v16 = v15.m128i_i64[0];
      *(_DWORD *)(v16 + 8) = v15.m128i_i32[2];
    }
    else
    {
LABEL_3:
      v9 = *(_QWORD *)(a1 + 40);
      v10 = *(_QWORD *)(v9 + 40);
      v11 = *(_DWORD *)(v9 + 48);
      v12 = *(_QWORD *)(a3 + 48);
      *(_QWORD *)v12 = v10;
      *(_DWORD *)(v12 + 8) = v11;
      if ( *(_DWORD *)(a3 + 8) != *(_DWORD *)(v10 + 24) )
        return 0;
      v17 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v10 + 40));
      v18 = *(_QWORD *)(a3 + 16);
      *(_QWORD *)v18 = v17.m128i_i64[0];
      *(_DWORD *)(v18 + 8) = v17.m128i_i32[2];
      if ( !(unsigned __int8)sub_32657E0(a3 + 24, *(_QWORD *)(*(_QWORD *)(v10 + 40) + 40LL)) )
      {
        v23 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v10 + 40) + 40LL));
        v24 = *(_QWORD *)(a3 + 16);
        *(_QWORD *)v24 = v23.m128i_i64[0];
        *(_DWORD *)(v24 + 8) = v23.m128i_i32[2];
        if ( !(unsigned __int8)sub_32657E0(a3 + 24, **(_QWORD **)(v10 + 40)) )
          return 0;
      }
      if ( *(_BYTE *)(a3 + 44) && *(_DWORD *)(a3 + 40) != (*(_DWORD *)(a3 + 40) & *(_DWORD *)(v10 + 28)) )
        return 0;
      v19 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a1 + 40));
      v20 = *(_QWORD *)(a3 + 56);
      *(_QWORD *)v20 = v19.m128i_i64[0];
      *(_DWORD *)(v20 + 8) = v19.m128i_i32[2];
    }
    result = 1;
    if ( *(_BYTE *)(a3 + 68) )
      return (*(_DWORD *)(a3 + 64) & *(_DWORD *)(a1 + 28)) == *(_DWORD *)(a3 + 64);
  }
  return result;
}
