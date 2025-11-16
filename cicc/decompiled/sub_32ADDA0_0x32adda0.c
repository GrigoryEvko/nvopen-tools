// Function: sub_32ADDA0
// Address: 0x32adda0
//
bool __fastcall sub_32ADDA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  bool result; // al
  __int64 *v6; // rax
  int v7; // edx
  __int64 v8; // r13
  __int64 v9; // r13
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __m128i v15; // [rsp-68h] [rbp-68h]
  __m128i v16; // [rsp-58h] [rbp-58h]
  __m128i v17; // [rsp-48h] [rbp-48h]
  __m128i v18; // [rsp-38h] [rbp-38h]

  if ( *(_DWORD *)a4 != *(_DWORD *)(a1 + 24) )
    return 0;
  v6 = *(__int64 **)(a1 + 40);
  v7 = *(_DWORD *)(a4 + 8);
  v8 = *v6;
  if ( v7 == *(_DWORD *)(*v6 + 24) )
  {
    v10 = a4 + 24;
    v11 = *(_QWORD *)(a4 + 16);
    v18 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v8 + 40));
    *(_QWORD *)v11 = v18.m128i_i64[0];
    *(_DWORD *)(v11 + 8) = v18.m128i_i32[2];
    if ( (unsigned __int8)sub_32657E0(a4 + 24, *(_QWORD *)(*(_QWORD *)(v8 + 40) + 40LL))
      || (v13 = *(_QWORD *)(a4 + 16),
          v17 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v8 + 40) + 40LL)),
          *(_QWORD *)v13 = v17.m128i_i64[0],
          *(_DWORD *)(v13 + 8) = v17.m128i_i32[2],
          (unsigned __int8)sub_32657E0(v10, **(_QWORD **)(v8 + 40))) )
    {
      v6 = *(__int64 **)(a1 + 40);
      if ( *(_BYTE *)(a4 + 44) && *(_DWORD *)(a4 + 40) != (*(_DWORD *)(a4 + 40) & *(_DWORD *)(v8 + 28)) )
        goto LABEL_12;
      if ( (unsigned __int8)sub_32657E0(a4 + 48, v6[5]) )
        goto LABEL_17;
    }
    v6 = *(__int64 **)(a1 + 40);
LABEL_12:
    v7 = *(_DWORD *)(a4 + 8);
  }
  v9 = v6[5];
  if ( v7 != *(_DWORD *)(v9 + 24) )
    return 0;
  v12 = *(_QWORD *)(a4 + 16);
  v16 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v9 + 40));
  *(_QWORD *)v12 = v16.m128i_i64[0];
  *(_DWORD *)(v12 + 8) = v16.m128i_i32[2];
  if ( !(unsigned __int8)sub_32657E0(a4 + 24, *(_QWORD *)(*(_QWORD *)(v9 + 40) + 40LL)) )
  {
    v14 = *(_QWORD *)(a4 + 16);
    v15 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v9 + 40) + 40LL));
    *(_QWORD *)v14 = v15.m128i_i64[0];
    *(_DWORD *)(v14 + 8) = v15.m128i_i32[2];
    if ( !(unsigned __int8)sub_32657E0(a4 + 24, **(_QWORD **)(v9 + 40)) )
      return 0;
  }
  if ( *(_BYTE *)(a4 + 44) && *(_DWORD *)(a4 + 40) != (*(_DWORD *)(a4 + 40) & *(_DWORD *)(v9 + 28))
    || !(unsigned __int8)sub_32657E0(a4 + 48, **(_QWORD **)(a1 + 40)) )
  {
    return 0;
  }
LABEL_17:
  result = 1;
  if ( *(_BYTE *)(a4 + 68) )
    return (*(_DWORD *)(a4 + 64) & *(_DWORD *)(a1 + 28)) == *(_DWORD *)(a4 + 64);
  return result;
}
