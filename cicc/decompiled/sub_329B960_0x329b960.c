// Function: sub_329B960
// Address: 0x329b960
//
__int64 __fastcall sub_329B960(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v5; // r13
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  _DWORD *v10; // rcx
  __m128i v11; // [rsp-48h] [rbp-48h]
  __m128i v12; // [rsp-38h] [rbp-38h]

  if ( *(_DWORD *)a4 != *(_DWORD *)(a1 + 24) )
    return 0;
  v5 = a4 + 16;
  v7 = *(_QWORD *)(a4 + 8);
  v12 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a1 + 40));
  *(_QWORD *)v7 = v12.m128i_i64[0];
  *(_DWORD *)(v7 + 8) = v12.m128i_i32[2];
  if ( !(unsigned __int8)sub_32657E0(a4 + 16, *(_QWORD *)(*(_QWORD *)(a1 + 40) + 40LL)) )
  {
    v9 = *(_QWORD *)(a4 + 8);
    v11 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a1 + 40) + 40LL));
    *(_QWORD *)v9 = v11.m128i_i64[0];
    *(_DWORD *)(v9 + 8) = v11.m128i_i32[2];
    if ( !(unsigned __int8)sub_32657E0(v5, **(_QWORD **)(a1 + 40)) )
      return 0;
  }
  v8 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 80LL);
  if ( *(_DWORD *)(v8 + 24) != 8 || *(_BYTE *)(a4 + 36) && *(_DWORD *)(a4 + 32) != *(_DWORD *)(v8 + 96) )
    return 0;
  v10 = *(_DWORD **)(a4 + 40);
  result = 1;
  if ( v10 )
    *v10 = *(_DWORD *)(v8 + 96);
  return result;
}
