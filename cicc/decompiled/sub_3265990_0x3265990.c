// Function: sub_3265990
// Address: 0x3265990
//
bool __fastcall sub_3265990(__int64 a1, __int64 a2)
{
  bool result; // al
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // r13
  __m128i v6; // xmm0
  __int64 v7; // rax
  _DWORD *v8; // rax
  __int64 v9; // r13
  int v10; // r14d
  __int64 v11; // rax
  __int64 v12; // rdx
  int v13; // r14d
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rax
  __m128i v18; // [rsp-38h] [rbp-38h]

  if ( *(_DWORD *)a1 != *(_DWORD *)(a2 + 24) )
    return 0;
  v3 = *(_QWORD *)(a1 + 8);
  v18 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  *(_QWORD *)v3 = v18.m128i_i64[0];
  *(_DWORD *)(v3 + 8) = v18.m128i_i32[2];
  v4 = *(_QWORD *)(a2 + 40);
  v5 = *(_QWORD *)(v4 + 40);
  if ( *(_DWORD *)(a1 + 24) == *(_DWORD *)(v5 + 24) )
  {
    v10 = *(_DWORD *)(v4 + 48);
    if ( (unsigned __int8)sub_32657E0(a1 + 32, **(_QWORD **)(v5 + 40)) )
    {
      v11 = *(_QWORD *)(a1 + 48);
      v12 = *(_QWORD *)(v5 + 40);
      if ( *(_QWORD *)(v12 + 40) == *(_QWORD *)v11
        && *(_DWORD *)(v12 + 48) == *(_DWORD *)(v11 + 8)
        && (!*(_BYTE *)(a1 + 60) || *(_DWORD *)(a1 + 56) == (*(_DWORD *)(a1 + 56) & *(_DWORD *)(v5 + 28))) )
      {
        v17 = *(_QWORD *)(a1 + 16);
        *(_QWORD *)v17 = v5;
        *(_DWORD *)(v17 + 8) = v10;
        goto LABEL_16;
      }
    }
    v4 = *(_QWORD *)(a2 + 40);
  }
  v6 = _mm_loadu_si128((const __m128i *)(v4 + 40));
  v7 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)v7 = v6.m128i_i64[0];
  *(_DWORD *)(v7 + 8) = v6.m128i_i32[2];
  v8 = *(_DWORD **)(a2 + 40);
  v9 = *(_QWORD *)v8;
  if ( *(_DWORD *)(a1 + 24) != *(_DWORD *)(*(_QWORD *)v8 + 24LL) )
    return 0;
  v13 = v8[2];
  if ( !(unsigned __int8)sub_32657E0(a1 + 32, **(_QWORD **)(v9 + 40)) )
    return 0;
  v14 = *(_QWORD *)(a1 + 48);
  v15 = *(_QWORD *)(v9 + 40);
  if ( *(_QWORD *)(v15 + 40) != *(_QWORD *)v14
    || *(_DWORD *)(v15 + 48) != *(_DWORD *)(v14 + 8)
    || *(_BYTE *)(a1 + 60) && *(_DWORD *)(a1 + 56) != (*(_DWORD *)(a1 + 56) & *(_DWORD *)(v9 + 28)) )
  {
    return 0;
  }
  v16 = *(_QWORD *)(a1 + 16);
  *(_QWORD *)v16 = v9;
  *(_DWORD *)(v16 + 8) = v13;
LABEL_16:
  result = 1;
  if ( *(_BYTE *)(a1 + 68) )
    return (*(_DWORD *)(a1 + 64) & *(_DWORD *)(a2 + 28)) == *(_DWORD *)(a1 + 64);
  return result;
}
