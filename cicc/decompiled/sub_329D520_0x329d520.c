// Function: sub_329D520
// Address: 0x329d520
//
__int64 __fastcall sub_329D520(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // eax
  __int64 result; // rax
  _QWORD *v6; // rax
  int v7; // esi
  __int64 v8; // rdx
  __int64 v9; // rax
  __m128i v10; // xmm0
  __int64 v11; // rax
  __int64 v12; // r9
  __int64 v13; // r8
  __int64 v14; // r10
  _QWORD *v15; // rsi
  __int64 v16; // rax
  __m128i v17; // xmm0
  __int64 v18; // rdx
  __int64 v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // r8
  __m128i v22; // xmm0
  __int64 v23; // rax
  __m128i v24; // xmm0
  __int64 v25; // rdx
  __int64 v26; // rsi
  __int64 v27; // rdx
  __int64 v28; // r8
  __m128i v29; // xmm0
  __int64 v30; // rax
  __m128i v31; // xmm0
  __int64 v32; // rax

  v4 = *(_DWORD *)(a1 + 24);
  if ( *(_DWORD *)(a4 + 64) != v4 )
    goto LABEL_2;
  v6 = *(_QWORD **)(a1 + 40);
  v7 = *(_DWORD *)(a4 + 72);
  v8 = *v6;
  if ( v7 == *(_DWORD *)(*v6 + 24LL) )
  {
    v10 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v8 + 40));
    v11 = *(_QWORD *)(a4 + 80);
    *(_QWORD *)v11 = v10.m128i_i64[0];
    *(_DWORD *)(v11 + 8) = v10.m128i_i32[2];
    v12 = *(_QWORD *)(v8 + 40);
    v13 = *(_QWORD *)(a4 + 88);
    v14 = *(_QWORD *)(v12 + 40);
    if ( v13 )
    {
      v6 = *(_QWORD **)(a1 + 40);
      v15 = v6;
      if ( v14 == v13 && *(_DWORD *)(v12 + 48) == *(_DWORD *)(a4 + 96) )
      {
LABEL_30:
        v6 = v15;
        if ( !*(_BYTE *)(a4 + 108) || *(_DWORD *)(a4 + 104) == (*(_DWORD *)(a4 + 104) & *(_DWORD *)(v8 + 28)) )
        {
          v31 = _mm_loadu_si128((const __m128i *)(v15 + 5));
          v32 = *(_QWORD *)(a4 + 112);
          *(_QWORD *)v32 = v31.m128i_i64[0];
          *(_DWORD *)(v32 + 8) = v31.m128i_i32[2];
          goto LABEL_25;
        }
      }
    }
    else
    {
      if ( v14 )
      {
        v15 = *(_QWORD **)(a1 + 40);
        goto LABEL_30;
      }
      v6 = *(_QWORD **)(a1 + 40);
    }
    v7 = *(_DWORD *)(a4 + 72);
  }
  v9 = v6[5];
  if ( *(_DWORD *)(v9 + 24) == v7 )
  {
    v24 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v9 + 40));
    v25 = *(_QWORD *)(a4 + 80);
    *(_QWORD *)v25 = v24.m128i_i64[0];
    *(_DWORD *)(v25 + 8) = v24.m128i_i32[2];
    v26 = *(_QWORD *)(v9 + 40);
    v27 = *(_QWORD *)(a4 + 88);
    v28 = *(_QWORD *)(v26 + 40);
    if ( v27 )
    {
      if ( v28 != v27 || *(_DWORD *)(v26 + 48) != *(_DWORD *)(a4 + 96) )
        goto LABEL_6;
    }
    else if ( !v28 )
    {
      goto LABEL_6;
    }
    if ( *(_BYTE *)(a4 + 108) && *(_DWORD *)(a4 + 104) != (*(_DWORD *)(a4 + 104) & *(_DWORD *)(v9 + 28)) )
      goto LABEL_6;
    v29 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a1 + 40));
    v30 = *(_QWORD *)(a4 + 112);
    *(_QWORD *)v30 = v29.m128i_i64[0];
    *(_DWORD *)(v30 + 8) = v29.m128i_i32[2];
LABEL_25:
    result = *(unsigned __int8 *)(a4 + 124);
    if ( !(_BYTE)result )
      return 1;
    if ( *(_DWORD *)(a4 + 120) == (*(_DWORD *)(a4 + 120) & *(_DWORD *)(a1 + 28)) )
      return result;
  }
LABEL_6:
  v4 = *(_DWORD *)(a1 + 24);
LABEL_2:
  if ( *(_DWORD *)a4 != v4 )
    return 0;
  v16 = **(_QWORD **)(a1 + 40);
  if ( *(_DWORD *)(a4 + 8) != *(_DWORD *)(v16 + 24) )
    return 0;
  v17 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v16 + 40));
  v18 = *(_QWORD *)(a4 + 16);
  *(_QWORD *)v18 = v17.m128i_i64[0];
  *(_DWORD *)(v18 + 8) = v17.m128i_i32[2];
  v19 = *(_QWORD *)(v16 + 40);
  v20 = *(_QWORD *)(a4 + 24);
  v21 = *(_QWORD *)(v19 + 40);
  if ( v20 )
  {
    if ( v21 != v20 || *(_DWORD *)(v19 + 48) != *(_DWORD *)(a4 + 32) )
      return 0;
    goto LABEL_14;
  }
  if ( v21 )
  {
LABEL_14:
    if ( !*(_BYTE *)(a4 + 44) || *(_DWORD *)(a4 + 40) == (*(_DWORD *)(a4 + 40) & *(_DWORD *)(v16 + 28)) )
    {
      v22 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a1 + 40) + 40LL));
      v23 = *(_QWORD *)(a4 + 48);
      *(_QWORD *)v23 = v22.m128i_i64[0];
      *(_DWORD *)(v23 + 8) = v22.m128i_i32[2];
      if ( !*(_BYTE *)(a4 + 60) || *(_DWORD *)(a4 + 56) == (*(_DWORD *)(a4 + 56) & *(_DWORD *)(a1 + 28)) )
        return 1;
    }
    return 0;
  }
  return 0;
}
