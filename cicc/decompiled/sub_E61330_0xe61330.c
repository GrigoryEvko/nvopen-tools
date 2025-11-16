// Function: sub_E61330
// Address: 0xe61330
//
__int64 __fastcall sub_E61330(__int64 a1, const __m128i *a2)
{
  __int64 v2; // r8
  __int64 v4; // r12
  unsigned __int32 v5; // r13d
  unsigned __int64 v6; // r14
  unsigned __int64 v7; // r9
  unsigned __int32 v8; // edx
  __int64 result; // rax
  _BOOL4 v10; // r10d
  __int64 v11; // rax
  __m128i *v12; // rsi
  _BOOL4 v13; // [rsp+Ch] [rbp-44h]
  __int64 v14; // [rsp+10h] [rbp-40h]
  unsigned __int64 v15; // [rsp+18h] [rbp-38h]

  v2 = a1 + 216;
  v4 = *(_QWORD *)(a1 + 224);
  v5 = a2->m128i_u32[2];
  v6 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(*(_QWORD *)(a1 + 264) - *(_QWORD *)(a1 + 256)) >> 3);
  v7 = v6 + 1;
  if ( v4 )
  {
    while ( 1 )
    {
      v8 = *(_DWORD *)(v4 + 32);
      result = *(_QWORD *)(v4 + 24);
      if ( v5 < v8 )
        result = *(_QWORD *)(v4 + 16);
      if ( !result )
        break;
      v4 = result;
    }
    if ( v5 < v8 )
    {
      if ( *(_QWORD *)(a1 + 232) != v4 )
        goto LABEL_15;
    }
    else if ( v5 <= v8 )
    {
      goto LABEL_17;
    }
    v10 = 1;
    if ( v4 == v2 )
      goto LABEL_10;
    goto LABEL_21;
  }
  v4 = a1 + 216;
  if ( v2 == *(_QWORD *)(a1 + 232) )
  {
    v10 = 1;
    goto LABEL_10;
  }
LABEL_15:
  result = sub_220EF80(v4);
  v7 = v6 + 1;
  if ( v5 <= *(_DWORD *)(result + 32) )
  {
    v4 = result;
LABEL_17:
    *(_QWORD *)(v4 + 48) = v7;
    v12 = *(__m128i **)(a1 + 264);
    if ( v12 != *(__m128i **)(a1 + 272) )
      goto LABEL_11;
    return sub_E61180((const __m128i **)(a1 + 256), v12, a2);
  }
  v2 = a1 + 216;
  if ( !v4 )
  {
    MEMORY[0x30] = 0;
    BUG();
  }
  v10 = 1;
  if ( v4 != a1 + 216 )
LABEL_21:
    v10 = v5 < *(_DWORD *)(v4 + 32);
LABEL_10:
  v13 = v10;
  v14 = v2;
  v15 = v7;
  v11 = sub_22077B0(56);
  *(_DWORD *)(v11 + 32) = v5;
  *(_QWORD *)(v11 + 40) = v6;
  *(_QWORD *)(v11 + 48) = v15;
  result = sub_220F040(v13, v11, v4, v14);
  ++*(_QWORD *)(a1 + 248);
  v12 = *(__m128i **)(a1 + 264);
  if ( v12 == *(__m128i **)(a1 + 272) )
    return sub_E61180((const __m128i **)(a1 + 256), v12, a2);
LABEL_11:
  if ( v12 )
  {
    *v12 = _mm_loadu_si128(a2);
    result = a2[1].m128i_i64[0];
    v12[1].m128i_i64[0] = result;
    v12 = *(__m128i **)(a1 + 264);
  }
  *(_QWORD *)(a1 + 264) = (char *)v12 + 24;
  return result;
}
