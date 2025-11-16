// Function: sub_77F070
// Address: 0x77f070
//
_BOOL8 __fastcall sub_77F070(__int64 a1, __int64 a2, _QWORD *a3)
{
  _BYTE *v6; // rax
  __int64 v7; // rdi
  __int64 v8; // r13
  unsigned int v9; // ecx
  __int64 v10; // rsi
  unsigned int v11; // edx
  __m128i *v12; // rax
  __m128i v13; // xmm0
  __m128i *v14; // rax
  int v15; // eax
  __m128i *v16; // rsi
  const __m128i *v17; // rcx
  __m128i *v19; // rax

  v6 = sub_724D50(0);
  *a3 = v6;
  v7 = *(_QWORD *)(a2 + 24);
  v8 = (__int64)v6;
  v9 = *(_DWORD *)(a1 + 8);
  v10 = *(_QWORD *)a1;
  v11 = v9 & (*(_QWORD *)(a2 + 24) >> 3);
  v12 = (__m128i *)(*(_QWORD *)a1 + 16LL * v11);
  if ( v12->m128i_i64[0] )
  {
    v13 = _mm_loadu_si128(v12);
    v12->m128i_i64[0] = v7;
    v12->m128i_i64[1] = v8;
    do
    {
      v11 = v9 & (v11 + 1);
      v14 = (__m128i *)(v10 + 16LL * v11);
    }
    while ( v14->m128i_i64[0] );
    *v14 = v13;
  }
  else
  {
    v12->m128i_i64[0] = v7;
    v12->m128i_i64[1] = v8;
  }
  v15 = *(_DWORD *)(a1 + 12) + 1;
  *(_DWORD *)(a1 + 12) = v15;
  if ( 2 * v15 > v9 )
    sub_7704A0(a1);
  v16 = *(__m128i **)(a2 + 24);
  v17 = (const __m128i *)v16[-1].m128i_i64[1];
  if ( (*(_BYTE *)(a2 + 8) & 0x40) != 0 )
  {
    v19 = sub_73C570(v17, 1);
    v16 = *(__m128i **)(a2 + 24);
    v17 = v19;
  }
  return sub_77D750(a1, v16, (__int64)v16, (__int64)v17, v8);
}
