// Function: sub_2434600
// Address: 0x2434600
//
__m128i *__fastcall sub_2434600(
        __m128i *a1,
        __int64 *a2,
        __int64 *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 *a8)
{
  __int64 *v8; // r14
  __int64 v9; // r15
  __int64 *v10; // r12
  __int64 *v11; // r13
  __int64 v12; // rbx
  __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rsi
  __m128i v21; // xmm0

  if ( a2 != a3 )
  {
    v8 = (__int64 *)*((_QWORD *)&a7 + 1);
    v9 = a7;
    v10 = a2;
    v11 = a8;
    do
    {
      v12 = *v10;
      v13 = sub_AD64C0(*(_QWORD *)(v9 + 152), *v8, 0);
      v14 = v12 - 32LL * (*(_DWORD *)(v12 + 4) & 0x7FFFFFF);
      if ( *(_QWORD *)v14 )
      {
        v15 = *(_QWORD *)(v14 + 8);
        **(_QWORD **)(v14 + 16) = v15;
        if ( v15 )
          *(_QWORD *)(v15 + 16) = *(_QWORD *)(v14 + 16);
      }
      *(_QWORD *)v14 = v13;
      if ( v13 )
      {
        v16 = *(_QWORD *)(v13 + 16);
        *(_QWORD *)(v14 + 8) = v16;
        if ( v16 )
          *(_QWORD *)(v16 + 16) = v14 + 8;
        *(_QWORD *)(v14 + 16) = v13 + 16;
        *(_QWORD *)(v13 + 16) = v14;
      }
      v17 = *v11;
      v18 = v12 + 32 * (1LL - (*(_DWORD *)(v12 + 4) & 0x7FFFFFF));
      if ( *(_QWORD *)v18 )
      {
        v19 = *(_QWORD *)(v18 + 8);
        **(_QWORD **)(v18 + 16) = v19;
        if ( v19 )
          *(_QWORD *)(v19 + 16) = *(_QWORD *)(v18 + 16);
      }
      *(_QWORD *)v18 = v17;
      if ( v17 )
      {
        v20 = *(_QWORD *)(v17 + 16);
        *(_QWORD *)(v18 + 8) = v20;
        if ( v20 )
          *(_QWORD *)(v20 + 16) = v18 + 8;
        *(_QWORD *)(v18 + 16) = v17 + 16;
        *(_QWORD *)(v17 + 16) = v18;
      }
      ++v10;
    }
    while ( a3 != v10 );
  }
  v21 = _mm_loadu_si128((const __m128i *)&a7);
  a1[1].m128i_i64[0] = (__int64)a8;
  *a1 = v21;
  return a1;
}
