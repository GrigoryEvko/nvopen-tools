// Function: sub_D2F780
// Address: 0xd2f780
//
bool __fastcall sub_D2F780(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 a8)
{
  __int64 *v8; // rdx
  __int64 *v9; // rax
  __int64 v10; // rdx
  __m128i *v11; // rax
  __m128i *v12; // rdx
  __int64 v13; // rax
  bool v14; // dl
  __int64 v15; // rax
  __int64 v17; // rdx
  unsigned __int64 v18; // rcx
  _QWORD *v19; // rax
  __m128i v20; // [rsp+0h] [rbp-30h] BYREF
  __int64 v21; // [rsp+10h] [rbp-20h]

  v8 = *(__int64 **)(a1 + 8);
  v21 = a8;
  v9 = *(__int64 **)a1;
  v10 = *v8;
  v20 = _mm_loadu_si128((const __m128i *)&a7);
  if ( !(unsigned __int8)sub_98CF40(a2, *v9, v10, 0) )
    return 0;
  if ( v20.m128i_i32[0] == 86 )
  {
    v11 = *(__m128i **)(a1 + 16);
    v12 = v11;
    if ( v20.m128i_i64[1] > (unsigned __int64)v11->m128i_i64[1] )
      v12 = &v20;
  }
  else
  {
    if ( v20.m128i_i32[0] != 90 )
      goto LABEL_10;
    v11 = *(__m128i **)(a1 + 24);
    v12 = v11;
    if ( v20.m128i_i64[1] > (unsigned __int64)v11->m128i_i64[1] )
      v12 = &v20;
  }
  *v11 = _mm_loadu_si128(v12);
  v11[1].m128i_i64[0] = v12[1].m128i_i64[0];
LABEL_10:
  v13 = *(_QWORD *)(a1 + 16);
  v14 = 0;
  if ( *(_DWORD *)v13 )
    v14 = *(_QWORD *)(v13 + 8) >= (unsigned __int64)(1LL << **(_BYTE **)(a1 + 40));
  **(_BYTE **)(a1 + 32) |= v14;
  if ( !**(_BYTE **)(a1 + 32) )
    return 0;
  v15 = *(_QWORD *)(a1 + 24);
  if ( !*(_DWORD *)v15 )
    return 0;
  v17 = *(_QWORD *)(a1 + 48);
  v18 = *(_QWORD *)(v15 + 8);
  v19 = *(_QWORD **)v17;
  if ( *(_DWORD *)(v17 + 8) > 0x40u )
    v19 = (_QWORD *)*v19;
  return v18 >= (unsigned __int64)v19;
}
