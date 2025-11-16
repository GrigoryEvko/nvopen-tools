// Function: sub_2134ED0
// Address: 0x2134ed0
//
__int64 *__fastcall sub_2134ED0(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v6; // rax
  __int64 v7; // rsi
  __m128i v8; // xmm0
  __m128i v9; // xmm1
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  _QWORD *v15; // r13
  __int128 v16; // rax
  unsigned int v18; // [rsp+Ch] [rbp-54h] BYREF
  __m128i v19; // [rsp+10h] [rbp-50h] BYREF
  __m128i v20; // [rsp+20h] [rbp-40h] BYREF
  __int64 v21; // [rsp+30h] [rbp-30h] BYREF
  int v22; // [rsp+38h] [rbp-28h]

  v6 = *(_QWORD *)(a2 + 32);
  v7 = *(_QWORD *)(a2 + 72);
  v8 = _mm_loadu_si128((const __m128i *)v6);
  v9 = _mm_loadu_si128((const __m128i *)(v6 + 40));
  v21 = v7;
  v10 = *(_QWORD *)(v6 + 80);
  v19 = v8;
  LODWORD(v10) = *(_DWORD *)(v10 + 84);
  v20 = v9;
  v18 = v10;
  if ( v7 )
    sub_1623A60((__int64)&v21, v7, 2);
  v22 = *(_DWORD *)(a2 + 64);
  sub_2133C10(a1, (unsigned __int64 *)&v19, (__int64)&v20, &v18, (__int64)&v21, v8, v9, a5);
  if ( v21 )
    sub_161E7C0((__int64)&v21, v21);
  if ( !v20.m128i_i64[0] )
    return (__int64 *)v19.m128i_i64[0];
  v15 = *(_QWORD **)(a1 + 8);
  *(_QWORD *)&v16 = sub_1D28D50(v15, v18, v11, v12, v13, v14);
  return sub_1D2E2F0(v15, (__int64 *)a2, v19.m128i_i64[0], v19.m128i_i64[1], v20.m128i_i64[0], v20.m128i_i64[1], v16);
}
