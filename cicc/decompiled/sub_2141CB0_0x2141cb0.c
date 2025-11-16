// Function: sub_2141CB0
// Address: 0x2141cb0
//
unsigned __int64 __fastcall sub_2141CB0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        _QWORD *a4,
        double a5,
        __m128i a6,
        __m128i a7)
{
  __int64 v8; // rsi
  __m128i *v9; // rsi
  __int64 v10; // rbx
  unsigned int v11; // r13d
  __m128i v12; // xmm0
  unsigned __int8 *v13; // rax
  __int64 v14; // rax
  int v15; // eax
  unsigned __int64 result; // rax
  __int32 v17; // edx
  __int64 v20; // [rsp+20h] [rbp-70h] BYREF
  int v21; // [rsp+28h] [rbp-68h]
  __m128i v22; // [rsp+30h] [rbp-60h] BYREF
  __int64 v23[10]; // [rsp+40h] [rbp-50h] BYREF

  v8 = *(_QWORD *)(a2 + 72);
  v20 = v8;
  if ( v8 )
    sub_1623A60((__int64)&v20, v8, 2);
  v9 = *(__m128i **)a1;
  v21 = *(_DWORD *)(a2 + 64);
  v10 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL);
  v11 = **(unsigned __int8 **)(a2 + 40);
  v12 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 32));
  v22 = v12;
  v13 = (unsigned __int8 *)(*(_QWORD *)(v12.m128i_i64[0] + 40) + 16LL * v12.m128i_u32[2]);
  sub_1F40D10((__int64)v23, (__int64)v9, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), *v13, *((_QWORD *)v13 + 1));
  if ( LOBYTE(v23[0]) == 8 )
  {
    v22.m128i_i64[0] = sub_2125740(a1, v22.m128i_u64[0], v22.m128i_i64[1]);
    v22.m128i_i32[2] = v17;
  }
  v14 = *(_QWORD *)(v22.m128i_i64[0] + 40) + 16LL * v22.m128i_u32[2];
  v15 = sub_1F40000(*(_BYTE *)v14, *(_QWORD *)(v14 + 8), v11);
  sub_20BE530(
    (__int64)v23,
    *(__m128i **)a1,
    *(_QWORD *)(a1 + 8),
    v15,
    v11,
    v10,
    v12,
    a6,
    a7,
    (__int64)&v22,
    1u,
    1u,
    (__int64)&v20,
    0,
    1);
  result = sub_200E870(a1, v23[0], v23[1], a3, a4, v12, *(double *)a6.m128i_i64, a7);
  if ( v20 )
    return sub_161E7C0((__int64)&v20, v20);
  return result;
}
